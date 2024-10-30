import json
from typing import *
from tqdm import tqdm
from config import DIR, FFLM_EMBED, RESPONDED_DATA, davinci
from concurrent.futures import as_completed, ThreadPoolExecutor
from fake_concurrent import SingleThreadExecutor
from model_api_logprob import LogProbModel_API, token_probs

#! varies per task
separator = " The corresponding code in another language: "

def prepare_input(target: str, source: str = ""):
    # since Davinci-002 cannot generate logProb for the first token, we always add a separator
    return source + separator + target, len(source) + len(separator)


def invoke(source: str, target: str, index: int, bar: tqdm):
    inputs, start = prepare_input(target, source)
    prior_inputs, prior_start = prepare_input(target)
    prefix_inputs, prefix_start = prepare_input(target, target + " " + source)

    inputs_doc, start_doc = prepare_input(source, target)
    prior_inputs_doc, prior_start_doc = prepare_input(source)

    s2s, lm, prefix, s2s_doc, lm_doc = map(
        LogProbModel_API(davinci).do_inference,
        [inputs, prior_inputs, prefix_inputs, inputs_doc, prior_inputs_doc],
    )

    bar.update()
    return index, [token_probs(*s2s, start), token_probs(*lm, prior_start), token_probs(*prefix, prefix_start), token_probs(*s2s_doc, start_doc), token_probs(*lm_doc, prior_start_doc)]


def compute(sources: List[str], targets: List[str]):
    s2s_logp, lm_logp, prefix_logp, s2s_doc_logp, lm_doc_logp = [], [], [], [], []

    with tqdm(total=len(sources), desc='>>> Computing FFLM score') as bar, ThreadPoolExecutor(max_workers=20) as ex:
        futures = as_completed([ex.submit(invoke, source, target, index, bar) for index, (source, target) in enumerate(zip(sources, targets))])

    for _, items in sorted([future.result() for future in futures]):
        for logp, item in zip([s2s_logp, lm_logp, prefix_logp, s2s_doc_logp, lm_doc_logp], items):
            logp += [item]

    return s2s_logp, lm_logp, prefix_logp, s2s_doc_logp, lm_doc_logp # each is a list of lists of log token probabilities


def main():
    sources, targets = [], []
    with open(f"{DIR}/{RESPONDED_DATA[0]}") as fin:
        for line in fin:
            js = json.loads(line)
            sources += ["```" + js["input"] + "```"]
            targets += ["```" + js["output"] + "```"]

    s2s_logp, lm_logp, prefix_logp, s2s_doc_logp, lm_doc_logp = compute(sources, targets)
    with open(f"{DIR}/{FFLM_EMBED}", "w") as fout:
        for s2s, lm, prefix, s2s_doc, lm_doc in zip(s2s_logp, lm_logp, prefix_logp, s2s_doc_logp, lm_doc_logp):
            print(json.dumps({
                "s2s_logp": s2s,
                "lm_logp": lm,
                "prefix_logp": prefix,
                "s2s_doc_logp": s2s_doc,
                "lm_doc_logp": lm_doc,
            }), file=fout)


if __name__ == "__main__":
    main()
