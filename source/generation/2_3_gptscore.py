import json
from config import davinci, RESPONDED_DATA, DIR, GS_RES
from typing import *
from tqdm import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor
from fake_concurrent import SingleThreadExecutor
from model_api_logprob import LogProbModel_API, token_probs

LIM = 10

#! varies per task
template = """Given the following requirement:
```
{}
```

Here is a readable and functionally correct implementation in {}:
```
{}
```

An alternative readable and functionally correct implementation in {} can be:
```
"""

def gptscore(input: str, output: str):
    logp = token_probs(*LogProbModel_API(davinci).do_inference(input + output + '```'), len(input))
    return sum(logp) / len(logp)

def obtain_score(lang: str, context: str, pred: str, ref: str, index: int, bar: tqdm):
    #! params vary per task
    ref_pred = gptscore(template.format(context, lang, ref, lang), pred) # ref->pred
    pred_ref = gptscore(template.format(context, lang, pred, lang), ref) # pred->ref

    avg_f = (ref_pred + pred_ref) / 2
    harm_f = 2 * ref_pred * pred_ref / (ref_pred + pred_ref)
    # we use harmonic mean of the two probabilities for now

    tqdm.write(f'>>> Success: Ref -> Pred {ref_pred:.3f} & Pred -> Ref {pred_ref:.3f}\nOverall: {harm_f:.3f}\n')
    bar.update()
    return index, harm_f

def main():
    contexts, preds, refs, langs = [], [], [], []
    with open(f'{DIR}/{RESPONDED_DATA[0]}') as fin:
        for line in fin:
            js = json.loads(line)
            langs += [js['lang']]
            contexts += [js['input']]
            preds += [js['output']]
            refs += [js['gold']]

    with tqdm(total=len(contexts), desc='>>> Progress') as bar, ThreadPoolExecutor(max_workers=20) as ex:
        futures = as_completed([ex.submit(obtain_score, lang, context, pred, ref, idx, bar) for idx, (lang, context, pred, ref) in enumerate(zip(langs, contexts, preds, refs))])

    with open(f'{DIR}/{GS_RES}', 'w') as fout:
        for _, score in sorted([future.result() for future in futures]):
            print(score, file=fout)


if __name__ == "__main__":
    main()
