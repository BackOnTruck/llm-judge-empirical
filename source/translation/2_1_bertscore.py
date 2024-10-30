# modified from https://github.com/Tiiiger/bert_score/blob/master/bert_score/scorer.py
from collections import defaultdict
import torch
from transformers import AutoTokenizer

from bertscore_utils import bert_cos_score_idf, get_idf_dict, get_model

def bertscore(
    cands,
    refs,
    model_type,
    idf,
    device=None,
    batch_size=128,
    nthreads=4
):
    """
    BERTScore metric.

    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str or list of list of str): reference sentences
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
        - :param: `device` (str): on which the contextual embedding model will be allocated on.
                  If this argument is None, the model lives on cuda:0 if cuda is available.
        - :param: `nthreads` (int): number of threads
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`.

    Return:
        - :param: Tensor of shape (N); N = number of input candidate reference pairs.
    """
    assert len(cands) == len(refs), "Different number of candidates and references"

    ref_group_boundaries = None
    if not isinstance(refs[0], str):
        ref_group_boundaries = []
        ori_cands, ori_refs = cands, refs
        cands, refs = [], []
        count = 0
        for cand, ref_group in zip(ori_cands, ori_refs):
            cands += [cand] * len(ref_group)
            refs += ref_group
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)

    print(">>> Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = get_model(model_type)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(">>> Loading IDF...")
    if not idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    elif isinstance(idf, dict):
        idf_dict = idf
    else:
        idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)

    print(">>> Computing scores...")
    all_preds = bert_cos_score_idf(
        model,
        refs,
        cands,
        tokenizer,
        idf_dict,
        verbose=False,
        device=device,
        batch_size=batch_size
    ).cpu()

    if ref_group_boundaries is not None:
        max_preds = []
        for beg, end in ref_group_boundaries:
            max_preds.append(all_preds[beg:end].max(dim=0)[0])
        all_preds = torch.stack(max_preds, dim=0)

    return all_preds[..., 2].tolist()

# end of bertscore.py

import json
from config import BS_RES, DIR, RESPONDED_DATA

def main():
    with open(f'{DIR}/{RESPONDED_DATA[0]}') as fin, open(f'{DIR}/{BS_RES}', 'w') as fout:
        refs, preds = [], []
        for line in fin:
            js = json.loads(line)
            refs += [js['gold']]
            preds += [js['output']]

        for score in bertscore(preds, refs, 'model/unixcoder-base', True):
            print(score, file=fout)


if __name__ == '__main__':
    main()
