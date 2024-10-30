# summary-level: faithfulness rating
import json
import numpy as np
from config import FFLM_EMBED, FFLM_RES, DIR
from typing import *


def score_calculation(content: Dict[str, List[float]]):
    s2s, s2s_doc, lm, lm_doc, prefix = map(np.array, [
        content["s2s_logp"],
        content["s2s_doc_logp"],
        content["lm_logp"],
        content["lm_doc_logp"],
        content["prefix_logp"]
    ])

    # Q = log P => exp Q = P; -Q = -log P = loss
    logp = [s2s, s2s_doc, lm, lm_doc, prefix]
    return *map(lambda x: np.exp(x), logp), *map(lambda x: -x, logp)


def main():
    with open(f'{DIR}/{FFLM_EMBED}') as fin:
        fflm = []
        for line in fin:
            s2s, s2s_doc, lm, lm_doc, prefix, s2s_loss, s2s_loss_doc, lm_loss, lm_loss_doc, prefix_loss = score_calculation(json.loads(line))

            #! these are inconsistent with the paper: np.exp(1 - s2s) (code) vs. np.exp(s2s) (paper)
            score_1 = np.mean((lm_loss - s2s_loss) * np.exp(1 - s2s))
            score_2 = np.mean((prefix_loss - s2s_loss) * np.exp(1 - s2s))
            score_3 = np.mean((lm_loss_doc - s2s_loss_doc) * np.exp(1 - s2s_doc))

            # harim.append(np.mean(-(1-s2s)*(1-(s2s-lm))))
            # cop.append(np.mean(prefix_loss)-np.mean(s2s_loss))
            fflm.append(0.25 * score_1 + 0.5 * score_2 + 0.25 * score_3)

    with open(f'{DIR}/{FFLM_RES}', 'w') as fout:
        for score in fflm:
            print(score, file=fout)


if __name__ == "__main__":
    main()
