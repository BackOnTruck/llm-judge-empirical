from config import *
from scipy.stats import spearmanr, pearsonr, kendalltau

N = 150
VANILLA = list(SFT_RES.values())
ALL = [[HUMAN_RES], [*SIM_RES.values()], [BS_RES, MS_RES], [GS_RES, FFLM_RES], [GEVAL_RES, BATCH_RES, *VANILLA[4:]], VANILLA[:4]]
names = ['human', 'sim', 'embed', 'prob', 'infer', 'sft']

def stats(f: Callable[[List[float], List[float]], Any], x: List[float], y: List[float]) -> float:
    return f(x, y).statistic * 100

tasks = ['codetransocean', 'complexcodeeval', 'codexglue']

for task in tasks:
    scores = {}
    print(f'\n--- {task} ---')

    for category in ALL:
        for metric in category:
            with open(f'data/_scoring/{task}/{metric}') as fin:
                scores[metric] = [float(fin.readline()) for _ in range(N)]

    for name, category in zip(names, ALL):
        if name in ('human', 'sim'):
            continue

        human = conv = other = inner = float('-inf')
        for metric in category:
            human = max(human, stats(spearmanr, scores[metric], scores[HUMAN_RES]))

            for conv_metric in ALL[1]:
                conv = max(conv, stats(spearmanr, scores[metric], scores[conv_metric]))

            for other_category in ALL[2:]:
                if category != other_category:
                    for other_metric in other_category:
                        other = max(other, stats(spearmanr, scores[metric], scores[other_metric]))

            for metric_2 in category:
                if metric != metric_2:
                    inner = max(inner, stats(spearmanr, scores[metric], scores[metric_2]))

        print(f'{name:7}: {human:6.2f} & {conv:6.2f} & {other:6.2f} & {inner:6.2f}')

    small, large, small_large = [], [], []
    s_llm = ['dsc2-lite', 'llama2', 'autoj', 'mixtral', 'prometheus']
    l_llm = ['dsc2.5', 'gpt4o', 'geval', 'batcheval']
    for idx, metric in enumerate(ALL[4] + ALL[5]):
        for metric_2 in ALL[4] + ALL[5]:
            name1, name2 = metric[9:-6], metric_2[9:-6]
            if name1 == name2:
                continue

            res = stats(spearmanr, scores[metric], scores[metric_2])
            if name1 in s_llm and name2 in s_llm:
                small += [res]

            elif name1 in l_llm and name2 in l_llm:
                large += [res]

            else:
                small_large += [res]

    print(f'\n- S+S & {min(small):6.2f} & {max(small):6.2f}')
    print(f'- L+L & {min(large):6.2f} & {max(large):6.2f}')
    print(f'- S+L & {min(small_large):6.2f} & {max(small_large):6.2f}')
