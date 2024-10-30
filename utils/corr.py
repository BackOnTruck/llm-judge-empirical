from config import *
from scipy.stats import spearmanr, pearsonr, kendalltau
from collections import Counter
from random import uniform, choice

ALL = [HUMAN_RES, *SIM_RES.values(), BS_RES, MS_RES, GS_RES, FFLM_RES, GEVAL_RES, BATCH_RES, *SFT_RES.values()]
CMP_ALL = [HUMAN_RES, GEVAL_RES, BATCH_RES, *SFT_RES.values()]
TIE_THRESHOLD = [0.5] + [0.7 for _ in ALL[1:]]
tasks = ['codetransocean', 'complexcodeeval', 'codexglue']

def stats(f: Callable[[List[float], List[float]], Any], x: List[float], y: List[float]) -> float:
    return f(x, y).statistic * 100

def cmp_stats(score: float, threshold: float):
    if -threshold < score < threshold:
        return 0.0

    return 1.0 if score > 0 else -1.0

def scoring(task: str, all_metrics: List[str]):
    scores = {}
    print(f'\n--- {task}: scoring ---')

    for key in all_metrics:
        with open(f'data/_scoring/{task}/{key}') as fin: # change
            scores[key] = [float(line) for line in fin]

    for key in all_metrics[1:]:
        current, reference = scores[key], scores[all_metrics[0]]
        print(f'{key[9:-6]:12}: Spearman = {stats(spearmanr, current, reference):6.2f}, Pearson = {stats(pearsonr, current, reference):6.2f}, Kendall = {stats(kendalltau, current, reference):6.2f}')

    print(f'\n--- {task}: score-based comparison ---')
    for key in all_metrics[1:]:
        current, reference = scores[key], scores[all_metrics[0]]

        all_cmp = []
        for i in range(0, len(current), 3):
            cur, ref = current[i:i+3], reference[i:i+3]
            for j in range(3):
                for k in range(3):
                    if j != k:
                        all_cmp += [[cur[j] - cur[k], cmp_stats(ref[j] - ref[k], TIE_THRESHOLD[0])]]

        all_cmp.sort(key=lambda x: abs(x[0]))
        ans = 0.0
        for threshold, _ in all_cmp:
            threshold = abs(threshold)
            correct = 0

            for cur, ref in all_cmp:
                verdict = cmp_stats(cur, threshold)
                correct += verdict == ref

            ans = max(ans, correct / len(all_cmp))

        print(f'{key[9:-6]:12}: Acc = {ans * 100:.2f}')

def cmp_work(predictions: List[float], references: List[float], orig_predictions: List[float]):
    correct = acc_total = agree = agr_total = agree_score = 0
    for i in range(0, len(predictions), 6):
        pred, ref, pred_orig = predictions[i:i + 6], references[i: i + 6], orig_predictions[i: i + 6]
        # response pair: 0: (0, 1), 1: (0, 2), 2: (1, 0), 3: (1, 2), 4: (2, 0), 5: (2, 1)
        # judgment pair for agreement: (0, 2), (1, 4), (3, 5)
        for j in range(6):
            acc_total += 1
            correct += pred[j] == ref[j]

        for j, k in [(0, 2), (1, 4), (3, 5)]:
            agr_total += 1
            agree += pred[j] == pred[k]
            agree_score += 1 - abs(pred_orig[j] - pred_orig[k]) / 2 # pred for integer difference; pred_orig for floating-point difference

    return correct / acc_total, agree / agr_total, agree_score / agr_total


def comparing(task: str, all_metrics: List[str]):
    scores, orig_scores = {}, {}
    print(f'\n--- {task}: comparison ---')

    for key, threshold in zip(all_metrics, TIE_THRESHOLD):
        with open(f'data/_comparison/{task}/{key}') as fin:
            orig_scores[key] = [float(line) for line in fin]
            scores[key] = [cmp_stats(score, threshold) for score in orig_scores[key]]
            print(f"Stats of {key[9:-6]:12}: {sorted(list(Counter(scores[key]).items()))}")

    for key in all_metrics[1:]:
        correct, agree, agree_score = cmp_work(scores[key], scores[ALL[0]], orig_scores[key])
        print(f'{key[9:-6]:12}: Acc = {correct * 100:5.2f}, Agr = {agree * 100:5.2f}, Agr Score = {agree_score * 100:5.2f}')

def main():
    for task in tasks:
        scoring(task, ALL)

    for task in tasks:
        comparing(task, CMP_ALL)

    # pairwise comparison: random choice baseline with correlation
    for task in tasks:
        print(f'--- Random baseline for {task} ---')
        desc = ['Uniform', 'Choice']

        with open(f'data/_comparison/{task}/{HUMAN_RES}') as fin:
            h_scores = list(map(float, fin.readlines()))
            h_results = list(map(lambda x: cmp_stats(x, TIE_THRESHOLD[0]), h_scores))

        scores1 = [choice([-1.0, 0.0, 1.0]) for _ in h_scores]
        scores2 = [uniform(-1.0, 1.0) for _ in h_scores]
        scores2_results = list(map(lambda x: cmp_stats(x, 1 / 3), scores2))

        for s, (scores, verdicts) in zip(desc, [(scores1, scores1), (scores2, scores2_results)]):
            print(f'- {s:7}: Spearman = {stats(spearmanr, scores, h_scores):6.2f}, Pearson = {stats(pearsonr, scores, h_scores):6.2f}, Kendall = {stats(kendalltau, scores, h_scores):6.2f}')
            correct, agree, agree_score = cmp_work(verdicts, h_results, scores)
            print(f'- {s:7}: Acc = {correct * 100:5.2f}, Agr = {agree * 100:5.2f}, Agr Score = {agree_score * 100:5.2f}')


if __name__ == '__main__':
    main()
