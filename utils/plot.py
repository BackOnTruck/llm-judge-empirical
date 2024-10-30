from scipy.stats import gaussian_kde, spearmanr, pearsonr, kendalltau
import numpy as np
import matplotlib.pyplot as plt
from config import *
from typing import *

ALL = [HUMAN_RES, *SIM_RES.values(), BS_RES, MS_RES, GS_RES, FFLM_RES, GEVAL_RES, BATCH_RES, *SFT_RES.values()]
names = [
    'Human',
    'BLEU', 'ROUGE-L', 'METEOR', 'ChrF++', 'CrystalBLEU',
    'BERTScore', 'MoverScore', 'GPTScore', 'FFLM',
    'G-Eval', 'BatchEval',
    'Llama2', 'Auto-J', 'Mixtral', 'Prometheus', 'DSC2-Lite', 'DS2.5', 'GPT-4o'
]
tasks = ['codetransocean', 'complexcodeeval', 'codexglue']

def plot_distrib(scores: Tuple[List[float], List[float], List[float]], name: str):
    min_score, max_score = np.min(scores), np.max(scores)
    for task_scores in scores:
        for i in range(len(task_scores)):
            task_scores[i] = (task_scores[i] - min_score) / (max_score - min_score)

    assert np.min(scores) == 0 and np.max(scores) == 1, str(scores)

    means = [np.mean(score) for score in scores]
    variances = [np.var(score) for score in scores]
    x_values = np.linspace(0, 1, 200)
    trans, gen, summ = scores

    plt.figure(figsize=(7, 5))
    plt.plot(x_values, gaussian_kde(trans)(x_values), color='blue', linestyle='-', label='Translation')
    plt.plot(x_values, gaussian_kde(gen)(x_values), color='green', linestyle='--', label='Generation')
    plt.plot(x_values, gaussian_kde(summ)(x_values), color='red', linestyle=':', label='Summarization')

    def to_str(x: List[float]):
        return ', '.join([f'{value:.3f}' for value in x])

    plt.xlabel('Score', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title(f'μ = {to_str(means)}\nσ² = {to_str(variances)}', fontsize=18) # we don't need a title for now
    plt.legend()
    plt.show()
    plt.savefig(f'utils/image/{name}.png', transparent=True)

    return variances

def main():
    rho, R, tau, variances = [], [], [], []
    for metric, name in zip(ALL, names):
        task_scores = [[float(line) for line in open(f'data/_scoring/{task}/{metric}')] for task in tasks]

        variance = plot_distrib(task_scores, name)
        if metric != HUMAN_RES:
            for s, var, ref in zip(task_scores, variance, human_scores):
                rho += [spearmanr(s, ref).statistic]
                R += [pearsonr(s, ref).statistic]
                tau += [kendalltau(s, ref).statistic]
                variances += [var]

        else:
            human_scores = task_scores

    for coef in (rho, R, tau):
        print(f'𝜌 = {spearmanr(coef, variances).statistic * 100:.2f}, R = {pearsonr(coef, variances).statistic * 100:.2f}, 𝜏 = {kendalltau(coef, variances).statistic * 100:.2f}')

if __name__ == '__main__':
    main()
