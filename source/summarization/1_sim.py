import json
from config import RESPONDED_DATA, SIM_RES, DIR, GPT4O_TK, ORIGINAL_DATA
from tqdm import tqdm, trange
from typing import *
import tiktoken
from crystalbleu import corpus_bleu
from nltk.util import ngrams

# evalute.load() tries to download scripts every time it launches; we manually retrieve the scripts instead
from metrics.bleu.bleu import Bleu
from metrics.rouge.rouge import Rouge
from metrics.meteor.meteor import Meteor
from metrics.chrf.chrf import ChrF

print(">>> Loading evaluators...")
tk = tiktoken.get_encoding(GPT4O_TK)

def bleu(preds: List[str], refs: List[str]):
    func = Bleu()._compute
    return [func(predictions=[pred], references=[[ref]], tokenizer=tk.encode)['bleu'] * 100 for (pred, ref) in tqdm(zip(preds, refs), desc='>>> BLEU')]

def rouge_l(preds: List[str], refs: List[str]):
    func = Rouge()._compute
    return [func(predictions=[pred], references=[[ref]], tokenizer=tk.encode)['rougeL'] * 100 for (pred, ref) in tqdm(zip(preds, refs), desc='>>> ROUGE-L')]

def meteor(preds: List[str], refs: List[str]):
    func = Meteor()._compute
    return [func(predictions=[pred], references=[[ref]])['meteor'] * 100 for (pred, ref) in tqdm(zip(preds, refs), desc='>>> METEOR')]

def chrf_pp(preds: List[str], refs: List[str]):
    func = ChrF()._compute
    return [func(predictions=[pred], references=[[ref]], word_order=2)['score'] for (pred, ref) in tqdm(zip(preds, refs), desc='>>> ChrF++')]

def crystalbleu(preds: List[str], refs: List[str]):
    corpus = []
    with open(f'{DIR}/{ORIGINAL_DATA[0]}') as fin:
        for line in tqdm(fin, desc='>>> Loading corpus for CrystalBLEU'):
            js = json.loads(line)
            corpus += tk.encode(js['input']) + tk.encode(js['gold'])

    all_ngrams = []
    for n in trange(1, 5, desc='>>> Collecting n-grams'):
        all_ngrams += list(ngrams(corpus, n))

    trival_ngrams = dict(Counter(all_ngrams).most_common(500))
    return [corpus_bleu([[tk.encode(ref)]], [tk.encode(pred)], ignoring=trival_ngrams) * 100 for (pred, ref) in tqdm(zip(preds, refs), desc='>>> CrystalBLEU')]


metrics = {
    'bleu': bleu,
    'rouge-l': rouge_l,
    'meteor': meteor,
    'chrf++': chrf_pp,
    'crystalbleu': crystalbleu
}

def main():
    with open(f'{DIR}/{RESPONDED_DATA[0]}') as fin:
        refs, preds = [], []
        for line in fin:
            js = json.loads(line)
            refs += [js['gold']]
            preds += [js['output']]

        for metric, func in metrics.items():
            with open(f'{DIR}/{SIM_RES[metric]}', 'w') as fout:
                for score in func(preds, refs):
                    print(score, file=fout)

if __name__ == '__main__':
    main()
