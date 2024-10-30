from tiktoken import get_encoding
import json
from typing import *
from tqdm import tqdm
from config import minlen_context, maxlen_context, maxlen_resp, DIR, ORIGINAL_DATA, FILTERED_DATA, GPT4O_TK

tk = get_encoding(GPT4O_TK)
allowed_lang = ['Python', 'Java', 'C++', 'C']
def check(lang: str, code: str, maxlen: int):
    return lang in allowed_lang and minlen_context < len(tk.encode(code)) < maxlen

class EvalDataset:
    def __init__(self, file: str):
        self.data = []
        with open(file) as fin:
            for line in (bar := tqdm(fin)):
                js: dict = json.loads(line)
                lang, text, code = js['lang'], js['input'], js['gold']

                if check(lang, text, maxlen_context) and check(lang, code, maxlen_resp):
                    self.data += [js]
                    bar.set_description(f'>>> {len(self.data)} samples loaded')

    def save(self, file: str):
        with open(file, 'w') as fout:
            for item in self.data:
                print(json.dumps(item), file=fout)

if __name__ == '__main__':
    for original_data, filtered_data in zip(ORIGINAL_DATA, FILTERED_DATA):
        tqdm.write(f'>>> Processing {original_data}...')
        dataset = EvalDataset(f'{DIR}/{original_data}')
        dataset.save(f'{DIR}/{filtered_data}')
