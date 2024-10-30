from tiktoken import get_encoding
import json
from typing import *
from tqdm import tqdm
from random import random
from config import minlen_context, maxlen_context, maxlen_resp, DIR, ORIGINAL_DATA, FILTERED_DATA, GPT4O_TK

tk = get_encoding(GPT4O_TK)
allowed_lang = ['Python', 'Java', 'C++', 'C']
def check(lang: str, code: str, maxlen: int):
    return lang in allowed_lang and minlen_context < len(tk.encode(code)) < maxlen

# unused for now; might be useful to prepend a line comment describing the code's purpose
def add_desc(lang: str, code: str, desc: str):
    return f'// {desc}\n{code}' if lang in ['Java', 'C', 'C++'] else f'# {desc}\n{code}'

class EvalDataset:
    def __init__(self, file: str):
        self.data = []
        with open(file) as fin:
            for line in (bar := tqdm(fin)):
                js: dict = json.loads(line)

                description = js['name'] # we do not use descriptions for now
                del js['name'], js['id']

                (src_lang, src_code), (tgt_lang, tgt_code) = js.items()
                if random() < 0.5:
                    # 50% chance to swap source and target
                    src_lang, tgt_lang = tgt_lang, src_lang
                    src_code, tgt_code = tgt_code, src_code

                if check(src_lang, src_code, maxlen_context) and check(tgt_lang, tgt_code, maxlen_resp):
                    self.data += [{'lang': f'{src_lang}, {tgt_lang}', 'input': src_code.strip(), 'gold': tgt_code.strip()}]
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
