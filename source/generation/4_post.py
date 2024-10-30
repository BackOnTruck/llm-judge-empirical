# This file is used to process any unexpected scoring format from the LLMs.
# Please edit this as you need.
import json, re, os
from config import SFT_RES, DIR

rules = [
    r'Overall:\s*(\d+(?:\.\d+)?)',
    r'Overall Score:\s*(\d+(?:\.\d+)?)',
    r'Score:\s*(\d+(?:\.\d+)?)',
    r'Rating:\s*(\d+(?:\.\d+)?)',
    r'\[RESULT\]\s*(\d+(?:\.\d+)?)'
    r'\[\[(\d+(?:\.\d+)?)\]\]',
    r'\[(\d+(?:\.\d+)?)\]',
    r'(\d+(?:\.\d+)?)'
]

def main():
    for file in SFT_RES.values():
        in_file = f'{DIR}/{file}.log'
        if not os.path.isfile(in_file):
            print(f'>>> {in_file}: skipped')
            continue

        with open(in_file) as fin, open(f'{DIR}/{file}', 'w') as fout:
            for line in fin:
                js = json.loads(line)
                if js['score'] == -1:
                    reply = js['judgment'].replace('*', '')
                    for rule in rules:
                        if matches := re.findall(rule, reply):
                            break

                    # you can add more patterns here as you see fit
                    s = float(matches[-1]) if matches else -1.0
                    if s > 10:
                        print(f'{file}: invalid score {s:.2f}; replacing with -1.0')
                        s = -1.0

                    js['score'] = s

                print(js['score'], file=fout)

if __name__ == '__main__':
    main()
