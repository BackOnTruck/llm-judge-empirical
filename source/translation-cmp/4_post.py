# This file is used to process any unexpected scoring format from the LLMs.
# Please edit this as you need.
import json, re, os
from config import SFT_RES, DIR

rules = [
    r'Overall:\s*(FIRST|SECOND|TIE|Response 1|Response 2|A|B)',
    r'Overall Score:\s*(FIRST|SECOND|TIE|Response 1|Response 2|A|B)',
    r'Score:\s*(FIRST|SECOND|TIE|Response 1|Response 2|A|B)',
    r'Rating:\s*(FIRST|SECOND|TIE|Response 1|Response 2|A|B)',
    r'\[RESULT\]\s*(FIRST|SECOND|TIE|Response 1|Response 2|A|B)',
    r'\[\[(FIRST|SECOND|TIE|Response 1|Response 2|A|B)\]\]',
    r'\[(FIRST|SECOND|TIE|Response 1|Response 2|A|B)\]',
    r'final decision is (FIRST|SECOND|TIE|Response 1|Response 2|A|B)',
    r'(TIE|Response 1|Response 2)',
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
                    reply = js['judgment'].replace('*', '').replace('`', '')
                    for rule in rules:
                        if matches := re.findall(rule.lower(), reply.lower()):
                            break

                    # you can add more patterns here as you see fit
                    final = matches[-1] if matches else 'tie'
                    js['score'] = 0 if final == 'tie' else 1 if final in ['first', 'response 1', 'a'] else -1

                print(js['score'], file=fout)

if __name__ == '__main__':
    main()
