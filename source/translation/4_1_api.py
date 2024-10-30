import json, re
from model_api import Model_API
from tqdm import tqdm
from config import deepseek_25, gpt4o, DIR, RESPONDED_DATA, SFT_RES
from concurrent.futures import as_completed, ThreadPoolExecutor
from vanilla_prompt import system_prompt, template


def solve(index: int, query: str, bar: tqdm, llm: str):
    model = Model_API(system_prompt, llm)
    try:
        reply = model.generate(query)[0]
        tqdm.write(f">>> Success: {len(reply)} characters")

    except Exception as e:
        reply = ''
        tqdm.write(f'>>> Failed: ({type(e)}) {e}')

    matches = re.findall(r'Overall:\s*([0-5](?:\.\d+)?)\/5', reply)
    if not matches:
        matches = re.findall(r'Overall Score:\s*([0-5](?:\.\d+)?)\/5', reply)

    if not matches:
        matches = re.findall(r'Score:\s*([0-5](?:\.\d+)?)\/5', reply)

    score = float(matches[-1]) if matches else -1.0

    bar.update()
    return index, reply, score

def main():
    prompts, instances = [], []
    with open(f'{DIR}/{RESPONDED_DATA[0]}') as fin:
        for line in fin:
            js = json.loads(line)

            #! varies per task
            src_lang, tgt_lang = js['lang'].split(', ')
            prompts += [template.format(src_lang, tgt_lang, src_lang, js['input'], tgt_lang, js['output'])]
            instances += [js]

    for llm in (deepseek_25, gpt4o):
        with tqdm(total=len(prompts), desc='>>> Progress') as bar, ThreadPoolExecutor(max_workers=50) as ex:
            futures = as_completed([ex.submit(solve, index, prompt, bar, llm) for index, prompt in enumerate(prompts)])

        with open(f'{DIR}/{SFT_RES[llm]}.log', 'w') as fout:
            for (_, reply, score), instance in zip(sorted([future.result() for future in futures]), instances):
                instance['judgment'] = reply
                instance['score'] = score
                print(json.dumps(instance), file=fout)


if __name__ == '__main__':
    main()
