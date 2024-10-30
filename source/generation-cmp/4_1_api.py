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

    bar.update()
    return index, reply, -1.0

def main():
    with open(f'{DIR}/{RESPONDED_DATA[0]}') as fin:
        prompts, instances = [], []
        lines = list(fin)

        for i in range(0, len(lines), 3):
            samples = [json.loads(line) for line in lines[i:i + 3]]
            for j in range(3):
                for k in range(3):
                    if j != k:
                        #! params vary per task
                        prompts += [template.format(samples[0]['input'], samples[j]['output'], samples[k]['output'])]
                        instances += [{}]

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
