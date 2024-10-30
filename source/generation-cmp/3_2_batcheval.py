import json, re, random
from config import gpt4o, RESPONDED_DATA, DIR, BATCH_RES, CRITERIA
from model_api import Model_API
from tqdm import trange, tqdm
from typing import *
from concurrent.futures import as_completed, ThreadPoolExecutor


def get_score(response: str, num: int):
    response = response.lower()
    s = response.split("scores:")[-1]

    try:
        matches = re.findall(r"\[[^\]]*\]", s)[0]
        s = matches[1:-1].split(",")
        s = [t.split(":")[-1].strip(" ") for t in s]
        for t in range(len(s)):
            try:
                _ = float(s[t])

            except:
                s[t] = "0"

        s = [float(t) for t in s]

    except:
        try:
            s = response.split("results:")[-1]
            scores = re.findall(r"sample\s*\d+:\s*([\d\.]+)", s)
            s = [float(t) for t in scores]

        except:
            s = []

    s = s[-num:]
    if len(s) != num and len(s) != 5:
        s = [-1.0 for _ in range(num)]

    return s

TEMP_DIR = 'batcheval-temp'
ITERS = 5
T = 0.2
BATCH_SIZE = 10

def generate_new_data(samples: List[Dict[str, Union[str, float]]], n_iter: int):
    samples.sort(key=lambda x: x['score'], reverse=True)
    last_batch_idx = random.sample(range(len(samples)), len(samples) % BATCH_SIZE)
    grouped = [samples[i] for i in range(len(samples)) if i not in last_batch_idx]

    batched = []
    bottle = len(grouped) // BATCH_SIZE
    for i in range(bottle):
        cur_batch = [grouped[bottle * j + i] for j in range(BATCH_SIZE)]
        random.shuffle(cur_batch)
        batched += [cur_batch]

    if last_batch_idx:
        batched += [[samples[t] for t in last_batch_idx]]

    with open(f'{TEMP_DIR}/{n_iter}.json', 'w') as fout:
        json.dump(batched, fout, indent=4)

#! varies per task
system_prompt = "You are an expert software developer."
template = f"""
You will be given a batch of {BATCH_SIZE} samples.
Each sample contains a requirement, and two code snippets implementing the requirement.
You are to compare the quality of both snippets, and decide which one is noticeably better or declare a tie because their difference in quality is insignificant.
Note that the developer may provide explanations or comments around the code, which should not affect your judgment of the code.

Code Snippets to be evaluated:
{{}}

You should analyze the code based on the following aspects:
{CRITERIA}

To fill the evaluation form, start with "I will do my best to provide individual analysis for each sample. Analysis:" to analyze the given samples regarding the evaluation criteria as concise as possible, without giving your scores during this step.
After analyzing all the samples, please give all the overall comparison results in order following the template "Results: [Sample1:result of Sample1,...,Sample10:result of Sample10]".

Use an integer to represent the comparison result, where 2 means the first response is better, 1 means a tie, and 0 means the second response is better.

Evaluation Form
"""

#! varies per task
template_per_sample = """
Sample{}:

## Requirement:
```
{}
```

## First Implementation:
```
{}
```

## Second Implementation:
```
{}
```
"""

def solve(batch: List[Dict[str, Union[str, float]]], bar: tqdm):
    Model = Model_API(system_prompt, gpt4o)
    contexts = ''
    for i, sample in enumerate(batch, start=1):
        #! params vary per task
        contexts += template_per_sample.format(i, sample['input'], sample['out1'], sample['out2'])

    try:
        scores = get_score(Model.generate(template.format(contexts), T=T)[0], len(batch))

    except Exception as e:
        print(f'Failed: ({type(e)}) {e}')
        scores = [-1.0 for _ in range(len(batch))]

    samples = []
    for sample, score in zip(batch, scores):
        sample['score'] = score
        samples += [sample]

    bar.update()
    return samples

def main():
    # 1. convert from dataset format to batcheval format
    samples = []
    with open(f'{DIR}/{RESPONDED_DATA[0]}') as fin:
        index, lines = 0, list(fin)

        for i in range(0, len(lines), 3):
            samples_3 = [json.loads(line) for line in lines[i:i + 3]]
            for j in range(3):
                for k in range(3):
                    if j != k:
                        #! params vary per task
                        inp, out1, out2 = samples_3[0]['input'], samples_3[j]['output'], samples_3[k]['output']
                        samples += [{'input': inp, 'out1': out1, 'out2': out2, 'score': 0.0, 'index': index}]
                        index += 1

    generate_new_data(samples, 0)

    # 2. batcheval
    for n_iter in trange(ITERS, desc='Iteration'):
        with open(f'{TEMP_DIR}/{n_iter}.json') as fin:
            batches = json.load(fin)

        with tqdm(total=len(batches), desc='>>> Progress') as bar, ThreadPoolExecutor(max_workers=40) as ex:
            futures = as_completed([ex.submit(solve, batch, bar) for batch in batches])

        samples = []
        for future in futures:
            samples += future.result()

        generate_new_data(samples, n_iter + 1)

    # 3. convert from batcheval format to dataset format
    with open(f'{TEMP_DIR}/{ITERS}.json') as fin:
        batches = json.load(fin)

    samples = [sample for batch in batches for sample in batch]
    with open(f'{DIR}/{BATCH_RES}', 'w') as fout:
        for sample in sorted(samples, key=lambda x: x['index']):
            print(sample['score'] - 1, file=fout)

if __name__ == "__main__":
    main()
