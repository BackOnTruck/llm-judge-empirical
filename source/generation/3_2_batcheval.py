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
            s = response.split("scores:")[-1]
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
Each sample contains a requirement, and a code snippet implementing the requirement.
You are to evaluate the quality of the code.
Note that the developer may provide explanations or comments around the code, which should not affect your judgment of the code.

Code Snippets to be evaluated:
{{}}

You should analyze the code based on the following aspects:
{CRITERIA}

To fill the evaluation form, start with "I will do my best to provide individual analysis for each sample. Analysis:" to analyze the given samples regarding the evaluation criteria as concise as possible, without giving your scores during this step.
After analyzing all the samples, please give all the float overall scores in order following the template "Float Scores: [Sample1:score of Sample1,...,Sample10:score of Sample10]".

Evaluation Form
"""

#! varies per task
template_per_sample = """
Sample{}:

## Requirement:
```
{}
```

## Implementation:
```
{}
```
"""

def solve(batch: List[Dict[str, Union[str, float]]], bar: tqdm):
    Model = Model_API(system_prompt, gpt4o)
    contexts = ''
    for i, sample in enumerate(batch, start=1):
        #! params vary per task
        contexts += template_per_sample.format(i, sample['input'], sample['output'])

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
        for line in fin:
            sample = json.loads(line)
            sample['score'] = 0.0
            samples += [sample]

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
            print(sample['score'], file=fout)

if __name__ == "__main__":
    main()
