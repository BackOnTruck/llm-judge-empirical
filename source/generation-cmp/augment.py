from model_api import Model_API
from config import SAMPLED_DATA, AUGMENTED_DATA, DIR, gpt4o
from concurrent.futures import as_completed, ThreadPoolExecutor
from tqdm import tqdm
from typing import *
import json

sys_prompt = 'You are a professional developer.'
template = """
Given the following requirement:
```
{}
```

And the following reference code:
```
{}
```

You are to refine the "Functional Dependencies" section in the requirement. For each dependency, please:
1. Retain the original dependency name as provided, instead of changing to method name of full qualified name.
2. Generate a signature describing its full qualified name and return type, along with the input parameters and their presumed type.
3. Provide a brief one-sentence description for its functionality.
4. If the original dependency name contains a local variable defined in the required function, clearly indicate this fact.

Now, print the entire refined requirement including the required function signature and requirement details without modification, followed by the (augmented) list of dependencies.
Do not output anything else or use any markdown formatting.
"""

def solve(index: int, request: Tuple[str, Dict[str, str]], bar: tqdm):
    model = Model_API(sys_prompt, gpt4o)
    query, js = request

    try:
        reply = model.generate(query)[0]
        tqdm.write(f">>> Success: {len(reply)} characters")

    except Exception as e:
        reply = ''
        tqdm.write(f'>>> Failed: ({type(e)}) {e}')

    bar.update()
    return index, reply, js

def main():
    requests = []
    for in_f, out_f in zip(SAMPLED_DATA, AUGMENTED_DATA):
        with open(f'{DIR}/{in_f}') as fin:
            for line in fin:
                js = json.loads(line)
                requests += [(template.format(js['input'], js['gold']), js)]

    with tqdm(total=len(requests), desc='>>> Progress') as bar, ThreadPoolExecutor(max_workers=30) as ex:
        futures = as_completed([ex.submit(solve, idx, request, bar) for idx, request in enumerate(requests)])

    with open(f'{DIR}/{out_f}', 'w') as fout:
        for _, reply, js in sorted([future.result() for future in futures]):
            js['input'] = reply
            print(json.dumps(js), file=fout)

if __name__ == '__main__':
    main()
