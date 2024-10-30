from model_api import Model_API
import json, re
from config import DIR, RESPONDED_DATA, GEVAL_RES, gpt4o, CRITERIA
from concurrent.futures import as_completed, ThreadPoolExecutor
from tqdm import tqdm

#! varies per task
# system prompt; disabled for LLMs not supporting it
system_prompt = "You are an expert software developer."

#! varies per task; evaluation steps generated by 3_1_geval_prep.py
# user prompts; EN template: for LLMs, 1 response; CN template: for humans, 3 responses
template = f"""A developer has written two summaries for the following code.
You are to compare the quality of the two summaries, without considering the quality of the code. You should decide which one is noticeably better or declare a tie due to insufficient difference in their quality.

## Code:
```
{{}}
```

## First Summary:
```
{{}}
```

### Second Summary:
```
{{}}
```

You should analyze the summaries based on the following aspects:
{CRITERIA}

To evaluate the summaries based on the given aspects, you can break down the process as follows:

### Step 1: Readability Analysis
- Clarity: Determine how clear each summary is in explaining what's happening in the code. Look for straightforward language and easy-to-follow explanations.
- Conciseness: Assess whether each summary is brief yet comprehensive, avoiding unnecessary detail or verbosity.
- Fluency: Check for smoothness in the text, ensuring it reads naturally without awkward phrasing or grammatical issues.

### Step 2: Consistency Analysis
- Accuracy: Evaluate how accurately each summary describes the key functionalities and objectives of the code.
- Completeness: Check if the summary captures all significant features and functions of the code, without omitting important points.
- Relevance: Ensure that the summary focuses on the most critical aspects of the code, avoiding unimportant details.

### Step 3: Comparison
- Compare both summaries based on their performance in the readability and consistency analyses.
- Decide which summary is better, considering the impact of any strengths or weaknesses identified, or declare a tie if differences are negligible.

Now, please fill out the evaluation form by giving the comparison result first by responding "FIRST" "SECOND" or "TIE" to indicate which summary is better, before comparing them on each aspect and making explanations.
## Evaluation form:
- Overall score: """

N = 20
mapping = {
    'first': 1,
    'tie': 0,
    'second': -1
}
def solve(index: int, query: str, bar: tqdm):
    model = Model_API(system_prompt, gpt4o)
    try:
        replies = model.generate(query, T=1.0, N=N, maxlen=20)
        tqdm.write(f">>> Success: {sum(len(reply) for reply in replies)} characters")

    except Exception as e:
        replies = None
        tqdm.write(f">>> Failed: ({type(e)}) {e}")

    bar.update()
    matches = list(filter(lambda x: x, [re.findall(r'(first|second|tie)', reply.lower()) for reply in replies]))
    if not matches:
        matches = [['tie']]

    return index, model.message, sum(mapping[match[0]] for match in matches) / len(matches), matches

def main():
    with open(f'{DIR}/{RESPONDED_DATA[0]}') as fin:
        batch = []
        lines = list(fin)

        for i in range(0, len(lines), 3):
            samples = [json.loads(line) for line in lines[i:i + 3]]
            for j in range(3):
                for k in range(3):
                    if j != k:
                        #! params vary per task
                        batch += [template.format(samples[0]['input'], samples[j]['output'], samples[k]['output'])]

    with tqdm(total=len(batch), desc='>>> Progress') as bar, ThreadPoolExecutor(max_workers=40) as ex:
        futures = as_completed([ex.submit(solve, idx, sample, bar) for idx, sample in enumerate(batch)])

    path = f'{DIR}/{GEVAL_RES}'
    with open(path, 'w') as fout, open(f'{path}.log', 'w') as flog, open(f'{path}.txt', 'w') as fscores:
        for _, message, score, all_scores in sorted([future.result() for future in futures]):
            print(json.dumps(message), file=flog)
            print(score, file=fout)
            print(json.dumps(all_scores), file=fscores)

if __name__ == '__main__':
    main()
