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
template = f"""Given a code snippet in {{}}, a developer has made two translations in {{}}.
You are to compare the quality of the two translated code snippets, without considering the quality of the original code, and decide which one is noticeably better or declare a tie because their difference in quality is insignificant.
Note that the developer may provide explanations or comments around the translated code, which should not affect your judgment of the code.

## Original {{}} Code:
```
{{}}
```

## First Translated {{}} Code:
```
{{}}
```

## Second Translated {{}} Code:
```
{{}}
```

You should analyze the translations based on the following aspects:
{CRITERIA}

To evaluate the translated code snippets and determine which one is of better quality or if they are equally good, you can consider the following steps based on the criteria provided:

### Step-by-Step Evaluation:

1. Readability & Idiomatic Usage:
   - Consistent Naming Conventions: Check if variables, functions, and classes in both translations follow the naming conventions typical in the [[target language]]. For instance, some languages prefer camelCase, while others prefer snake_case or PascalCase.
   - Structure and Organization: Evaluate the structure of the code. Is it organized in a way that is typically expected in the [[target language]]? Look for code indentation, spacing, and line breaks.
   - Use of Language Constructs: Determine if the translations make use of language-specific constructs or features that enhance clarity and are idiomatic, like list comprehensions in Python or lambda expressions in JavaScript.
   - Clarity and Simplicity: Is the code easy to read and understand? Does it avoid overly complex constructs that aren’t necessary?

2. Consistency with Source:
   - Functional Equivalence: Ensure that both translations achieve the same functionality as the original code. Check if the logic, algorithm, or intended effect is preserved.
   - Structural Similarity: Analyze whether the translation maintains a structure similar to the original, which can mean similar flow, sequence of operations, and dependency management.
   - Error Handling & Edge-Cases: Check if any error handling or edge-case considerations from the original code are preserved in the translated versions.

### Comparative Review:

1. Side-by-Side Comparison:
   - Compare both translations side-by-side to observe the differences in readability and idiomatic usage.
   - Identify any specific areas where one translation may handle certain operations more clearly or efficiently.

2. Evaluate Alignment with Source Code:
   - For each translation, identify any deviations from the original code in terms of functionality. Note if changes have been made that impact the output or purpose of the code.

3. Assessment Summary:
   - Based on the analysis above, summarize the strengths and weaknesses of each translated snippet regarding readability and consistency with the source.
   - Decide if one translation is significantly better or if they can be considered a tie in terms of quality.

After completing these evaluations, you can render a verdict on which translation is superior or if they are equally well-done. You'd then provide your reasoning based on the observations made during this stepwise analysis.

Now, please fill out the evaluation form by giving the comparison result first by responding "FIRST" "SECOND" or "TIE" to indicate which translation is better, before comparing them on each aspect and making explanations.
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
                        s0, s1 = samples[j], samples[k]
                        src_lang, tgt_lang = s0['lang'].split(', ')

                        batch += [template.format(src_lang, tgt_lang, src_lang, s0['input'], tgt_lang, s0['output'], tgt_lang, s1['output'])]

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
