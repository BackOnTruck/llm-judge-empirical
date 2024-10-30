# LLM-as-a-Judge Empirical Study

This is the repository for paper `Can LLMs replace Human Evaluators? An Empirical Study of
LLM-as-a-Judge in Software Engineering Tasks`.



## Hardware Requirement
Local LLM inference are executed on a Ubuntu server with 4x A100 GPUs.
Online APIs can be invoked on any Linux PCs with Internet connection, including Windows Subsystem for Linux.



## Customization

If you want to use another task or dataset, please modify the following variables:

- `config.py`:
    - `DIR`: location of dataset
    - `os.environ["TIKTOKEN_CACHE_DIR"]`: location of Tiktoken cache
    - `maxlen_context`, `maxlen_resp`, `maxlen_output`: maximum length of input, response and judgment (in tokens)
    - `CRITERIA`: evaluation aspects and criteria for each score

- `filter.py`:
    - rewrite to handle the format of your dataset file

- `respond.py`:
    - `infer_sysprompt`: describes LLM's role to generate response
    - `infer_template`: prompt template for response generation
    - `postproc(resp: str)`: post-processing of LLM response; may be modified depending on the output format
    - formatting params

- `0_human_proc.py`:
    - `template_cn`: Chinese version of the evaluation form, for human evaluators
    - `intro_cn`: Chinese version of the evaluation criteria, for human evaluators
    - formatting params

- `1_sim.py`:
    - `metrics`: whether to use text or code similarity metrics
    - `crystalbleu()`: training corpus

- `2_3_gptscore.py`:
    - `template`: prompt template
    - formatting params

- `2_4_fflm_prep.py`:
    - `separator`: separates input and output in the prompt

- `3_1_geval_prep.py`:
    - `system_prompt`: describes LLM's role
    - `template`: prompt template to generate evaluation steps

- `3_1_geval.py`:
    - `system_prompt`: describes LLM's role
    - `template`: prompt template
    - formatting params

- `3_2_batcheval.py`:
    - `system_prompt`: describes LLM's role
    - `template`: prompt template
    - `template_per_sample`: context template
    - formatting params

- `4_1_api.py`:
    - formatting params

- `4_2_local.py`:
    - formatting params

- `vanilla_prompt.py`:
    - `system_prompt`: describes LLM's role
    - `template`: prompt template
    - formatting params



## Task & Dataset
* (Code-Code) Code Translation: CodeTransOcean-MultilingualTrans, `data/codetransocean`

* (Code-Text) Code Summarization: CodeXGLUE, `data/codexglue`

* (Text-Code) Code Generation: ComplexCodeEval, `data/complexcodeeval`



## Pipeline
### A. Preprocess (if necessary)

- Study the format of your custom dataset, preprocess it into *-A-initial.jsonl, and edit filter.py accordingly

### B. Remove long inputs

- `filter.py`
← `*-A-initial.jsonl`
→ `*-B-filtered.jsonl`: `keys = {'lang', 'input', 'gold'}`

### C. Select a number of samples for training, validation, and testing

- `sample.py`
← `*-B-filtered.jsonl`
→ `*-C-sampled.jsonl`: no additional keys

### D. Generate responses with various LLMs

- `respond.py`
← `*-C-sampled.jsonl`
→ `*-D-responded.jsonl`: `old_keys = {'lang', 'input', 'gold'}, new_keys = {'index', 'llm', 'output', 'context'}`
Note that responses from different LLMs with the same input are placed adjancently.

### E. Evaluate responses
← `test-D-responded.jsonl`
→` test-<#>-<method>.jsonl`: each line contains a float number indicating the score

#### E-0. Curate human judgments
← `test-D-responded.jsonl`
→ `test-0-human-in.md`
online evaluation → `test-0-human.jsonl`

#### E-1. Generate conventional scores

- `1_sim.py`
→ `test-1-*-*.jsonl`

#### E-2. Generate embedding & probability scores
If using FFLM, please preprocess:

- `2_4_fflm_prep.py`
→ `test-99-fflm-embed.jsonl`

- `2_*_*.py`
→ `test-2-*.jsonl`

#### E-3. Generate LLM scores

- `3_1_geval_prep.py`
→ Evaluation steps (Please include this in the prompt in `3_1_geval.py`)

- `3_*_*.py`
→ `test-3-*.jsonl`

#### E-4. Generate more LLM scores

- `4_*_*.py`
→ `test-4-*.jsonl.log` (might contain unexpected scoring format)

- `4_post.py`
→ `test-4-*.jsonl` (contains finalized scores)



## Evaluator I/O format

```json
// Input *.jsonl file:
{
    // filter.py
    "lang": "Python",                  // language(s) involved
    "input": "Implement hello world.", // input
    "gold": "print('Hello, world!')",  // ground-truth output
    
    // sample.py does not add any additional keys
    // respond.py
    "index": 3,                        // index in dataset
    "llm": "ds-55m"                    // abbreviation of llm that generates the output
    "output": "print('helloworld')",   // output

    // 4_*_*.py
    "judgment": "Overall: 4.5/5",      // LLM-annotated evaluation
    "score": 4.5,                      // score extracted from 'judgment'
}
```

