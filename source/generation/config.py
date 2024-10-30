import os, warnings
from typing import *

#* Models & General options
warnings.simplefilter('ignore', category=FutureWarning) # vLLM code causes annoying FutureWarnings
os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'
# os.environ["TIKTOKEN_CACHE_DIR"] = '' #! may need modification

# maximum context length for evaluation (input + output + judgment; instructions don't count)
context_len = 4096
# we filter extra-long data beforehand
# UniXcoder is the only model that needs truncation, and receives only the reference and output to classify
minlen_context, maxlen_context, maxlen_resp, maxlen_output = 5, 1536, 1536, 1024
# number of inputs per task to be evaluated manually
# inputs are sampled porportionally from each language group, removing extra samples from the most common languages
# 3 outputs to be generated per input
NUM_SAMPLES = 50, 5, 5000
RESP_PER_INST = 3


#^ online LLMs
GPT4O_TK = 'o200k_base'

API_MODELS = {
    (deepseek_25 := 'deepseek-chat'): 'DeepSeek-V2.5',
    (gpt4o := 'gpt-4o'): 'GPT-4o',
    (davinci := 'davinci-002'): 'Davinci-002 (Legacy Completion with log probability)',
    (openai_embed := 'text-embedding-3-large'): 'Text Embedding 3 Large (Embedding)',
}
CHAT_APIS = {deepseek_25, gpt4o}

API_KEY = {
    deepseek_25: '',
    gpt4o: '',
    davinci: '',
    openai_embed: '',
}

API_URL = {
    deepseek_25: 'https://api.deepseek.com',
    gpt4o: 'https://api.openai.com/v1/models',
    davinci: 'https://api.openai.com/v1/models',
    openai_embed: 'https://api.openai.com/v1/models',
}

#^ offline LLMs
MODELS = [
    codellama_7b := 'cl-7b',
    codellama_13b := 'cl-13b',
    codellama_34b := 'cl-34b',

    dscoder_1b := 'dsc-1b',
    dscoder_7b := 'dsc-7b',
    dscoder_33b := 'dsc-33b',

    qwen25_1500m := 'q2.5-1.5b',
    qwen25_7b := 'q2.5-7b',
    # qwen25_32b := 'q2.5-32b', # to be released

    dscoder_2_lite := 'dsc2-16b',
    magicoder_7b := 'magic-7b',
    codestral_22b := 'cs-22b',
    codegeex4_9b := 'cg4-9b',
]

EVAL_MODELS = {
    llama2_13b := 'l2-13b',
    autoj_13b := 'autoj-13b',
    mixtral_8x7b := 'mix-8x7b',
    prometheus2_8x7b := 'prom-8x7b',
    dscoder_2_lite
}

no_system_prompt = {magicoder_7b}

path = {
    codellama_7b: 'model/codellama-instruct/7b',
    codellama_13b: 'model/codellama-instruct/13b',
    codellama_34b: 'model/codellama-instruct/34b',

    dscoder_1b: 'model/deepseek-coder-instruct/1.3b',
    dscoder_7b: 'model/deepseek-coder-instruct/6.7b',
    dscoder_33b: 'model/deepseek-coder-instruct/33b',

    qwen25_1500m: 'model/qwen2.5-coder-instruct/1.5b',
    qwen25_7b: 'model/qwen2.5-coder-instruct/7b',
    # qwen25_32b: 'model/qwen2.5-coder-instruct/32b'

    dscoder_2_lite: 'model/deepseek-coder-v2-instruct/lite-16b',
    magicoder_7b: 'model/magicoder-s-ds-6.7b',
    codestral_22b: 'model/codestral-22b-v0.1',
    codegeex4_9b: 'model/codegeex4-all-9b',

    llama2_13b: 'model/_non-code-llms/llama2-13b-chat',
    autoj_13b: 'model/_non-code-llms/autoj-13b',
    mixtral_8x7b: 'model/_non-code-llms/mixtral-instruct-8x7b-v0.1',
    prometheus2_8x7b: 'model/_non-code-llms/prometheus-bgb-8x7b-v2.0',
}

gen_config = {
    'greedy': {
        'temperature': 0.0
    },
    'sampling': {
        'temperature': 1.0,
        'top_p': 0.9,
        'top_k': 50,
        'n': 5
    }
}

#! Code Generation Dataset
DIR = 'data/complexcodeeval'

#^ evaluator inputs: filter -> sample -> respond
ORIGINAL_DATA = ['test-A-initial.jsonl']
FILTERED_DATA = ['test-B-filtered.jsonl']
SAMPLED_DATA = ['test-C-sampled.jsonl']
AUGMENTED_DATA = ['test-D-augmented.jsonl']
RESPONDED_DATA = ['test-E-responded.jsonl']

#^ evaluator outputs: scores
# group 1: human & non-LLM approaches
HUMAN_IN = 'test-0-human-in.md'
HUMAN_IN2 = 'test-0-human-in2.md'
HUMAN_RES = 'test-0-human.jsonl'
SIM_RES = {
    'bleu': 'test-1-1-bleu.jsonl',
    'rouge-l': 'test-1-2-rouge.jsonl',
    'meteor': 'test-1-3-meteor.jsonl',
    'chrf++': 'test-1-4-chrf.jsonl',
    'crystalbleu': 'test-1-5-crystalbleu.jsonl', # not intended for text evaluation
}

# group 2: embedding & probability approaches
BS_RES = 'test-2-1-bertscore.jsonl'
MS_RES = 'test-2-2-moverscore.jsonl'
GS_RES = 'test-2-3-gptscore.jsonl'
FFLM_EMBED = 'test-99-fflm-embed.jsonl'
FFLM_RES = 'test-2-4-fflm.jsonl'

# group 3: prompting approaches
GEVAL_RES = 'test-3-1-geval.jsonl'
BATCH_RES = 'test-3-2-batcheval.jsonl'

# group 4: SFT vs. non-SFT model + vanilla prompt
SFT_RES = {
    llama2_13b: 'test-4-1-llama2.jsonl',
    autoj_13b: 'test-4-2-autoj.jsonl',
    mixtral_8x7b: 'test-4-3-mixtral.jsonl',
    prometheus2_8x7b: 'test-4-4-prometheus.jsonl',
    dscoder_2_lite: 'test-4-5-dsc2-lite.jsonl',
    deepseek_25: 'test-4-6-dsc2.5.jsonl',
    gpt4o: 'test-4-7-gpt4o.jsonl',
}

#! varies per task
CRITERIA = """
1. Functional Correctness: How well does the code meet the task’s requirements and perform as expected?

- 5/5: Fully satisfies the task’s requirements, performs correctly in all expected cases.
- 4/5: Mostly correct, with minor issues that don't greatly affect the overall functionality.
- 3/5: Partially correct, but contains significant errors in logic or functionality.
- 2/5: Major issues that prevent the code from functioning correctly, though the intent is somewhat clear.
- 1/5: Completely incorrect or fails to meet the task requirements in any meaningful way.

2. Readability: How easy is the generated code to read and understand, considering clarity, simplicity, and structure?

- 5/5: Extremely clear and concise, with excellent structure and naming conventions that make it easy to follow.
- 4/5: Generally easy to read, with minor issues in clarity or structure that don’t hinder understanding.
- 3/5: Somewhat readable, but with several unclear or awkward sections that require interpretation.
- 2/5: Difficult to read due to poor structure, unclear naming, or overly complex code.
- 1/5: Very confusing and hard to follow, with significant clarity or structural issues that make understanding the code difficult.
"""
