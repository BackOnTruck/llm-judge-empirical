import os, warnings
from typing import *

#* Models & General options
warnings.simplefilter('ignore', category=FutureWarning) # vLLM code causes annoying FutureWarnings
os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'
# os.environ["TIKTOKEN_CACHE_DIR"] = '' #! may need modification

# maximum context length for evaluation (input + output + judgment; instructions don't count)
context_len = 4096
#! code-code: 1536 + 1536 + 1024; text-code: 1024 + 2048 + 1024; code-text: 2048 + 1024 + 1024
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

#! Code Translation Dataset
DIR = 'data/codetransocean'

#^ evaluator inputs: filter -> sample -> respond
ORIGINAL_DATA = ['test-A-initial.jsonl', 'valid-A-initial.jsonl', 'train-A-initial.jsonl']
FILTERED_DATA = ['test-B-filtered.jsonl', 'valid-B-filtered.jsonl', 'train-B-filtered.jsonl']
SAMPLED_DATA = ['test-C-sampled.jsonl', 'valid-C-sampled.jsonl', 'train-C-sampled.jsonl']
RESPONDED_DATA = ['test-D-responded.jsonl', 'valid-D-responded.jsonl', 'train-D-responded.jsonl']

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

CRITERIA = """
1. Readability & Idiomatic Usage: How easy is it to read and understand the translated code, and how well does it adhere to the idiomatic practices of the target language?
2. Consistency with Source: How closely does the translated code maintain the same meaning and functionality as the original, while maintaining a similar structure?
"""
