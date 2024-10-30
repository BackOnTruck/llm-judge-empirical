from typing import *
import torch
from config import path, gen_config, context_len, no_system_prompt
from vllm import LLM, SamplingParams
from tqdm import tqdm
from random import randrange

class Model:
    def __init__(self, llm: str, sys_prompt: str, max_tokens: int, batch_size: Optional[int]=None):
        file = path[llm]
        self.llm, self.sys_prompt = llm, sys_prompt
        if batch_size is not None:
            self.reset(batch_size)

        print(f">>> Loading {file}...")
        self.model = LLM(file, trust_remote_code=True, dtype=torch.bfloat16, max_model_len=context_len, tensor_parallel_size=4)
        self.decoding('greedy', max_tokens)

    def add_message(self, role: str, content: str, index: int):
        self.message[index] += [{'role': role, 'content': content}]

    def reset(self, batch_size: int):
        self.message = [[] for _ in range(batch_size)]
        if self.llm not in no_system_prompt and self.sys_prompt:
            for i in range(batch_size):
                self.add_message('system', self.sys_prompt, i)

    def decoding(self, strategy: str, max_tokens: int):
        print(f">>> Using {strategy}, max tokens = {max_tokens}")
        self.param = SamplingParams(max_tokens=max_tokens, **gen_config[strategy])

    def generate(self, queries: List[str]):
        assert len(queries) == len(self.message), f"Size mismatch: {len(self.message)} messages, {len(queries)} queries"
        model, tk, param = self.model, self.model.get_tokenizer(), self.param
        print(f">>> Generating with T = {param.temperature:.2}, p = {param.top_p:.2}, k = {param.top_k}, N = {param.n}")

        for i, query in enumerate(queries):
            self.add_message('user', query, i)

        messages = tk.apply_chat_template(self.message, add_generation_prompt=True, tokenize=False)
        responses = [[output.text for output in x.outputs] for x in model.generate(messages, param)]

        for i, response in enumerate(responses):
            self.add_message('assistant', response[0], i)

        index = randrange(len(responses))
        tqdm.write(f'{self.llm} - Sample query\n{queries[index].strip()}\n')
        tqdm.write(f'{self.llm} - Sample response\n{responses[index][0].strip()}')
        return self.message, responses
        # msg: history + 1st response, (batch_size, turns) of dict
        # resp: all N responses, (batch_size, N) of str
