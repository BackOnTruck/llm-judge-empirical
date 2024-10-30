from typing import *
import openai
from config import maxlen_output, CHAT_APIS, API_KEY, API_URL
from tqdm import tqdm
from time import sleep

LIM = 10 # attempts before failure

class Model_API:
    def __init__(self, sys_prompt: str, llm: str):
        assert llm in CHAT_APIS, f'{llm} does not support chat completions'

        self.sys_prompt, self.llm = sys_prompt, llm
        self.client = openai.OpenAI(api_key=API_KEY[llm], base_url=API_URL[llm])
        self.reset()

    def reset(self):
        self.message = []
        if self.sys_prompt:
            self.add_message('system', self.sys_prompt)

    def add_message(self, role: str, content: str):
        self.message += [{'role': role, 'content': content}]

    def generate(self, query: str, T: float=0.0, N: int=1, maxlen: int=maxlen_output):
        self.add_message('user', query)

        for n_failed in range(LIM):
            try:
                responses = [choice.message.content.strip() for choice in self.client.chat.completions.create(
                        model=self.llm,
                        messages=self.message,
                        temperature=T,
                        max_tokens=maxlen,
                        n=N,
                        timeout=30
                    ).choices]

                self.add_message('assistant', responses[0])
                return responses
                # msg: history + 1st response, [turns] of dict
                # resp: all N responses, [N] of str

            except Exception as e:
                tqdm.write(f"Error #{n_failed}: ({type(e)}) {e}")
                sleep(2)

        raise RuntimeError('API invocation failure')
