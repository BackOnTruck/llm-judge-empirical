import openai
from time import sleep
from tqdm import tqdm
from config import API_KEY, API_URL
from typing import *

LIM = 10

class LogProbModel_API:
    def __init__(self, llm: str):
        self.llm = llm
        self.client = openai.OpenAI(api_key=API_KEY[llm], base_url=API_URL[llm])

    def do_inference(self, context: str):
        try:
            result = self.generate(context)

        except Exception as e:
            tqdm.write(f">>> Failed: ({type(e)}) {e}")
            raise

        info = result.choices[0].logprobs
        logprobs, offsets = info.token_logprobs, info.text_offset
        tqdm.write(f">>> Success: {len(logprobs)} tokens")
        return logprobs, offsets

    def generate(self, prompt: str):
        for n_failed in range(LIM):
            try:
                response = self.client.completions.create(
                    model=self.llm,
                    prompt=prompt,
                    max_tokens=0,
                    temperature=0,
                    logprobs=0,
                    echo=True,
                    timeout=15,
                )
                return response

            except Exception as e:
                tqdm.write(f"Error #{n_failed}: ({type(e)}) {e}")
                sleep(2)

        raise RuntimeError("API invocation failure")


def locate(L: List[int], x: int):
    for i in range(len(L)):
        if L[i] > x:
            return i - 1

    return len(L) - 1


def token_probs(logProb: List[float], offsets: List[int], start_pos: int):
    return logProb[locate(offsets, start_pos):]
