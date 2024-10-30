from config import DIR, RESPONDED_DATA, SFT_RES, EVAL_MODELS, maxlen_output, llama2_13b, autoj_13b, mixtral_8x7b, prometheus2_8x7b, dscoder_2_lite
import json, os
from psutil import process_iter
from signal import SIGKILL
from typing import *
from model import Model
from multiprocessing import Queue, Process
from time import sleep
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex as getHandle, nvmlDeviceGetMemoryInfo as getMem
from vanilla_prompt import system_prompt as sys_prompt0, temp_format

# prometheus2 somehow does not support system prompt; we instead prepend it to the user prompt
# auto-j paper does not offer a system prompt; we leave it blank
system_prompt = {
    llama2_13b: '',
    autoj_13b: '',
    mixtral_8x7b: '',
    prometheus2_8x7b: '',
    dscoder_2_lite: sys_prompt0,
}

template = {
    llama2_13b: temp_format[1],
    autoj_13b: temp_format[1],
    mixtral_8x7b: temp_format[2],
    prometheus2_8x7b: temp_format[2],
    dscoder_2_lite: temp_format[0],
}

nvmlInit()
cooldown = 15

def generate(llm: str, batch: List[str], maxlen: int, q: Queue):
    print(f'\n\n------ Processing {llm} ------')
    try:
        m = Model(llm, system_prompt[llm], maxlen * 2, len(batch)) # allow more tokens to avoid truncation
        q.put([x[0] for x in m.generate(batch)[1]])

    except Exception as e:
        print(f'-- Unexpected exception: {type(e)} {e}')
        q.put(None)

def available(device_id: int):
    memory = getMem(getHandle(device_id))
    return memory.free > 4e10 # approx. 37.3 GiB

def main():
    for model in EVAL_MODELS:
        sampled_data, responded_data = RESPONDED_DATA[0], f'{SFT_RES[model]}.log'

        print(f'>>> Processing {responded_data}...')
        instances = []
        with open(f'{DIR}/{sampled_data}') as fin:
            for instance in [json.loads(line) for line in fin]:
                #! varies per task
                langs, codes = instance['lang'].split(', '), (instance['input'], instance['output'])
                prompt = template[model](*langs, *codes)
                instances += [[prompt, instance]]

        print(f'>>> Model {model}: {len(instances)} instances')

        while 1:
            # ensure that VRAM is properly released
            for i in range(4):
                if not available(i):
                    input(f"--- WARNING (PID = {os.getpid()}) ---\nGPU {i} unavailable, please manually kill subprocesses and press enter...")
                    break

            # vLLM might mess up if we don't start a new process for each LLM
            q = Queue()
            proc = Process(target=generate, args=(model, [x[0] for x in instances], maxlen_output, q))
            proc.start()

            if (outputs := q.get()) is not None: # automatically waits for q.put()
                for output, instance in zip(outputs, instances):
                    instance += [output]

                print(f">>> {model} finished. Waiting for vLLM to terminate...")
                proc.join(cooldown)

                if proc.is_alive():
                    print(">>> vLLM fails to terminate itself. Killing vLLM...")
                    proc.kill() # vLLM sucks in shutting down so we kill it; terminate() / `kill <PID>` / SIGTERM may not work

                    for proc in process_iter(['pid', 'cmdline']):
                        cmdline, pid = proc.info['cmdline'], proc.info['pid']
                        if type(cmdline) == list and 'python 4_2_local.py' in ' '.join(cmdline) and proc.info['pid'] != os.getpid():
                            #! remember to modify the command line when renaming this file
                            os.kill(pid, SIGKILL)
                            print(f">>> Process {pid} killed, command line: {cmdline}")

                print(f">>> Sleeping for {cooldown} secs to ensure the release of VRAM...")
                sleep(cooldown)
                break

            print(f">>> {model} fails. Trying again...")

        with open(f'{DIR}/{responded_data}', 'w') as fout:
            for instance in instances:
                cur_sample = instance[1]
                cur_sample['judgment'] = instance[2]
                cur_sample['score'] = -1.0 # we leave it to 4_post.py to extract the score
                print(json.dumps(cur_sample), file=fout)


if __name__ == '__main__':
    main()
