from config import DIR, SAMPLED_DATA, RESPONDED_DATA, MODELS, maxlen_resp, RESP_PER_INST
import json, os
from psutil import process_iter
from signal import SIGKILL
from typing import *
from random import sample
from model import Model
from multiprocessing import Queue, Process
from time import sleep
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex as getHandle, nvmlDeviceGetMemoryInfo as getMem

nvmlInit()
cooldown = 15

#! varies per task
infer_sysprompt = 'You are an expert software developer.'
infer_template = """Given the following code:
{}

Please write a brief summary of its functionality, focusing on the main purpose without elaborating on too many details.
"""

def generate(llm: str, batch: List[str], maxlen: int, q: Queue):
    print(f'\n\n------ Processing {llm} ------')
    try:
        m = Model(llm, infer_sysprompt, maxlen * 2, len(batch)) # allow more tokens to avoid truncation
        q.put([x[0] for x in m.generate(batch)[1]])

    except Exception as e:
        print(f'-- Unexpected exception: {type(e)} {e}')
        q.put(None)

def available(device_id: int):
    memory = getMem(getHandle(device_id))
    return memory.free > 4e10 # approx. 37.3 GiB

def main():
    for sampled_data, responded_data in zip(SAMPLED_DATA, RESPONDED_DATA):
        print(f'>>> Processing {responded_data}...')
        instances = {model: [] for model in MODELS}
        with open(f'{DIR}/{sampled_data}') as fin:
            for instance, models in [[json.loads(line), sample(MODELS, RESP_PER_INST)] for line in fin]:
                #! varies per task
                prompt = infer_template.format(instance['input'])

                for model in models:
                    instances[model] += [[prompt, instance.copy()]] # instance MUST be copied to avoid being shared among LLMs

        for model in MODELS:
            print(f'>>> Model {model}: {len(instances[model])} instances')

        for model in MODELS:
            if not instances[model]:
                print(f'>>> No instances for {model}. Continuing...')
                continue

            while 1:
                # ensure that VRAM is properly released
                for i in range(4):
                    if not available(i):
                        input(f"--- WARNING (PID = {os.getpid()}) ---\nGPU {i} unavailable, please manually kill subprocesses and press enter...")
                        break

                # vLLM might mess up if we don't start a new process for each LLM
                q = Queue()
                proc = Process(target=generate, args=(model, [x[0] for x in instances[model]], maxlen_resp, q))
                proc.start()

                if (outputs := q.get()) is not None: # automatically waits for q.put()
                    for output, instance in zip(outputs, instances[model]):
                        instance += [output]

                    print(f">>> {model} finished. Waiting for vLLM to terminate...")
                    proc.join(cooldown)

                    if proc.is_alive():
                        print(">>> vLLM fails to terminate itself. Killing vLLM...")
                        proc.kill() # vLLM sucks in shutting down so we kill it; terminate() / `kill <PID>` / SIGTERM may not work

                        for proc in process_iter(['pid', 'cmdline']):
                            cmdline, pid = proc.info['cmdline'], proc.info['pid']
                            if type(cmdline) == list and 'python respond.py' in ' '.join(cmdline) and proc.info['pid'] != os.getpid():
                                #! remember to modify the command line when renaming this file
                                os.kill(pid, SIGKILL)
                                print(f">>> Process {pid} killed, command line: {cmdline}")

                    print(f">>> Sleeping for {cooldown} secs to ensure the release of VRAM...")
                    sleep(cooldown)
                    break

                print(f">>> {model} fails. Trying again...")

        with open(f'{DIR}/{responded_data}', 'w') as fout:
            samples = []
            for model, instances_model in instances.items():
                for instance in instances_model:
                    cur_sample = instance[1]
                    cur_sample['llm'] = model
                    cur_sample['output'] = instance[2]
                    samples += [cur_sample]

            samples.sort(key=lambda x: x['input'])
            for cnt, cur_sample in enumerate(samples):
                cur_sample['index'] = cnt
                print(json.dumps(cur_sample), file=fout)


if __name__ == '__main__':
    main()
