from model_api import Model_API
from config import gpt4o, CRITERIA

#! varies per task
system_prompt = "You are an expert software developer."
template = f"""A developer has implemented the following requirement twice.
You are to compare the quality of the two code snippets, and decide which one is noticeably better or declare a tie because their difference in quality is insignificant.
Note that the developer may provide explanations or comments around the code, which should not affect your judgment of the code.

## Requirement:
```
[[requirement]]
```

## First Implementation:
```
[[first code snippet]]
```

## Second Implementation:
```
[[second code snippet]]
```

You should analyze the implementations based on the following aspects:
{CRITERIA}

First, generate evaluation steps:
"""

def main():
    model = Model_API(system_prompt, gpt4o)
    print(model.generate(template)[0])

if __name__ == '__main__':
    main()
