from model_api import Model_API
from config import gpt4o, CRITERIA

#! varies per task
system_prompt = "You are an expert software developer."
template = f"""A developer has implemented the following requirement.
You are to evaluate the quality of the code.
Note that the developer may provide explanations or comments around the code, which should not affect your judgment of the code.

## Requirement:
```
[[requirement]]
```

## Implementation:
```
[[code]]
```

You should analyze the implementation based on the following aspects:
{CRITERIA}

Evaluation Steps:
"""

def main():
    model = Model_API(system_prompt, gpt4o)
    print(model.generate(template)[0])

if __name__ == '__main__':
    main()
