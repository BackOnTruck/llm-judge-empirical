from model_api import Model_API
from config import gpt4o, CRITERIA

#! varies per task
system_prompt = "You are an expert software developer."
template = f"""Given a code snippet in [[source language]], a developer has translated it into [[target language]].
You are to evaluate the quality of the translated code, without considering the quality of the original code.
Note that the developer may provide explanations or comments around the translated code, which should not affect your judgment of the code.

## Original [[source language]] Code:
```
[[source code]]
```

## Translated [[target language]] Code:
```
[[target code]]
```

You should analyze the translated code based on the following aspects:
{CRITERIA}

Evaluation Steps:
"""

def main():
    model = Model_API(system_prompt, gpt4o)
    print(model.generate(template)[0])

if __name__ == '__main__':
    main()
