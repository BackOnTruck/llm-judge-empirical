from model_api import Model_API
from config import gpt4o, CRITERIA

#! varies per task
system_prompt = "You are an expert software developer."
template = f"""Given a code snippet in [[source language]], a developer has made two translations in [[target language]].
You are to compare the quality of the two translated code snippets, without considering the quality of the original code, and decide which one is noticeably better or declare a tie because their difference in quality is insignificant.
Note that the developer may provide explanations or comments around the translated code, which should not affect your judgment of the code.

## Original [[source language]] Code:
```
[[source code]]
```

## First Translated [[target language]] Code:
```
[[first target code]]
```

## Second Translated [[target language]] Code:
```
[[second target code]]
```

You should analyze the translated code snippets based on the following aspects:
{CRITERIA}

First, generate the evaluation steps:
"""

def main():
    model = Model_API(system_prompt, gpt4o)
    print(model.generate(template)[0])

if __name__ == '__main__':
    main()
