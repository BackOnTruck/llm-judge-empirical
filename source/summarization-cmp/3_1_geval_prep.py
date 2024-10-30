from model_api import Model_API
from config import gpt4o, CRITERIA

#! varies per task
system_prompt = "You are an expert software developer."
template = f"""A developer has written two summaries for the following code.
You are to compare the quality of the two summaries, without considering the quality of the code. You should decide which one is noticeably better or declare a tie due to insufficient difference in their quality.

## Code:
```
[[code]]
```

## First Summary:
```
[[first summary]]
```

### Second Summary:
```
[[[second summary]]]
```

You should analyze the summaries based on the following aspects:
{CRITERIA}

First, generate the evaluation steps:
"""

def main():
    model = Model_API(system_prompt, gpt4o)
    print(model.generate(template)[0])

if __name__ == '__main__':
    main()
