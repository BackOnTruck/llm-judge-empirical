from model_api import Model_API
from config import gpt4o, CRITERIA

#! varies per task
system_prompt = "You are an expert software developer."
template = f"""A developer has written a summary for the following code.
You are to evaluate the quality of the summary.

## Code:
```
[[code]]
```

## Summary:
```
[[summary]]
```

You should analyze the summary based on the following aspects:
{CRITERIA}

Evaluation Steps:
"""

def main():
    model = Model_API(system_prompt, gpt4o)
    print(model.generate(template)[0])

if __name__ == '__main__':
    main()
