from config import CRITERIA

#! varies per task
# for APIs
system_prompt = "You are an expert software developer."

template = f"""A developer has implemented the following requirement.
You are to evaluate the quality of the code.
Note that the developer may provide explanations or comments around the code, which should not affect your judgment of the code.

## Requirement:
```
{{}}
```

## Implementation:
```
{{}}
```

You should analyze the code based on the following aspects:
{CRITERIA}

For each aspect, provide a brief analysis of the code's strengths and weaknesses, before assigning a score.
After scoring each aspect, assign an overall score using the format `Overall: X/5` based on either the average or a holistic assessment of the code's quality.
"""

#! varies per task
# for local SFT'd LLMs and their base LLMs
template1 = """Write critiques for a submitted response on a given user’s query, and grade the response:

[BEGIN DATA]
***
[Query]:
Given the following requirement:
```
{}
```
Please implement the required function or method in {}.
Note: Please output ONLY the code without any explanation, comments, or code block delimiters. Ensure that the output contains only the code itself.
***
[Response]:
{}
***
[END DATA]

Write critiques for this response. After that, you should give a final rating for the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
"""

template2 = f"""You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows:
"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.
###The instruction to evaluate:
Given the following requirement:
```
{{}}
```
Please implement the required function or method in {{}}.
Note: Please output ONLY the code without any explanation, comments, or code block delimiters. Ensure that the output contains only the code itself.
###Response to evaluate:
{{}}
###Score Rubrics:
Please assign a score for each evaluation aspect.
{CRITERIA}
3. Overall: Based on the aspects above, provide an overall score at the end of your feedback.
###Feedback:
"""

def temp_format1(lang: str, src: str, tgt: str):
    return template1.format(src, lang, tgt)

def temp_format2(lang: str, src: str, tgt: str):
    return template2.format(src, lang, tgt)

def temp_format0(lang: str, src: str, tgt: str):
    return template.format(src, tgt)

temp_format = [temp_format0, temp_format1, temp_format2]
