from config import CRITERIA

#! varies per task
# for APIs
system_prompt = "You are an expert software developer."

template = f"""A developer has written two summaries for the following code.
You are to compare the quality of the two summaries, without considering the quality of the code, and decide which one is noticeably better or declare a tie because their difference in quality is insignificant.

## Code:
```
{{}}
```

## First Summary:
```
{{}}
```

## Second Summary:
```
{{}}
``

You should analyze the summaries based on the following aspects:
{CRITERIA}

For each aspect, provide a brief comparison of the two summaries' strengths and weaknesses, before deciding which is better or declaring a tie.
After comparing on each aspect, assign an overall comparison verdict using the format `Overall: X` based on either the average or a holistic assessment of these aspects, where X is either "FIRST", "SECOND", or "TIE" to indicate the better response if there is one.
"""

#! varies per task
# for local SFT'd LLMs and their base LLMs
template1 = """You are assessing two submitted responses on a given user’s query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]:
Given the following code:
{}

Please write a brief summary of its functionality, focusing on the main purpose without elaborating on too many details.
***
[Response 1]:
{}
***
[Response 2]:
{}
***
[END DATA]

Here are the instructions to assess and compare the two responses:
1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you’ve provided.
"""

template2 = f"""You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.
###Instruction:
Given the following code:
{{}}

Please write a brief summary of its functionality, focusing on the main purpose without elaborating on too many details.
###Response A:
{{}}
###Response B:
{{}}
###Score Rubrics:
Please assign a score for each evaluation aspect.
{CRITERIA}
3. Overall: Based on the aspects above, provide an overall comparison at the end of your feedback.
###Feedback:
"""

def temp_format1(lang: str, src: str, tgt: str, tgt2: str):
    return template1.format(src, tgt, tgt2)

def temp_format2(lang: str, src: str, tgt: str, tgt2: str):
    return template2.format(src, tgt, tgt2)

def temp_format0(lang: str, src: str, tgt: str, tgt2: str):
    return template.format(src, tgt, tgt2)

temp_format = [temp_format0, temp_format1, temp_format2]
