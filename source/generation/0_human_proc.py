from config import RESPONDED_DATA, HUMAN_IN, DIR, RESP_PER_INST, HUMAN_IN2
import json

template_cn = """
# 案例 {}
## 需求:
```
{}
```

以下是一份高质量的参考代码，仅供参考，待评测代码不需要和参考代码高度相似。
## 参考代码
```{}
{}
```
"""

template2_cn = """
# 案例 {}
## 代码1
```{}
{}
```
1. 功能正确性：[-]/5
2. 可读性：[-]/5

总分：[-]/5

## 代码2
```{}
{}
```
1. 功能正确性：[-]/5
2. 可读性：[-]/5

总分：[-]/5

## 代码3
```{}
{}
```
1. 功能正确性：[-]/5
2. 可读性：[-]/5

总分：[-]/5
"""

intro_cn = """# 代码生成人工评测
给定需求，一位开发员已编程将其实现。
你需要评估其编写的代码质量。
注意：开发者可能会在代码周围提供解释或注释，这些内容不应影响你对代码质量的评判。

你需要从以下角度分析代码质量：

1. 功能正确性：代码在多大程度上满足任务的需求，能在预期情况下正确执行？

- 5/5：完全满足任务需求，在所有预期情况下正确运行。
- 4/5：大部分正确，仅有轻微问题，对整体功能没有太大影响。
- 3/5：部分正确，但包含显著的逻辑或功能性错误。
- 2/5：存在重大问题，导致代码无法正常运行，尽管能大致推断出代码的意图。
- 1/5：完全错误，对提出的需求毫无意义。

2. 可读性：代码的清晰度、简洁性和结构如何，是否易于阅读和理解？

- 5/5：非常清晰简洁，结构和命名规范优秀，易于理解。
- 4/5：总体上易于阅读，仅有轻微的清晰度或结构问题，不影响理解。
- 3/5：部分可读，但有多个不清晰或不顺畅的部分，需花时间解释。
- 2/5：结构差，命名不清晰，或代码过于复杂，阅读困难。
- 1/5：非常混乱且难以理解，存在显著的清晰度或结构问题，几乎无法理解。

对于每个角度，先简要分析评论优缺点，再给分。最后给出总分。
"""


with open(f'{DIR}/{RESPONDED_DATA[0]}') as fin, open(f'{DIR}/{HUMAN_IN}', 'w', encoding='utf-8') as fout1, open(f'{DIR}/{HUMAN_IN2}', 'w', encoding='utf-8') as fout2:
    print(intro_cn, '\n', file=fout1)
    lines = list(fin)

    for index in range(0, len(lines), RESP_PER_INST):
        js = [json.loads(line) for line in lines[index:index + RESP_PER_INST]]
        source, targets, reference, lang = js[0]['input'], [s['output'] for s in js], js[0]['gold'], js[0]['lang']

        #! params vary per task
        format_params = ()
        for target in targets:
            format_params += lang, target.replace('```', '@@@')

        print(template_cn.format(index, source, lang, reference), file=fout1)
        print(template2_cn.format(index, *format_params), file=fout2)
