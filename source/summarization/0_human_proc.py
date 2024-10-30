from config import RESPONDED_DATA, HUMAN_IN, DIR, RESP_PER_INST, HUMAN_IN2
import json

template_cn = """
# 案例 {}
## 代码:
```{}
{}
```

以下是一份高质量的摘要，仅供参考，待评测摘要不必和参考摘要高度相似。
## 参考摘要
```
{}
```
"""


template2_cn = """
# 案例{}
## 摘要1
```
{}
```
1. 可读性：[-]/5
2. 一致性：[-]/5

总分：[-]/5

## 摘要2
```
{}
```
1. 可读性：[-]/5
2. 一致性：[-]/5

总分：[-]/5

## 摘要3
```
{}
```
1. 可读性：[-]/5
2. 一致性：[-]/5

总分：[-]/5
"""

intro_cn = """# 代码摘要人工评测
给定代码，一位开发员编写了该代码的文字摘要。
你需要评估该摘要的质量，无需考虑代码本身的质量。

你需要从以下角度分析摘要的质量：

1. 可读性：摘要的清晰度、简洁性和流畅性如何？

- 5/5：非常清晰、简洁且结构良好，非常容易理解。
- 4/5：基本清晰简洁，仅有轻微的可读性问题。
- 3/5：可以理解，但可能包含一些不清晰或措辞不自然的地方。
- 2/5：由于语言不清晰或结构不佳，难以理解。
- 1/5：非常混乱，存在明显的语言或结构问题。

2. 一致性：摘要是否描述了代码的关键功能，既没有遗漏重要细节，也没有添加相对不重要的内容？

- 5/5：与代码完全一致，准确捕捉了所有关键功能，没有遗漏重要细节或添加不必要的内容。
- 4/5：与代码基本一致，捕捉了关键功能，但添加了一些不太重要的细节。
- 3/5：较为一致，捕捉了整体功能，但遗漏了一些重要内容，或引入了过多不重要的细节。
- 2/5：不一致，明显遗漏了关键点或添加了不必要的内容，不太契合代码的关键功能。
- 1/5：非常不一致，完全遗漏了关键功能，只有不相关的内容，完全无法正确描述代码的功能。

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
            format_params += target.replace('```', '@@@').strip(),

        print(template_cn.format(index, lang, source.strip(), reference.strip()), file=fout1)
        print(template2_cn.format(index, *format_params), file=fout2)
