from config import RESPONDED_DATA, HUMAN_IN, DIR, RESP_PER_INST, HUMAN_IN2
import json

template_cn = """
# 案例 {}（{} → {}）
## 原始代码:
```{}
{}
```

以下是一份高质量的参考代码，仅供参考，待评测代码不需要和参考代码高度相似。
## 参考代码
```{}
{}
```
"""

template2_cn = """
# 案例{}
## 代码1
```{}
{}
```
1. 可读性与习惯用法：[-]/5
2. 与原始代码的一致性：[-]/5

总分：[-]/5

## 代码2
```{}
{}
```
1. 可读性与习惯用法：[-]/5
2. 与原始代码的一致性：[-]/5

总分：[-]/5

## 代码3
```{}
{}
```
1. 可读性与习惯用法：[-]/5
2. 与原始代码的一致性：[-]/5

总分：[-]/5
"""

intro_cn = """# 代码翻译人工评测
给定一份原始代码，一位开发员已将其翻译为另一种语言。
你需要评估翻译后代码的质量，无需考虑原始代码的质量。
注意：开发者可能会在代码周围提供解释或注释，这些内容不应影响你对代码质量的评判。

你需要从以下角度分析代码质量：

1. 可读性与习惯用法：翻译后的代码是否易于阅读理解？是否遵循目标语言习惯用法？

- 5/5：非常容易阅读，完全符合习惯用法，结构清晰，符合语言习惯。
- 4/5：较易理解，基本符合习惯用法，仅有轻微的问题，略微偏离语言习惯。
- 3/5：基本可读，但在结构和编程习惯上存在明显缺陷，可能会妨碍理解。
- 2/5：由于可读性差或使用了不合习惯的结构，难以阅读，仅部分内容可理解。
- 1/5：存在严重的可读性问题，使用了不符合惯用用法或不自然的模式，几乎无法理解。

2. 与原始代码的一致性：翻译后的代码是否保持与原始代码相同的意义、功能和结构？

- 5/5：在意义和功能上高度相似，没有差异或遗漏。
- 4/5：一致，仅有轻微差异，不影响整体功能。
- 3/5：基本一致，但具有明显差异，可能会影响代码解释或行为。
- 2/5：明显的差异或遗漏改变了代码的含义或功能，只保留了总体的意图。
- 1/5：与原始代码完全不同，改变了核心功能或含义。

对于每个角度，先简要分析评论优缺点，再给分。最后给出总分。
"""


with open(f'{DIR}/{RESPONDED_DATA[0]}') as fin, open(f'{DIR}/{HUMAN_IN}', 'w', encoding='utf-8') as fout1, open(f'{DIR}/{HUMAN_IN2}', 'w', encoding='utf-8') as fout2:
    print(intro_cn, '\n', file=fout1)
    lines = list(fin)

    for index in range(0, len(lines), RESP_PER_INST):
        js = [json.loads(line) for line in lines[index:index + RESP_PER_INST]]

        src_lang, tgt_lang = js[0]['lang'].split(', ')
        source, targets, reference = js[0]['input'], [s['output'] for s in js], js[0]['gold']

        #! params vary per task
        format_params = ()
        for target in targets:
            format_params += tgt_lang, target

        print(template_cn.format(index, src_lang, tgt_lang, src_lang, source, tgt_lang, reference), file=fout1)
        print(template2_cn.format(index, *format_params), file=fout2)
