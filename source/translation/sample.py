import json
from random import sample, randrange, shuffle
from config import DIR, FILTERED_DATA, SAMPLED_DATA, NUM_SAMPLES

for filtered_data, sampled_data, num_samples in zip(FILTERED_DATA, SAMPLED_DATA, NUM_SAMPLES):
    with open(f'{DIR}/{filtered_data}') as fin, open(f'{DIR}/{sampled_data}', 'w') as fout:
        contents = {}
        for line in fin:
            js = json.loads(line)
            lang = js['lang']
            if lang not in contents:
                contents[lang] = []

            contents[lang] += [js]

        contents = dict(sorted(contents.items(), key=lambda item: len(item[1])))
        all_items = []
        for i, (lang, items) in enumerate(contents.items()):
            remaining = len(contents) - i
            goal = min((num_samples + remaining - 1) // remaining, len(items))

            num_samples -= goal
            all_items += sample(items, goal)
            print(f"- Language '{lang}': {goal} samples; {num_samples} more needed")

        shuffle(all_items)
        for content in all_items:
            print(json.dumps(content), file=fout)

        print(f"{sampled_data}: {len(all_items)} samples total\n")
