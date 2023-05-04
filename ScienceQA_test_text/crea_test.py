import json

new_dict = {}
with open('./problems.json', encoding="utf-8") as f:
    data = json.load(f)
    for rid, row in data.items():
        data_typ = row['split']
        if data_typ == 'test':
            new_dict[rid] = row


with open('test.json', 'w') as fp:
    json.dump(new_dict, fp, indent=4)