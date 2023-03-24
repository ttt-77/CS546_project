import json, jsonlines
import collections
def extract(filename,output_file):
    dictionary = collections.defaultdict(dict)
    with open(filename, 'r') as f:
        for line in f.readlines():
            d = json.loads(line.strip())
            question = d['metadata']['question']
            if question not in dictionary:
                dictionary[question]['ground_truth'] = d['metadata']['ground_truth']
                dictionary[question]['sample'] = []
            for sam in d['samples']:
                dictionary[question]['sample'].append(sam)
    res = []
    for key in dictionary.keys():
        temp_dict = {}
        temp_dict['question'] = key
        temp_dict['ground_truth'] = dictionary[key]['ground_truth']
        temp_dict['sample'] = dictionary[key]['sample']
        res.append(temp_dict)
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(res)
    print(len(res))

if __name__ == '__main__':
    extract('./train.jsonl','./math_train.json')
    extract('./test.jsonl', './math_test.json')
