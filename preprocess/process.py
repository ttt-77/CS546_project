# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from stanfordcorenlp import StanfordCoreNLP
import collections
import re
import json, jsonlines

def sentenceRelation(nlp_parser, sen):
    Dependency = nlp_parser.dependency_parse(' '.join(sen))
    pos = nlp_parser.pos_tag(' '.join(sen))
    relationDict = collections.defaultdict(dict)
    relationDict[0]['value'] = 'ROOT'
    relationDict[0]['pos'] = 'ROOT'
    relationDict[0]['child'] = []
    i = 1
    new_sen = []
    while i < len(pos)+1:
        new_sen.append(pos[i-1][0])
        relationDict[i]['value'] = pos[i-1][0]
        relationDict[i]['pos'] = pos[i - 1][1]
        relationDict[i]['child'] = []
        i += 1
    for item in Dependency:
        relationDict[item[1]]['child'].append((item[0],item[2]))
        relationDict[item[2]]['parent'] = (item[0],item[1])
    return relationDict, new_sen

def trim(relation, index):
    cells = []
    cell = []
    for key in relation.keys():
        if relation[key]['pos'] == 'CD':
            if cell:
                cells.append(list(set(cell)))
            cell = []
            cell.append(key)
            parent = relation[key]['parent']
            if parent[0] == 'nummod':
                cell.append(parent[1])
                nummod_child =  relation[parent[1]]['child']
                for item in nummod_child:
                    if item[0] == 'compound' or 'amod':
                        cell.append(item[1])
            temp_key = parent[1]
            temp_list = [temp_key]
            while len( relation[temp_key]['pos']) < 2 or (relation[temp_key]['pos'][:2] != 'VB' and relation[temp_key]['pos'][:2] != 'RO'):
                parent = relation[temp_key]['parent']
                temp_key = parent[1]
                temp_list.append(temp_key)
            if relation[temp_key]['pos'][:2] == 'VB':
                for item in temp_list:
                    cell.append(item)
            #check 'per', 'every', 'each'
            for temp in relation.keys():
                if relation[temp]['value'].lower() in ['per', 'every', 'each']:
                    temp_list = [temp]
                    temp_parent =  relation[temp]['parent']
                    temp_key = temp_parent[1]
                    temp_list.append(temp_key)
                    while len(relation[temp_key]['pos']) < 2 or (
                            relation[temp_key]['pos'][:2] != 'VB' and relation[temp_key]['pos'][:2] != 'RO'):
                        temp_parent = relation[temp_key]['parent']
                        temp_key = temp_parent[1]
                        temp_list.append(temp_key)
                    if relation[temp_key]['pos'][:2] == 'VB' and temp_key in cell:
                        for temp_item in temp_list:
                            cell.append(temp_item)
        if relation[key]['value'] == '?':
            temp_key = relation[key]['parent'][1]
            temp_key_list = [temp_key]
            for child in relation[temp_key]['child']:
                if child[0] == 'nsubj' or child[0] == 'obj' or child[0] == 'obl':
                    temp_key_list.append(child[1])
                    for sec_child in relation[child[1]]['child']:
                        if sec_child[0] == 'amod' or sec_child[0] == 'compound' or sec_child[0] == 'nmod' or sec_child[0] == 'nmod:poss':
                            temp_key_list.append(sec_child[1])
            for item in temp_key_list:
                cell.append(item)



    if cell:
        cells.append(list(set(cell)))
    TreeList = []
    for cell in cells:
        newTree = {}
        for item in cell:
            new_key = item + index
            newTree[new_key] = {}
            newTree[new_key]['value'] =  relation[item]['value']
            newTree[new_key]['pos'] = relation[item]['pos']
            newTree[new_key]['child'] = []
            if relation[item]['parent'][1] in cell:
                newTree[new_key]['parent'] = (relation[item]['parent'][0],relation[item]['parent'][1]+index)
            else:
                newTree[new_key]['parent'] = (None,None)
            for child in relation[item]['child']:
                if child[1] in cell:
                    newTree[new_key]['child'].append((child[0],child[1]+index))
        TreeList.append(newTree)
    return TreeList

def answer_process(sentence):
    answer = re.findall('####[\s|\d|\.]*', sentence)
    for item in answer:
        sentence= sentence.replace(item,'')
    if answer:
        answer = answer[0].replace("####", '').strip()
    else:
        answer = None
    sentence = sentence.replace('\n', '. ')
    result = re.findall('<<[\d|\+|\-|\*|\/|\s|\=|\.|\(|\)|\[|\]|\{|\}]*>>',sentence)
    for item in result:
        sentence= sentence.replace(item,'')
    return sentence,answer

def denpendency(raw_data):
    # Use a breakpoint in the code line below to debug your script.

    nlp_parser = StanfordCoreNLP('http://localhost', port=9000, timeout=15000)
    pos = nlp_parser.pos_tag(raw_data)
    question = {}
    sentences = []
    sen = []
    graph = []
    index = 0
    for i in range(len(pos)):
        sen.append(pos[i][0])
        if pos[i][1] == '.':
            relation,new_sen = sentenceRelation(nlp_parser, sen)
            sentences.extend(new_sen)
            TreeList = trim(relation, index)
            index += len(new_sen)
            graph.extend(TreeList)
            sen = []
    question['sentence'] = sentences
    question['graph'] = graph
    edges = []
    for item in graph:
        for key in item.keys():
            for child in item[key]['child']:
                edges.append((key,child[1]))
    question['edges'] = edges
    return question
def process(filename,output_file):
    result = []
    index = 0
    with open(filename) as f:
        for line in f:
            print(index)
            if index <1104:
                index += 1
                continue
            index += 1
            temp = {}
            d = json.loads(line.strip())
            question = d['question']
            ground_truth = d['ground_truth']
            sample = d['sample']
            processed_question = denpendency(question)
            new_ground_truth, truth = answer_process(ground_truth)
            processed_ground_truth = denpendency(new_ground_truth)

            processed_sample = []
            for item in sample:
                new_item, answer = answer_process(item)
                if answer == None:
                    continue
                temp_item = denpendency(new_item)
                if answer == truth:
                    temp_item['label'] = 1
                else:
                    temp_item['label'] = 0
                temp_item['answer'] = answer
                temp_item['truth'] = truth
                processed_sample.append(temp_item)
            temp['question'] = processed_question
            temp['ground_truth'] = processed_ground_truth
            temp['sample'] = processed_sample
            with jsonlines.open(output_file, 'a') as writer:
                writer.write_all([temp])
            result.append(temp)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    process('../math_test.json', '../test901_1000.json')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
