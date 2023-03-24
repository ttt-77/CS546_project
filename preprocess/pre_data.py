import json, jsonlines
class vocab():
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0
    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            word = word.lower()
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {'PAD':0, '<s>':1, '</s>':2, 'UNKN':3}
        self.word2count = {}
        self.index2word = ['PAD', '<s>', '</s>', 'UNKN']
        self.n_words = 4  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

def Data_loader(train_file, test_file, max_len, output_train, output_test):
    vo = vocab()

    train_dataset = []
    test_dataset = []
    with open(train_file) as f:
        for line in f:
            d = json.loads(line.strip())
            vo.add_sen_to_vocab(d['question']['sentence'])
            for item in d['sample']:
                vo.add_sen_to_vocab(item['sentence'])
    vo.trim(2)
    word2index = vo.word2index
    index2word = vo.index2word
    with open(train_file) as f:
        for line in f:
            d = json.loads(line.strip())
            question = ['<s>'] + d['question']['sentence'] + ['<\s>']
            new_question = []
            for word in question:
                if word in index2word:
                    new_question.append(word2index[word])
                else:
                    new_question.append(word2index['UNKN'])
            if len(new_question) < max_len:
                new_question.extend([0]*(max_len-len(new_question)))
            else:
                print(1)
            for item in d['sample']:
                answer = ['<s>'] + item['sentence'] + ['<\s>']
                new_answer = []
                for word in answer:
                    if word in index2word:
                        new_answer.append(word2index[word])
                    else:
                        new_answer.append(word2index['UNKN'])
                if len(new_answer) < max_len:
                    new_answer.extend([0] * (max_len - len(new_answer)))
                else:
                    print(0)
                temp = {}
                temp['question'] = new_question
                temp['question_edges'] = d['question']['edges']
                temp['answer'] = new_answer
                temp['answer_edges'] = item['edges']
                temp['label'] = item['label']
                train_dataset.append(temp)
    with open(test_file) as f:
        for line in f:
            d = json.loads(line.strip())
            question = ['<s>'] + d['question']['sentence'] + ['<\s>']
            new_question = []
            for word in question:
                if word in index2word:
                    new_question.append(word2index[word])
                else:
                    new_question.append(word2index['UNKN'])
            if len(new_question) < max_len:
                new_question.extend([0]*(max_len-len(new_question)))
            else:
                print(0)
            for item in d['sample']:
                answer = ['<s>'] + item['sentence'] + ['<\s>']
                new_answer = []
                for word in answer:
                    if word in index2word:
                        new_answer.append(word2index[word])
                    else:
                        new_answer.append(word2index['UNKN'])
                if len(new_answer) < max_len:
                    new_answer.extend([0] * (max_len - len(new_answer)))
                else:
                    print(0)
                temp = {}
                temp['question'] = new_question
                temp['question_edges'] = d['question']['edges']
                temp['answer'] = new_answer
                temp['answer_edges'] = item['edges']
                temp['label'] = item['label']
                temp['output_result'] = item['answer']
                test_dataset.append(temp)


    with jsonlines.open(output_train, 'w') as writer:
        writer.write_all(train_dataset)
    with jsonlines.open(output_test, 'w') as writer:
        writer.write_all(test_dataset)

def merge(file_list):
    res = []
    for file in file_list:
        with open(file) as f:
            for line in f:
                d=json.loads(line.strip())
                res.append(d)
    with jsonlines.open('all_test.json', 'w') as writer:
        writer.write_all(res)
if __name__ == '__main__':
    #file_list = ['./train1_100.json', './train101_200.json', './train201_300.json', './train301_400.json', './train401_500.json', './train501_600.json', './train601_700.json', './train701_800.json', './train801_900.json', './train901_1000.json']
    #merge(file_list)
    #test_file = ['./test1_100.json', './test101_200.json', './test168_300.json', './test301_500.json','./test501_700.json', './test701_900.json', './test901_1000.json']
    #merge(test_file)
    Data_loader('./all_train.json', './all_test.json', 256, 'output_all_train.json', 'output_all_test.json')



