import torch
import torch.nn as nn
import random
import glob
import numpy as np
import torch.nn.functional as F
import copy
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
from tqdm import tqdm, trange
from layers import AttentionModule, TenorNetworkModule
from utils import process_pair, calculate_loss, calculate_normalized_ged
import json, jsonlines
import collections
print("let's use ", torch.cuda.device_count(), "GPUs!")

def Data_load(filename):
    data = []
    with open(filename,'r') as f:
        for line in f:
            d = json.loads(line.strip())
            if d['answer_edges'] != [] and d['question_edges'] != []:
                data.append(d)
    return data

def evaluate(l):
    truth = 0
    false = 0
    major = 0
    pro = 0
    for d in l:
        pos = 0
        neg = 0
        pos_count =0
        neg_count = 0
        prediction = d['prediction']
        target = d['target']
        answer = d['answer']
        result_count = collections.Counter(answer)
        result_probability = collections.defaultdict(float)
        truth = None
        for i in range(len(target)):
            result_probability[answer[i]] +=prediction[i]
            if target[i] == 1:
              truth = answer[i]
        # print(result_count, result_probability)
        max_value1 = max(result_count, key=result_count.get)
        max_value2 = max(result_probability, key=result_probability.get)
        if max_value1 == truth:
          major += 1
        if max_value2 == truth:
          pro += 1
    print(major/len(l), pro/len(l))
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.lstm = nn.GRU(input_size = embedding_size, hidden_size = hidden_size, num_layers = n_layers,dropout=dropout, bidirectional = True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        embedded = self.em_dropout(embedded)
        outputs, _ = self.lstm(embedded)
        return  outputs

class SimGNN(nn.Module):
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.encoder1 = Encoder(self.args.vocab_size, self.args.embedding_size, self.args.hidden_size)
        self.encoder2 = Encoder(self.args.vocab_size, self.args.embedding_size, self.args.hidden_size)
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1, add_self_loops=True)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist / torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        output_1 = self.encoder1(features_1,256)
        output_2 = self.encoder2(features_2,256)
        abstract_features_1 = self.convolutional_pass(edge_index_1, output_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, output_2)

        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score

class SimGNNTrainer(object):
    """
    SimGNN model trainer.
    """
    def __init__(self, args, sequence_length):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.initial_label_enumeration()
        self.number_of_features = sequence_length
        self.setup_model()

    def setup_model(self):
        """
        Creating a SimGNN.
        """
        device = torch.device("cuda")
        self.model = SimGNN(self.args, self.number_of_features)
        self.model = self.model.to(device)
    def initial_label_enumeration(self):
        """
        Collecting the unique node idsentifiers.
        """
        print("\nEnumerating unique labels.\n")
        self.training_graphs = Data_load(self.args.training_graphs)
        self.testing_graphs = Data_load(self.args.testing_graphs)

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph:graph + self.args.batch_size])
        return batches

    def transfer_to_torch(self, data):
        """
        Transferring the data to torch and creating a hash table.
        Including the indices, features and target.
        :param data: Data dictionary.
        :return new_data: Dictionary of Torch Tensors.
        """
        device = torch.device("cuda")
        new_data = dict()
        edges_1 = data["question_edges"] + [[y, x] for x, y in data["question_edges"]]

        edges_2 = data["answer_edges"] + [[y, x] for x, y in data["answer_edges"]]
        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)
        edges_1 = edges_1.to(device)
        edges_2 = edges_2.to(device)

        question = torch.from_numpy(np.array(data["question"], dtype=np.int64).T).type(torch.long)
        answer = torch.from_numpy(np.array(data["answer"], dtype=np.int64).T).type(torch.long)
        question = question.to(device)
        answer = answer.to(device)



        new_data["edge_index_1"] = edges_1
        new_data["edge_index_2"] = edges_2

        new_data["features_1"] = question
        new_data["features_2"] = answer

        target = torch.from_numpy(np.array(data["label"], dtype=np.int64).T).type(torch.long)
        new_data["target"] = target.to(device)
        
        return new_data

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        lo = torch.nn.BCELoss()
        losses = 0
        device = torch.device("cuda")
        for graph_pair in batch:
            data = self.transfer_to_torch(graph_pair)
            target = data["target"]
            prediction = self.model(data)
            target = torch.tensor([[target.item()]])
            target = target.to(device)
            losses = losses + lo(prediction, target.float())
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        best_accuracy = 0 
        for epoch in epochs:
            batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum / main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
            if epoch % 5 == 4:
                accuracy = self.score()
                if accuracy > best_accuracy:
                    print(epoch,accuracy)
                    best_accuracy = accuracy
                    self.save('./best_model3')
                with open('accuracy.json','a') as f:
                    f.write(str(epoch))
                    f.write('\t')
                    f.write(str(accuracy))
                    f.write('\n')
                self.model.train()    

    def score(self):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.scores = []
        self.ground_truth = []
        result = []
        temp_question = ''
        temp = {'prediction':[], 'target':[], 'answer':[]}
        for graph_pair in tqdm(self.testing_graphs):
            data = self.transfer_to_torch(graph_pair)
            # self.ground_truth.append(calculate_normalized_ged(data))
            target = data["target"]
            prediction = self.model(data)
            new_question = graph_pair['question']
            if new_question != temp_question:
                if temp['prediction']:
                    result.append(temp)
                    temp = {'prediction':[], 'target':[], 'answer':[]}
                temp_question = new_question
            temp['prediction'].append(prediction.item())
            temp['target'].append(target.item())
            temp['answer'].append(graph_pair['output_result'])
        accuracy = self.print_evaluation(result)
        with jsonlines.open('result.json', 'w') as writer:
            writer.write_all(result)
        return accuracy
    def print_evaluation(self,l):
        """
        Printing the error rates.
        """
        # norm_ged_mean = np.mean(self.ground_truth)
        # base_error = np.mean([(n - norm_ged_mean) ** 2 for n in self.ground_truth])
        # model_error = np.mean(self.scores)
        # print("\nBaseline error: " + str(round(base_error, 5)) + ".")
        # print("\nModel test error: " + str(round(model_error, 5)) + ".")
        truth = 0
        false = 0
        major = 0
        pro = 0
        for d in l:
            pos = 0
            neg = 0
            pos_count =0
            neg_count = 0
            prediction = d['prediction']
            target = d['target']
            answer = d['answer']
            result_count = collections.Counter(answer)
            result_probability = collections.defaultdict(float)
            truth = None
            for i in range(len(target)):
                result_probability[answer[i]] +=prediction[i]
                if target[i] == 1:
                  truth = answer[i]
            # print(result_count, result_probability)
            max_value1 = max(result_count, key=result_count.get)
            max_value2 = max(result_probability, key=result_probability.get)
            if max_value1 == truth:
              major += 1
            if max_value2 == truth:
              pro += 1
        print('major:',major/len(l), 'pro:',pro/len(l))
        return pro/len(l)
    def save(self,path = None):
        if not path:
            path = self.args.save_path
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load(self.args.load_path))