# -*-coding:utf-8 -*-
import math
import random
#import os; os.chdir('./exp_4')
random.seed(0)
import multiprocessing,time

def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
    
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)
        return error

    def test(self):
        cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
        labels = [[0], [1], [1], [0]]
        self.setup(2, 5, 1)
        self.train(cases, labels, 10000, 0.05, 0.1)
        for case in cases:
            print(self.predict(case))

def line_process(input):
    case = [-1 for i in range(len(input) - 1)]
    label = [0] if input[-1] == 'no' else [1]
    base = (('no-recurrence-events', 'recurrence-events'),\
            ('10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'),
            ('lt40', 'ge40', 'premeno'),
            ('0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44','45-49', '50-54', '55-59'),
            ('0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26','27-29', '30-32', '33-35', '36-39'),
            ('yes', 'no'),('1','2','3'),('left','right'),('left_up', 'left_low', 'right_up','right_low', 'central'))
    for index,t in enumerate(input[:-1]):
        for num,content in enumerate(base[index]):
            if content == t:case[index]=num
    #remove lost data
    for i in case:
        if i <0:
            return 0,0
    return case,label
if __name__ == '__main__':
    data_set = [];labels = []
    with open('./breast-cancer.data.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            if line:
                i = line.strip().split(',')
                case,label = line_process(i)
                if case:data_set.append(case);labels.append(label)

    def newTrain(mtp,limit,learn,correct):
        print('NN_{} begin, learn={}, correct={}, limit = {}'.\
        format(mtp,learn,correct,limit))
        t = time.time()
        nn = BPNeuralNetwork()
        nn.setup(9, 2, 1)
        error = nn.train(data_set,labels,limit=limit,learn=learn,correct=correct)
        dt = round(time.time() - t,2)
        print('NN_{} done , spent {}s, with error {}'\
        .format(mtp,dt,round(error,4)))

    a = multiprocessing.Process(target=newTrain,args=('test',1000000,0.1,0.1))
    a.start()
    '''
    for i in range(1,10):
        a = multiprocessing.Process(target=newTrain,args=(i,100000,i/10.0,0.1))
        a.start()
    '''
    