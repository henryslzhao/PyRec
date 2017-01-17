from InputData import InputData
from Evaluator import Evaluator
import numpy as np


def sigmoid(x):
    import math
    return math.exp(x) / (1 + math.exp(x))
def delta(x):
    return sigmoid(x) * (1 - sigmoid(x))

class EmbeddingWarp(InputData):
    def __init__(self, train_file, test_file, item_vector_file, gamma=0.005, C=1, num_iteration=5, num_factor=10, epson=0.3, topK = 20):
        InputData.__init__(self, train_file, test_file)
        num_item = len(self.item_hash)
        num_user = len(self.uid_hash)
        self.gamma = gamma
        self.C = C
        self.num_iteration = num_iteration
        self.epson = epson
        self.topK = topK
        self.num_item = num_item
        self.num_user = num_user
        self.uid_predict = {}
        self.uid_recommend = {}
        self.evaluator = None
        self.user_vectors = np.random.normal(size=(self.num_user, num_factor))
        self.item_vectors = self.load_item_vectors(item_vector_file)
        self.train()
        self.recommend()

    def load_item_vectors(self, fname):
        try:
            ifile = open(fname, 'r')
        except:
            print 'cannot open %s' % fname
        count = 0
        vocab_vector = {}
        while True:
            line = ifile.readline().strip()
            if not line:
                break
            line = line.split()
            if count == 0:
                num_vocab = int(line[0])
                num_dim = int(line[1])
            else:
                word = line[0]
                item = self.item_hash[word]
                vec = np.array([float(e) for e in line[1:]])
                vocab_vector[item] = vec
            count += 1
        ifile.close()
        return  vocab_vector

    def __get_user_item_pair(self, ):
        user_item_pairs = []
        for user in self.train_tuple:
            for item in self.train_tuple[user]:
                user_item_pairs.append((user, item))
        return  user_item_pairs

    def __get_rating(self, user, item):
        return np.dot(self.user_vectors[user], self.item_vectors[item])

    def recommend(self, ):
        for i in xrange(self.num_user):
            self.uid_predict[i] = {}
            for j in xrange(self.num_item):
                if j not in self.train_tuple[i]:
                    self.uid_predict[i][j] = self.__get_rating(i, j)
            predict = self.uid_predict[i]
            predict = sorted(predict.iteritems(), key=lambda e: e[1], reverse=True)
            recommend_result = predict[:self.topK]
            recommend_result = [elem[0] for elem in recommend_result]
            self.uid_recommend[i] = recommend_result

    def train(self,):
        user_item_pairs = []
        for user in self.train_tuple:
            for item in self.train_tuple[user]:
                user_item_pairs.append((user, item))

        for iter in range(self.num_iteration):
            import random
            random.shuffle(user_item_pairs)
            self.__iteration(user_item_pairs, self.gamma, self.C)


    def __iteration(self, user_item_pairs, gama, C):
        for uid, item in user_item_pairs:
            y = self.__get_rating(uid, item)
            num_sample = 0
            while num_sample < self.num_item:
                num_sample += 1
                sample_item = np.random.randint(0, self.num_item - 1)
                y_sample = self.__get_rating(uid, sample_item)
                x = self.train_tuple[uid][item]
                if sample_item not in self.train_tuple[uid]:
                    x_sample = 0
                else:
                    x_sample = self.train_tuple[uid][sample_item]
                if (x - x_sample) * (y_sample + self.epson - y) > 0:
                    break
            eta = self.num_item / num_sample * delta(y_sample + self.epson - y)
            self.user_vectors[uid] += gama * eta * (self.item_vectors[item] - self.item_vectors[sample_item])
            # projection
            self.user_vectors[uid] /= np.linalg.norm(self.user_vectors[uid]) / C

    def evaluation(self,):
        self.evaluator = Evaluator(self.test_tuple, self.uid_recommend, self.num_user, self.num_item, self.topK)
        self.evaluator.prec_recall()



