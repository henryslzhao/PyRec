import numpy as np
import time
import sys
from math import exp
from InputData import InputData
from Evaluator import Evaluator


class BPRMF(InputData):
    '''
    BPRMF: implicit matrix factorization, Bayesian Personalized Ranking
    '''
    def __init__(self, train_file, test_file, topK=20, num_factor=30, num_iteration=10, learning_rate=0.05, bias_reg_param = 1, reg_param = 0.0025, \
                 neg_reg_param = 0.00025, num_neg_sample=10):
        InputData.__init__(self, train_file, test_file)
        num_item = len(self.item_hash)
        num_user = len(self.uid_hash)
        self.topK = topK
        self.num_item = num_item
        self.num_user = num_user
        self.num_factor = num_factor
        self.num_iteration = num_iteration
        self.learning_rate = learning_rate
        self.bias_reg_param = bias_reg_param
        self.reg_param = reg_param
        self.neg_reg_param = neg_reg_param
        self.num_neg_sample = num_neg_sample

        self.counts = None
        self.uid_predict = {}
        self.uid_recommend = {}
        self.evaluator = None
        self.user_vectors = np.random.random_sample((self.num_user, self.num_factor))
        self.item_vectors = np.random.random_sample((self.num_item, self.num_factor))
        self.item_bias = np.zeros(self.num_item)
        print "number of user % i" % self.num_user
        print "number of item %i " % self.num_item

        self.recommend()

    def recommend(self, ):
        self.__train_model()
        for i in xrange(self.num_user):
            self.uid_predict[i] = {}
            for j in xrange(self.num_item):
                if j not in self.train_tuple[i]:
                    self.uid_predict[i][j] = np.dot(self.user_vectors[i], self.item_vectors[j]) + self.item_bias[j]
            predict = self.uid_predict[i]
            predict = sorted(predict.iteritems(), key=lambda e: e[1], reverse=True)
            recommend_result = predict[:self.topK]
            recommend_result = [elem[0] for elem in recommend_result]
            self.uid_recommend[i] = recommend_result

    def evaluation(self, ):
        self.evaluator = Evaluator(self.test_tuple, self.uid_recommend, self.num_user, self.num_item, self.topK)
        self.evaluator.prec_recall()

    def __update(self, u, i, j):
        x = self.item_bias[i] - self.item_bias[j] + np.dot(self.user_vectors[u, :], self.item_vectors[i, :]-self.item_vectors[j, :])
        if x > 9:
            z = 0
        elif x < -9:
            z = 1
        else:
            z = 1.0 / (1.0 + exp(x))
        #update parameters
        self.item_bias[i] += self.learning_rate * (z - self.bias_reg_param * self.item_bias[i])
        self.item_bias[j] += self.learning_rate * (-z - self.bias_reg_param * self.item_bias[j])
        self.user_vectors[u, :] += self.learning_rate * ((self.item_vectors[i, :] - self.item_vectors[j, :]) * z - self.reg_param * self.user_vectors[u, :])
        self.item_vectors[i, :] += self.learning_rate * (self.user_vectors[u, :] * z - self.reg_param * self.item_vectors[i, :])
        self.item_vectors[j, :] += self.learning_rate * (-self.user_vectors[u, :] * z - self.neg_reg_param * self.item_vectors[j, :])

    def __iteration(self):
        for u in xrange(self.num_user):
            for i in self.train_tuple[u]:
                num_sample = self.num_neg_sample
                while num_sample:
                    num_sample -= 1
                    j = np.random.randint(0, self.num_item)
                    while j in self.train_tuple[u]:
                        j = np.random.randint(0, self.num_item)
                    self.__update(u, i, j)

    def __random_iteration(self,):
        user_item_pairs = []
        for u in xrange(self.num_user):
            for i in self.train_tuple[u]:
                user_item_pairs.append((u, i))
        import random
        random.shuffle(user_item_pairs)
        for u, i in user_item_pairs:
            num_sample = self.num_neg_sample
            while num_sample:
                num_sample -= 1
                j = np.random.randint(0, self.num_item)
                while j in self.train_tuple[u]:
                    j = np.random.randint(0, self.num_item)
                self.__update(u, i, j)


    def __train_model(self):
        for i in xrange(self.num_iteration):
            t0 = time.time()
            self.__iteration()
            print 'iteration %i costs time %f' % (i+1, time.time() - t0)



