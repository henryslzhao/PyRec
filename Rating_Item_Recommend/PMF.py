import numpy as np
import time
import sys
from math import exp
from InputData import InputData
from Evaluator import Evaluator


class PMF(InputData):
    '''
    PMF: probabilistic matrix factorization,nips08
    '''
    def __init__(self, train_file, test_file, topK=20, num_factor=10,  learning_rate=0.0002, reg_param = 0.025):
        InputData.__init__(self, train_file, test_file)
        num_item = len(self.item_hash)
        num_user = len(self.uid_hash)
        self.topK = topK
        self.num_item = num_item
        self.num_user = num_user
        self.num_factor = num_factor
        #self.num_iteration = num_iteration
        self.learning_rate = learning_rate
        self.reg_param = reg_param

        self.uid_predict = {}
        self.uid_recommend = {}
        self.evaluator = None
        self.converged = False
        self.user_vectors = np.random.random((self.num_user, self.num_factor))
        self.item_vectors = np.random.random((self.num_item, self.num_factor))

        self.new_user_vectors = np.random.random((self.num_user, self.num_factor))
        self.new_item_vectors = np.random.random((self.num_item, self.num_factor))

        print "number of user % i" % self.num_user
        print "number of item %i " % self.num_item
        self.recommend()

    def recommend(self, ):
        self.__train_model()
        for i in xrange(self.num_user):
            self.uid_predict[i] = {}
            for j in xrange(self.num_item):
                if j not in self.train_tuple[i]:
                    self.uid_predict[i][j] = np.dot(self.user_vectors[i], self.item_vectors[j])
            predict = self.uid_predict[i]
            predict = sorted(predict.iteritems(), key=lambda e: e[1], reverse=True)
            recommend_result = predict[:self.topK]
            recommend_result = [elem[0] for elem in recommend_result]
            self.uid_recommend[i] = recommend_result

    def evaluation(self, ):
        self.evaluator = Evaluator(self.test_tuple, self.uid_recommend, self.num_user, self.num_item, self.topK)
        self.evaluator.prec_recall()


    def __train_model(self,):
        updates_o = np.zeros((self.num_user, self.num_factor))
        updates_d = np.zeros((self.num_item, self.num_factor))
        init_likelihood = self.__likelihood(self.user_vectors, self.item_vectors)
        print init_likelihood
        while (not self.converged ):
            updates_o, updates_d = self.__update_od(updates_o, updates_d)
            self.__try_update(updates_o ,updates_d)
            final_likelihood = self.__likelihood(self.new_user_vectors, self.new_item_vectors)
            print  final_likelihood
            if final_likelihood > init_likelihood:
                self.__apply_update()
                self.learning_rate *= 1.25
                if final_likelihood - init_likelihood < 0.1:
                    self.converged = True
                init_likelihood = final_likelihood
            else:
                self.learning_rate *= 0.5

            if self.learning_rate < 1e-10:
                self.converged = True


    def __update_od(self, updates_o, updates_d):
        for user in xrange(self.num_user):
            for item in self.train_tuple[user]:
                rating = self.train_tuple[user][item]
                r_hat = np.dot(self.user_vectors[user], self.item_vectors[item])
                updates_o[user] += self.item_vectors[item] * (rating - r_hat)
                updates_d[item] += self.user_vectors[user] * (rating - r_hat)
        return  updates_o, updates_d


    def __try_update(self, updates_o, updates_d):
        alpha = self.learning_rate
        beta = -self.reg_param
        for i in xrange(self.num_user):
            self.new_user_vectors[i] = self.user_vectors[i] + alpha * (beta * self.user_vectors[i] + updates_o[i])
        for i in xrange(self.num_item):
            self.new_item_vectors[i] = self.item_vectors[i] + alpha * (beta * self.item_vectors[i] + updates_d[i])

    def __apply_update(self,):
        for i in xrange(self.num_user):
            self.user_vectors[i] = self.new_user_vectors[i]
        for i in xrange(self.num_item):
            self.item_vectors[i] = self.new_item_vectors[i]

    def __likelihood(self, user_vectors, item_vectors ):
        sq_error = 0
        L2_norm = 0
        for user in xrange(self.num_user):
            for item in self.train_tuple[user]:
                r_hat = np.dot(user_vectors[user], item_vectors[item])
                rating = self.train_tuple[user][item]
                sq_error = (r_hat - rating) ** 2
                L2_norm += np.dot(user_vectors[user], user_vectors[user]) + np.dot(item_vectors[item], item_vectors[item])
        return -0.5 * (sq_error + L2_norm * self.reg_param)


