import numpy as np
from InputData import InputData
from Evaluator import Evaluator
class MostPop(InputData):
    def __init__(self, train_file, test_file, topK=20):
        InputData.__init__(self, train_file, test_file)
        num_item = len(self.item_hash)
        num_user = len(self.uid_hash)
        self.topK = topK
        self.num_item = num_item
        self.num_user = num_user
        self.uid_recommend = {}
        self.evaluator = None
        self.recommend()

    def recommend(self,):
        item_pop = {item:0 for item in range(self.num_item)}
        for uid in xrange(self.num_user):
            try:
                item_rating = self.train_tuple[uid]
                for item in item_rating:
                    item_pop[item] += 1
            except:
                print str(uid) + 'containing no training data'
        item_pop =  sorted(item_pop.iteritems(), key = lambda e:e[1], reverse=True)
        recommend = [item[0] for item in item_pop[:self.topK]]
        for uid in xrange(self.num_user):
            self.uid_recommend[uid] = recommend

    def evaluation(self, ):
        self.evaluator = Evaluator(self.test_tuple, self.uid_recommend, self.num_user, self.num_item, self.topK)
        self.evaluator.prec_recall()

