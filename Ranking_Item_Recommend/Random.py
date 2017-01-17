from InputData import InputData
from Evaluator import Evaluator
import numpy as np
class Random(InputData):
    def __init__(self, train_file, test_file, topK = 20):
        InputData.__init__(self, train_file, test_file)
        num_item = len(self.item_hash)
        num_user = len(self.uid_hash)
        self.topK = topK
        self.num_item = num_item
        self.num_user = num_user
        self.uid_predict = {}
        self.uid_recommend = {}
        self.evaluator = None
        self.recommend()

    def recommend(self, ):
        for i in xrange(self.num_user):
            self.uid_predict[i] = {}

            for j in xrange(self.num_item):
                if j not in self.train_tuple[i]:
                    self.uid_predict[i][j] = np.random.rand()
            predict = self.uid_predict[i]
            predict = sorted(predict.iteritems(), key=lambda e: e[1], reverse=True)
            recommend_result = predict[:self.topK]
            recommend_result = [elem[0] for elem in recommend_result]
            self.uid_recommend[i] = recommend_result


    def evaluation(self,):
        self.evaluator = Evaluator(self.test_tuple, self.uid_recommend, self.num_user, self.num_item, self.topK)
        self.evaluator.prec_recall()



