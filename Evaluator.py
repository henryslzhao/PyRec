class Evaluator:
    def __init__(self, test_data, recommend_data, num_user, num_item, topK=20):
        self.topK = topK
        self.num_user = num_user
        self.num_item = num_item
        self.test_tuple = test_data
        self.uid_recommend = recommend_data

    def prec_recall(self, ):
        sum_prec = [0] * self.topK
        sum_recall = [0] * self.topK
        count = 0
        for k in xrange(self.topK):
            for uid in xrange(self.num_user):
                if len(set(self.test_tuple[uid].keys())) == 0:
                    continue
                correct_recommend = set(self.uid_recommend[uid][:k + 1]) & set(self.test_tuple[uid].keys())
                prec = len(correct_recommend) * 1.0 / (k + 1)
                recall = len(correct_recommend) * 1.0 / len(set(self.test_tuple[uid].keys()))
                if k == 0:
                    count += 1
                sum_prec[k] += prec
                sum_recall[k] += recall
        avg_prec = [item / count for item in sum_prec]
        avg_recall = [item / count for item in sum_recall]
        print count
        print avg_prec
        print avg_recall
