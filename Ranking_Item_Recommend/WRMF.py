import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import time
from InputData import InputData
from Evaluator import Evaluator


class WRMF(InputData):
    '''
    WRMF: implicit matrix factorization, http://labs.yahoo.com/files/HuKorenVolinsky-ICDM08.pdf.
    '''

    def __init__(self, train_file, test_file, topK=20, num_factor=40, num_iteration=10, reg_param=0.2):
        InputData.__init__(self, train_file, test_file)
        num_item = len(self.item_hash)
        num_user = len(self.uid_hash)
        self.topK = topK
        self.num_item = num_item
        self.num_user = num_user
        self.num_factor = num_factor
        self.num_iteration = num_iteration
        self.reg_param = reg_param
        self.counts = None
        self.uid_predict = {}
        self.uid_recommend = {}
        self.evaluator = None
        self.user_vectors = None
        self.item_vectors = None

        self.recommend()

    def recommend(self, ):
        self.__load_matrix()
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

    def __load_matrix(self):
        t0 = time.time()
        self.counts = np.zeros((self.num_user, self.num_item))
        total = 0.0
        num_zeros = self.num_item * self.num_user
        for uid in xrange(self.num_user):
            for item in self.train_tuple[uid]:
                self.counts[uid][item] = self.train_tuple[uid][item]
                total += self.train_tuple[uid][item]
                num_zeros -= 1
        alpha = num_zeros / total
        print 'alpha %.2f' % alpha
        self.counts *= alpha
        self.counts = sparse.csr_matrix(self.counts)
        t1 = time.time()
        print "finishing load matrix in %f seconds" % (t1-t0)

    def __train_model(self,):
        self.user_vectors = np.random.normal(size=(self.num_user, self.num_factor))
        self.item_vectors = np.random.normal(size=(self.num_item, self.num_factor))
        for i in xrange(self.num_iteration):
            t0 = time.time()
            self.user_vectors = self.__iteration(True, sparse.csr_matrix(self.item_vectors))
            self.item_vectors = self.__iteration(False, sparse.csr_matrix(self.user_vectors))
            t1 = time.time()
            print 'iteration %i finished in %f seconds' % (i+1, t1-t0)

    def __iteration(self, update_user, fixed_vecs):
        num_solve = self.num_user if update_user else self.num_item
        num_fixed = fixed_vecs.shape[0]
        YTY = fixed_vecs.T.dot(fixed_vecs)
        eye = sparse.eye(num_fixed)
        lambda_eye = self.reg_param * sparse.eye(self.num_factor)
        solve_vecs = np.zeros((num_solve, self.num_factor))
        for i in xrange(num_solve):
            if update_user:
                counts_i = self.counts[i].toarray()
            else:
                counts_i = self.counts[:, i].T.toarray()
            CuI = sparse.diags(counts_i, [0])
            pu = counts_i.copy()
            pu[np.where(pu != 0)] = 1.0
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            solve_vecs[i] = xu
        return  solve_vecs





    def evaluation(self, ):
        self.evaluator = Evaluator(self.test_tuple, self.uid_recommend, self.num_user, self.num_item, self.topK)
        self.evaluator.prec_recall()
