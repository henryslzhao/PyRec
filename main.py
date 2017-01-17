import argparse
import warnings
import math
import sys
import numpy as np
import time

from Ranking_Item_Recommend.Random import Random
from Ranking_Item_Recommend.MostPop import MostPop
from Ranking_Item_Recommend.WRMF import WRMF
from Ranking_Item_Recommend.BPRMF import BPRMF
from Rating_Item_Recommend.PMF import PMF

def run(train_file, test_file, model):
    if model == 'Random':
        r = Random(train_file, test_file)
        r.evaluation()
    elif model == 'MostPop':
        r = MostPop(train_file, test_file)
        r.evaluation()
    elif model == 'WRMF':
        r = WRMF(train_file, test_file)
        r.evaluation()
    elif model == 'BPRMF':
        r = BPRMF(train_file, test_file)
        r.evaluation()
    elif model == 'PMF':
        r = PMF(train_file, test_file)
        r.evaluation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training file', dest='train_file', required=True)
    parser.add_argument('-test', help='Test file', dest='test_file', required=True)
    parser.add_argument('-model', help='recommender model, e.g., Random, MostPop, MF, BiasedMF, BPR, WRMF', dest='model', default='Random')

    args = parser.parse_args()

    run(args.train_file, args.test_file, args.model)