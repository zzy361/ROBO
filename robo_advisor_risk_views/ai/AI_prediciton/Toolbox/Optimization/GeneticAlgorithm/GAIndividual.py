import numpy as np
from jfquant.Optimization.ObjFunction import obj_function


class GAIndividual:
    '''
    individual of genetic algorithm
    '''

    def __init__(self, vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.

    def generate(self):
        '''
        generate a random chromsome for genetic algorithm
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0][i] + (self.bound[1][i] - self.bound[0][i]) * rnd[i]

    def calculateFitness(self, chrom, backtest_para_dict, opt_para_name):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness = obj_function(chrom, backtest_para_dict, opt_para_name)
