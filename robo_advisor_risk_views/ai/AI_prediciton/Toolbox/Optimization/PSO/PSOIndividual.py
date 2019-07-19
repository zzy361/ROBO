import numpy as np
from Toolbox.Optimization.ObjFunction import obj_function
import copy


class PSOIndividual:
    '''
    individual of PSO
    '''

    def __init__(self, vardim, bound, back_para, back_test_dict):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.
        self.back_para = back_para
        self.back_test_dict = back_test_dict

    def generate(self):
        '''
        generate a random chromsome
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        self.velocity = np.random.random(size=len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0, i] + \
                            (self.bound[1, i] - self.bound[0, i]) * rnd[i]
        self.bestPosition = np.zeros(len)
        self.bestFitness = 0.

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness = obj_function(self.chrom, self.back_para, self.back_test_dict)
