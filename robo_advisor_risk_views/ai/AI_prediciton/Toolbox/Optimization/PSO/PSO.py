import numpy as np
from Toolbox.Optimization.PSO.PSOIndividual import PSOIndividual
import random
import copy
import matplotlib.pyplot as plt


class PSO:
    '''
    the class for Particle Swarm Optimization
    '''

    def __init__(self, optimizer_para_dict,backtest_para_dict, bound, opt_para_dict):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        params: algorithm required parameters, it is a list which is consisting of[w, c1, c2]
        '''
        self.sizepop = optimizer_para_dict['pop_size']
        self.vardim = len(bound[0])
        self.bound = bound
        self.MAXGEN = optimizer_para_dict['iter_num']
        self.params = list(optimizer_para_dict['para_dict'].values())
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.opt_para_dict = opt_para_dict
        self.backtest_para_dict = backtest_para_dict
    def initialize(self):
        '''
        initialize the population of pso
        '''
        for i in range(0, self.sizepop):
            ind = PSOIndividual(self.vardim, self.bound, self.opt_para_dict, self.backtest_para_dict)
            ind.generate()
            self.population.append(ind)

    def evaluation(self):
        '''
        evaluation the fitness of the population
        '''
        for i in range(0, self.sizepop):
            print(i)
            self.population[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness
            if self.population[i].fitness > self.population[i].bestFitness:
                self.population[i].bestFitness = self.population[i].fitness
                self.population[i].bestIndex = copy.deepcopy(
                    self.population[i].chrom)

    def update(self):
        '''
        update the population of pso
        '''
        for i in range(0, self.sizepop):
            self.population[i].velocity = self.params[0] * self.population[i].velocity + self.params[1] * np.random.random(self.vardim) * (
                self.population[i].bestPosition - self.population[i].chrom) + self.params[2] * np.random.random(self.vardim) * (self.best.chrom - self.population[i].chrom)
            self.population[i].chrom = self.population[i].chrom + self.population[i].velocity

            for j in range(len(self.population[i].chrom)):
                if self.population[i].chrom[j] < self.bound[0][j]:
                    self.population[i].chrom[j] = self.bound[0][j]
                elif self.population[i].chrom[j] > self.bound[1][j]:
                    self.population[i].chrom[j] = self.bound[1][j]

    def solve(self, plot_result):
        '''
        the evolution process of the pso algorithm
        '''
        self.t = 0
        self.initialize()
        self.evaluation()
        print(self.fitness)
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t, 0] = self.best.fitness
        self.trace[self.t, 1] = self.avefitness
        print("Generation %d: optimal function value is: %f; average function value is %f" % (
            self.t, self.best.fitness, self.avefitness))
        print("Optimal solution is:")
        print(self.best.chrom)
        while self.t < self.MAXGEN - 1:
            self.t += 1
            self.update()
            self.evaluation()
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)

            self.trace[self.t, 0] = self.best.fitness
            self.trace[self.t, 1] = self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
            print("Optimal solution is:")
            print(self.best.chrom)
        print("Optimal function value is: %f; " % self.trace[self.t, 0])
        print("Optimal solution is:")
        print(self.best.chrom)
        if plot_result == True:
            self.printResult()
        else:
            pass

    def printResult(self):
        '''
        plot the result of pso algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Particle Swarm Optimization algorithm for function optimization")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    bound = np.array([[5, 0.03], [30, 0.15]])
    pso = PSO(60, 25, bound, 1000, [0.7298, 1.4962, 1.4962])
    pso.solve()
