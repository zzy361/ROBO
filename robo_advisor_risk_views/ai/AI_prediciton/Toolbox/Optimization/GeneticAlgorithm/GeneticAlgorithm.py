import numpy as np
from Optimization.GeneticAlgorithm.GAIndividual import GAIndividual
import random
import copy
import matplotlib.pyplot as plt

class GeneticAlgorithm:

    def __init__(self, optimizer_para_dict, backtest_para_dict, opt_para_name):
        self.pop_size = optimizer_para_dict['pop_size']
        self.iteration_num = optimizer_para_dict['iteration_num']
        self.dimension = optimizer_para_dict['dimension']
        self.bound = optimizer_para_dict['bound']
        self.population = []
        self.fitness = np.zeros((self.pop_size, 1))
        self.trace = np.zeros((self.iteration_num, 2))
        self.params = [0.9, 0.6, 0.1]
        self.backtest_para_dict = backtest_para_dict
        self.opt_para_name = opt_para_name
    def initialize(self):
        for i in range(0, self.pop_size):
            ind = GAIndividual(self.dimension, self.bound)
            ind.generate()
            self.population.append(ind)

    def evaluate(self):
        for i in range(0, self.pop_size):
            self.population[i].calculateFitness(self.population[i].chrom, self.backtest_para_dict)
            self.fitness[i] = self.population[i].fitness

    def solve(self, drow_opt_line):
        self.t = 0
        self.initialize()
        self.evaluate()
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.trace[self.t, 0] = self.best.fitness
        self.trace[self.t, 1] = np.mean(self.fitness)
        print("Generation %d: optimal function value is: %f;" % (self.t, self.trace[self.t, 0]))
        while self.t < self.iteration_num-1:
            self.t += 1
            self.selectionOperation()
            self.crossoverOperation()
            self.mutationOperation()
            self.evaluate()
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.trace[self.t, 0] = self.best.fitness
            self.trace[self.t, 1] = np.mean(self.fitness)
            print("Generation %d: optimal function value is: %f" % (self.t, self.trace[self.t, 0]))

        print("Optimal function value is: %f; " % self.trace[self.t, 0])
        print("Optimal solution is:", self.best.chrom)
        if drow_opt_line == 1:
            self.printResult()
        else:
            pass

    def selectionOperation(self):
        newpop = []
        totalFitness = np.sum(self.fitness)
        accuFitness = np.zeros((self.pop_size, 1))

        sum1 = 0.
        for i in range(0, self.pop_size):
            accuFitness[i] = sum1 + self.fitness[i] / totalFitness
            sum1 = accuFitness[i]

        for i in range(0, self.pop_size):
            r = random.random()
            idx = 0
            for j in range(0, self.pop_size - 1):
                if j == 0 and r < accuFitness[j]:
                    idx = 0
                    break
                elif r >= accuFitness[j] and r < accuFitness[j + 1]:
                    idx = j + 1
                    break
            newpop.append(self.population[idx])
        self.population = newpop

    def crossoverOperation(self):
        newpop = []
        for i in range(0, self.pop_size, 2):
            idx1 = random.randint(0, self.pop_size - 1)
            idx2 = random.randint(0, self.pop_size - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, self.pop_size - 1)
            newpop.append(copy.deepcopy(self.population[idx1]))
            newpop.append(copy.deepcopy(self.population[idx2]))
            r = random.random()
            if r < self.params[0]:
                crossPos = random.randint(1, self.dimension - 1)
                for j in range(crossPos, self.dimension):
                    newpop[i].chrom[j] = newpop[i].chrom[
                        j] * self.params[2] + (1 - self.params[2]) * newpop[i + 1].chrom[j]
                    newpop[i + 1].chrom[j] = newpop[i + 1].chrom[j] * self.params[2] + \
                        (1 - self.params[2]) * newpop[i].chrom[j]
        self.population = newpop

    def mutationOperation(self):
        newpop = []
        for i in range(0, self.pop_size):
            newpop.append(copy.deepcopy(self.population[i]))
            r = random.random()
            if r < self.params[1]:
                mutatePos = random.randint(0, self.dimension - 1)
                theta = random.random()
                if theta > 0.5:
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[
                        mutatePos] - (newpop[i].chrom[mutatePos] - self.bound[0, mutatePos]) * (1 - random.random() ** (1 - self.t / self.iteration_num))
                else:
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[
                        mutatePos] + (self.bound[1, mutatePos] - newpop[i].chrom[mutatePos]) * (1 - random.random() ** (1 - self.t / self.iteration_num))
        self.population = newpop

    def printResult(self):
        x = np.arange(0, self.iteration_num)
        y1 = self.trace[:, 0]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Genetic algorithm for function optimization")
        plt.legend()
        plt.show()
