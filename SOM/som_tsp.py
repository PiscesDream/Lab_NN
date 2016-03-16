#http://mnemstudio.org/neural-networks-som4.htm
import matplotlib.pyplot as plt
import numpy

import math
import random
import sys

# Coordinates for example 1.  Ring-shaped map.
CITIES = [[7.0, 2.0], [12.0, 5.0], [9.0, 10.0], [5.0, 10.0], [2.0, 5.0]]
CITIES = numpy.random.uniform(0, 10, size=(20, 2)) 

EPOCHS = 3000
NUM_CITIES = len(CITIES) 
NUM_NEURONS = NUM_CITIES * 2
NEAR = 0.05
MOMENTUM = 0.995
THETA = 0.5
PHI = 0.5

# Coordinates for example 2.
#CITIES = [[3.0, 3.0], [12.0, 2.0], [6.0, 6.0], [7.0, 5.0], [6.0, 12.0]]

class SOM_Class4:
    def __init__(self, numEpochs, numCities, numNeurons, near, momentum, theta, phi, cities):
        self.mNumEpochs = numEpochs
        self.mNumCities = numCities
        self.mNumNeurons = numNeurons
        self.mNear = near
        self.mMomentum = momentum
        self.city = cities
        self.neuronXY = []
        self.weight = []
        self.r = []
        self.gridInterval = 0.0
        self.theta = theta
        self.phi = phi
        return

    def initialize_arrays(self):
        for i in range(self.mNumNeurons):
            self.r.append([0.0] * self.mNumNeurons)

        for i in range(self.mNumNeurons):
            newRow = []
            newRow.append(0.5 + 0.5 * math.cos(self.gridInterval))
            newRow.append(0.5 + 0.5 * math.sin(self.gridInterval))
            self.neuronXY.append(newRow)

            self.gridInterval += math.pi * 2.0 / float(NUM_NEURONS)

            newRow = []
            newRow.append(random.random())
            newRow.append(random.random())
            self.weight.append(newRow)

        self.compute_matrix(self.theta)
        return

    def get_distance(self, index, index2):
        dx = self.neuronXY[index][0] - self.neuronXY[index2][0]
        dy = self.neuronXY[index][1] - self.neuronXY[index2][1]
        return math.sqrt(dx * dx + dy * dy)

    def compute_matrix(self, theta):
        for i in range(self.mNumNeurons):
            self.r[i][i] = 1.0
            for j in range(i + 1, self.mNumNeurons):
                self.r[i][j] = math.exp(-1.0 * (self.get_distance(i, j) * self.get_distance(i, j)) / (2.0 * math.pow(theta, 2)))
                self.r[j][i] = self.r[i][j]
        return

    def find_minimum(self, location1, location2):
        minimumDistance = 100000.0 # or any giant number.
        minimumIndex = -1

        for i in range(self.mNumNeurons):
            distance = (math.pow((location1 - self.weight[i][0]), 2)) + (math.pow((location2 - self.weight[i][1]), 2))
            if distance < minimumDistance:
                minimumDistance = distance
                minimumIndex = i

        return minimumIndex

    def print_ring(self):
        for i in range(self.mNumNeurons):
            sys.stdout.write("(" + "{:03.3f}".format(self.weight[i][0]) + ", " + "{:03.3f}".format(self.weight[i][1]) + ") ")
        draw_tsp(self.weight, 'blue')
        sys.stdout.write("\n")
        return

    def algorithm(self):
        index = 0
        minimumIndex = 0
        loc1 = 0.0
        loc2 = 0.0
        count = 0

        while count < self.mNumEpochs:
            # Pick a city for random comparison.
            index = int(math.floor(random.random() * self.mNumCities))
            loc1 = self.city[index][0] + (random.random() * self.mNear) - self.mNear / 2.0
            loc2 = self.city[index][1] + (random.random() * self.mNear) - self.mNear / 2.0

            minimumIndex = self.find_minimum(loc1, loc2)

            # Update all weights.
            for i in range(self.mNumNeurons):
                self.weight[i][0] += (self.phi * self.r[i][minimumIndex] * (loc1 - self.weight[i][0]))
                self.weight[i][1] += (self.phi * self.r[i][minimumIndex] * (loc2 - self.weight[i][1]))

            self.phi *= self.mMomentum
            self.theta *= self.mMomentum

            self.compute_matrix(self.theta)

            if count % 20 == 0:
                self.print_ring()

            count += 1

        return

def draw(l, color='red', marker='', ls='-'):
    print l
    plt.plot(map(lambda x: x[0], l),
             map(lambda x: x[1], l),
             color = color, marker=marker, ls=ls)
    plt.plot([l[-1][0], l[0][0]],
             [l[-1][1], l[0][1]],
             color = color, marker=marker, ls=ls)
    plt.draw()

def draw_tsp(l, color='red'):
    plt.clf()
    draw(CITIES, marker='x', ls=' ')
    draw(l, color=color)

if __name__ == '__main__':
    plt.ion()

    som = SOM_Class4(EPOCHS, NUM_CITIES, NUM_NEURONS, NEAR, MOMENTUM, THETA, PHI, CITIES)
    som.initialize_arrays()
    som.algorithm()

    raw_input('all done')
