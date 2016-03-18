#http://mnemstudio.org/neural-networks-art1-example-1.htm

import math
import sys

N = 4 # Number of components in an input vector.
M = 3 # Max number of clusters to be formed.
VIGILANCE = 0.4
PATTERNS = 7
TRAINING_PATTERNS = 4 # Use this many for training, the rest are for tests.

PATTERN_ARRAY = [[1, 1, 0, 0], 
                 [0, 0, 0, 1], 
                 [1, 0, 0, 0], 
                 [0, 0, 1, 1], 
                 [0, 1, 0, 0], 
                 [0, 0, 1, 0], 
                 [1, 0, 1, 0]]

class ART1_Example1:
    def __init__(self, inputSize, numClusters, vigilance, numPatterns, numTraining, patternArray):
        self.mInputSize = inputSize
        self.mNumClusters = numClusters
        self.mVigilance = vigilance
        self.mNumPatterns = numPatterns
        self.mNumTraining = numTraining
        self.mPatterns = patternArray
        
        self.bw = [] # Bottom-up weights.
        self.tw = [] # Top-down weights.

        self.f1a = [] # Input layer.
        self.f1b = [] # Interface layer.
        self.f2 = []
        return
    
    def initialize_arrays(self):
        # Initialize bottom-up weight matrix.
        sys.stdout.write("Weights initialized to:")
        for i in range(self.mNumClusters):
            self.bw.append([0.0] * self.mInputSize)
            for j in range(self.mInputSize):
                self.bw[i][j] = 1.0 / (1.0 + self.mInputSize)
                sys.stdout.write(str(self.bw[i][j]) + ", ")
            
            sys.stdout.write("\n")
        
        sys.stdout.write("\n")
        
        # Initialize top-down weight matrix.
        for i in range(self.mNumClusters):
            self.tw.append([0.0] * self.mInputSize)
            for j in range(self.mInputSize):
                self.tw[i][j] = 1.0
                sys.stdout.write(str(self.tw[i][j]) + ", ")
            
            sys.stdout.write("\n")
        
        sys.stdout.write("\n")
        
        self.f1a = [0.0] * self.mInputSize
        self.f1b = [0.0] * self.mInputSize
        self.f2 = [0.0] * self.mNumClusters
        return
    
    def get_vector_sum(self, nodeArray):
        total = 0
        length = len(nodeArray)
        for i in range(length):
            total += nodeArray[i]
        
        return total
    
    def get_maximum(self, nodeArray):
        maximum = 0;
        foundNewMaximum = False;
        length = len(nodeArray)
        done = False
        
        while not done:
            foundNewMaximum = False
            for i in range(length):
                if i != maximum:
                    if nodeArray[i] > nodeArray[maximum]:
                        maximum = i
                        foundNewMaximum = True
            
            if foundNewMaximum == False:
                done = True
        
        return maximum
    
    def test_for_reset(self, activationSum, inputSum, f2Max):
        doReset = False
        
        if(float(activationSum) / float(inputSum) >= self.mVigilance):
            doReset = False # Candidate is accepted.
        else:
            self.f2[f2Max] = -1.0 # Inhibit.
            doReset = True # Candidate is rejected.
        
        return doReset
    
    def update_weights(self, activationSum, f2Max):
        # Update bw(f2Max)
        for i in range(self.mInputSize):
            self.bw[f2Max][i] = (2.0 * float(self.f1b[i])) / (1.0 + float(activationSum))
        
        for i in range(self.mNumClusters):
            for j in range(self.mInputSize):
                sys.stdout.write(str(self.bw[i][j]) + ", ")
            
            sys.stdout.write("\n")
        sys.stdout.write("\n")
        
        # Update tw(f2Max)
        for i in range(self.mInputSize):
            self.tw[f2Max][i] = self.f1b[i]
        
        for i in range(self.mNumClusters):
            for j in range(self.mInputSize):
                sys.stdout.write(str(self.tw[i][j]) + ", ")
            
            sys.stdout.write("\n")
        sys.stdout.write("\n")
        
        return
    
    def ART1(self):
        inputSum = 0
        activationSum = 0
        f2Max = 0
        reset = True
        
        sys.stdout.write("Begin ART1:\n")
        for k in range(self.mNumPatterns):
            sys.stdout.write("Vector " + str(k) + ': ' + repr(self.mPatterns[k]) + "\n\n")
            
            # Initialize f2 layer activations to 0.0
            for i in range(self.mNumClusters):
                self.f2[i] = 0.0
            
            # Input pattern() to f1 layer.
            for i in range(self.mInputSize):
                self.f1a[i] = self.mPatterns[k][i]
            
            # Compute sum of input pattern.
            inputSum = self.get_vector_sum(self.f1a)
            sys.stdout.write("InputSum (si) = " + str(inputSum) + "\n\n")
            
            # Compute activations for each node in the f1 layer.
            # Send input signal from f1a to the fF1b layer.
            for i in range(self.mInputSize):
                self.f1b[i] = self.f1a[i]
            
            # Compute net input for each node in the f2 layer.
            for i in range(self.mNumClusters):
                sys.stdout.write("f2["+str(i)+"]: ")
                for j in range(self.mInputSize):
                    self.f2[i] += self.bw[i][j] * float(self.f1a[j])
                    sys.stdout.write(str(self.f2[i]) + " -> ")
                
                sys.stdout.write("\n")
            sys.stdout.write("\n")
            
            reset = True
            while reset == True:
                # Determine the largest value of the f2 nodes.
                f2Max = self.get_maximum(self.f2)
                
                # Recompute the f1a to f1b activations (perform AND function)
                for i in range(self.mInputSize):
                    sys.stdout.write(str(self.f1b[i]) + " * " + str(self.tw[f2Max][i]) + " = " + str(self.f1b[i] * self.tw[f2Max][i]) + "\n")
                    self.f1b[i] = self.f1a[i] * math.floor(self.tw[f2Max][i])
                
                # Compute sum of input pattern.
                activationSum = self.get_vector_sum(self.f1b)
                sys.stdout.write("ActivationSum (x(i)) = " + str(activationSum) + "\n\n")
                
                reset = self.test_for_reset(activationSum, inputSum, f2Max)
            
            # Only use number of TRAINING_PATTERNS for training, the rest are tests.
            if k < self.mNumTraining:
                self.update_weights(activationSum, f2Max)
            
            sys.stdout.write("Vector #" + str(k) + " belongs to cluster #" + str(f2Max) + "\n\n")
                
        return
    
    def print_results(self):
        sys.stdout.write("Final weight values:\n")
        
        for i in range(self.mNumClusters):
            for j in range(self.mInputSize):
                sys.stdout.write(str(self.bw[i][j]) + ", ")
            
            sys.stdout.write("\n")
        sys.stdout.write("\n")
        
        for i in range(self.mNumClusters):
            for j in range(self.mInputSize):
                sys.stdout.write(str(self.tw[i][j]) + ", ")
            
            sys.stdout.write("\n")
        sys.stdout.write("\n")
        return

if __name__ == '__main__':
    art1 = ART1_Example1(N, M, VIGILANCE, PATTERNS, TRAINING_PATTERNS, PATTERN_ARRAY)
    art1.initialize_arrays()
    art1.ART1()
    art1.print_results()
    
    
