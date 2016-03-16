import math
import sys

A1 = [0, 0, 1, 1, 0, 0, 0, 
      0, 0, 0, 1, 0, 0, 0, 
      0, 0, 0, 1, 0, 0, 0, 
      0, 0, 1, 0, 1, 0, 0, 
      0, 0, 1, 0, 1, 0, 0, 
      0, 1, 1, 1, 1, 1, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      1, 1, 1, 0, 1, 1, 1]

B1 = [1, 1, 1, 1, 1, 1, 0, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 1, 1, 1, 1, 0, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      1, 1, 1, 1, 1, 1, 0]

C1 = [0, 0, 1, 1, 1, 1, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 0, 1, 1, 1, 1, 0]

D1 = [1, 1, 1, 1, 1, 0, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 1, 0, 
      1, 1, 1, 1, 1, 0, 0]

E1 = [1, 1, 1, 1, 1, 1, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 0, 0, 
      0, 1, 0, 1, 0, 0, 0, 
      0, 1, 1, 1, 0, 0, 0, 
      0, 1, 0, 1, 0, 0, 0, 
      0, 1, 0, 0, 0, 0, 0, 
      0, 1, 0, 0, 0, 0, 1, 
      1, 1, 1, 1, 1, 1, 1]

J1 = [0, 0, 0, 1, 1, 1, 1, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 0, 1, 1, 1, 0, 0]

K1 = [1, 1, 1, 0, 0, 1, 1, 
      0, 1, 0, 0, 1, 0, 0, 
      0, 1, 0, 1, 0, 0, 0, 
      0, 1, 1, 0, 0, 0, 0, 
      0, 1, 1, 0, 0, 0, 0, 
      0, 1, 0, 1, 0, 0, 0, 
      0, 1, 0, 0, 1, 0, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      1, 1, 1, 0, 0, 1, 1]

A2 = [0, 0, 0, 1, 0, 0, 0, 
      0, 0, 0, 1, 0, 0, 0, 
      0, 0, 0, 1, 0, 0, 0, 
      0, 0, 1, 0, 1, 0, 0, 
      0, 0, 1, 0, 1, 0, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 1, 1, 1, 1, 1, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 1, 0, 0, 0, 1, 0]

B2 = [1, 1, 1, 1, 1, 1, 0, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 1, 1, 1, 1, 1, 0, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 1, 1, 1, 1, 1, 0]

C2 = [0, 0, 1, 1, 1, 0, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 0, 1, 1, 1, 0, 0]

D2 = [1, 1, 1, 1, 1, 0, 0, 
      1, 0, 0, 0, 0, 1, 0, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 0, 0, 0, 0, 1, 0, 
      1, 1, 1, 1, 1, 0, 0]

E2 = [1, 1, 1, 1, 1, 1, 1, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 1, 1, 1, 1, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 1, 1, 1, 1, 1, 1]

J2 = [0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 0, 1, 1, 1, 0, 0]

K2 = [1, 0, 0, 0, 0, 1, 0, 
      1, 0, 0, 0, 1, 0, 0, 
      1, 0, 0, 1, 0, 0, 0, 
      1, 0, 1, 0, 0, 0, 0, 
      1, 1, 0, 0, 0, 0, 0, 
      1, 0, 1, 0, 0, 0, 0, 
      1, 0, 0, 1, 0, 0, 0, 
      1, 0, 0, 0, 1, 0, 0, 
      1, 0, 0, 0, 0, 1, 0]

A3 = [0, 0, 0, 1, 0, 0, 0, 
      0, 0, 0, 1, 0, 0, 0, 
      0, 0, 1, 0, 1, 0, 0, 
      0, 0, 1, 0, 1, 0, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 1, 1, 1, 1, 1, 0, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 1, 0, 0, 0, 1, 1]

B3 = [1, 1, 1, 1, 1, 1, 0, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 1, 1, 1, 1, 0, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      1, 1, 1, 1, 1, 1, 0]

C3 = [0, 0, 1, 1, 1, 0, 1, 
      0, 1, 0, 0, 0, 1, 1, 
      1, 0, 0, 0, 0, 0, 1, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 0, 1, 1, 1, 0, 0]

D3 = [1, 1, 1, 1, 0, 0, 0, 
      0, 1, 0, 0, 1, 0, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 1, 0, 0, 1, 0, 0, 
      1, 1, 1, 1, 0, 0, 0]

E3 = [1, 1, 1, 1, 1, 1, 1, 
      0, 1, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 1, 0, 0, 
      0, 1, 1, 1, 1, 0, 0, 
      0, 1, 0, 0, 1, 0, 0, 
      0, 1, 0, 0, 0, 0, 0, 
      0, 1, 0, 0, 0, 0, 0, 
      0, 1, 0, 0, 0, 0, 1, 
      1, 1, 1, 1, 1, 1, 1]

J3 = [0, 0, 0, 0, 1, 1, 1, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 1, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 0, 1, 1, 1, 0, 0]

K3 = [1, 1, 1, 0, 0, 1, 1, 
      0, 1, 0, 0, 0, 1, 0, 
      0, 1, 0, 0, 1, 0, 0, 
      0, 1, 0, 1, 0, 0, 0, 
      0, 1, 1, 0, 0, 0, 0, 
      0, 1, 0, 1, 0, 0, 0, 
      0, 1, 0, 0, 1, 0, 0, 
      0, 1, 0, 0, 0, 1, 0, 
      1, 1, 1, 0, 0, 1, 1]

N = 63 # Number of components in an input vector.
M = 10 # Max number of clusters to be formed.
VIGILANCE = 0.3
FONT_WIDTH = 7
TRAINING_PATTERNS = 21 # How many PATTERNS for training weights.
IN_PATTERNS = 21 # Total input patterns.

pattern = []

class ART1_Example2:
    def __init__(self, inputSize, numClusters, vigilance, fontWidth, numPatterns, numTraining, patternArray):
        self.mInputSize = inputSize
        self.mNumClusters = numClusters
        self.mVigilance = vigilance
        self.mFontWidth = fontWidth
        self.mNumPatterns = numPatterns
        self.mNumTraining = numTraining
        self.mPatterns = patternArray

        self.bw = [] # Bottom-up weights.
        self.tw = [] # Top-down weights.

        self.f1a = [] # Input layer.
        self.f1b = [] # Interface layer.
        self.f2 = []

        self.membership = []
        return

    def initialize_arrays(self):
        # Initialize bottom-up weight matrix.
        for i in range(self.mNumClusters):
            self.bw.append([0.0] * self.mInputSize)
            for j in range(self.mInputSize):
                self.bw[i][j] = 1.0 / (1.0 + self.mInputSize)

        # Initialize top-down weight matrix.
        for i in range(self.mNumClusters):
            self.tw.append([0.0] * self.mInputSize)
            for j in range(self.mInputSize):
                self.tw[i][j] = 1.0

        self.f1a = [0.0] * self.mInputSize
        self.f1b = [0.0] * self.mInputSize
        self.f2 = [0.0] * self.mNumClusters
        self.membership = [0] * self.mNumPatterns
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
        
        # Update tw(f2Max)
        for i in range(self.mInputSize):
            self.tw[f2Max][i] = self.f1b[i]
        
        return

    def ART1(self):
        inputSum = 0
        activationSum = 0
        f2Max = 0
        reset = True
        
        for k in range(self.mNumPatterns):
            # Initialize f2 layer activations to 0.0
            for i in range(self.mNumClusters):
                self.f2[i] = 0.0
            
            # Input pattern() to f1 layer.
            for i in range(self.mInputSize):
                self.f1a[i] = self.mPatterns[k][i]
            
            # Compute sum of input pattern.
            inputSum = self.get_vector_sum(self.f1a)
            
            # Compute activations for each node in the f1 layer.
            # Send input signal from f1a to the fF1b layer.
            for i in range(self.mInputSize):
                self.f1b[i] = self.f1a[i]
            
            # Compute net input for each node in the f2 layer.
            for i in range(self.mNumClusters):
                for j in range(self.mInputSize):
                    self.f2[i] += self.bw[i][j] * float(self.f1a[j])
            
            reset = True
            while reset == True:
                # Determine the largest value of the f2 nodes.
                f2Max = self.get_maximum(self.f2)
                
                # Recompute the f1a to f1b activations (perform AND function)
                for i in range(self.mInputSize):
                    self.f1b[i] = self.f1a[i] * math.floor(self.tw[f2Max][i])
                
                # Compute sum of input pattern.
                activationSum = self.get_vector_sum(self.f1b)
                
                reset = self.test_for_reset(activationSum, inputSum, f2Max)
            
            # Only use number of TRAINING_PATTERNS for training, the rest are tests.
            if k < self.mNumTraining:
                self.update_weights(activationSum, f2Max)
            
            # Record which cluster the input vector is assigned to.
            self.membership[k] = f2Max;
        return
    
    def print_results(self):
        k = 0
        
        sys.stdout.write("Input vectors assigned to each cluster:\n\n")
        for i in range(self.mNumClusters):
            sys.stdout.write("Cluster # " + str(i) + ": ")
            for j in range(self.mNumPatterns):
                if self.membership[j] == i:
                    sys.stdout.write(str(j) + ", ")
            
            sys.stdout.write("\n")
        
        sys.stdout.write("Final weight values for each cluster:\n\n")
        for i in range(self.mNumClusters):
            for j in range(self.mInputSize):
                if self.tw[i][j] >= 0.5:
                    sys.stdout.write("#")
                else:
                    sys.stdout.write(".")
                
                k += 1
                if k == self.mFontWidth:
                    sys.stdout.write("\n")
                    k = 0
            
            sys.stdout.write("\n")
        
        return

if __name__ == '__main__':
    pattern = []
    pattern.append(A1)
    pattern.append(B1)
    pattern.append(C1)
    pattern.append(D1)
    pattern.append(E1)
    pattern.append(J1)
    pattern.append(K1)
    pattern.append(A2)
    pattern.append(B2)
    pattern.append(C2)
    pattern.append(D2)
    pattern.append(E2)
    pattern.append(J2)
    pattern.append(K2)
    pattern.append(A3)
    pattern.append(B3)
    pattern.append(C3)
    pattern.append(D3)
    pattern.append(E3)
    pattern.append(J3)
    pattern.append(K3)

    art1 = ART1_Example2(N, M, VIGILANCE, FONT_WIDTH, IN_PATTERNS, TRAINING_PATTERNS, pattern)
    art1.initialize_arrays()
    art1.ART1()
    art1.print_results()
    
