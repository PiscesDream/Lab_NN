import sys

MAX_ITEMS = 11
MAX_CUSTOMERS = 14
TOTAL_PROTOTYPE_VECTORS = 10

BETA = 1.0 # Small positive number.
VIGILANCE = 0.6 # 0 <= VIGILANCE < 1

DATABASE = [[0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0], 
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1], 
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], 
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], 
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], 
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], 
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], 
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]

class ART1_MTJones1:
    def __init__(self, numItems, numCustomers, maxPrototypes, beta, vigilance, database):
        self.mNumItems = numItems
        self.mNumCustomers = numCustomers
        self.mMaxPrototypes = maxPrototypes
        self.mBeta = beta
        self.mVigilance = vigilance
        self.mDatabase = database
        
        self.prototypeVectors = 0 # Total populated prototype vectors.
        
        self.prototypeVector = []
        
        # Number of occupants of cluster.
        self.members = []
        
        # Identifies which cluster a member belongs to.
        self.membership = []
        return
    
    def initialize_arrays(self):
        for i in range(self.mMaxPrototypes):
            self.prototypeVector.append([0] * self.mNumItems)
        
        for i in range(self.mNumCustomers):
            self.membership.append(-1)
        
        #members[] = new int[TOTAL_PROTOTYPE_VECTORS]
        for i in range(self.mMaxPrototypes):
            self.members.append(0)
        return
    
    def vector_bitwise_AND(self, v, vRow, w, wRow):
        ANDResult = []
        for i in range(self.mNumItems):
            ANDResult.append(v[vRow][i] * w[wRow][i])
        
        return ANDResult
    
    def vectorMagnitude(self, vector, vRow = -1):
        # This function counts up all the 1's in a given vector.
        # Note that vector can be a single or 2-D array.
        totalOnes = 0
        for i in range(self.mNumItems):
            # If vRow gets used, then vector is a 2-D array
            if vRow > -1:
                if vector[vRow][i] == 1:
                    totalOnes += 1
            else:
                if vector[i] == 1:
                    totalOnes += 1
        
        return totalOnes
    
    def update_prototype_vectors(self, Cluster):
        first = True
        if Cluster >= 0:
            for i in range(self.mNumItems):
                self.prototypeVector[Cluster][i] = 0
            
            for i in range(self.mNumCustomers):
                if self.membership[i] == Cluster:
                    if first:
                        for j in range(self.mNumItems):
                            self.prototypeVector[Cluster][j] = self.mDatabase[i][j]
                        
                        first = False
                    else:
                        for j in range(self.mNumItems):
                            self.prototypeVector[Cluster][j] = self.prototypeVector[Cluster][j] * self.mDatabase[i][j]
        
        return
    
    def create_new_prototype_vector(self, vector, vRow):
        cluster = 0
        for i in range(self.mMaxPrototypes):
            if self.members[i] == 0:
                cluster = i
        
        if cluster == self.mMaxPrototypes - 1:
            self.prototypeVectors += 1
        
        for i in range(self.mNumItems):
            self.prototypeVector[cluster][i] = vector[vRow][i]
        
        self.members[cluster] = 1
        
        return cluster
    
    def perform_art1(self):
        magPE = 0
        magP = 0
        magE = 0
        result = 0.0
        test = 0.0
        done = False
        Count = 50
        
        while not done:
            done = True
            for i in range(self.mNumCustomers):
                for j in range(self.mMaxPrototypes):
                    # Check to see if this vector has any members.
                    if self.members[j] > 0:
                        ANDResult = self.vector_bitwise_AND(self.mDatabase, i, self.prototypeVector, j)
                        
                        magPE = self.vectorMagnitude(ANDResult)
                        magP = self.vectorMagnitude(self.prototypeVector, j)
                        magE = self.vectorMagnitude(self.mDatabase, i)
                        
                        result = float(magPE) / (self.mBeta + float(magP))
                        
                        test = float(magE) / (self.mBeta + float(MAX_ITEMS))
                        
                        if result > test:
                            # Test for vigilance acceptability.
                            if (float(magPE) / float(magE)) < self.mVigilance:
                                old = 0
                                # Ensure this is a different cluster.
                                if self.membership[i] != j:
                                    # Move customer to the new cluster.
                                    old = self.membership[i]
                                    self.membership[i] = j
                                    
                                    if old >= 0:
                                        self.members[old] -= 1
                                        if self.members[old] == 0:
                                            self.prototypeVectors -= 1
                                        
                                        self.members[j] += 1
                                        # Recalculate the prototype vectors for the old and new clusters.
                                        if old >= 0 and old < self.mMaxPrototypes:
                                            self.update_prototype_vectors(old)
                                        
                                        self.update_prototype_vectors(j)
                                        done = False
                                        break
                
                # Check to see if the current vector was processed.
                if self.membership[i] == -1:
                    # No prototype vector was found to be close to the example
                    # vector.  Create a new prototype vector for this example.
                    self.membership[i] = self.create_new_prototype_vector(self.mDatabase, i)
                    done = False
            
            if Count <= 0:
                break
            else:
                # Not Done yet.
                Count -= 1
        
        return
    
    def display_customer_clusters(self):
        for i in range(self.mMaxPrototypes):
            sys.stdout.write("ProtoType Vector: " + str(i) + ":\t")
            for j in range(self.mNumItems):
                sys.stdout.write(str(self.prototypeVector[i][j]))
            
            sys.stdout.write("\n")
            
            for j in range(self.mNumCustomers):
                if self.membership[j] == i:
                    sys.stdout.write("Customer: " + str(j) + ":\t\t\t")
                    for k in range(self.mNumItems):
                        sys.stdout.write(str(self.mDatabase[j][k]))
                    
                    sys.stdout.write("\n")
            
            sys.stdout.write("\n")
        sys.stdout.write("\n")
        return
    
    def display_memberships(self):
        for i in range(self.mNumCustomers):
            sys.stdout.write("Customer: " + str(i) + " belongs to cluster: " + str(self.membership[i]) + "\n")
        
        sys.stdout.write("\n")
        return
    
    def display_customer_database(self):
        for i in range(self.mNumCustomers):
            sys.stdout.write("\n")
            sys.stdout.write("Customer: " + str(i) + ":\t")
            for j in range(self.mNumItems):
                sys.stdout.write(str(self.mDatabase[i][j]))
        
        return

if __name__ == '__main__':
    art1 = ART1_MTJones1(MAX_ITEMS, MAX_CUSTOMERS, TOTAL_PROTOTYPE_VECTORS, BETA, VIGILANCE, DATABASE)
    art1.initialize_arrays()
    art1.perform_art1()
    art1.display_customer_clusters()
    art1.display_memberships()
    art1.display_customer_database()
    
    
    
    
    
    
