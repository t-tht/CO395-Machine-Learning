import math
from collections import defaultdict
import numpy as np

from lib.Test import Test
from lib.Helpers import find_split_point, find_mode

class Node:

    def __init__ (self):
        self.dict = defaultdict(int)

    # Count the number of nodes in a subtree from the current node
    def nodeCount(self):
        if self.leaf:
            return 1
        return 1 + self.dict["left"].nodeCount() + self.dict["right"].nodeCount()
    

    # Calculate depth of subtree from current node
    def maxHeight(self):   
        if self.leaf:
            return 0
        return 1+max(self.dict["left"].maxHeight(), self.dict["right"].maxHeight())


    # Make node a decision node and set parameters
    def isDecision(self, threshold, attribute):
        self.leaf = False
        self.dict["value"] = threshold
        self.dict["attribute"] = attribute


    # Make node a leaf node and set parameters
    def isLeaf(self, val):
        self.leaf = True
        self.dict["value"] = val


    # Add a branch to the node
    def addBranch(self, cnode):
        self.leaf = False
        if "left" not in self.dict:
            self.dict["left"] = cnode
        else:
            self.dict["right"] = cnode


    # Print tree -> used with PrettyPrint to create .latex file
    def print_tree(self, depth, f):
      indent = "  "*depth

      if self.leaf == False:
          #decision nodes
          f.write(indent + "[{" + "attr" + str(self.dict["attribute"]) + " < " + str(-self.dict["value"]) + "}")
          if self.dict["right"]:
              self.dict["right"].print_tree(depth+1,f)
          if self.dict["left"]:
              self.dict["left"].print_tree(depth+1,f)
          f.write("]")

      if self.leaf == True:
          f.write(indent + "[" + str(self.dict["value"]))
          if self.dict["right"]:
              self.dict["right"].print_tree(depth+1,f)
          if self.dict["left"]:
              self.dict["left"].print_tree(depth+1,f)
          f.write("]")


    #               Pruning with Error Estimation
    ################################################################


    # Called from the root node for tree to be pruned
    def prune_tree(self, validation_set):
        # Set z value, corresponds to confidence level
        z = 0.9
        # Sort and split dataset
        data = sorted(validation_set, key=lambda x: x[self.dict["attribute"]])
        point = find_split_point(data, self.dict["attribute"], self.dict["value"])
        subset1 = data[ : point]
        subset2 = data[point : ]
        # Call for subtrees to be pruned
        self.dict["left"].prune(subset1, z)
        self.dict["right"].prune(subset2, z)
        return self

    # Recursive function that does the pruning
    def prune(self, data, z):
        # If no data, return. Actual return value doesn't matter
        if len(data) == 0:
            return 0, 0
        # If leaf, then calculate eror rate and error estimate and return tuple
        elif self.leaf:
            incorrect = 0
            for d in data:
                if self.dict["value"] != d[-1]:
                    incorrect += 1
            return incorrect/len(data), self.estimate_error(z, incorrect/len(data), len(data))

        # First sort, split data and call prune on subtrees
        data = sorted(data, key=lambda x: x[self.dict["attribute"]])
        point = find_split_point(data, self.dict["attribute"], self.dict["value"])

        subset1 = data[ : point]
        subset2 = data[point : ]

        f1, e1 = self.dict["left"].prune(subset1, z)
        f2, e2 = self.dict["right"].prune(subset2, z)

        # Calculate weighted error estimates for subtree and node
        error_estimate_subtree = (len(subset1)/len(data)) * e1
        error_estimate_subtree += (len(subset2)/len(data)) * e2
        node_error = (len(subset1)/len(data)) * f1 + (len(subset2)/len(data)) * f2
        error_estimate_node = self.estimate_error(z, node_error, len(data))

        # If error estimate of node is > error estimate of subtree then prune
        if error_estimate_node > error_estimate_subtree:
            mode = find_mode(data)
            self.leaf = True
            self.dict["value"] = mode
            self.dict["left"] = None
            self.dict["right"] = None

            incorrect = 0
            for d in data:
                if self.dict["value"] != d[-1]:
                    incorrect += 1
            return incorrect/len(data), self.estimate_error(z, incorrect/len(data), len(data))

        return node_error, error_estimate_node

    # Error estimate formula applied
    def estimate_error(self, z, f, N):
        est = (f + (z**2)/(2*N) + z*((f/N - f/(N**2) + (z**2)/(4*(N)**2)))) / (1 + (z**2)/N)
        if est > 1:
            return 1
        elif est < 0:
            return 0
        return est
            

