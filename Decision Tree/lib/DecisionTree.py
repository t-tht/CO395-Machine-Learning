import numpy as np
import math
import sys

from lib.Node import Node
from lib.Helpers import find_split_point, find_mode

from collections import defaultdict



class DecisionTree:

    def decision_tree_learning(self, data, depth, pre_prune=0):
        if len(data) < pre_prune:
            mode = find_mode(data)
            curr = Node()
            curr.isLeaf(mode)
            return curr, depth

        if self.same_label(data):
            # Base case
            curr = Node()
            curr.isLeaf(data[0][-1])
            return curr, depth
        
        # Assume dataset can be split
        split_arrays, threshold, col = self.find_split(data)
        # split_arrays holds the partitioned tables
        # readings holds the threshold value for making the decision
        # col is the column in table the decision node will consider

        if col is None:
            mode = find_mode(data)
            curr = Node()
            curr.isLeaf(mode)
            return curr, depth

        # If one subarray is empty then find mode and create leaf
        if len(split_arrays[0]) == 0 or len(split_arrays[1]) == 0:
            # Base case
            arr = []
            if len(split_arrays[0]) == 0:
                arr = split_arrays[1]
            else:
                arr = split_arrays[0]
            mode = find_mode(arr)
            curr = Node()
            curr.isLeaf(mode)
            return curr, depth

        # Otherwise create decision node and recursively look at subsets
        maxDepth = depth
        curr = Node()
        curr.isDecision(threshold, col)
  

        for i in range(len(split_arrays)):
            cnode, cdepth = self.decision_tree_learning(split_arrays[i], depth+1, pre_prune=pre_prune)
            if cdepth > maxDepth:
                maxDepth = cdepth
            curr.addBranch(cnode)
            
        return curr, maxDepth        


    #               Methods to find the column to split on
    ################################################################

    
    def find_split(self, data):
        max_gain = 1-sys.maxsize
        partition_column = None
        threshold = None

        # Find column with max gain and split threshold
        for col in range(len(data[0])-1):
            thresh, gain = self.find_threshold(data, col)
            # listify -> update best column only if the column has more than one type of value
            if gain > max_gain and len(self.listify(data, col)) > 1:
                max_gain = gain
                partition_column = col
                threshold = thresh

        
        if partition_column is None:
            return None, None, None

        # Sort data on the chosen column
        data = sorted(data, key=lambda x: x[partition_column])

        # Find index at which to split
        point = find_split_point(data, partition_column, threshold)

        new_arrays = []

        new_arrays.append(data[ : point])
        new_arrays.append(data[point : ])

        # Return the new subsets, threshold value and the column split on
        return new_arrays, threshold, partition_column

    
    # Find entropy of the Room column
    def find_entropy(self, data):
        hist = defaultdict(int)
        tcount = 0

        for d in data:
            hist[d[-1]] += 1
            tcount += 1

        entropy = 0

        for k in hist.keys():
            p = hist[k] / tcount
            if p > 0:
                entropy += p * math.log(p, 2)

        return -1*entropy


    #          Methods to determine where to split on a column
    #######################################################################


    # Generate possible threshold values and find gain for each
    def find_threshold(self, data, attribute):
        values = self.listify(data, attribute)
        max_gain = 1-sys.maxsize
        optimal_threshold = 0

        # Find gain at each midpoint of values
        for i in range(len(values)-1):
            mpoint = (values[i] + values[i+1]) / 2
            gain = self.find_threshold_gain(data, attribute, mpoint)
            if gain > max_gain:
                max_gain = gain
                optimal_threshold = mpoint
        
        return optimal_threshold, max_gain


    # Find the gain of the set given a threshold
    def find_threshold_gain(self, data, attribute, threshold):
        lower = 0
        higher = 0

        # Keep track of the number of variables both < and >= the threshold
        for d in data:
            if d[attribute] < threshold:
                lower += 1
            else:
                higher += 1
        
        remainder = 0

        dsize = len(data)

        lower_entropy, higher_entropy = self.find_threshold_entropy(data, attribute, threshold)

        remainder = (lower/dsize)*lower_entropy + (higher/dsize)*higher_entropy

        gain = self.find_entropy(data) - remainder

        return gain

    
    # Find the entropy of a set given the threshold value
    def find_threshold_entropy(self, data, attribute, threshold):
        lowerhist = defaultdict(int)
        higherhist = defaultdict(int)
        lowercount = 0
        highercount = 0

        # Keep track of frequency of (Room | value < threshold) and (Room | value >= threshold)
        for d in data:
            if d[attribute] < threshold:
                lowerhist[d[-1]] += 1
                lowercount += 1
            else:
                higherhist[d[-1]] += 1
                highercount += 1
        
        lower_entropy = 0
        higher_entropy = 0
        # Calculate entropy for each 'half' of the set
        for k in lowerhist.keys():
            p = lowerhist[k] / lowercount
            if p > 0:
                lower_entropy += p * math.log(p, 2)
        
        for k in higherhist.keys():
            p = higherhist[k] / highercount
            if p > 0:
                higher_entropy += p * math.log(p, 2)

        return -1*lower_entropy, -1*higher_entropy
        

    #               Helper methods
    ################################################## 


    # Check if all decision values (room no's) in a dataset are the same
    def same_label(self, data):
        if len(data) == 1:
            return True
        
        for i in range(len(data)-1):
            if data[i][-1] != data[i+1][-1]:
                return False
        return True


    # Convert a column into an ordered list of its distinct consituents
    def listify(self, data, attribute):
        attr = defaultdict(int)

        for d in data:
            attr[d[attribute]] = 1
        attr = list(attr.keys())
        attr.sort()
        return attr
