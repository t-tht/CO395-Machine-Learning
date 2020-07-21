import math
from collections import defaultdict
import numpy as np

from lib.Test import Test
from lib.Node import Node
from lib.Helpers import find_split_point


class Pruning:
    
    def __init__ (self, val_set):
        self.val_set = val_set


    # Function that is called to initiate pruning
    def prune(self, root):
        if root.leaf:
            return
        self.root = root

        # Find initial metrics on validation set
        self.test = Test(self.val_set)
        c, n, p = self.test.run_tests(root)

        # Sort the tree and split according to the threshold value
        self.val_set = sorted(self.val_set, key=lambda x: x[root.dict["attribute"]])
        point = find_split_point(self.val_set, root.dict["attribute"], root.dict["value"])

        subset1 = self.val_set[ : point]
        subset2 = self.val_set[point : ]

        # Save error rate as member data and recursively call prunify
        self.err_rate = n/p
        self.prunify(subset1, self.root.dict["left"])
        self.prunify(subset2, self.root.dict["right"])


    # Run a test on validation set and return error rate
    def check(self):
        self.test = Test(self.val_set)
        c,n,p = self.test.run_tests(self.root)
        return n/p


    # Recursively prune the tree
    def prunify(self, data, node):
        # In case of leaf, return
        if node.leaf:
            return
        
        # First sort the data on attribute column and split into subsets
        data = sorted(data, key=lambda x: x[node.dict["attribute"]])
        point = find_split_point(data, node.dict["attribute"], node.dict["value"])

        subset1 = data[ : point]
        subset2 = data[point : ]

        # Recursivel call prunify 
        self.prunify(subset1, node.dict["left"])
        self.prunify(subset2, node.dict["right"])

        # If both children are leaves
        if node.dict["left"].leaf and node.dict["right"].leaf:
            # Save the decision node's threshold value
            tmp = node.dict["value"]
            # Covnert decision node to a leaf node
            node.leaf = True
            best_err = 1
            best_val = None
            # Find possible values for node from its' leaves' values, evaluate new error rate with these as end values
            mlist = [node.dict["left"].dict["value"], node.dict["right"].dict["value"]]
            for v in mlist:
                node.dict["value"] = v
                err = self.check()
                if err < self.err_rate and err < best_err:
                    best_err = err
                    best_val = v
            # If improvement, then prune
            if best_val is not None:
                self.err_rate = best_err
                node.dict["value"] = best_val
                node.dict["left"] = None
                node.dict["right"] = None
            # Otherwise restore the node to decision node
            else:
                node.dict["value"] = tmp
                node.leaf = False
            return
