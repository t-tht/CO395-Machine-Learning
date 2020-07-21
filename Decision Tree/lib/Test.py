import numpy as np
from collections import defaultdict

# Test class for autmating tests
# Provide a trained tree and the intended test set
# Returns the confusion matrix, number of incorrect guesses and the total number of test data

class Test:
    
    def __init__ (self, file):
        if isinstance(file, str):
            self.test_data = np.loadtxt(file)
        else:
            self.test_data = file
        # The number of possible rooms -> assuming four rooms only
        self.nums = 4
    

    # Run tests on given tree
    def run_tests(self, root):
        # Create empty confusion matrix
        confusion_matrix = np.array([[0]*self.nums]*self.nums)
        incorrect = 0
        # Test each data row and compare to expected result
        for i in range(len(self.test_data)):
            curr_data = self.test_data[i]
            correct_result = curr_data[-1]
            result = self.traverseTree(root, self.test_data[i])
            # Update confusion matrix
            confusion_matrix[int(result)-1][int(correct_result)-1] += 1
            if result != correct_result:
                incorrect += 1

        return confusion_matrix, incorrect, len(self.test_data)
    

    # Traverse tree with data from provided root node, return the tree-predicted value
    def traverseTree(self, root, drow):
        node = root
        while not node.leaf:
            reading = drow[node.dict["attribute"]]
            if reading < node.dict["value"]:
                node = node.dict["left"]
            else:   
                node = node.dict["right"]

        return node.dict["value"]

    # Confusion matrix structure - 2D array
    #                           Actual room
    #                       R1      R2      R3      R4
    # Predicted room    R1  a       b       c       d
    #                   R2  b       c       d       a   
    #                   R3  a       b       c       d
    #                   R4  b       c       d       a