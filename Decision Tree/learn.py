import numpy as np

from lib.DecisionTree import DecisionTree
from lib.Test import Test
from lib.PrettyPrint import PrettyPrint

# Column headers are:
# Wifi1,     Wifi2  ...  Wifi7,   Label

# Feel free to modify this stuff
train_path = "data/clean_dataset.txt"
test_path = "data/clean_dataset.txt"



# But please leave this as is
data = np.loadtxt(train_path)
tree = DecisionTree()
root, depth = tree.decision_tree_learning(data, 0)
print("Depth is ", depth)
print("Running long tests")
t = Test(test_path)
conf_matrix, incorrect_results, tsize = t.run_tests(root)
print(incorrect_results/tsize * 100, "% misclassified") # Percentage of incorrect results
PrettyPrint(root); #writes latex file and compiles it
