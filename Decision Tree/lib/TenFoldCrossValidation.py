import numpy as np
from collections import defaultdict
import math

from lib.DecisionTree import DecisionTree
from lib.Node import Node
from lib.Test import Test
from lib.Pruning import Pruning

    # Confusion matrix structure - 2D array
    #                           Actual room
    #                       R1      R2      R3      R4
    # Predicted room    R1  a       b       c       d
    #                   R2  b       c       d       a
    #                   R3  a       b       c       d
    #                   R4  b       c       d       a

class TenFoldCrossValidation:

    def run(self, datasets, logging=False, pre_prune=0):
        self.datasets = datasets
        self.num_folds = 10
        # Open a file for logging if option enabled
        if logging:
            self.file = open("verbose_results.txt", "w+")
        else:
            self.file = None

        # Run cross validation for array of datasets
        for d in self.datasets:
            self.write_file("\nUsing dataset " + d)
            print("Using dataset " + d)

            # Initialise metrics and datasets
            data = np.loadtxt(d)
            np.random.shuffle(data)
            interval = int(len(data)/self.num_folds)
            pre_confusion_matrix = np.zeros(shape=(4,4))
            post_confusion_matrix = np.zeros(shape=(4,4))
            pre_depth = 0
            post_depth = 0

            for i in range(self.num_folds):
                # Construct training and test sets
                rest = np.concatenate((data[0 : interval*i], data[interval+interval*i : ]), 0)
                test_set = data[interval*i : interval+interval*i]
                lenset = len(rest)
                np.random.shuffle(rest)

                for j in range(0,int(lenset/interval)):

                    # Generate validation set and remove from training set
                    validation_set = rest[interval*j : interval+interval*j]
                    training_set = np.concatenate((rest[0 : interval*j], rest[interval+interval*j : ]), 0)

                    # Initialise pruning class to the validation set
                    p = Pruning(validation_set)

                    # Train tree and run test
                    tree = DecisionTree()
                    root, depth = tree.decision_tree_learning(training_set, 0, pre_prune=5)
                    pre_depth = depth

                    # Evaluate the tree on the test set
                    pre_classification_rate = self.evaluate(test_set, root)
                    pre_conf_matrix = self.confusion_matrix

                    # Prune the tree
                    root.prune_tree(validation_set)
                    p.prune(root)

                    # Find the pruned depth
                    post_depth = root.maxHeight()

                    # Test pruned tree on test data
                    post_classification_rate = self.evaluate(test_set, root)
                    post_conf_matrix = self.confusion_matrix

                pre_confusion_matrix = np.add(pre_confusion_matrix, pre_conf_matrix)
                post_confusion_matrix = np.add(post_confusion_matrix, post_conf_matrix)

            self.write_file("\nPre prune metrics: \n")
            print("\nPre prune metrics: \n")
            self.write_file("\n" + str(d) + " has average pre prune depth " + str(int(pre_depth/self.num_folds*(lenset/interval))) + "\n")
            # Get pre prune metrics
            self.trawl_confusion_matrix(pre_confusion_matrix)
            self.write_file("\nPost prune metrics:\n")
            print("\nPost prune metrics:\n")
            self.write_file("\n" + str(d) + " has average post prune depth " + str(int(post_depth/self.num_folds*(lenset/interval))) + "\n")
            self.trawl_confusion_matrix(post_confusion_matrix)
            self.write_file("\n\n##################################################")
            print("\n\n##################################################")

        if self.file is not None:
            self.file.close()


    # Run tests on a trained tree with a provided dataset and return classification rate
    def evaluate(self, test_set, trained_tree):
        test = Test(test_set)
        confusion_matrix, incorrect_results, tsize = test.run_tests(trained_tree)
        error_rate = incorrect_results/tsize
        self.confusion_matrix = confusion_matrix

        return 1-error_rate


    # Find measures from confusion matrix, calculate F1 value and report in logs
    def trawl_confusion_matrix(self, conf_matrix):
        self.write_file(str(conf_matrix) + "\n")

        avg_recall = 0
        avg_precision = 0
        avg_f1 = 0
        correct = 0
        incorrect = 0

        for i in range(len(conf_matrix[0])):
            for j in range(len(conf_matrix)):
                if i == j:
                    correct += conf_matrix[j][i]
                else:
                    incorrect += conf_matrix[j][i]
            
        # Error rate
        print("Error rate is: " + str(1-correct/(correct+incorrect)))
        print("Classification rate is: " + str(correct/(correct+incorrect)))
        # If logging not enabled, stop
        if self.file is None:
            return
        self.write_file("Error rate is: " + str(1-correct/(correct+incorrect))+ "\n")
        self.write_file("Classification rate is: " + str(correct/(correct+incorrect)) + "\n")

        for i in range(len(conf_matrix)):
                num = conf_matrix[i][i]

                # Calculate recall rate
                recdem = 0
                for j in range(len(conf_matrix[i])):
                    recdem += conf_matrix[i][j]
                recall_rate = num / recdem
                avg_recall += recall_rate
                self.write_file("Recall rate for Room " + str(i+1) + " : " + str(recall_rate) + "\n")
                
                # Calculate precision rate
                precdem = 0
                for j in range(len(conf_matrix)):
                    precdem += conf_matrix[j][i]
                precision_rate = num / precdem
                avg_precision += precision_rate
                self.write_file("Precision rate for Room " +  str(i+1) +  " : " + str(precision_rate) + "\n")
                
                # Calculate F1 rate
                f1 = 2*(precision_rate * recall_rate) / (precision_rate + recall_rate)
                avg_f1 += f1
                self.write_file("F1 rate for Room " + str(i+1) + " : " + str(f1) + "\n")
        
        # Display average rates
        self.write_file("Average recall rate: " + str(avg_recall/len(conf_matrix))+ "\n")
        self.write_file("Average precision rate: " + str(avg_precision/len(conf_matrix))+ "\n")
        self.write_file("Average F1 value: " + str(avg_f1/len(conf_matrix))+ "\n")


    def write_file(self, s):
        if self.file is not None:
            self.file.write(s)
