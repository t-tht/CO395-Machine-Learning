Authors:
PS3416

Note:
The tree of the Noisy dataset may be too small to see well in the report. Both are viewable as separate pdf in this folder.



Running the code on a user specified training and test dataset:

1. Open the "learn.py" file
2. Replace the train_path variable with the path to the training dataset.
3. Replace the test_path variable with the path to the test dataset.
4. Run the program with "python3 learn.py". 
5. The program will print out the error rate.



Running Ten Fold Cross Validation:

1. Open the "tfcv.py" file
2. Replace the strings in the "sets" array with the path/name.txt for the datasets to be used.
3. Set 'verbose_logging' to True if you want to the detailed metrics such as the Confusion Matrices, pre-prune and post-prune depths etc.
4. Run the program with "python3 tfcv.py"
5. The program will print out the average classification rate for each dataset.
6. If 'verbose_logging' is enabled then a more detailed report is available in 'verbose_results.txt'.



Visualising the tree:

Currently, visualising the tree is only available in "learn.py", and requires the import of PrettyPrint from lib.PrettyPrint.
In order to use it, call PrettyPrint and pass in the root of the Decision Tree as an argument. 
Visualising the tree requires "pdflatex" to be installed, which converts the latex file into a pdf.
Printing the tree can be included anywhere so long as the module in included.





Structure of the confusion matrix

    # Confusion matrix structure - 2D array
    #                           Actual room
    #                       R1      R2      R3      R4
    # Predicted room    R1  a       b       c       d
    #                   R2  b       c       d       a   
    #                   R3  a       b       c       d
    #                   R4  b       c       d       a