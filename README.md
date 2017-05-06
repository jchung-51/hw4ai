# Justin Chung jchung51
# Ryan Demo rdemo1
# HW 4
# Artificial Intelligence

Algorithm types:
decision_tree
naive_bayes
neural_network


1. Decision Tree

Training includes recursive call to add to decision tree. We go through every attribute and caculate the gain which includes finding the entropy. This returns the most likely value for an instance. Ran into a bug with pruning, and could only get non-pruning to function correctly.

To call information gain ratio for training...
python classify.py --data [train_file] --mode train --model-file name.model --algorithm [algorithm_type] --gain ratio

Data               gain        gainRatio

House Votes        64/87                63/87
Iris               23/30                12/30
Monks1             374/432              264/432
Monks2             272/432              297/432
Monks3             380/432              266/432

Information gain ratio is the information gain divided by the range of values of the attributes. Thus if the range of values of attributes is greater, the information gain is less. This makes attributes with a smaller range of variance more valuable.

2. Naive Bayes

We assume that each predictor is independent of one another. For training, we store a dictionary where the unique labels match to a list of mean and standard deviation pairs. Then when we predict an instance, we calculate the probabilities for each attribute and determine the best classification. 

Data            

House Votes         78/87
Iris                29/30
Monks1              292/432
Monks2              228/432
Monks3              390/432

3. Neural Network

House Votes	79/87
Iris		29/30
Monks1		292/432
Monks2		290/432
Monks3		338/432
