# HW 4 Artificial Intelligence
Justin Chung jchung51

Ryan Demo rdemo1

### Run
`python classify.py --data [train_file] --mode train --model-file name.model --algorithm [algorithm_type]
`

Algorithm types:
- decision_tree
- naive_bayes
- neural_network

We include all commands needed to reproduce our results in `all-commands.txt`

# Errors
Pruning does not work fully, so we omitted it so that it wouldn't break the algorithm.




## Decision Tree
Training includes recursive call to add to decision tree. We go through every attribute and caculate the gain which includes finding the entropy. This returns the most likely value for an instance. Ran into a bug with pruning, and could only get non-pruning to function correctly.

To call information gain ratio for training:

`python classify.py --data [train_file] --mode train --model-file name.model --algorithm [algorithm_type] --gain ratio`

### Naive Bayes

We assume that each predictor is independent of one another. For training, we store a dictionary where the unique labels match to a list of mean and standard deviation pairs. Then when we predict an instance, we calculate the probabilities for each attribute and determine the best classification. 

### Neural Network
We iterate over the number of epochs, but parallelize all of the operations for a set of training cases instead of using iteration. Using gradient descent, we update the learning rate as the epochs increase.
