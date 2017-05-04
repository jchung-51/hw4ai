from abc import ABCMeta, abstractmethod
import math
import numpy as np

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        #self.label_num = int(label)
        self.label_str = str(label)
        return
        
    def __str__(self):
        return self.label_str

# the feature vectors will be stored in dictionaries so that they can be sparse structures
class FeatureVector:
    def __init__(self):
        self.feature_vec = {}
        pass
        
    def add(self, index, value):
        self.feature_vec[index] = value
        pass
        
    def get(self, index):
        val = self.feature_vec[index]
        return val
    
    def __str__(self):
        s = ""
        for i in self.feature_vec:
            s += str(self.feature_vec[i]) + " "
        return s
    
    def length(self):
        return len(self.feature_vec.keys())

    def arr(self):
        return np.array([self.feature_vec[i] for i in range(self.length())])
    
    
class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label
        
    def getLabel(self):
        return str(self._label)
    
    def getFeature(self):
        return self._feature_vector
    
    def length(self):
        return self._feature_vector.length()

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass
    
        

"""
TODO: you must implement additional data structures for
the three algorithms specified in the hw4 PDF

for example, if you want to define a data structure for the
DecisionTree algorithm, you could write

class DecisionTree(Predictor):
	# class code

Remember that if you subclass the Predictor base class, you must
include methods called train() and predict() in your subclasses
"""

class DecisionTree(Predictor):
    
    def __init__(self):
        self.tree = {}
        self.labels = set()
        
    def getTree(self):
        return self.tree
      
    # TRAIN  
    def train(self, instances): 
        instances_copy = instances[:]
        attributes = list(range(instances[0].length()))
        default = self.default(instances)
        self.tree = self.dtl(instances_copy, attributes)
        print self.labels
        #print self.labels
        #print self.tree
    
    def dtl(self, data, attr):
        most, counter, counter[most] = self.default(data)
        #print most, counter[most]
        if len(data) == 0:
            return most 
        elif len(counter.keys()) == 1:
            self.labels.add(counter.keys()[0])
            return counter.keys()[0]
        elif len(attr) - 1 <= 0:
            return most
        else:
            best = self.choose_attribute(attr, data)
            print best
            tree = {best:{}}
            #print best,self.get_val(data, best)
            for val in self.get_val(data, best):
                ex = self.get_ex(data, best, val)
                #print len(ex), len(data)
                attr_copy = list(attr)
                attr_copy.remove(best)
                #print attr, best , attr_copy
                subtree = self.dtl(ex, attr_copy)
                tree[best][val] = subtree
            return tree
        
    def get_ex(self, data, attr, val):
        exs = []
        for entry in data:
            if (entry.getFeature().get(attr) == val):
                exs.append(entry)
        return exs
        
    def choose_attribute(self, attrs, data):
        best = None
        best_val = 0.0
        for attr in attrs:
            x = self.gain(data, attrs, attr)
            if x > best_val:
                best_val = x
                best = attr
                #print "gain", x
        return best
    
    def get_val(self, data, attr):
        vals = set()
        for entry in data:
            vals.add(entry.getFeature().get(attr))
        return vals
    
    
    def entropy(self, data, attr):
        vals     = {}
        entropy = 0.0

        for entry in data:
            if (vals.has_key(entry.getFeature().get(attr))):
                vals[entry.getFeature().get(attr)] += 1.0
            else:
                vals[entry.getFeature().get(attr)]  = 1.0
    
        for count in vals.values():
            entropy += (-count/len(data)) * math.log(count/len(data), 2) 
            
        return entropy
    
    def gain(self, data, attrs, attr):
        vals = {}
        entropy = 0.0
    
        for entry in data:
            if (vals.has_key(entry.getFeature().get(attr))):
                vals[entry.getFeature().get(attr)] += 1.0
            else:
                vals[entry.getFeature().get(attr)]  = 1.0
    
        # Calculate the sum of the entropy for each subset of records weighted
        # by their probability of occuring in the training set.
        for val in vals.keys():
            prob = vals[val] / sum(vals.values())
            subset = [entry for entry in data if entry.getFeature().get(attr) == val]
            entropy += prob * self.entropy(subset, attr)
    
        # Subtract the entropy of the chosen attribute from the entropy of the
        # whole data set with respect to the target attribute (and return it)
        return (self.entropy(data, attr) - entropy)
        

    def default(self,instances):
        counter = {}
        most = None
        count = 0
        
        for instance in instances:
            if not counter.has_key(instance.getLabel()):
                counter[instance.getLabel()] = 1
            else:
                counter[instance.getLabel()] += 1

        for label in counter.keys():
            if counter[label] > count:
                most = label

        return most, counter, counter[most]
    #TEST
    def predict(self, instance): 
        
        return self.test(self.tree, instance)
        
    def test(self, tree, instance):
        if not tree:
            return None
        if not isinstance(tree, dict ):
            return tree
        
        attr = list(tree.keys())[0]
        vals = list(tree.values())[0]
        instance_val = instance.getFeature().get(attr)
        print attr, instance_val
        if instance_val not in vals:
            return None
        return self.test(vals[instance_val], instance)
    

class NaiveBayes(Predictor):

    def __init__(self):
        return


class NeuralNetwork(Predictor):

    def __init__(self, shape, classes):
        # Layer info
        self.layers = len(shape) - 1
        self.shape = shape
        self.classes = []
        self.tprobs = []
        self.targets = dict()  # maps class to target probabilities

        # Generate desired output for each class
        for index, c in enumerate(classes):
            probs = np.ones(len(classes)) * 0.05
            probs[index] = 0.95
            self.targets[c] = index
            self.tprobs.append(probs)
            self.classes.append(c)

        # Init run data
        self.layerIn = []
        self.layerOut = []

        self.weights = []
        np.random.seed(1) # Constant random seed makes testing deterministic

        # Init small random weights
        for (l1, l2) in zip(shape[:-1], shape[1:]):
            self.weights.append(np.random.uniform(-0.01, 0.01, (l2, l1+1))) # l1+1 for bias node
            # self.weights.append(np.random.normal(scale = 0.1, size = (l2, l1+1))) # l1+1 for bias node



    def target(self, instance):
        return self.tprobs[self.targets[instance.getLabel()]]

    def targetMat(self, instances):
        return np.matrix([self.tprobs[self.targets[instance.getLabel()]] for instance in instances])

    def train(self, instances):

        epochs = 1000
        learningRate = 1
        deltas = []

        trainingSet = np.matrix([instance.getFeature().arr() for instance in instances]).T
        trainingTargets = self.targetMat(instances).T
        cases = trainingSet.shape[1]

        for epoch in range(epochs):

            if epoch % 500 == 0:
                print epoch

            # Propagate inputs forward to compute outputs
            self.predictParallel(trainingSet)

            # Propagate deltas backward from output layer to input layer
            for layer in reversed(range(self.layers)):

                # Compare to target
                if layer == self.layers - 1:
                    diff = self.layerOut[layer] - trainingTargets
                    deltas.append(np.multiply(diff, dsigmoid(self.layerOut[layer])))

                # Compare to following layer
                else:
                    deltaProp = self.weights[layer + 1].T.dot(deltas[-1])
                    deltas.append(np.multiply(deltaProp[:-1,:], dsigmoid(self.layerOut[layer])))


            for layer in range(self.layers):

                deltaIndex = self.layers - 1 - layer

                # Get nodes for a layer as an array of column vectors
                if layer == 0:
                    prevLayer = np.vstack([trainingSet, np.ones([1, cases])])
                else:
                    prevLayer = np.vstack([self.layerOut[layer - 1], np.ones([1, self.layerOut[layer - 1].shape[1]])])

                # Get change in weight for each test case as an array of weight matrices
                weightDeltas = np.multiply(np.expand_dims(prevLayer, 0).transpose(2,0,1), np.expand_dims(deltas[deltaIndex], 0).transpose(2,1,0))

                # Flatten the 3D weight matrix into 2D by summing element-wise to get overall change in weight
                weightDelta = np.sum(weightDeltas, axis = 0)

                # Add weight deltas to weights for this layer
                self.weights[layer] -= learningRate * weightDelta

            deltas = []


    def predictParallel(self, trainingSet):

        cases = trainingSet.shape[1]

        self.layerIn = []
        self.layerOut = []
        for layer in range(self.layers):
            # Dot weights with biased columns (1 column = 1 training case's features)
            layerIn = self.weights[layer].dot(np.vstack([(trainingSet if layer == 0 else self.layerOut[-1]), np.ones([1, cases])]))
            self.layerIn.append(layerIn)
            self.layerOut.append(sigmoid(layerIn))


    def predict(self, instance):

        features = instance.length()

        # Clear intermediates
        self.layerIn = []
        self.layerOut = []

        # Iterate through the layers
        for layer in range(self.layers):
            out = instance.getFeature().arr() if layer == 0 else self.layerOut[-1]
            layerIn = self.weights[layer].dot(biased(out)) # add bias node and dot with weight matrix
            self.layerIn.append(layerIn)
            self.layerOut.append(sigmoid(layerIn))

        return self.classes[np.argmax(self.layerOut[-1])]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return np.multiply(sigmoid(x), (1 - sigmoid(x)))

def biased(arr):
    return np.append(arr, 1)
