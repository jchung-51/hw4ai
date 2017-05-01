from abc import ABCMeta, abstractmethod
import math
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

class DecisionTree():
    
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
    
    
    
    
    
    