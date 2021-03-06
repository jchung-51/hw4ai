import os
import argparse
import sys
import pickle
from Methods import ClassificationLabel, FeatureVector, Instance, Predictor, DecisionTree, NaiveBayes, NeuralNetwork

def load_data(filename):
	instances = []
	labels = set()
	with open(filename) as reader:
		for line in reader:
			if len(line.strip()) == 0:
				continue
			
			# Divide the line into features and label.
			split_line = line.split(",")
			label_string = split_line[0]

			label = ClassificationLabel(label_string)
			feature_vector = FeatureVector()
			
			if label not in labels:
				labels.add(str(label))
			
			index = 0
			for item in split_line[1:]:  
				value = float(item)

				feature_vector.add(index, value)
				index += 1

			instance = Instance(feature_vector, label)
			instances.append(instance)
			
			#print label, feature_vector
			
	#print labels
	return instances, labels

def get_args():
	parser = argparse.ArgumentParser(description="This allows you to specify the arguments you want for classification.")

	parser.add_argument("--data", type=str, required=True, help="The data files you want to use for training or testing.")
	parser.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="Mode: train or test.")
	parser.add_argument("--model-file", type=str, required=True, help="Filename specifying where to save or load model.")
	parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")
	parser.add_argument("--gain", type=str, help="info gain")

	args = parser.parse_args()
	check_args(args)

	return args

def predict(predictor, instances, args):
	results = []
	actual = []
	labelsCorrect = dict()
	predictionLabels = dict()
	instanceLabels = dict()
	correct = 0.0
	total_count = len(instances)
	for instance in instances:
		label = predictor.predict(instance)
		actual.append(instance.getLabel())
		results.append(label)
		if (instance.getLabel() == label):
			correct += 1.0
			if not label in labelsCorrect:
				labelsCorrect[label] = 1
			else:
				labelsCorrect[label] += 1

		if not label in predictionLabels:
			predictionLabels[label] = 1
		else:
			predictionLabels[label] += 1

		if not instance.getLabel() in instanceLabels:
			instanceLabels[instance.getLabel()] = 1
		else:
			instanceLabels[instance.getLabel()] += 1

		print(instance.getLabel(), label)

	printAllStats = False

	if printAllStats:
		print args.algorithm.lower()
		print args.model_file.lower().replace(".model", "")
		print "Accuracy: ", str(correct / total_count), "\t\t", correct, "/", total_count
		for label, numCorrect in labelsCorrect.iteritems():
			print "Label", label, ":"
			print "\tPrecision:", float(numCorrect) / predictionLabels[label], "\t", numCorrect, "/", predictionLabels[label]
			print "\tRecall:   ", float(numCorrect) / instanceLabels[label], "\t", numCorrect, "/", instanceLabels[label]
	else:
		print correct, total_count, correct / total_count

	return results

def check_args(args):
	if args.mode.lower() == "train":
		if args.algorithm is None:
			raise Exception("--algorithm must be specified in mode \"train\"")
	else:
		if not os.path.exists(args.model_file):
			raise Exception("model file specified by --model-file does not exist.")

def train(instances, labels, algorithm, ratio):
	"""
	This is where you tell classify.py what algorithm to use for training
	The actual code for training should be in the Predictor subclasses
	For example, if you have a subclass DecisionTree in Methods.py
	You could say
	if algorithm == "decision_tree":
		predictor = DecisionTree()
	"""
	if algorithm == "decision_tree":
		if not ratio:
			predictor = DecisionTree()
		else:
			print "gain"
			predictor = DecisionTree(ratio)
	elif algorithm == "naive_bayes":
		predictor = NaiveBayes()
	elif algorithm == "neural_network":
		features = instances[0].length()
		classes = len(labels)
		shape = (features, features, classes)
		predictor = NeuralNetwork(shape, labels)
	predictor.train(instances)
	return predictor

def main():
	args = get_args()
	if args.mode.lower() == "train":
		# Load training data.
		instances, labels = load_data(args.data)
		predictor = None
		if args.gain == "ratio":
			predictor = train(instances, labels, args.algorithm, True)
		else:
			# Train
			predictor = train(instances, labels, args.algorithm, None)
		try:
			with open(args.model_file, 'wb') as writer:
				pickle.dump(predictor, writer)
		except IOError:
			raise Exception("Exception while writing to the model file.")
		except pickle.PickleError:
			raise Exception("Exception while dumping pickle.")

	elif args.mode.lower() == "test":
		# Load the test data.
		instances, labels = load_data(args.data)

		predictor = None
		# Load model
		try:
			with open(args.model_file, 'rb') as reader:
				predictor = pickle.load(reader)
		except IOError:
			raise Exception("Exception while reading the model file.")
		except pickle.PickleError:
			raise Exception("Exception while loading pickle.")

		predict(predictor, instances, args)
	else:
		raise Exception("Unrecognized mode.")

if __name__ == "__main__":
	main()
