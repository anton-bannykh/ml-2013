import os
from sys import stdin, stdout, stderr

from util import *
from perceptron import *

def collect():
	dataset = load()
	trainset, testset = split(dataset)
	w = train(trainset, TRAIN_ITERATIONS)
	outcome = []

	for (x, y) in testset:
		yy = classify(w, x)
		
		if y == yy:
			if y == 1:
				outcome.append(Outcome.TP)
			else:
				outcome.append(Outcome.TN)
		else:
			if y == 1:
				outcome.append(Outcome.FN)
			else:
				outcome.append(Outcome.FP)
	return outcome

def count():	
	outcome = collect()
	
	tp = outcome.count(Outcome.TP)
	fp = outcome.count(Outcome.FP)
	fn = outcome.count(Outcome.FN)
	all = len(outcome)
	
	return ((fp + fn) / all), (tp / (tp + fp)), (tp / (tp + fn))
	
err_rate, prec, rec = count()
print("Error rate = %3.5f" % err_rate)
print("Precision = %3.5f" % prec)
print("Recall = %3.5f" % rec)	
	