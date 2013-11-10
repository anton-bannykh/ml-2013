import numpy

from util import *
import random	

def weights(dataset):
	xx = [x for (x, _) in dataset]
	yy = [y for (_, y) in dataset]
	
	return numpy.dot(numpy.linalg.pinv(xx), yy)

def classify(w, x):
	return sign(numpy.inner(w, x))
	
def train(dataset, n):
	w = weights(dataset)
	
	best_w = None
	res = 0
	
	i = 0
	while i < n:
		mc = [(xx, yy) for (xx, yy) in dataset if not equals(yy, classify(w, xx))]
		
		if res > len(mc) or best_w is None:
			res = len(mc)
			best_w = w.copy()
		
		if res == 0:
			break
		
		x, y = random.choice(mc)
		w += y * numpy.array(x)
		i += 1

	return best_w