import numpy
import math
import random

g = 0.8
def h(x):
	res = numpy.zeros(len(x))
	for i in range(len(x)):
		res[i] = 1.0/(1.0 + math.exp(-x[i]))
	return res
		
def net(w, o):
	res = numpy.zeros(len(w[0]))
	for j in range(len(w[0])):
		for i in range(len(o)):
			res[j] += w[i][j] * o[i]
	return res
		
def train(x, y, c):
	n = len(x[0])
	w1 = [[random.random() for j in range(c)]for i in range(n + 1)]
	w2 = [[random.random()] for i in range(c + 1)]	
	for j in range(len(x)):
		x[j] = numpy.append(x[j], [1])
	for times in range(200):
		dw1 = [numpy.zeros(c) for i in range(n + 1)]
		dw2 = [0 for i in range(c + 1)]
		for t in range(len(x)):
			o1 = h(net(w1, x[t]))
			o1 = numpy.append(o1, 1.0)
			o2 = h(net(w2, o1))
			d2 = -o2[0] * (1. - o2[0]) * (o2[0] - (y[t] + 1.)/2.)
#			print o2[0], y[t], d2
			d1 = numpy.zeros(c)
			for i in range(c):
				d1[i] = o1[i] * (1. - o1[i]) * w2[i][0] * d2
			for i in range(c + 1):
				dw2[i] += g * o1[i] * d2
			for i in range(n + 1):
				for j in range(c):
					dw1[i][j] += g * x[t][i] * d1[j]
		for i in range(c + 1):
			w2[i][0] += dw2[i]
		for i in range(n + 1):
			for j in range(c):
				w1[i][j] += dw1[i][j]
	return w1, w2

def test(x, y, w1, w2):
	for j in range(len(x)):
		x[j] = numpy.append(x[j], [1])
	res = [numpy.zeros(2),numpy.zeros(2)]
	while res[0][0] + res[0][1] == 0 or res[0][0] + res[1][0] == 0:
		res = [numpy.zeros(2),numpy.zeros(2)]
		for t in range(len(x)):
			o1 = h(net(w1, x[t]))
			o1 = numpy.append(o1, [1])
			o2 = h(net(w2, o1))
			if o2[0] > 0.5:
				tr = 1
			else:
				tr = -1 
			if y[t] == 1:
				if tr == 1:
					res[0][0]+=1
				else :
					res[0][1]+=1
			else:
				if tr == 1:
					res[1][0]+=1
				else:
					res[1][1]+=1
	return res[0][0]/(res[0][0] + res[0][1]), res[0][0]/(res[0][0] + res[1][0])

def get_c(x, y):
	bc = 5
	bf1 = 0
	b = int(len(y) * 0.5)
	train_x, train_y, test_x, test_y = x[:b], y[:b], x[b:], y[b:]
	c = 10
	while c <= 40:
		w1, w2 = train(train_x[:], train_y, c)
		p, r = test(test_x[:], test_y, w1, w2)
		f1 = 2 * p * r/ (p + r)
		if f1 > bf1:
			bf1, bc = f1, c
#		print p, r
#		print f1
		c += 10
	return bc
