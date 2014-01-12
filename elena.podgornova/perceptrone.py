import numpy

def g(x, y):
	if numpy.dot(x, y) >= 0: 
		return 1 
	else: 
		return -1

def train(x, y):
	l = len(x)
	t = numpy.zeros(len(x[0]))	
	for i in range(1000):
		for j in range(l):
			if g(t, x[j]) != y[j]:
				t += y[j] * x[j]
	return t

def test(x, y, t):
	res = [numpy.zeros(2),numpy.zeros(2)]
	while res[0][0] + res[0][1] == 0 or res[0][0] + res[1][0] == 0:
		res = [numpy.zeros(2),numpy.zeros(2)]
		for i in range(len(x)):
			tr = g(t, x[i])
			if y[i] == 1:
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
