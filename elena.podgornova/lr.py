import numpy
from scipy.optimize import minimize

def train(x, y, c):
	def optim(tmp):
		t, t0 = tmp[:-1], tmp[-1]
		return 0.5 * sum(t**2) + c * sum([numpy.log(1.0 + numpy.exp(- y[i] *(numpy.dot(t, x[i]) + t0))) for i in range(len(y))])
	return (minimize(optim, numpy.zeros(len(x[0]) + 1))).x

def g(x, x0, y):
	if numpy.dot(x, y) + x0 >= 0: 
		return 1 
	else: 
		return -1

def test(x, y, tmp):
	t, t0 = tmp[:-1], tmp[-1]
	res = [numpy.zeros(2),numpy.zeros(2)]
	for i in range(len(x)):
		tr = g(t, t0, x[i])
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

def get_c(x, y):
	bc = 0
	bf1 = 0
	b = int(len(y) * 0.5)
	train_x, train_y, test_x, test_y = x[:b], y[:b], x[b:], y[b:]
	for deg in range(-5,3):
		c = 2.0 ** deg
		t = train(train_x, train_y, c)
		p, r = test(test_x, test_y, t)
		f1 = 2 * p * r/ (p + r)
		if f1 > bf1:
			bc, bf1 = c, f1
	return bc
			
