import numpy
import random
import kernels

def train(x, y, c, kernel, tol, mp):
	it = len(x)
	k = numpy.zeros((it, it))
	for i in range(it):
		for j in range(it):
			k[i][j] = kernel(x[i], x[j])
	a, b = numpy.zeros(it), 0
	p = 0
	def f(i):
		return numpy.dot(a * y, k[i]) + b
#		return numpy.dot(a, k[i]) + b
	while (p < mp):
		ca = 0
		for i in range(it):
			ei = f(i) - y[i]
			if (y[i]*ei < -tol and a[i] < c) or (y[i]*ei > tol and a[i] > 0):
				j = random.randint(0, it - 1)
				while (i == j):
					j = random.randint(0, it - 1)
				ej = f(j) - y[j]
				ai, aj = a[i], a[j]
				if y[i] != y[j]:
					l, h = max(0, aj-ai), min(c, c+aj-ai)
				else:
					l, h = max(0, ai+aj-c), min(c, ai+aj)
				n = 2 * k[i][j] - k[i][i] - k[j][j]
				if n >= 0:
					continue
				a[j] = aj - y[j]*(ei-ej)/n
				if a[j] > h:
					a[j] = h
				elif a[j] < l:
					a[j] = l
				if abs(a[j] - aj) < 10**(-5):
					continue
				a[i] = ai + y[i] * y[j] * (aj - a[j])
				b1 = b - ei - y[i] * (a[i] - ai) * k[i][i] - y[j]* (a[j] - aj) * k[i][j]
				b2 = b - ej - y[i] * (a[i] - ai) * k[i][j] - y[j] * (a[j] - aj) * k[j][j]
				if 0 < a[i] < c:
					b = b1
				elif 0 < a[j] < c:
					b = b2
				else:
					b = (b1 + b2) / 2
				ca += 1
		if ca == 0:
			p += 1
		else:
			p = 0	
	return a, b		

def test(train_x, train_y, a, b, x, y, kernel):
	def f(xt):
		return 1 if (b + sum([a[i]*train_y[i]*kernel(train_x[i],xt) for i in range(len(train_y))])) > 0 else -1
#		return 1 if (b + sum([a[i]*kernel(train_x[i],xt) for i in range(len(train_y))])) > 0 else -1
	res = [numpy.zeros(2),numpy.zeros(2)]
	for i in range(len(x)):
		tr = f(x[i])
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
	
def get_c(x, y, kernel):
	bc = 0
	bf1 = 0
	b = int(len(y) * 0.5)
	train_x, train_y, test_x, test_y = x[:b], y[:b], x[b:], y[b:]
	for deg in range(-5,3):
		c = 2. ** deg
		a, b = train(train_x, train_y, c, kernel, 1e-3, 50)
		p, r = test(train_x, train_y, a, b, test_x, test_y, kernel)
		f1 = 2 * p * r/ (p + r)
		if f1 > bf1:
			bc, bf1 = c, f1
	return bc
	
