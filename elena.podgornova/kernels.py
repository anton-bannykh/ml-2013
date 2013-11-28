import numpy

def scalar(xi, x):
	return numpy.dot(xi, x);

def poly(xi, x, p=2):
	return (1 + numpy.dot(xi, x))**p

def gauss(xi, x, g = 100):
	return (-g * sum(xi - x) ** 2)
