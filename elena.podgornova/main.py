import numpy
import perceptrone
import svm
import svm_smo
import kernels
import lr

def split(data, part):
	temp_data = [x.split(',') for x in data]
	numpy.random.shuffle(temp_data)
	b = int(len(temp_data) * part)
	train_x = [numpy.array([float(i) for i in x[2:]]) for x in temp_data[:b]]
	train_y = [1.0 if x[1] == 'M' else -1.0 for x in temp_data[:b]]
	test_x = [numpy.array([float(i) for i in x[2:]]) for x in temp_data[b:]]
	test_y = [1.0 if x[1] == 'M' else -1.0 for x in temp_data[b:]]
	return train_x, train_y, test_x, test_y

def main():
	f = open('wdbc.data')
	lines = f.readlines()
	data = [x for x in lines]
	return data

data = main()
train_x, train_y, test_x, test_y = split(data, 0.4)

c = svm.get_c(train_x, train_y)
tsvm = svm.train(train_x, train_y, c)
psvm, rsvm = svm.test(test_x, test_y, tsvm)
esvm = 2 * psvm * rsvm / (psvm + psvm)

print("svm:")
print("\tF1 %.3f " %esvm)
print("\tprecision %.3f, recall %.3f" %(psvm, rsvm))

tp = perceptrone.train(train_x, train_y)
pp, rp = perceptrone.test(test_x, test_y, tp) 
ep = 2 * pp * rp / (pp + rp)

print("lp:")
print("\tF1 %.3f " %ep)
print("\tprecision %.3f, recall %.3f" %(pp, rp))

c = svm_smo.get_c(train_x, train_y, kernels.poly)
a, b = svm_smo.train(train_x, train_y, c, kernels.poly, 1e-3, 50)
psmo, rsmo = svm_smo.test(train_x, train_y, a, b, test_x, test_y, kernels.poly)
esmo = 2 * psmo*rsmo/(psmo + rsmo)

print("svm_smo_poly:")
print("\tF1 %.3f " %esmo)
print("\tprecision %.3f, recall %.3f" %(psmo, rsmo))

c = svm_smo.get_c(train_x, train_y, kernels.gauss)
a, b = svm_smo.train(train_x, train_y, c, kernels.gauss, 1e-3, 50)
psmo, rsmo = svm_smo.test(train_x, train_y, a, b, test_x, test_y, kernels.gauss)
esmo = 2 * psmo*rsmo/(psmo + rsmo)

print("svm_smo_gauss:")
print("\tF1 %.3f " %esmo)
print("\tprecision %.3f, recall %.3f" %(psmo, rsmo))

c = lr.get_c(train_x, train_y)
t = lr.train(train_x, train_y, c)
plr, rlr = lr.test(test_x, test_y, t)
elr = 2 * plr*rlr/(plr + rlr)

print("lr:")
print("\tF1 %.3f " %elr)
print("\tprecision %.3f, recall %.3f" %(plr, rlr))
