import numpy
import perceptrone

def split(data, part):
	temp_data = [x.split(',') for x in data]
	numpy.random.shuffle(temp_data)
	b = int(len(temp_data) * part)
	train_data = [numpy.array([float(i) for i in x[2:]]) for x in temp_data[:b]]
	train_y = [1.0 if x[1] == 'M' else -1.0 for x in temp_data[:b]]
	test_data = [numpy.array([float(i) for i in x[2:]]) for x in temp_data[b:]]
	test_y = [1.0 if x[1] == 'M' else -1.0 for x in temp_data[b:]]
	return train_data, train_y, test_data, test_y

#def main():
f = open('wdbc.data')
lines = f.readlines()
data = [x for x in lines]	
train_data, train_y, test_data, test_y = split(data, 0.1)
t = perceptrone.train(train_data, train_y)
precision, recall, error = perceptrone.test(test_data, test_y, t) 
print("error %.3f " %error)
print("precision %.3f, recall %.3f" %(precision, recall))
