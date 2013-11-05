from urllib.request import urlopen
import random

DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
LOCAL_DATA = "load/wdbc.data"
LOCAL_TRAIN = "data/train.data"
LOCAL_TEST = "data/test.data"
FRACTION = 0.1
EPS = 1e-5
TRAIN_ITERATIONS = 1000

class Outcome:
    TP = "true positive"
    TN = "true negative"
    FP = "false positive"
    FN = "false negative"

def load():
	res = []
	
	for data in urlopen(DATA_URL).readlines():
		str = data.decode("utf-8").split(',')
		id = int(str[0])
		diag = 1 if str[1] == 'M' else -1
		res.append(([1.0] + [float(s) for s in str[2:]], diag))
	
	return res;
	
def split(dataset):
	train = int(FRACTION * len(dataset))
	random.shuffle(dataset)
	return dataset[:train], dataset[train:]

def equals(a, b):
	return abs(a - b) < EPS

def sign(a):
	return -1 if a <= 0 else 1 