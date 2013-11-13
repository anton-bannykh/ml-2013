from urllib.request import urlopen

class Vector:
    def __init__(self, data, label):
        self.data = data
        self.label = label

def split_data(data, fraction):
	perm = np.random.permutation(vecs.size)
	data1_size = int(fraction * data.size)
	data1, data2 = perm[:train_size], perm[train_size:]
	
	return data1, data2

def load_data(train_fraction):
	try:
		DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
		inf = open('wdbc.data')
	except IOError:
		inf = urlopen(DATA_URL)
		
	ss = inf.readlines()
    vecs = np.array([Vector(np.array(list(map(float, s.split(',') [2:]))), 1 if s.split(',')[1] == 'M' else -1) for s in ss])
    tr, t = split_data(vecs, train_fraction)
    inf.close()

    return (vecs[tr], vecs[t])
	
	
