import numpy

from perceptron import Entry

# downloads data set to a temporary file
def get_data(url, local_path):
    import urllib.request
    import shutil

    with urllib.request.urlopen(url) as fu, open(local_path, 'wb') as f:
        shutil.copyfileobj(fu, f)

# loads data set from the temporary file and converts it appropriately
def load_data(local_path):
    res = []
    with open(local_path) as f:
        for line in f.readlines():
            l = line.split(',')
            id = int(l[0])
            diagnosis = -1 if l[1] == 'B' else 1
            ff = [float(x) for x in l[2: 32]]
            ff.append(0.0) # bias
            features = numpy.array(ff)
            res.append(Entry(id = id, correct = diagnosis, features = features))
    return res

# exact inverse of the 'load_data' function
def write_data(data_set, local_path):
    with open(local_path, 'w') as f:
        for de in data_set:
            l = []
            l.append(str(de.id))
            l.append('B' if de.correct == -1 else 'M')
            l.extend([str(x) for x in de.features])
            f.write(','.join(l) + "\n")
