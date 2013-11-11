#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

_DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

def retrieve_data(as_list=False, negative_label=-1, positive_label=1):
    xs, ys = [], []
    file = urlopen(_DATA_URL)
    for line in file.readlines():
        fields = line.decode('utf-8').strip().split(',')

        # negative_label is for benign, 1 is for malignant
        ys.append(negative_label if fields[1] == 'B' else positive_label)

        numeric_fields = [float(x) for x in fields[2:]]
        numeric_fields.insert(0, 1.0)
        if not as_list:
            numeric_fields = np.array(numeric_fields)
        xs.append(numeric_fields)
    file.close()
    return xs, ys