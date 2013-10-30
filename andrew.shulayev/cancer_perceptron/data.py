#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from urllib.request import urlopen

_DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

def retrieve_data():
    xs, ys = [], []
    with urlopen(_DATA_URL) as file:
        for line in file.readlines():
            fields = line.decode('utf-8').strip().split(',')

            # -1 is for benign, 1 is for malignant
            ys.append(-1 if fields[1] == 'B' else 1)

            numeric_fields = [float(x) for x in fields[2:]]
            numeric_fields.insert(0, 1.0)
            xs.append(np.array(numeric_fields))
    return xs, ys