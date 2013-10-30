#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from urllib.request import urlopen

_DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

def retrieve_data(as_list=False, negative_label=-1):
    xs, ys = [], []
    with urlopen(_DATA_URL) as file:
        for line in file.readlines():
            fields = line.decode('utf-8').strip().split(',')

            # negative_label is for benign, 1 is for malignant
            ys.append(negative_label if fields[1] == 'B' else 1)

            numeric_fields = [float(x) for x in fields[2:]]
            numeric_fields.insert(0, 1.0)
            if not as_list:
                numeric_fields = np.array(numeric_fields)
            xs.append(numeric_fields)
    return xs, ys