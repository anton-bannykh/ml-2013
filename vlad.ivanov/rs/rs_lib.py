from collections import defaultdict
from random import uniform as rand
from math import sqrt
f_size = 10
M1 = 0.006
M2 = 0.002
EPS = 0.001
def read_stat(path_data, path_ans):
	train_data = defaultdict(lambda : defaultdict(int))
	test_data = defaultdict(lambda : defaultdict(int))
	f = open(path_data)
	r, users_num, items_num, train_r_num, test_r_num = map(int, f.readline().split())
	r_sum = 0
	for t in range(train_r_num):
		u, i, r = map(int, f.readline().split())	
		train_data[u][i] = r
		r_sum += r

	MU = r_sum * 1.0 / train_r_num			
	f1 = open(path_ans)
	for t in range(test_r_num):
		r = int(f1.readline())
		u, i = map(int, f.readline().split())
		test_data[u][i] = r


	return ((train_data, test_data, items_num, users_num, MU))

def random_v(size):
	temp = []
	for t in range(size):
		temp.append(rand(-1, 1))
	return temp

def dot(x, y):
	temp = 0
	for t in range(f_size):
		temp += x[t] * y[t]
	return temp

def norm(v):
	return sum(map(lambda x: x * x, v))

def predict(u, i, b_i, b_u, items_v, users_v, MU):
	return (MU + b_i[i] + b_u[u] + dot(items_v[i], users_v[u]))

def update_vectors(u, i, delta, b_i, b_u, items_v, users_v):
	b_i[i] += M1 * (delta - M2 * b_i[i])
	b_u[u] += M1 * (delta - M2 * b_u[u])
	users_v_old = users_v[u]
	for t in range(f_size):
		users_v[u][t] += M1 * (delta * items_v[i][t] - M2 * users_v[u][t])
	for t in range(f_size):
		items_v[i][t] += M1 * (delta * users_v_old[t] - M2 * items_v[i][t])
	return b_i, b_u, items_v, users_v
def reg_error(train_data, b_i, b_u, items_v, users_v, MU):
	err = 0
	n = 0
	for u, d in train_data.items():
		for i, r in d.items():
			n += 1
			err += (r - MU - b_i[i] - b_u[u] - dot(users_v[u], items_v[i])) * (r - MU - b_i[i] - b_u[u] - dot(users_v[u], items_v[i])) + M2 * (b_i[i] * b_i[i] + b_u[u] * b_u[u] + norm(users_v[u]) + norm(items_v[i]))
			
	return err / n

def deviation(data, b_i, b_u, items_v, users_v, MU):
	err = 0
	n = 0
	for u, d in data.items():
		for i, r in d.items():
			n += 1
			r_t = predict(u, i, b_i, b_u, items_v, users_v, MU)
			err += (r - r_t) * (r - r_t)		
	return sqrt(err / n)




def training(train_data, items_num, users_num, MU):
	items_v = []
	for i in range(items_num):
		items_v.append(random_v(f_size))
	users_v = []
	for i in range(users_num):
		users_v.append(random_v(f_size))

	b_u = [0] * users_num
	b_i = [0] * items_num
	
	error_delta = 1
	prev_error = 10
	while error_delta > EPS:
		for u, d in train_data.items():
			for i, r in d.items():
				r_t = predict(u, i, b_i, b_u, items_v, users_v, MU)
				b_i, b_u, items_v, users_v = update_vectors(u, i, r - r_t, b_i, b_u, items_v, users_v)
		cur_error = reg_error(train_data, b_i, b_u, items_v, users_v, MU)
		error_delta = prev_error - cur_error
		prev_error = cur_error
		print(cur_error)
	return ((b_i, b_u, items_v, users_v))


















