from rs_lib import *

train_data, test_data, items_num, users_num, MU = read_stat("itmo-recsys-data/movielensfold5.txt", "itmo-recsys-data/movielensfold5ans.txt")

b_i, b_u, items_v, users_v = training(train_data, items_num, users_num, MU)

print(deviation(test_data, b_i, b_u, items_v, users_v, MU))
