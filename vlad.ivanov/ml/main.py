from perc import Perceptron
'''
def createTData():
	ans = list(map(int, open("train_labels").read().split()))
	for i in range(NUM_TRAIN):
		print(i)
		x = list(map(int, open("train/" + str(i + 1) + ".txt").read().split()))
		if ans[i] != cur_d:
			tData.append((x, -1))
		else:
			tData.append((x, 1))

	for i in range(NUM_TRAIN, NUM_TRAIN + 20000):
		if ans[i] == cur_d:
			x = list(map(int, open("train/" + str(i + 1) + ".txt").read().split()))
			tData.append((x, 1))
NUM_TRAIN = 20000



#tData = []
#createTData()
#tData = tData[::-1]
#print(len(tData))
#print('training')
'''
cur_d = 2
p = Perceptron()
#p.train(tData)
#p.extract('stoaandator2')

p.insert('stoaandator' + str(cur_d))

print("check")

fw = open("perc_" + str(cur_d), 'w')

for t in range(10000):
	if t % 100 == 0:
		print(t)
	if p.check(list(map(int, open("test/" + str(t + 1) + ".txt").read().split()))) == 1:
		fw.write(str(t + 1) + '\n')	
