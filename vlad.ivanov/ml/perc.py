from random import randint



class Perceptron(object):
	def __init__(self, s_len = None, a_len = None, sToA = None, aToR = None, trainA = None):
		if sToA == None:
			self.sToA = []
		if aToR == None:	
			self.aToR = []
		if trainA == None:
			self.trainA = []
		if s_len == None:
			self.s_len = 0
		if a_len == None:
			self.a_len = 0

	def findA(self, vector):
		temp = [0] * self.a_len
		
#		for t in range(len(self.aToR)):
#			temp.append(0)
		for t in range(self.s_len):
			if vector[t] != 0:
				for g in range(self.a_len):
					temp[g] += vector[t] * self.sToA[t][g]
		for t in range(self.a_len):
			if temp[t] > 0:
				temp[t] = 1
			else:
				temp[t] = 0
		return temp

	def findR(self, vector):
		sum = 0

		for t in range(self.a_len):
			sum += vector[t] * self.aToR[t]
		if sum > 0:
			return 1
		elif sum <= 0:
			return -1

	def changeAtor(self, vector, res):
		if res == 1:
			for t in range(self.a_len):
				self.aToR[t] += vector[t]
		else:
			for t in range(self.a_len):
				self.aToR[t] -= vector[t]
	def createStoa(self):
		self.sToA = []
		for i in range(self.s_len):
			self.sToA.append([])		
			for g in range(self.a_len):
				self.sToA[i].append(randint(0, 1) * 2 - 1)
	
	def createTrainA(self, tData):
		self.trainA = []
		w = 0
		for t in tData:
			print(w)
			w += 1
			self.trainA.append(self.findA(t[0]))
			

	def train(self, tData):
		self.createStoa()
		print("create ATOR")
		self.aToR = [0] * self.a_len
		
		self.createTrainA(tData)
#		self.checkTrainA()
		print('ok')
		flag = True
		poi = 0
		while flag:
			poi += 1
			flag = False
			u = 0
			w2 = 0
			w1 = 0
			for vector, res in tData:
				v = self.trainA[u]
				
				u += 1
				#print(u)
				ans = self.findR(v)
			
				if ans != res:
					self.changeAtor(v, res)
					flag = True
				#print(self.aToR, ans, res, v)
			if poi % 5 == 0:
				u = 0
				for vector, res in tData:
					v = self.trainA[u]
				
					u += 1
					#print(u)
					ans = self.findR(v)
			
					if ans != res:
						if res == 1:
							w2 += 1
						else:
							w1 += 1
				print(w1, w2)
			
			
	def check(self, vector):
		return(self.findR(self.findA(vector)))

	def extract(self, path):
		f = open(path, 'w')
		for t in range(len(self.sToA)):
			f.write((' ').join("{0}".format(n) for n in self.sToA[t]))
			f.write('\n')
		f.write('\n')
		f.write((' ').join("{0}".format(n) for n in self.aToR))
		
	def insert(self, path):
		lst = open(path).readlines()
		self.sToA = []
		self.aToR = []
		for t in lst[:-2]:
			self.sToA.append(list(map(int, t[:-1].split())))
		self.aToR = list(map(int, lst[-1].split()))
		self.a_len = len(self.aToR)
		self.s_len = len(self.sToA)

	def checkTrainA(self):
		for t in range(5000):
			for g in range(5000):
				if t != g and self.trainA[t] == self.trainA[g] and tData[t][1] != tData[g][1]:
					print(t, g)







