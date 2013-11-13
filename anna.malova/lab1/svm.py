from sklearn import svm
from data import *

def train_SVM(train_data):
	best_score = None, best_C = None, best_classifier = None
	for C in range(1000):
		classifier = svm.LinearSVC(C = C)
		train, test = split_data(train_data, 0.5)
		classifier.fit([v.data for v in train], [v.label for v in train])
		score = score([v.data for v in test], [v.label for v in test])
		if best_score == None or score > best_score:
			best_score = score
			best_C = C
			best_classifier = classifier
	return (best_classifier, best_C, best_score)		
		
def main(train_fraction):
	train, test = load_data(train_fraction)
	lin_svm = train_SVM(train)
	print("Best C = " + str(lin_svm[2]))
	print("Score on train data = " + str(lin_svm[1]))
	score = lin_svm[0].score([v.data for v in test], [v.label for v in test])
	print("Score on test data = " + str(score))
	
