import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle
import os

def read_data(filename = "hog_feature.pkl"):
	with open(filename, 'rb') as f:
		dataset = pickle.load(f)
	return dataset

def train_svm(dataset, kernel):
	tuned_parameters = {'rbf':[{'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],'kernel':['rbf']}],
						'linear':[{'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],'kernel': ['linear']}],
						'poly':[{'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],'kernel': ['poly']}]}
	print("start training svm " + kernel)
	train_x, train_y , test_x, test_y = dataset[0], dataset[1], dataset[2], dataset[3]
	clf = svm.SVC(kernel = kernel, gamma = 'scale', probability=True)
	# clf = GridSearchCV(svm.SVC(), tuned_parameters[kernel], n_jobs = -1, verbose = 1)
	clf.fit(X = train_x, y = train_y)
	return clf

def load_model(kernel):
	with open("svm_"+kernel+"_model.pkl","rb")as f:
		return pickle.load(f)

def save_model(model,kernel):
	with open("svm_"+kernel+"_model.pkl","wb")as f:
		pickle.dump(model, f)
	return 


def predict(model, dataset):
	train_res = model.predict(dataset[0])
	train_ac = np.mean(train_res==dataset[1])
	print("The accuracy on train_set is {ac}".format(ac = train_ac))

	test_res = model.predict(dataset[2])
	test_ac = np.mean(test_res==dataset[3])
	print("The accuracy on test_set is {ac}".format(ac = test_ac))
	return train_ac, test_ac

def output_support_vectors(model):
	sv = model.support_vectors_
	print("support_vectors:")
	print(sv)
	print("shape:" + str(sv.shape))
	with open('support_vector.txt','w')as f:
		f.write("support_vectors:"+'\n')
		f.write(str(sv) + '\n')
		f.write("shape:" + str(sv.shape) + '\n')
	return


def main():
	dataset = read_data()
	dataset[1][dataset[1]==0] = -1
	dataset[3][dataset[3]==0] = -1
	print(dataset[1])
	kernels = ["rbf", "linear","sigmoid","poly"]
	models = []
	res = []
	for k in kernels:
		# if os.path.exists("svm_"+k+"_model.pkl"):
		# 	model = load_model(k)
		# else:
		model = train_svm(dataset, k)
		save_model(model, k)
		train_ac, test_ac = predict(model, dataset)
		res.append([train_ac, test_ac])
	return
main()