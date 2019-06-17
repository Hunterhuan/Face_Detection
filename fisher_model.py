import numpy as np
import pickle



def read_data(filename = "hog_feature.pkl"):
	with open(filename, 'rb') as f:
		dataset = pickle.load(f)
	return dataset

def divide_dataset(dataset):
	train_x, train_y , test_x, test_y = dataset[0], dataset[1], dataset[2], dataset[3]
	train_x_pos = []
	train_x_neg = []
	test_x_pos = []
	test_x_neg = []
	for i in range(len(train_x)):
		if train_y[i]==1:
			train_x_pos.append(train_x[i])
		else:
			train_x_neg.append(train_x[i])
	for i in range(len(test_x)):
		if test_y[i]==1:
			test_x_pos.append(test_x[i])
		else:
			test_x_neg.append(test_x[i])
	train_x_pos, train_x_neg, test_x_pos, test_x_neg = np.array(train_x_pos), np.array(train_x_neg), np.array(test_x_pos), np.array(test_x_neg)
	return [train_x_pos, train_x_neg, test_x_pos, test_x_neg]

def compute_beta(dataset):
    train_x_pos, train_x_neg, test_x_pos, test_x_neg = dataset[0], dataset[1], dataset[2], dataset[3]
    mean_x_pos = np.reshape(np.mean(train_x_pos, axis = 0),(900,1))
    mean_x_neg = np.reshape(np.mean(train_x_neg, axis = 0),(900,1))
    cov_pos = np.dot(train_x_pos.T - mean_x_pos, train_x_pos-mean_x_pos.T)
    cov_neg = np.dot(train_x_neg.T - mean_x_neg, train_x_neg-mean_x_neg.T)
    S_W = cov_pos + cov_neg
    S_W_inv = np.linalg.inv(S_W)
    beta = np.dot(S_W_inv,mean_x_pos - mean_x_neg)

    intra_var = np.dot((mean_x_pos-mean_x_neg).T, beta)[0][0]**2
    inter_var = (np.dot(np.dot(beta.T, cov_pos), beta) + np.dot(np.dot(beta.T, cov_neg), beta))[0][0]**2

    print("intra variance is {a}".format(a = intra_var))
    print("inter variance is {a}".format(a = inter_var))
    return beta

def save_model(threshold, beta):
	with open("fisher_model.pkl", "wb") as f:
		pickle.dump([beta, threshold], f)
	return 

def predict(dataset, divided_dataset, beta):
	train_x, train_y , test_x, test_y = dataset[0], dataset[1], dataset[2], dataset[3]
	predict_pos = np.dot(divided_dataset[0], beta)
	predict_neg = np.dot(divided_dataset[1], beta)

	threshold = (np.mean(predict_pos) + np.mean(predict_neg))/2
	train_correct = np.sum(predict_pos>=threshold) + np.sum(predict_neg<=threshold)
	train_accuracy = train_correct/dataset[0].shape[0]
	print("accuracy on trainset is {a}".format(a=train_accuracy))

	predict_pos = np.dot(divided_dataset[2], beta)
	predict_neg = np.dot(divided_dataset[3], beta)
	test_correct = np.sum(predict_pos>=threshold) + np.sum(predict_neg<=threshold)
	test_accuracy = test_correct/dataset[2].shape[0]
	print("accuracy on testset is {a}".format(a=test_accuracy))
	save_model(threshold, beta)


def main():
	dataset = read_data()
	divided_dataset = divide_dataset(dataset)

	beta = compute_beta(divided_dataset)

	predict(dataset, divided_dataset, beta)


main()