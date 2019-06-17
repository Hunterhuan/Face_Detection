import numpy as np
import random
import pickle
import math

np.random.seed(222)

def read_data(filename = "hog_feature.pkl"):
	with open(filename, 'rb')as f:
		features = pickle.load(f)
	return features

# output shape (n,1)
def sigmoid(h):
	exw = np.exp(h)
	return exw/(1+exw)

# output shape is just number
def compute_loss(theta, x, y):
	xw = np.dot(x, theta)
	return np.mean(y*xw - np.log(1+np.exp(xw)))

# output shape is (m,1)
def compute_gradient(theta, x, y):
	xw = np.dot(x, theta)
	return np.dot(x.T,(y-sigmoid(xw)))


# output is just float
def compute_ac(theta, x, y):
	xw = np.dot(x, theta)
	res = sigmoid(xw)
	res = ((res >= 0.5)==y)
	return np.mean(res)

# t = np.array([[3],[4]])
# x = np.array([[5,6],[3,4],[7,8]])
# y = np.array([[1],[0],[1]])
# print(compute_loss(t,x,y))
# print(compute_ac(t,x,y))


def train_sgd(dataset, steps, lr):
	# x-shape is (n, 900), y-shape is (n, 1)
	train_x, train_y , test_x, test_y = dataset[0], dataset[1], dataset[2], dataset[3]
	print("train_x's shape is", str(train_x.shape))
	print("train_y's shape is", str(train_y.shape))
	print("test_x's shape is", str(test_x.shape))
	print("test_y's shape is", str(test_y.shape))
	(n,m) = train_x.shape
	weights = np.random.randn(m,)
	step = 0
	for i in range(steps):
		index = np.arange(train_x.shape[0])
		np.random.shuffle(index)
		for j in range(train_x.shape[0]):
			weights = weights + lr * compute_gradient(weights, train_x[index[j]], train_y[index[j]])
			step += 1
			if step%100==0:
				loss_train = compute_loss(weights, train_x, train_y)
				ac_train = compute_ac(weights, train_x, train_y)
				loss_test = compute_loss(weights, test_x, test_y)
				ac_test = compute_ac(weights, test_x, test_y)
				print("step of {a}".format(a = step))
				print("ac of train: {a}, ac of test: {b}".format(a = ac_train, b = ac_test))
	return weights


def train_gd(dataset, steps, lr):
	# x-shape is (n, 900), y-shape is (n, 1)
	train_x, train_y , test_x, test_y = dataset[0], dataset[1], dataset[2], dataset[3]
	print("train_x's shape is", str(train_x.shape))
	print("train_y's shape is", str(train_y.shape))
	print("test_x's shape is", str(test_x.shape))
	print("test_y's shape is", str(test_y.shape))
	(n,m) = train_x.shape
	weights = np.random.randn(m,)
	step = 0
	for i in range(steps):
		# weights = weights + lr * compute_gradient(weights, train_x, train_y)/train_x.shape[0]
		weights = weights + lr * compute_gradient(weights, train_x, train_y)
		step += 1
		if step%100==0:
			loss_train = compute_loss(weights, train_x, train_y)
			ac_train = compute_ac(weights, train_x, train_y)
			loss_test = compute_loss(weights, test_x, test_y)
			ac_test = compute_ac(weights, test_x, test_y)
			print("step of {a}".format(a = step))
			print("ac of train: {a}, ac of test: {b}".format(a = ac_train, b = ac_test))
	return weights


def train_langevin_dynamics(dataset, steps, delta_t, epsion):
	train_x, train_y , test_x, test_y = dataset[0], dataset[1], dataset[2], dataset[3]
	print("train_x's shape is", str(train_x.shape))
	print("train_y's shape is", str(train_y.shape))
	print("test_x's shape is", str(test_x.shape))
	print("test_y's shape is", str(test_y.shape))
	(n,m) = train_x.shape
	weights = np.random.randn(m,)
	step = 0
	for i in range(steps):
		index = np.arange(train_x.shape[0])
		np.random.shuffle(index)
		for j in range(train_x.shape[0]):
			weights = weights + delta_t * compute_gradient(weights, train_x[index[j]], train_y[index[j]])/2 + math.sqrt(delta_t)*np.random.normal(loc = 0, scale = epsion, size = m)
			step += 1
			if step%100==0:
				loss_train = compute_loss(weights, train_x, train_y)
				ac_train = compute_ac(weights, train_x, train_y)
				loss_test = compute_loss(weights, test_x, test_y)
				ac_test = compute_ac(weights, test_x, test_y)
				print("step of {a}".format(a = step))
				print("ac of train: {a}, ac of test: {b}".format(a = ac_train, b = ac_test))
	return weights

def main():
	dataset = read_data()
	
	sgd_weights = train_sgd(dataset, 10000, 0.01)
	gd_weights = train_gd(dataset, 10000, 0.001)
	ld_weights = train_langevin_dynamics(dataset, 10000, 0.05, 0.01)

	with open("logistic_model.pkl",'wb')as f:
		pickle.dump(sgd_weights, f)
	return 

main()