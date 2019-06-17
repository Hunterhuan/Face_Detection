import numpy as np
import cv2
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def extract_data():
	data = []
	label = []
	folders = ['train_pos/','train_neg/','test_pos/','test_neg/']
	for folder in folders:
		tmp = []
		tmp2 = []
		filenames = os.listdir(folder)
		count = 0
		for filename in filenames:
			path = folder + filename
			img = cv2.imread(path,cv2.IMREAD_COLOR)
			tmp.append(np.array(img))
			if 'pos' in folder:
				tmp2.append(1)
			else:
				tmp2.append(0)
			# if count>100:
			# 	break
			# count += 1
		data.append(tmp)
		label.append(tmp2)
	train_set = np.array(data[0] + data[1])
	train_set_label = np.array(label[0] + label[1])
	test_set = np.array(data[2] + data[3])
	test_set_label = np.array(label[2] + label[3])
	print(train_set.shape)
	with open('conv_img.pkl','wb')as f:
		pickle.dump([train_set, train_set_label, test_set, test_set_label],f)
extract_data()

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
from keras.models import Model

np.random.seed(2019)

IF_Train = True



def read_preprocess_data():
	with open('conv_img.pkl','rb')as f:
		data = pickle.load(f)
	return data

def preprocess_data(data):
	data[0] = data[0].astype(np.float32)/255
	data[2] = data[2].astype(np.float32)/255
	data[1] = keras.utils.to_categorical(data[1], num_classes = 2)
	data[3] = keras.utils.to_categorical(data[3], num_classes = 2)
	return data[0], data[1], data[2], data[3]

data = read_preprocess_data()
x_train, y_train, x_test, y_test = preprocess_data(data)

def build_model(inputshape, num_classes = 2):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=inputshape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation("softmax"))
	model.summary()
	return model
	
x_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

model = build_model(inputshape  = x_shape, num_classes = 2)



def train_model():
	sgd = SGD(lr=0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])
	model.fit(x_train, y_train, batch_size = 32, epochs = 20, validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test)
	return model

if IF_Train:
	model = train_model()
	model.save('keras_model.h5')
else:
	model = load_model('keras_model.h5')

print(model)
model.summary()

def extract_interoutput(model, layer_name):
	inter_model = Model(model.input, outputs = model.get_layer(layer_name).output)
	inter_output = inter_model.predict(x_train)
	with open('conv_output.pkl','wb')as f:
		pickle.dump([inter_output, y_train], f)
extract_interoutput(model, 'dense_1')