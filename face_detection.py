import os
import pickle
import numpy as np


from skimage.feature import hog
from skimage import io
from PIL import Image
import cv2


sliding_step = 10
start_scale = 40
scale_step = 10

Load = False

threshold = 0.97

def get_hog(image):
	feature , hog_image = hog(image, orientations = 9, 
								pixels_per_cell = (16,16),
								cells_per_block = (2,2),
								feature_vector = True,
								visualise = True)
	return feature


def sliding_windows(image):
	scale = start_scale

	images = []
	hog_features = []
	windows = []
	for k in range(scale_step):
		scale = int((image.shape[0] - start_scale)*k/scale_step + start_scale)
		print(scale)
		for i in range(0, image.shape[0]-scale, sliding_step):
			for j in range(0, image.shape[1] - scale, sliding_step):
				pic = cv2.resize(image[i:i+scale,j:j+scale, :], (96,96))
				images.append(pic)
				hog_features.append(get_hog(pic))
				# cv2.imshow("laji",image[i:i+scale,j:j+scale, :])
				# cv2.waitKey(0)
				windows.append([i, j , scale])
		scale += scale_step
	return images, hog_features, windows




image = cv2.imread("img_590.jpg", cv2.IMREAD_COLOR)

ret = image.copy()

if Load:
	with open('tmplaji.pkl','rb')as f:
		images, hog_features, windows = pickle.load(f)
		images = np.array(images)
		hog_features = np.array(hog_features)
else:
	print(image.shape)
	images, hog_features, windows = sliding_windows(image)

	with open('tmplaji.pkl','wb')as f:
		pickle.dump([images, hog_feature, windows], f)
# ret = image
# ret = cv2.rectangle(ret, (0,0), (200,200), (255,255,255), 2)
# cv2.imshow("caonima", ret)
# cv2.waitKey(0)

from sklearn import svm


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
from keras.models import Model


def logistic_predict(model, x):
	xw = np.dot(x, model)
	exw = np.exp(xw)
	res = exw/(1+exw)
	prob = res
	prob = np.reshape(prob, (-1,1))
	res = (res>threshold)
	res = np.reshape(res, (-1,1))
	return res, prob

def svm_predict(model, x):
	# res = model.predict(x)
	# res = (res==1)
	res = model.predict_proba(x)
	prob = res[:,1]
	res = res[:,1] > threshold
	res = np.reshape(res, (-1,1))
	prob = np.reshape(prob, (-1,1))
	return res, prob


def CNN_predict(model, x):
	res = model.predict(x)
	ex = np.exp(res)
	tmp = np.reshape(np.sum(ex, axis = 1), (res.shape[0], 1))
	res = ex/tmp
	prob = res[:,1]
	res = res[:,1]>threshold
	res = np.reshape(res, (-1,1))
	prob = np.reshape(prob, (-1,1))
	return res, prob

def load_models():
	with open("logistic_model.pkl",'rb')as f:
		logistic_model = pickle.load(f)
	with open("svm_linear_model.pkl",'rb')as f:
		svm_linear_model = pickle.load(f)
	with open("svm_rbf_model.pkl",'rb')as f:
		svm_rbf_model = pickle.load(f)
	CNN_model = load_model('keras_model.h5')
	CNN_model = Model(CNN_model.input, outputs = CNN_model.get_layer('dense_2').output)
	return logistic_model, svm_linear_model, svm_rbf_model, CNN_model




def predict():
	res1, prob1 = logistic_predict(logistic_model, hog_features)
	print(res1.shape)
	res2, prob2 = svm_predict(svm_linear_model, hog_features)
	print(res2.shape)
	res3, prob3 = svm_predict(svm_rbf_model, hog_features)
	print(res3.shape)
	# res4, prob4 = CNN_predict(CNN_model, images)
	# print(res4.shape)
	res = np.concatenate([ res1, res2, res3], axis = 1)
	prob = np.concatenate([prob1, prob2, prob3], axis = 1)
	print(res.shape)
	return res, prob

logistic_model, svm_linear_model, svm_rbf_model, CNN_model = load_models()

res, prob = predict()


model_nums = res.shape[1]

res = np.sum(res, axis = 1)
res = res >= model_nums

prob = np.mean(prob, axis = 1)


count = 0

windows_tmp = []

for i in range(len(res)):
	if res[i]:
		windows_tmp.append([prob[i],(windows[i][1], windows[i][0]), (windows[i][1]+windows[i][2], windows[i][0]+windows[i][2])])
		# ret = cv2.rectangle(ret, (windows[i][1], windows[i][0]), (windows[i][1]+windows[i][2], windows[i][0]+windows[i][2]), (0,0,255), 2)
		# cv2.imshow("caonima", ret)
		# cv2.waitKey(0)
		count += 1



def IOU(box1, box2):
    '''
    两个框（二维）的 iou 计算
    
    注意：边框以左上为原点
    
    box:[top, left, bottom, right]
    '''
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h<0 or in_w<0 else in_h*in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return iou



NMS_threshold = 0.6



def NMS(windows_nms):
	windows_nms.sort(reverse = True)
	index = 0
	while 1:
		if index >= len(windows_nms):
			break
		delete_list = []
		for i in range(index, len(windows_nms)):
			iou = IOU([windows_nms[index][1][1], windows_nms[index][1][0], windows_nms[index][2][1], windows_nms[index][2][0]],
				[windows_nms[i][1][1], windows_nms[i][1][0], windows_nms[i][2][1], windows_nms[i][2][0]])
			if iou > NMS_threshold:
				delete_list.append(i)
		windows_nms = [windows_nms[i] for i in range(len(windows_nms)) if i not in delete_list]

		index += 1
	return windows_nms


windows_res = NMS(windows_tmp)

for i in range(len(windows_res)):
	ret = cv2.rectangle(ret, windows_res[i][1], windows_res[i][2], (0,0,255), 2)

cv2.imshow("final_result", ret)
cv2.waitKey(0)
