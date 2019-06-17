import numpy as np
import os,sys
import cv2
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import io
from PIL import Image
import pickle

train_hog_array = []
train_label = []
test_hog_array = []
test_label = []


def read_dataset_path():
	set_list = ['01','02','03','04','05','06','07','08','09','10']
	data_set = []
	for index in set_list:
		# folder_txt_name
		filename = "FDDB-folds/"+"FDDB-fold-"+index+"-ellipseList.txt"
		with open(filename,'r')as f:
			data_set.append([])
			while(1):
				line = f.readline()
				if(line==''):
					break
				# path of picture 
				pic_path = line[:-1]
				# get the face number of this picture
				num = int(f.readline()[:-1])
				# get the face position in this picture
				label = []
				for i in range(num):
					line = f.readline().split()[:-1]
					line = [float(n) for n in line]
					label.append(line)
				print([pic_path, label])
				data_set[-1].append([pic_path, label])
	train_set = data_set[:-2].copy()
	test_set = data_set[-2:].copy()
	# with open('tmp.pkl','wb')as f:
	# 	pickle.dump([train_set, test_set], f)
	return train_set, test_set




def get_hog(image):
	# winSize = (image.shape[1], image.shape[0])
	# blockSize = (32,32)
	# blockStride = (1,1)
	# cellSize = (16,16)
	# nbins = 9
	# cv2.HOGDescriptor(winSize = winSize, 
	# 						blockStride = blockStride, 
	# 						cellSize = cellSize, 
	# 						nbins = nbins)
	# hist = hog.compute(image)
	# print(hist.shape)
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	feature , hog_image = hog(image, orientations = 9, 
								pixels_per_cell = (16,16),
								cells_per_block = (2,2),
								feature_vector = True,
								visualise = True)
	# print(hog_image.shape)
	# io.imshow(hog_image)
	# io.imshow(image)
	# print(hog_image)
	# io.imsave("hog_image.jpg",image)
	# io.show()
	# cv2.imwrite("hog_origin.jpg",image)
	# cv2.imshow("emm", image)
	# cv2.imshow("caonima",hog_image)
	# cv2.waitKey(0)
	# print("lalala")
	# print(feature.shape)
	return feature,hog_image


def generate_feature(dataset , if_train):
	pos_hog_feature = []
	neg_hog_feature = []
	pos_image_set = []
	neg_image_set = []
	neg_folder = []
	if if_train:
		neg_folder = [0,1,2,3]
	else:
		neg_folder = [1]
	count = 0
	for index in range(len(dataset)):
		for img in dataset[index]:
			image = cv2.imread(img[0]+'.jpg',cv2.IMREAD_COLOR)
			image_visual = image
			# for each face in this picture
			for face_label in img[1]:
				# generate pos label
				width = int(face_label[1]*2*(1+1.0/3))
				height = int(face_label[0]*2*(1+1.0/3))
				image_padding = cv2.copyMakeBorder(image,height,height,width,width,cv2.BORDER_REPLICATE)
				cur_pos = [int(face_label[-2]+width/2),int(face_label[-1]+height/2), width, height]
				cv2.rectangle(image_visual, (max(int(face_label[-2]-width/2),0),max(int(face_label[-1]-height/2),0)), 
											(min(int(face_label[-2]+width/2), image.shape[1]-1),min(int(face_label[-1]+height/2),image.shape[0]-1)),
											(255,0,0),1)
				# cv2.imshow("caonima",image_visual)
				# cv2.waitKey(0)
				# padding the picture
				image_face = image_padding[cur_pos[1]:cur_pos[1]+cur_pos[3],cur_pos[0]:cur_pos[0]+cur_pos[2]]
				image_pos = cv2.resize(image_face, (96,96))
				# add to the set list
				pos_image_set.append(image_pos)
				hog_feature, hog_image = get_hog(image_pos)
				pos_hog_feature.append(hog_feature)

				if if_train:
					cv2.imwrite('train_pos/' + str(index)+'_' +img[0].split('/')[-1] + '_' + str(count) + '.jpg', image_pos)
					count += 1
					print('train_pos/' + str(index)+'_'+ img[0].split('/')[-1] + '_' + str(count) + '.jpg')
				else:
					cv2.imwrite('test_pos/'+ str(index)+ '_' + img[0].split('/')[-1] + '_' + str(count) + '.jpg', image_pos)
					count += 1
					print('test_pos/' + str(index)+'_'+ img[0].split('/')[-1] + '_' + str(count) + '.jpg')
				# generate neg label
				if index in neg_folder:
					width = int(face_label[1]*2)
					height = int(face_label[0]*2)
					direc = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
					image_padding = cv2.copyMakeBorder(image,height,height,width,width,cv2.BORDER_REPLICATE)
					for j in range(len(direc)):
						cv2.rectangle(image_visual, (max(int(face_label[-2]-width/2+direc[j][0]*width/3),0),max(int(face_label[-1]-height/2+direc[j][1]*height/3),0)), 
													(min(int(face_label[-2]+width/2+direc[j][0]*width/3), image.shape[1]-1),min(int(face_label[-1]+height/2+direc[j][1]*height/3),image.shape[0]-1)),
													(255,255,0),1)
						cur_pos = [int(face_label[-2]+width/2+direc[j][0]*width/3),int(face_label[-1]+height/2+direc[j][1]*height/3),width, height]
						image_face = image_padding[cur_pos[1]:cur_pos[1]+cur_pos[3],cur_pos[0]:cur_pos[0]+cur_pos[2]]
						image_neg = cv2.resize(image_face, (96,96))
						# add to the set list
						neg_image_set.append(image_neg)
						hog_feature, hog_image = get_hog(image_neg)
						neg_hog_feature.append(hog_feature)
						if if_train:
							cv2.imwrite('train_neg/' + str(index)+'_'+ img[0].split('/')[-1] + '_' + str(count) + '_' + str(j) + '.jpg', image_neg)
							count += 1
							print('train_neg/' + str(index)+'_'+ img[0].split('/')[-1] + '_' + str(count) + '_' + str(j) + '.jpg')
						else:
							cv2.imwrite('test_neg/' + str(index)+'_'+ img[0].split('/')[-1] + '_' + str(count) + '_' + str(j) + '.jpg', image_neg)
							count += 1
							print('test_neg/' + str(index)+'_'+ img[0].split('/')[-1] + '_' + str(count) + '_' + str(j) + '.jpg')
			cv2.imwrite('label_visualize/' + img[0].split('/')[-1] + str(count) + '.jpg', image_visual)
	print(count)
	pos_hog_feature = np.array(pos_hog_feature)
	neg_hog_feature = np.array(neg_hog_feature)
	return pos_hog_feature, neg_hog_feature


train_set_path, test_set_path = read_dataset_path()

# print(len(train_set_path))
# print(len(test_set_path))

train_pos_hog, train_neg_hog = generate_feature(train_set_path, True)
test_pos_hog, test_neg_hog = generate_feature(test_set_path, False)

train_label = np.array([1 for i in range(len(train_pos_hog))] + [0 for i in range(len(train_neg_hog))])
test_label = np.array([1 for i in range(len(test_pos_hog))] + [0 for i in range(len(test_neg_hog))])

train_hog_x = np.concatenate((train_pos_hog,train_neg_hog),axis=0)
test_hog_x = np.concatenate((test_pos_hog,test_neg_hog),axis=0)
with open("hog_feature.pkl",'wb')as f:
	pickle.dump([train_hog_x, train_label, test_hog_x, test_label], f)