# _*_ coding:utf-8 -*-
from sklearn.manifold import TSNE
# from openTSNE.sklearn import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

def read_data(filename = "hog_feature.pkl"):
	with open(filename, 'rb') as f:
		dataset = pickle.load(f)
	return dataset

def tsne(dataset):
	if os.path.exists('tsne.pkl'):
		with open('tsne.pkl', 'rb')as f:
			t = pickle.load(f)
	else:
		t = TSNE(n_components=2, learning_rate=100).fit_transform(dataset[0])
		with open('tsne.pkl','wb')as f:
			pickle.dump(t, f)
	plt.figure(figsize = (12,6))
	plt.scatter(t[:,0], t[:,1], c = dataset[1], s = 1)
	plt.show()

def tsne_3D(dataset):
	if os.path.exists('tsne3D.pkl'):
		with open('tsne3D.pkl', 'rb')as f:
			t = pickle.load(f)
	else:
		t = TSNE(n_components=3, learning_rate=100).fit_transform(dataset[0])
		with open('tsne3D.pkl','wb')as f:
			pickle.dump(t, f)
	ax = plt.subplot(111, projection = '3d')
	ax.set_title('3D_curve_pca')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.scatter(t[:,0], t[:,1], t[:,2], c = dataset[1], s = 1)
	plt.show()

def pca(dataset):
	if os.path.exists('pca.pkl'):
		with open('pca.pkl', 'rb')as f:
			p = pickle.load(f)
	else:
		p = PCA(n_components=2).fit_transform(dataset[0])
		with open('pca.pkl','wb')as f:
			pickle.dump(p, f)
	print(p.shape)
	plt.figure(figsize=(12,6))
	plt.scatter(p[:,0], p[:,1], c = dataset[1],  s = 1)
	plt.show()

def pca_3D(dataset):
	if os.path.exists('pca3D.pkl'):
		with open('pca3D.pkl', 'rb')as f:
			p = pickle.load(f)
	else:
		p = PCA(n_components=3).fit_transform(dataset[0])
		with open('pca3D.pkl','wb')as f:
			pickle.dump(p, f)
	print(p.shape)
	ax = plt.subplot(111, projection = '3d')
	ax.set_title('3D_curve_pca')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.scatter(p[:,0], p[:,1], p[:,2], c = dataset[1], s = 1)
	plt.show()
def main():
	dataset = read_data()
	tsne(dataset)
	pca(dataset)
	# tsne_3D(dataset)
	# pca_3D(dataset)
main()