import urllib.request
import os
import tarfile


# get the url from the website.
url_images = "http://tamaraberg.com/faceDataset/originalPics.tar.gz"
url_annotations = "http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz"


print("Downloading dataset images...")
print("Please wait...")
urllib.request.urlretrieve(url_images, "originalPics.tar.gz")
print("images download Successfully")
print("Unzipping dataset images...")
t = tarfile.open("originalPics.tar.gz")
t.extractall()
print("images download Successfully")

print("Downloading dataset annotations...")
print("Please wait...")
urllib.request.urlretrieve(url_annotations, "FDDB-folds.tgz")
print("Unzipping dataset annotations")
t = tarfile.open("FDDB-folds.tgz")
t.extractall()
print("annotations download Successfully")




