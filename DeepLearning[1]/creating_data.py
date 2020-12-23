import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle


DataDir = r"C:\Users\sadi\PycharmProjects\DeepLearning_Cats_Dogs\PetImages\PetImages/"

categories = ["Dog", "Cat"]

# for i in categories:
#     path = os.path.join(DataDir, i)
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#         plt.imshow(img_array, cmap='gray') Plots pictures on the graph and makes it also gray
#         plt.show() Show the graph itself
#         break
#     break

img_size = 80

# new_array = cv2.resize(img_array, (img_size, img_size))
# plt.imshow(new_array, cmap='gray')
# plt.show()

training_data = []


def create_training_data():
    for i in categories:
        path = os.path.join(DataDir, i) # Creates a path to the needed folder (Dogs and Cats)
        class_num = categories.index(i) # Stores index to understand is it a dog or is it a cat

        for img in os.listdir(path): # os.listdir goes through all the files withing the directory
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # Here we turned image into array using cv2.imread. cv2.IMREAD_GRAYSCALE turnes picture into black and white as it weighs less
                new_array = cv2.resize(img_array, (img_size, img_size)) # We resize picture to make it weight less
                training_data.append([new_array, class_num]) # Append our array with photo and number to understand is it a dog or is it a cat

            except Exception: # Some images might be corrupted, to avoid code to stop, we use try/except method
                pass


create_training_data()
print(len(training_data))


import random

random.shuffle(training_data) # Command to shuffle all the files in the given array
for sample in training_data[:10]: # Just for the example print the first ten elements in the file to make sure that they are shuffled properly
    print(sample)

x = [] # For all the arrays, we call them features
y = [] # For all the labels 1 - Cat, 0 - Dog

for features, labels in training_data:
    x.append(features)
    y.append(labels)

# print(x[0].reshape(-1, img_size, img_size, 1))

x = np.array(x).reshape(-1, img_size, img_size, 1)

pickle_out = open(r"/PetImages/X.pickle", 'wb')
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open(r"/PetImages/y.pickle", 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

