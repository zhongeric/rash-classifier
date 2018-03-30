from keras.models import load_model
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os
import sys
import math
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop

from keras import backend as K
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.utils.data_utils import get_file

from sklearn.metrics import confusion_matrix


global cls_pred

import time
start_time = time.time()
# your script

import realrashy
#knifey.maybe_download_and_extract()
realrashy.copy_files()

train_dir = realrashy.train_dir
test_dir = realrashy.test_dir

image_input = '/Users/EricZhong/TensorFlow-Tutorials/data/input/IMG_3468.jpg'

def time_elapsed(string_chosen):
	print("----------------------------------")
	print(string_chosen)
	elapsed_time = time.time() - start_time
	time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
	print(elapsed_time)
	print("----------------------------------")

def path_join(dirname, filenames):
	return [os.path.join(dirname, filename) for filename in filenames]#

def load_images(image_paths):
	# Load the images from disk.
	images = [plt.imread(path) for path in image_paths]

	# Convert to a numpy array and return it.
	return np.asarray(images)

def plot_images(images, cls_true, cls_pred=None, smooth=True):

	assert len(images) == len(cls_true)

	# Create figure with sub-plots.
	fig, axes = plt.subplots(3, 3)

	# Adjust vertical spacing.
	if cls_pred is None:
		hspace = 0.3
	else:
		hspace = 0.6
	fig.subplots_adjust(hspace=hspace, wspace=0.3)

	# Interpolation type.
	if smooth:
		interpolation = 'spline16'
	else:
		interpolation = 'nearest'

	for i, ax in enumerate(axes.flat):
		# There may be less than 9 images, ensure it doesn't crash.
		if i < len(images):
			# Plot image.
			ax.imshow(images[i],
					  interpolation=interpolation)

			# Name of the true class.
			cls_true_name = class_names[cls_true[i]]

			# Show true and predicted classes.
			if cls_pred is None:
				xlabel = "True: {0}".format(cls_true_name)
			else:
				# Name of the predicted class.
				cls_pred_name = class_names[cls_pred[i]]

				xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

			# Show the classes as the label on the x-axis.
			ax.set_xlabel(xlabel)
		
		# Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])
	
	# Ensure the plot is shown correctly with multiple plots
	# in a single Notebook cell.
	plt.show()


model = VGG16(include_top=True, weights='imagenet')

datagen_test = ImageDataGenerator(rescale=1./255)

batch_size = 2
input_shape = model.layers[0].output_shape[1:3]


generator_test = datagen_test.flow_from_directory(directory=test_dir,
												  target_size=input_shape,
												  batch_size=batch_size,
												  shuffle=False)

image_paths_test = path_join(test_dir, generator_test.filenames)

steps_test = generator_test.n / batch_size

cls_test = generator_test.classes
#ls_train = generator_train.classes

class_names = list(generator_test.class_indices.keys())
class_names

print(class_names)

print("successfully compiled images")
######################################################################
#time_elapsed(string_chosen="time before model load")

path_model = 'big_unbalancedmodel.keras'

#time_elapsed(string_chosen="time after model load")

model3 = load_model(path_model)

#images = load_images(image_paths=image_paths_test[0:1])

print("successfully loaded model!")

## TEST INPUT VS TRAINEd

def predict(image_path):
	global test_shape

	# Load and resize the image using PIL.
	#img = PIL.Image.open(image_path)
	#img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

	# Plot the image.
	#plt.imshow(img_resized)
	#plt.show()

	img_path = image_path
	img = image.load_img(img_path, target_size=(224, 224))

	plt.imshow(img)
	plt.show()

	x = image.img_to_array(img)
	#x = np.expand_dims(x, axis=0)
	x = x/255
	#x = preprocess_input(x)
	print('Input image shape:', x.shape)

	time_elapsed(string_chosen="time before classification")

	# Convert the PIL image to a numpy-array with the proper shape.
	img_array = np.expand_dims(np.array(x), axis=0)

	# Use the VGG16 model to make a prediction.
	# This outputs an array with 1000 numbers corresponding to
	# the classes of the ImageNet-dataset.
	y_pred = model3.predict(img_array)
	cls_pred = np.argmax(y_pred,axis=1)
	# Decode the output of the VGG16 model.

	#print(y_pred)
	
	print(cls_pred)

	#print(test_shape)

	#ts = x[0] for x in test_shape

	#print(ts)

	cls_p = np.resize(cls_pred,(406,1))

	#print(cls_test)

	#print(cls_p.shape)

	cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
						  y_pred=cls_p)  # Predicted class.

	print("Confusion matrix:")
	
	# Print the confusion matrix as text.
	print(cm)
	
	# Print the class-names for easy reference.
	for i, class_name in enumerate(class_names):
		print("({0}) {1}".format(i, class_name))


	#cls_p_test = np.resize(y_pred,(1,1000))

	#cc = y_pred.flatten()

	#print(cc.shape)

	#cls_names = class_names.flatten()

	#print(cls_names.shape)

	#pred_decoded = decode_predictions(cls_p_test)[0] 

	CLASS_INDEX = class_names

	top = 3

	results = []

	for pred in y_pred:
		top_indices = pred.argsort()[-top:][::-1]
		result = [tuple(CLASS_INDEX[i]) + (pred[i],) for i in top_indices]
		result.sort(key=lambda x: x[2], reverse=True)
		results.append(result)
	
	#return results

	print(results)

	r = str(results)

	rr = [r.replace(',' , '') for item in results]

	new_results = r.split(")")

	array_length = len(new_results)

	#print(new_results)

	for i in range(array_length):
		print(new_results[i])

	#print(rr[0])

	time_elapsed(string_chosen="time after classification")

#nb = input('*********************** PRESS ENTER TO START ************************')

#if nb == 'GO':
#	print("Classifying images")
#else: 
#	time.sleep(.1)

#classify_input(image_path=image_input)
predict(image_path=image_input)

#getweights(image_path=image_input)

import coremltools
#convert to coreml model
coreml_model = coremltools.converters.keras.convert(model3, input_names='data', image_input_names='data',
													class_labels=class_names, red_bias=-1,
													blue_bias=-1, green_bias=-1, image_scale=2./255)

#set parameters of the mode
coreml_model.short_description = "Classify the type of rash"
#coreml_model.input_description["image"] = 'image to be classified'
#coreml_model.output_description["classification"] = "Levels Of Confidence For Prediction"
#save the model
coreml_model.save("rashClassifier.mlmodel")

print("successfuly saved model")
