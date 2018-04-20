import numpy as np
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

#%matplotlib inline


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))



#============
# GRADED FUNCTION: HappyModel
#np.random.seed(2018)

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    ### START CODE HERE ###
    print("input_shape = " + str(input_shape))
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    
    x = Input(input_shape)
    
    X = ZeroPadding2D(0)(x)
    # CONV -> BN -> RELU Block
    X = Conv2D(32, (3,3), padding='same')(X)
    #X = BatchNormalization()(X)
    X = Activation(activation='relu')(X)
    
    X = Conv2D(64, (3,3), padding='same')(X)
    #X = BatchNormalization()(X)
    X = Activation(activation='relu')(X)

    X = MaxPooling2D((2,2))(X)
    X = Dense(32)(X)
    X = Activation('relu')(X)
    X = Flatten()(X)
    X = Dense(2)(X)
    X = Activation('softmax')(X)
    
    model = Model(inputs = x, outputs = X, name = 'model')
    ### END CODE HERE ###
    model.summary()
    return model


happyModel = HappyModel(X_train.shape[1:])

happyModel.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

yytest=keras.utils.to_categorical(Y_test, 2)
yytrain=keras.utils.to_categorical(Y_train, 2)

happyModel.fit(x = X_train, y = yytrain, batch_size = 64, verbose = 2, epochs = 5)

preds = happyModel.evaluate(x=X_test, y=yytest)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))    



#=================


plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))
