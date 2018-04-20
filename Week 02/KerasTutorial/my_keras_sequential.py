import keras
import os

# two classes
num_classes = 2

# Y_train.shape = (num, 1) -> y_train.shape = (num, 2)
y_train = keras.utils.to_categorical(Y_train, num_classes)
y_test = keras.utils.to_categorical(Y_test, num_classes)

def myKerasModel():
    # input_shape should be the shape without example dim.
    # e.g. X_train.shape = (10000, 64, 64, 3)
    # input_shape should be (64, 64, 3)
    # then Input(input_shape) would be (None, 64, 64, 3)
    input_shape = X_train.shape[1:]
    
    print("input_shape = " + str(input_shape))

    model = keras.models.Sequential()
    model.add(Conv2D(32, (3,3), padding = 'same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    op = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(optimizer=op, loss='categorical_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    model.fit(x=X_train, y=y_train, batch_size=32, verbose=2, epochs=10)
    preds = model.evaluate(x = X_test, y = y_test)
    
    print()
    print(preds)
    
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'myKerasModel.h5'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

 
myKerasModel()
