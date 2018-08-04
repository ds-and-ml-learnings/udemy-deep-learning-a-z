from keras.models import Sequential
from keras.layers import Convolution2D # Adding the Convolution2D network (videos are in 3D bc of time)
from keras.layers import MaxPooling2D # Adding the Process 2: pooling
from keras.layers import Flatten
from keras.layers import Dense # Add the fully connected layers in a neural network
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model



def preprocessing_images(classifier):
    '''
    Randomly chanhes the image to avoid overfitting
    '''
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_datagen = ImageDataGenerator(rescale=1./255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)

    train_set = train_datagen.flow_from_directory('../archive/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

    test_set = test_datagen.flow_from_directory('../archive/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
    classifier.fit_generator(train_set,
                             steps_per_epoch = (8000/32),
                             epochs = 25,
                             validation_data = test_set,
                             validation_steps = (2000/32))
    classifier.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

    # returns a compiled model
    # identical to the previous one
    #model = load_model('my_model.h5')



def build_classifier():
    # Input size -> (3,64,64) we need to use 3 layers for the rgb color scheme (64 is the pixels)

    # Initializing the CNN
    classifier = Sequential()

    classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation='relu')) # 128 was set random by the author
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    preprocessing_images(classifier)


def build_classifier_2():
    classifier = Sequential()

    classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd Convolutional layers
    classifier.add(Convolution2D(32, (3, 3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))


    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation='relu')) # 128 was set random by the author
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    preprocessing_images(classifier)

'''
250/250 [==============================] - 165s 661ms/step - loss: 0.2120 - acc: 0.9104 - val_loss: 0.5755 - val_acc: 0.7975

'''
build_classifier_2()
