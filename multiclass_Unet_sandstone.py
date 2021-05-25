# https://youtu.be/XyX5HNuv-xE
"""
Author: Dr. Sreenivas Bhattiprolu

Multiclass semantic segmentation using U-Net

Including segmenting large images by dividing them into smaller patches 
and stiching them back

To annotate images and generate labels, you can use APEER (for free):
www.apeer.com 
"""

from simple_multi_unet_model import multi_unet_model #Uses softmax 

from keras.utils import normalize
from keras import backend as K
import glob

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def multicalssification():
    #Resizing images, if needed
    SIZE_X = 128 
    SIZE_Y = 128
    n_classes=4 #Number of classes for segmentation

    #Capture training image info as a list
    train_images = []

    for directory_path in glob.glob("static/database1/"):
        for img_path in glob.glob(os.path.join(directory_path, "*.bmp")):
            img = cv2.imread(img_path, 0)       
            img = cv2.resize(img, (SIZE_Y, SIZE_X))
            train_images.append(img)
    for directory_path in glob.glob("static/database2/"):
        for img_path in glob.glob(os.path.join(directory_path, "*.bmp")):
            img = cv2.imread(img_path, 0)       
            img = cv2.resize(img, (SIZE_Y, SIZE_X))
            train_images.append(img)
        
    #Convert list to array for machine learning processing        
    train_images = np.array(train_images)

    #Capture mask/label info as a list
    train_masks = [] 
    for directory_path in glob.glob("static/database1/"):
        for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
            mask = cv2.imread(mask_path, 0)
            # mask= np.random.randint(0,2, size=(SIZE_X,SIZE_Y))       
            mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
            train_masks.append(mask/255)
    for directory_path in glob.glob("static/database2/"):
        for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
            mask = cv2.imread(mask_path, 0)
            # mask= np.random.randint(0,2, size=(SIZE_X,SIZE_Y))       
            mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
            train_masks.append(mask/255)
            
    #Convert list to array for machine learning processing          
    train_masks = np.array(train_masks)

    ###############################################
    #Encode labels... but multi dim array so need to flatten, encode and reshape
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    n, h, w = train_masks.shape
    train_masks_reshaped = train_masks.reshape(-1,1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

    print(np.unique(train_masks))

    #################################################
    train_images = np.expand_dims(train_images, axis=3)
    train_images = normalize(train_images, axis=1)

    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

    #Create a subset of data for quick testing
    #Picking 10% for testing and remaining for training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size = 0.20, random_state = 0)

    

    print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

    from keras.utils import to_categorical
    train_masks_cat = to_categorical(y_train)
    n_classes= train_masks_cat.shape[3]
    print(n_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))



    test_masks_cat = to_categorical(y_test)
    y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))



    ###############################################################
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(train_masks_reshaped_encoded),
                                                    train_masks_reshaped_encoded)
    print("Class weights are...:", class_weights)


    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH  = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]

    def get_model():
        return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

    model = get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # If starting with pre-trained weights. 
    model.load_weights('test_binarize_database1_database2.hdf5')

    # K.set_value(model.optimizer.learning_rate, 0.001)

    # history = model.fit(X_train, y_train_cat, 
    #                     batch_size = 16, 
    #                     verbose=1, 
    #                     epochs=50, 
    #                     validation_data=(X_test, y_test_cat), 
    #                     #class_weight=class_weights,
    #                     shuffle=False)
                        


    # model.save('test.hdf5')
    # model.save('test_binarize.hdf5')
    # model.save('test_binarize_database1_database2.hdf5')
    #model.save('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')
    ############################################################
    #Evaluate the model
        # evaluate model
    # _, acc = model.evaluate(X_test, y_test_cat)
    # print("Accuracy is = ", (acc * 100.0), "%")


    ###
    #plot the training and validation accuracy and loss at each epoch
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, 'y', label='Training loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # acc = history.history['acc']
    # val_acc = history.history['val_acc']

    # plt.plot(epochs, acc, 'y', label='Training Accuracy')
    # plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    # plt.title('Training and validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()


    ##################################
    _, acc = model.evaluate(X_test, y_test_cat)
    print("Accuracy is = ", (acc * 100.0), "%") 
    import random
    test_img_number = random.randint(0, len(X_test)-1)
    test_img = X_test[test_img_number]
    ground_truth=y_test[test_img_number]
    test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img_norm, 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('database result')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img, cmap='jet')
    plt.show()