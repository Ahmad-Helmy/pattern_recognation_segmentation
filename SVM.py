import numpy as np
from numpy.core.fromnumeric import shape
import cv2
from sklearn.svm import SVC


# Normalization parameter

def read_img(imgname):
    img = cv2.imread(imgname)
    print(img)
    [h,w,d]=shape(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean=[]
    for i in range(h):
        for j in range(w):
            r = img[i,j,0]
            g = img[i,j,1]
            b = img[i,j,2]
          
            mean.append((r+g+ b)/3)

    return mean , h, w, d


# reding Background 1 pixels
def read_Object(imgname,category):

    print(imgname)
    bg = cv2.imread(imgname) 
    [h,w,d]=shape(bg)
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    cc=1 
    x=[]
    for i in range(h):
        for j in range(w):
            r = bg[i,j,0]
            g = bg[i,j,1]
            b = bg[i,j,2]
            x.append((r+g+ b)/3)
    xx = np.mean(x)
    yy = category

    return xx,yy  
def SVM_segmentation(img_num):
    #  '001.bmp' , '002.bmp' , '300.bmp'
    imag_name = ''
    objects=[]
    if(img_num==1):
        imag_name = '001.bmp'
        objects=['static/SVM/bg1.jpg','static/SVM/c1.jpg','static/SVM/n1.jpg']
    elif (img_num==2):
        imag_name = '002.bmp'
        objects=['static/SVM/Background2.JPG','static/SVM/cytoplasm2.JPG','static/SVM/nucleus2.JPG']
    elif (img_num==300):
        imag_name = '300.bmp'
        objects=['static/SVM/Background300.JPG','static/SVM/cytoplasm300.JPG','static/SVM/nucleus300.JPG']
    print('===========',imag_name,objects)
    imag_path='static/SVM/'+imag_name
    
    X_test , x_size, y_size, dim = read_img(imag_path)
    X_train = []

    # 1 : Background , 2 :cytoplasm , 3: nucleus
    Y_train = []

    for i in range(len(objects)) :

        x,y = read_Object(objects[i],i+1)
        X_train.append(x)
        Y_train.append(y)

    # print(X_train,Y_train)



    w = np.random.randint(-1, 1, 12)

    X_train = np.stack((np.asarray(X_train),np.ones(len(X_train))),axis=1)
    print(X_train)
    print(Y_train)
    svclassifier = SVC(kernel='linear', verbose=True, max_iter=50)
    svclassifier.fit((X_train), Y_train)
    y_pred=[]

    bias=np.ones(len(X_test))
    # for i in range(len(X_test)):
    #     y_pred = svclassifier.predict(X_test[i])
    X_test = np.stack((np.asarray(X_test),bias),axis=1)
    print(X_test)
    y_pred = svclassifier.predict(X_test)
    print(y_pred)

    output=[]

    for i in range(len(y_pred)):
    
        if y_pred[i] == 1:
            output.append([245, 220 , 178])
        elif y_pred[i] == 2:
            output.append([146 , 148, 148])
        elif y_pred[i] == 3:
            output.append([246 , 81, 88])

    output_image_shape = (shape(np.asmatrix(output)))

    img = np.array(output)
    output_image = img.reshape((x_size, y_size, dim))
    out_path='static/output/SVM/'+imag_name
    cv2.imwrite(out_path,output_image)

    return out_path
