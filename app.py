import flask
from flask import render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from FCM import FCM
from multiclass_Unet_sandstone import multicalssification
from SVM import SVM_segmentation
import cv2
import matplotlib.pyplot as plt
from utils import makedirs


app = flask.Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

app.config["DEBUG"] = True
socketio = SocketIO(app)






@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/database1', methods=['GET'])
def database1():
    paths=[]
    for i in range(1,301):
        path="000"
        path+=str(i)
        
        paths.append(path[-3:])
    return render_template('index.html',paths=paths,database="database1")

@app.route('/database2', methods=['GET'])
def database2():
    paths=[]
    for i in range(1,101):
        path="000"
        path+=str(i)
        paths.append(path[-3:])
    return render_template('index.html',paths=paths,database="database2")
    
    
@app.route('/database', methods=['GET'])
def database():
    return render_template('database.html')
    
@app.route('/setting1/<name>/<database>', methods=['GET'])
def setting1(name,database):
    return render_template('setting1.html',name=name,database=database)

@app.route('/setting2', methods=['GET'])
def setting2():
    return render_template('setting2.html')


@app.route('/setting3', methods=['GET'])
def setting3():
    return render_template('setting3.html')

@socketio.on('startP3')
def setting3Start(img_num):
    print(img_num)
    path=SVM_segmentation(int(img_num))
    socketio.emit('output3',path)

@socketio.on('startP2')
def setting2Start():
    multicalssification()
    
    
@socketio.on('start')
def clusterSocket(data):
    print("========",data)
    imgName=data['name']
    try:
        #--------------Lord image file--------------  
        inputPath="static/"+data['database']+"/"+imgName+".bmp"
        img= cv2.imread(inputPath, cv2.IMREAD_GRAYSCALE) # cf. 8bit image-> 0~255

        #--------------Clustering--------------  
        cluster = FCM(img, image_bit=8, n_clusters=data['clusters'], m=2, epsilon=data['epsilon'], max_iter=data['iterations'])
        cluster.form_clusters()
        result=cluster.result
                    
        #-------------------Plot and save result------------------------
        if True:
            makedirs("static/output/FCM/segmentation")            
            outputPath = "static/output/FCM/segmentation/"+imgName+".png"
            plt.imshow(result)
            plt.savefig(outputPath, dpi=300)
            plt.close()       
        emit('output','/'+outputPath)
    except IOError:
        print("Error")




socketio.run(app)