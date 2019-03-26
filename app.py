
from flask import Flask, render_template,request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import base64
import sys 
import os
sys.path.append(os.path.abspath("./model"))
from load import * 
app=Flask(__name__)
global model, graph
model,graph=init()
#dgt=np.array(['০','১','২','৩','৪','৫','৬','৭','৮','৯'])

def convertImage(imgData1):
    print(imgData1[22:])
    imgstr=imgData1[22:]
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr))
        
@app.route('/')
def index():
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
    imgData=request.get_data()
    convertImage(imgData)
    x=imread('output.png',mode='L')
    print(x.shape)
    x=imresize(x,(32,32))
    x=x.reshape(1,32,32,1)
    with graph.as_default():
        print(x.shape)
        out=model.predict(x)
        print(out)
        print(np.argmax(out,axis=1))
        #response=np.array_str(dgt[np.argmax(out,axis=1)])
        response=np.array_str(np.argmax(out,axis=1))
        
        return response	
	

if __name__=="__main__":
	port=int(os.environ.get('PORT',5000))
	app.run(host='0.0.0.0',port=port)
