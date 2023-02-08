from cProfile import label
from secrets import choice
from matplotlib.font_manager import json_dump
from requests import session
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from tensorflow import keras
import shap
from lime.lime_text import LimeTextExplainer
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM , Bidirectional,Dense,GlobalMaxPooling1D,Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from snowballstemmer import TurkishStemmer
import flask
from flask import Flask,request,render_template ,jsonify
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import keras_metrics as km
# from fastapi import FastAPI, Request
# from typing import Union
import random


with open(r"veri.json", encoding='utf-8') as file:
    data = json.load(file)
tags = []
patterns = []
responses={}

sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)


for intent in data['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['patterns']:
    patterns.append(lines)
    tags.append(intent['tag'])
data = pd.DataFrame({"patterns":patterns,
                     "tags":tags})
import string
data['patterns'] = data['patterns'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['patterns'] = data['patterns'].apply(lambda wrd: ''.join(wrd))

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=2000)#gtihubdan bak
tokenizer.fit_on_texts(data['patterns'])
train = tokenizer.texts_to_sequences(data['patterns'])


from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(train)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])#tags




input_shape = x_train.shape[1]
print(input_shape)

vocabulary = len(tokenizer.word_index)
print("number of unique words : ",vocabulary)
#output length
output_length = le.classes_.shape[0]
print("output length: ",output_length)


i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1,20)(i)
x = LSTM(20,return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length,activation="softmax")(x)
model  = Model(i,x)

model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

train = model.fit(x_train,y_train,epochs=100)


loss_train=train.history['loss']
epochs=range(1,501)
plt.plot(epochs,loss_train,'g',label='Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Chatbot Loss Grafiği')
plt.legend()
plt.show()

def getModel(cumle):
    texts_p = []
    prediction_input = cumle
  #removing punctuation and converting to lowercase
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)
    cumle=prediction_input
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    #print(prediction_input)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input],input_shape)
    global sess
    global graph
    with graph.as_default():
      set_session(sess)

      output = model.predict(prediction_input)
      
    output = output.argmax()

    response_tag = le.inverse_transform([output])[0]
    cevap= random.choice(responses[response_tag])
    return cevap






app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getbot')
def get_bot_response():
  message = request.args.get('msg')
  veri=[]
  sonuc=getModel(message)
  veri.append(str(sonuc))
  cvpp={"cevap": str(sonuc)}
  cvp={"data": [cvpp], "success":True,
  "message":"geldi" }
  resp = flask.Response(json.dumps(cvp,ensure_ascii=False))
  resp.headers["Access-Control-Allow-Origin"] = "*"
  return resp;
  #return json.dumps(cvp,ensure_ascii=False)


if __name__ == "__main__":
        app.run(debug=True ,port=8080,use_reloader=False)
        app.config['JSON_AS_ASCII'] = False 
