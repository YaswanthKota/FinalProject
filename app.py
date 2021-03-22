from flask import Flask, request, jsonify, render_template
import pickle
from joblib import load
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input,Conv1D,MaxPooling1D,Dense,GlobalMaxPooling1D,Embedding,concatenate,Dropout
from tensorflow.keras.models import Model
import re
from tensorflow.keras.models import load_model
from joblib import load
import tweepy
import pandas as pd
import time

app = Flask(__name__)

dmodel = load_model('finalmodel.h5')
MAX_SEQUENCE_LENGTH = 50
tokenizer = load('model_tokenizer')
label = ['Normal' , 'Depressive']

def clean_tweet(tweet):
    '''
    Function to clean tweet text by removing links, special characters using simple regex statements.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\d+)|(\w+:\/\/\S+)", " ", tweet).split())


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict' , methods=['POST'])
def predict():
    text = request.form['dtext']
    otext = text
    text = text.lower()
    text = clean_tweet(text)
    x = dmodel.predict(pad_sequences(tokenizer.texts_to_sequences([text]),maxlen=MAX_SEQUENCE_LENGTH))
    res = np.argmax(x,axis=1)
    li = res[0]
    return render_template('index.html' , prediction_text = label[li], index = li, text = otext)
def predict1(text):
    x = dmodel.predict(pad_sequences(tokenizer.texts_to_sequences([text]),maxlen=MAX_SEQUENCE_LENGTH))
    res = np.argmax(x,axis=1)
    li = res[0]
    
    return label[li]

@app.route('/twitter',methods=['GET','POST'])
def twitter():
    return render_template('twiiter.html')

@app.route('/tweets', methods=['POST'])
def tweets():
    username=request.form['username']
    #print(username)
    consumer_key = "Od7zy6m9390gqyUaFomXW9LSv"
    consumer_secret = "IqJYMgtHGaAAUJxzJlsxqgnBKdx8SKP2Q779Kdsz3oEpwQbGde"
    access_token = "1268057362788540416-aUbTXxF5JrRthdcVjXPjZSfTjmJi6n"
    access_token_secret = "Jtzsttach800vnLhAeJPG8UBbOka3v57UsltOcBJAyac0"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True)
    tweets = []
    count = 10
    try:      
        tweets = tweepy.Cursor(api.user_timeline,id=username).items(count)
        tweets_list = [[tweet.created_at, tweet.id, tweet.text] for tweet in tweets]
        tweets_df = pd.DataFrame(tweets_list,columns=['Datetime', 'Tweet Id', 'Text'])
        tweets_df.to_csv('{}-tweets.csv'.format(username), sep=',', index = False)
    except BaseException as e:
        print('failed on_status,',str(e))
        time.sleep(3)
    x=[]
    a={}
    filename=username+'-tweets.csv'
    data = pd.read_csv(filename)
    tweetslist=list(data.values)
    n=0
    d=0
    for i in tweetslist:
        m=clean_tweet(i[2])
        if m!='':
            x.append(m)
            y=predict1(m)
            a[m]=y
            if y=='Normal':
                n+=1
            elif y=='Depressive':
                d+=1
    npercent=(n/(n+d))*100
    dpercent=(d/(n+d))*100

        #print(y)
    #print(a)
    #print(x)
    return render_template('twitter1.html',tweetslist=a,npercent=round(npercent,1),dpercent=round(dpercent,1))


if __name__ == "__main__":
    app.run(debug=True)
