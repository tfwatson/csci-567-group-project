import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm.auto import tqdm

def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
    if chroma:
        stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

emotions={
  '01':'anger',
  '02':'disgust',
  '03':'fear',
  '04':'joy',
  '05':'neutral',
  '06':'sadness',
  '07':'surprise'
}

emotions_toindex={
  'anger': '01',
  'disgust': '02',
  'fear': '03',
  'joy': '04',
  'neutral': '05',
  'sadness': '06',
  'surprise': '07'
}
# Emotions to observe
observed_emotions=['anger','disgust','fear','joy','neutral','sadness','surprise']

def load_data(file_list, label, test_size=0.2):
    x,y=[],[]
    input_folder = r'Audio\train\wav'
    for i in  tqdm(range(len(file_list))):
        file_name=os.path.join(input_folder, file_list[i])
        #print(file_name)
        emotion= label[i]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file_name, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, train_size= 0.75,random_state=9)

import time

part = "train"  # data read "dev", "test"
csvfile = pd.read_csv(part + "_text.csv", encoding='utf-8')
label = csvfile['Emotion']  # if you want to try multiclass use csvfile['Emotion']
filenam = [f"{file}.wav" for file in csvfile['0']]


x_train,x_test,y_train,y_test=load_data(filenam, label, test_size=0.25)

print(f'Features extracted: {x_train.shape[1]}')

model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

import joblib

# save
joblib.dump(model, "voice_model44%.pkl")
#Features extracted: 180
#Accuracy: 43.33%
