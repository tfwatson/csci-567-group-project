import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm.auto import tqdm
import joblib

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

observed_sentiment = ['positive', 'neutral', 'negative']
def load_data(file_list, label, test_size=0.2):
    x,y=[],[]
    input_folder = r'Audio\dev\wav'
    for i in  tqdm(range(len(file_list))):
        file_name=os.path.join(input_folder, file_list[i])
        #print(file_name)
        emotion= label[i]
        if emotion not in observed_sentiment:
            print(emotion)
            continue
        feature=extract_feature(file_name, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return x, y
    #return train_test_split(np.array(x), y, test_size=test_size, train_size= .1,random_state=9)

part = "dev"  # data read "dev", "test"
csvfile = pd.read_csv(part + "_text.csv", encoding='utf-8')
label = csvfile['Sentiment']  # if you want to try multiclass use csvfile['Emotion']
filenam = [f"{file}.wav" for file in csvfile['0']]
#filenam = filenam[:10]

x, y = load_data(filenam, label, test_size=0.9)

#x_train,x_test,y_train,y_test=load_data(filenam, label, test_size=0.9)

model = joblib.load("voice_3classes.pkl")


pro = model.predict_proba(x)
weights_df = pd.DataFrame(pro)
weights_df.to_csv('dev_voice.csv',index=False, header=False)
# pred  = model.predict(x)
#
# neu_correct = 0
# pos_correct = 0
# neg_correct = 0
# for i in range(len(y)):
#     if y[i] == pred[i]:
#         if y[i] == 'positive':
#             pos_correct+=1
#         elif y[i] == 'negative':
#             neg_correct+=1
#         else:
#             neu_correct+=1
# import collections
# print(collections.Counter(y))
# print(neg_correct,neu_correct,pos_correct)

