import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm.auto import tqdm

import pandas as pd

batch_size = 32
Imgfolder_address = r"spectrograms\train" # location of img
num_of_img_read = 64 # control the number of img used
part = "train"  # data read "dev", "test"
binary_label = ["positive",  "neutral", "negative"]  #label that use in clip
model_id = "openai/clip-vit-base-patch32" # model used

#Read the csv file
csvfile = pd.read_csv(part + "_text.csv", encoding='utf-8')
# multi_emo_label = [f"a photo of a {Emo} audio spectrograms" for Emo in csvfile['Emotion']]
# binary_emo_label = [f"a photo of a {Emo} audio spectrograms" for Emo in csvfile['Sentiment']]
label = csvfile['Sentiment']  # if you want to try multiclass use csvfile['Emotion']
filenam = [f"{file}.png" for file in csvfile['0']]
spectrograms_path = "spectrograms/" + part

### apply label to model
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
label_tokens = processor(text=binary_label, padding=True, images=None, return_tensors='pt').to(device)
label_emb = model.get_text_features(**label_tokens)
label_emb = label_emb.detach().cpu().numpy()
label_emb = label_emb / np.linalg.norm(label_emb, axis=0)



### store the img info in to array
import os
im = []
for x in tqdm(range(num_of_img_read)): # len(filenam)
    file_address = os.path.join(Imgfolder_address, filenam[x])
    temp = Image.open(file_address)
    keep = temp.copy()
    im.append(keep)
    temp.close()

### get the output from mode
pred = [] # output store here
for i in tqdm(range(0,len(im),batch_size)):
    i_end = min(i + batch_size, len(im))
    image = processor(
        text=None,
        images= im[i:i_end],
        return_tensors='pt'
    )['pixel_values'].to(device)
    img_emb = model.get_image_features(image)
    #print(img_emb.shape)
    img_emb = img_emb.detach().cpu().numpy()
    score = np.dot(img_emb, label_emb.T)
    #print(score.shape)

    pred.extend(np.argmax(score, axis=1))




#accuracy
count = 0
for i in range(len(pred)):
    print(binary_label[pred[i]], label[i])
    if binary_label[pred[i]] == label[i]:
        count += 1
print("acc: ", count/len(pred))

