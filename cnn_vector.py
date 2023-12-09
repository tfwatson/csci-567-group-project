import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
import torchvision.transforms as transforms
import pandas as pd
import cv2
import os
batch_size = 32
label_dict = {"positive":0,  "neutral":1, "negative":2}
target_size = (100, 100)
part = 'train'
Imgfolder_address = r"spectrograms\train" # location of img
csvfile = pd.read_csv(part + "_text.csv", encoding='utf-8')
label = csvfile['Sentiment']  # if you want to try multiclass use csvfile['Emotion']
data_test = {'image':[], 'label':[]}
filenam = [f"{file}.png" for file in csvfile['0']]


list_of_file = filenam
file_num = len(list_of_file)

images_test = [cv2.resize(cv2.imread(os.path.join(Imgfolder_address, list_of_file[i])), target_size) for i in range(file_num)]
data_test['image'] = images_test
data_test['label'] = list(map(lambda x: label_dict[x], label[:file_num]))




preprocess = transforms.Compose([#transforms.Resize((img_size,img_size)),
                                 transforms.ToTensor(),
                                 ])
inputs_val = []
for i in range(len(data_test['image'])):
    image = data_test['image'][i]
    label = data_test['label'][i]

    input_tensor = preprocess(image)
    inputs_val.append([input_tensor, label])


dloader_val = torch.utils.data.DataLoader(
    inputs_val,batch_size=batch_size, shuffle=False
)

class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, padding=1)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=4, padding=1)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.conv_layer4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()

        self.conv_layer5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.max_pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropout6 = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(230400, 1024)
        self.relu6 = nn.ReLU()

        self.dropout7 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(1024,512)
        self.relu7 = nn.ReLU()

        self.dropout8 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(512, 256)
        self.relu8 = nn.ReLU()

        self.fc9 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu1(out)
        out = self.max_pool1(out)

        out = self.conv_layer2(out)
        out = self.relu2(out)
        out = self.max_pool2(out)

        out = self.conv_layer3(out)
        out = self.relu3(out)

        out = self.conv_layer4(out)
        out = self.relu4(out)

        out = self.conv_layer5(out)
        out = self.relu5(out)
        out = self.max_pool5(out)

        out = out.reshape(out.size(0), -1)

        out = self.dropout6(out)
        out = self.fc6(out)
        out = self.relu6(out)

        out = self.dropout7(out)
        out = self.fc7(out)
        out = self.relu7(out)

        out = self.dropout8(out)
        out = self.fc8(out)
        out = self.relu8(out)
        out = self.fc9(out)
        return out




model = torch.load('cnn.pt')

device = torch.device('cuda')



with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    all_val_loss = []
    for images, labels in dloader_val: # using train model
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        #dic = {"tensor": outputs.cpu(), "label": labels }
        predicted = torch.argmax(outputs, dim=1)
        if labels == predicted:
            print(labels)
        weights_df = pd.DataFrame(outputs.cpu())
        labels_df = pd.DataFrame(labels.cpu(),  columns=['Label'])
        combine =  pd.concat([weights_df, labels_df], axis=1)
        combine.to_csv('train_cnn.csv', mode='a', index=False, header=False)
        # total += labels.size(0)
        # predicted = torch.argmax(outputs, dim=1)
        # correct += (predicted==labels).sum().item()

    # mean_val_acc = 100 * (correct/ total)
    # print("Val_acc: {:.1f}%".format(
    #          mean_val_acc)
    # )