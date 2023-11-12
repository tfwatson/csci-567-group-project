import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
import torchvision.transforms as transforms
import pandas as pd
import cv2
batch_size = 32
Imgfolder_address = r"spectrograms\train" # location of img
#num_of_img_read = 1000 # control the number of img used
step = 512
part = "train"  # data read "dev", "test"
label_dict = {"positive":0,  "neutral":1, "negative":2}
binary_label = ["positive",  "neutral", "negative"]  #label that use in clip


csvfile = pd.read_csv(part + "_text.csv", encoding='utf-8')
label = csvfile['Sentiment']  # if you want to try multiclass use csvfile['Emotion']
filenam = [f"{file}.png" for file in csvfile['0']]
device = torch.device('cuda')
num_classes = len(binary_label)

import os
data_train = {'image':[], 'label':[]}


target_size = (100, 100)


batch_size = 100

# image_files = os.listdir(Imgfolder_address)
# num_batches = len(image_files) // batch_size
# test = []
# for i in tqdm(range(num_batches)):
#     batch_files = image_files[i * batch_size: min(len(image_files),(i + 1) * batch_size)]
#     test += [cv2.imread(os.path.join(Imgfolder_address, file)) for file in batch_files]

# for i in range(1):
#     images = cv2.resize(cv2.imread(os.path.join(Imgfolder_address, os.listdir(Imgfolder_address)[0])), target_size)
#     print(images.shape)
#     cv2.imshow('Image', images)
#     cv2.waitKey(0)
list_of_file = filenam
images_train = [np.array(cv2.resize(cv2.imread(os.path.join(Imgfolder_address, list_of_file[i])), target_size)) for i in range(len(list_of_file))]
data_train['image'] = images_train
data_train['label'] = list(map(lambda x: label_dict[x], label))

# for x in tqdm(range(3000)): # len(filenam)
#     file_address = os.path.join(Imgfolder_address, filenam[x])
#     y = label_dict[label[x]]
#     temp = Image.open(file_address)
#     keep = temp.copy()
#     if keep.mode == 'RGBA':
#         keep = keep.convert("RGB")
#     data_train['image'].append(keep)
#     data_train['label'].append(y)
#     temp.close()
##image_size: 3000*1200

part = 'test'
Imgfolder_address = r"spectrograms\test" # location of img
csvfile = pd.read_csv(part + "_text.csv", encoding='utf-8')
label = csvfile['Sentiment']  # if you want to try multiclass use csvfile['Emotion']
data_test = {'image':[], 'label':[]}
filenam = [f"{file}.png" for file in csvfile['0']]

list_of_file = filenam
images_test = [cv2.resize(cv2.imread(os.path.join(Imgfolder_address, list_of_file[i])), target_size) for i in range(len(list_of_file))]
data_test['image'] = images_test
data_test['label'] = list(map(lambda x: label_dict[x], label))

# for x in tqdm(range(3000)): # len(filenam)
#     file_address = os.path.join(Imgfolder_address, filenam[x])
#     y = label_dict[label[x]]
#     temp = Image.open(file_address)
#     keep = temp.copy()
#     if keep.mode == 'RGBA':
#         keep = keep.convert("RGB")
#     data_test['image'].append(keep)
#     data_test['label'].append(y)
#
#
#     temp.close()

##reshape image_size
img_size = 100
preprocess = transforms.Compose([
                                #[transforms.Resize((img_size,img_size)),
                                 transforms.ToTensor()
                                 ])
inputs_train = []
for i in range(len(data_train['image'])):
    image = data_train['image'][i]
    label = data_train['label'][i]

    input_tensor = preprocess(image)
    inputs_train.append([input_tensor, label])


np.random.seed(0)
idx = np.random.randint(0, len(data_train['label']), step)
print(idx)
tensors = torch.concat([inputs_train[i][0] for i in idx], axis=1)
print(tensors.shape)

tensors = tensors.swapaxes(0,1).reshape(3,-1).T
print(tensors.shape)

mean = torch.mean(tensors,axis=0)
std = torch.std(tensors, axis=0)
del tensors
preprocess = transforms.Compose([transforms.Normalize(mean=mean,std=std)])
for i in tqdm(range(len(inputs_train))):
    input_tensor = preprocess(inputs_train[i][0])
    inputs_train[i][0] = input_tensor

preprocess = transforms.Compose([#transforms.Resize((img_size,img_size)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=mean,std=std)
                                 ])
inputs_val = []
for i in range(len(data_test['image'])):
    image = data_test['image'][i]
    label = data_test['label'][i]

    input_tensor = preprocess(image)
    inputs_val.append([input_tensor, label])


dloader_train = torch.utils.data.DataLoader(
    inputs_train,batch_size=batch_size, shuffle=True
)
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
        self.fc6 = nn.Linear(30976, 1024)
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
model = ConvNeuralNet(num_classes).to(device)
loss_func = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(dloader_train):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        all_val_loss = []
        for images, labels in dloader_val:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            total += labels.size(0)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted==labels).sum().item()
            all_val_loss.append(loss_func(outputs, labels).item())
        mean_val_loss = sum(all_val_loss) / len(all_val_loss)
        mean_val_acc = 100 * (correct/ total)
        print("Epoch {}, loss: {:.4f}, Val_loss: {:.4f}, Val_acc: {:.1f}%".format(
                epoch+1, loss.item(), mean_val_loss, mean_val_acc)
        )

torch.save(model, "cnn.pt")