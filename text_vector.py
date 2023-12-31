# -*- coding: utf-8 -*-
"""text_processing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13zDuqKliYZdhORi2nyqAPPywGf4p7gi0
"""

import numpy as np
import torch
import torch.nn as nn
import itertools
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import pandas as pd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

vocab, embeddings = [], []
with open('glove.6B.100d.txt', 'rt', encoding='utf8') as fi:
    full_content = fi.read().strip().split('\n')
for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    vocab.append(i_word)
    embeddings.append(i_embeddings)

vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)

vocab_npa = np.insert(vocab_npa, 0, '[PAD]')
vocab_npa = np.insert(vocab_npa, 1, '[UNK]')

pad_emb_npa = np.zeros((1, embs_npa.shape[1]))  # embedding for '<pad>' token.
unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True)  # embedding for '<unk>' token.

# insert embeddings for pad and unk tokens at top of embs_npa.
embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, embs_npa))

my_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float(), freeze=True, padding_idx=0)

word2idx = {
    word: idx
    for idx, word in enumerate(vocab_npa)
}

print(len(word2idx.keys()))

df = pd.read_csv('train_text.csv')

from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Digits, Whitespace

pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])


def sentiment_to_tensor(s: str):
    if s == 'positive':
        return 0
    elif s == 'neutral':
        return 1
    else:
        return 2


def utterance_split(u: str):
    obj = pre_tokenizer.pre_tokenize_str(u)
    return [x[0].lower() for x in obj]


df = pd.concat(
    [df['Utterance'].map(lambda x: utterance_split(x)), df['Sentiment'].map(lambda x: sentiment_to_tensor(x))],
    keys=['data', 'labels'])

df['data'] = df['data'].map(lambda x: [word2idx.get(word, 1) for word in x])

# Tunable hyperparameter
batch_size = 32


class TextDataset(Dataset):
    def __init__(self, data, labels):
        # We have to transpose after pad_sequence since pad_sequence performs a transpose
        self.data = torch.transpose(nn.utils.rnn.pad_sequence(list(map(lambda x: torch.LongTensor(x), df['data']))), 0,
                                    1)

        self.labels = torch.LongTensor(list(labels))

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


text_dataset = TextDataset(df['data'], df['labels'])
text_dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False)

# Model architecture parameters
vocab_size = len(word2idx.keys())
embedding_size = 100
padding_index = 0
num_classes = 3

# Tunable hyperparameters
hidden_layers = 1
hidden_layer_size = 256
dropout_probability = 0.33
linear_output_size = 128
num_directions = 2
elu_alpha = 1
learning_rate = 0.001
scheduler_gamma = 0.9


class GLoVeLSTM(nn.Module):
    def __init__(self):
        super(GLoVeLSTM, self).__init__()

        self.embedding = my_embedding_layer
        self.lstm = nn.LSTM(input_size=embedding_size, num_layers=hidden_layers, hidden_size=hidden_layer_size,
                            bidirectional=(True if num_directions == 2 else False), batch_first=True)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.linear = nn.Linear(in_features=hidden_layer_size * num_directions, out_features=linear_output_size)
        self.activation = nn.ELU(alpha=elu_alpha)
        self.classifier = nn.Linear(in_features=linear_output_size, out_features=num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.classifier(x)
        return x[:, -1, :]


glove_lstm = torch.load('glove_lstm.pt')

# with torch.no_grad():
#     glove_lstm.eval()
#     correct = 0
#     total = 0
#     all_val_loss = []
#     for images, labels in text_dataloader:  # using train model
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = glove_lstm(images)
#         # dic = {"tensor": outputs.cpu(), "label": labels }
#         weights_df = pd.DataFrame(outputs.cpu())
#         labels_df = pd.DataFrame(labels.cpu(), columns=['Label'])
#         combine = pd.concat([weights_df, labels_df], axis=1)
# #         combine.to_csv('train_textmodel.csv', mode='a', index=False, header=False)
#
# df = pd.read_csv('dev_text.csv')
#
# df = pd.concat(
#     [df['Utterance'].map(lambda x: utterance_split(x)), df['Sentiment'].map(lambda x: sentiment_to_tensor(x))],
#     keys=['data', 'labels'])
#
# df['data'] = df['data'].map(lambda x: [word2idx.get(word, 1) for word in x])
#
# text_dataset = TextDataset(df['data'], df['labels'])
# text_dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False)
#
# with torch.no_grad():
#     glove_lstm.eval()
#     correct = 0
#     total = 0
#     all_val_loss = []
#     for images, labels in text_dataloader:  # using train model
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = glove_lstm(images)
#         # dic = {"tensor": outputs.cpu(), "label": labels }
#         weights_df = pd.DataFrame(outputs.cpu())
#         labels_df = pd.DataFrame(labels.cpu(), columns=['Label'])
#         combine = pd.concat([weights_df, labels_df], axis=1)
#         combine.to_csv('dev_textmodel.csv', mode='a', index=False, header=False)

df = pd.read_csv('test_text.csv')

df = pd.concat(
    [df['Utterance'].map(lambda x: utterance_split(x)), df['Sentiment'].map(lambda x: sentiment_to_tensor(x))],
    keys=['data', 'labels'])

df['data'] = df['data'].map(lambda x: [word2idx.get(word, 1) for word in x])

text_dataset = TextDataset(df['data'], df['labels'])
text_dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False)
dic = {0:0,1:0,2:0}
with torch.no_grad():
    glove_lstm.eval()
    correct = 0
    total = 0
    all_val_loss = []
    for images, labels in text_dataloader:  # using train model
        images = images.to(device)
        labels = labels.to(device)
        outputs = glove_lstm(images)
        # dic = {"tensor": outputs.cpu(), "label": labels }
        predicted = torch.argmax(outputs, dim=1)
        for i in range(len(labels)):
            if int(predicted[i]) != 1:
                print(predicted[i])
            if labels[i] == predicted[i]:
                dic[int(labels.cpu()[i])] += 1
        weights_df = pd.DataFrame(outputs.cpu())
        labels_df = pd.DataFrame(labels.cpu(), columns=['Label'])
        combine = pd.concat([weights_df, labels_df], axis=1)
        # combine.to_csv('test_textmodel.csv', mode='a', index=False, header=False)
    print(dic)