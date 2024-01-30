#!/usr/bin/env python
# coding: utf-8

# In[4]:


import CNNModel_MNIST
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as tf


# In[5]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[6]:


tran = tf.Compose([
    tf.ToTensor(),
    tf.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, transform=tran, download=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# In[7]:


loaders = {
    
    'train': DataLoader(train_dataset,
                       batch_size=128,
                       shuffle = True),
    
    'val': DataLoader(val_dataset,
                       shuffle = True),
}


# In[15]:


model = CNNModel_MNIST.CNN().cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss_fn = nn.CrossEntropyLoss()
losses = []
best_val_acc = 0

for epoch in range(1,11):
    model.train()
    for batch_idx, (data,target) in enumerate (loaders['train']):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output,target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % 75 == 0:
            print("Train Epoch", epoch ,"Loss: ", loss.item())
        
    model.eval()
    correct=0
    with torch.no_grad():
        for data, target in loaders['val']:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct+= pred.eq(target.view_as(pred)).sum().item()
            
    val_acc = correct/len(loaders['val'].dataset)
    print("Val Accuracy: ", val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'weights.pth')


# In[16]:


plt.plot(losses, label='Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:




