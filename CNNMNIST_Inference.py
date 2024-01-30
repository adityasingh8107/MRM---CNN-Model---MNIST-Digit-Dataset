#!/usr/bin/env python
# coding: utf-8

# In[2]:


import CNNModel_MNIST
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as tf
from sklearn.metrics import confusion_matrix, classification_report


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[4]:


tran = tf.Compose([
    tf.ToTensor(),
    tf.Normalize((0.5,), (0.5,))
])

test_data=datasets.MNIST(
    root='data',
    train=False,
    transform=tran,
    download = True
)


# In[5]:


loaders = {
    'test': DataLoader(test_data,
                      shuffle = True),
}


# In[6]:


model = CNNModel_MNIST.CNN().to(device)
model.load_state_dict(torch.load('weights.pth'))
model.eval()


# In[7]:


total=0
correct=0
labels=[]
pred_labels = []
with torch.no_grad():
    for data, target in loaders['test']:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        total += target.size(0)
        correct+= pred.eq(target.view_as(pred)).sum().item()
        labels.extend(target.cpu().numpy())
        pred_labels.extend(pred.cpu().numpy())
            
test_acc = correct/len(loaders['test'].dataset)
print("Test Accuracy: ", test_acc)


# In[8]:


conf_matrix = confusion_matrix(labels, pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[9]:


report = classification_report(labels, pred_labels)
print(report)


# In[ ]:




