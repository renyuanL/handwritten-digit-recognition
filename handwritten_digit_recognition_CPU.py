#!/usr/bin/env python
# coding: utf-8

# <td>
# <a target="_blank" href="https://colab.research.google.com/github/renyuanL/handwritten-digit-recognition/blob/master/handwritten_digit_recognition_CPU.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run this in Google Colab</a>
# </td>
#  

# - https://colab.research.google.com/github/renyuanL/handwritten-digit-recognition/blob/master/handwritten_digit_recognition_CPU.ipynb
# 
# 

# # Handwritten Digit Recognition
# ## 手寫數字辨識
# 
# - Authors= 
#     - Amitrajit Bose, 
#     - Renyuan Lyu from CGU, Taiwan
# - Dataset= MNIST
# - [Medium Article Link](https://medium.com/@amitrajit_bose/handwritten-digit-mnist-pytorch-977b5338e627)
# - Frameworks = PyTorch
# 

# ### Necessary Imports

# In[1]:


# Import necessary packages
# 引入(import) 必要的模組 (modules)

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time


# In[2]:


import os
#from google.colab import drive


# ### Download The Dataset & Define The Transforms

# In[3]:


# 蒐集、下載、整理 手寫數字資料 (MNIST)，以供實驗
# 第一次執行較花時間，第2次以後可關閉下載功能，以節省時間。

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset= datasets.MNIST('drive/My Drive/mnist/MNIST_data/', 
                         download= True,  # 第一次執行須 True, 以後就可改為False
                         train=True, 
                         transform=transform)
valset=   datasets.MNIST('drive/My Drive/mnist/MNIST_data/', 
                         download= True, 
                         train=False, transform=transform)

trainloader= torch.utils.data.DataLoader(trainset, 
                                         batch_size=64, 
                                         shuffle=True)
valloader=   torch.utils.data.DataLoader(valset,   
                                         batch_size=64, 
                                         shuffle=True)


# ### Exploring The Data

# In[4]:


# 觀賞一下資料庫

dataiter= iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)


# In[5]:


x= images[0].numpy().squeeze()
plt.imshow(x, cmap='gray_r');


# In[6]:


figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    x= images[index].numpy().squeeze()
    plt.imshow(x, cmap='gray_r')


# ### Defining The Neural Network

# ![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mlp_mnist.png)

# In[7]:


# 定義 神經網路的架構參數

from torch import nn

# Layer details for the neural network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)


# In[8]:


def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


# In[9]:


# 在訓練之前，先測試一下辨識效能：

images, labels = next(iter(valloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)


# In[10]:


# 全面測試辨識效能

correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))


# In[ ]:





# In[11]:


# 開始準備訓練

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images)
loss = criterion(logps, labels)


# In[12]:


loss, logps 


# In[13]:


print(f'Before backward pass: weight= {model[0].weight},\n weightGrad= {model[0].weight.grad}\n')


# In[14]:


loss.backward()

# 執行 .backward() 之後， weight 本身不變，但 weightGrad 被更新了！
print(f'Before backward pass: weight= {model[0].weight},\n weightGrad= {model[0].weight.grad}\n')


# In[15]:


# 指定 Optimization 的演算法為 SGD (Stochastic Gradient Decent)

from torch import optim

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# In[16]:


print('Initial weights - ', model[0].weight)

images, labels = next(iter(trainloader))
images.resize_(64, 784)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
output = model(images)
loss = criterion(output, labels)
loss.backward()
print('Gradient -', model[0].weight.grad)


# In[17]:


# Take an update step and few the new weights
optimizer.step()
print('Updated weights - ', model[0].weight)


# ### Core Training Of Neural Network

# In[ ]:


optimizer = optim.SGD(model.parameters(), 
                      lr= 0.003, 
                      momentum= 0.9)

time0 = time() # 監控一下時間，記錄當前時間

epochs= 15  # 這個 epochs 可改小，以免跑太久，但若要提升辨識率，則應該改大一點。

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vectors
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {e} - Training loss: {running_loss/len(trainloader)}")
    print(f"Acc Training Time (in minutes)= {(time()-time0)/60:.3f}\n")


# In[ ]:





# In[ ]:


# 訓練後，再測試一下辨識效能：

images, labels = next(iter(valloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)


# ### Model Evaluation

# In[ ]:


# 訓練後，全面測試辨識效能

correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))


# In[ ]:




