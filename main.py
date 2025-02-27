#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #For GPU
device='cpu' #For CPU


# In[2]:


print("##############################")
print("Name: Anup Patel")
print("SR: 15474")
print("Dept: CSA")
print("##############################")


# In[3]:


# torch.cuda.is_available()


# In[4]:


#transforming the PIL Image to tensors
trainset = torchvision.datasets.FashionMNIST(root = "./data", train = True, download = True, transform = transforms.ToTensor())
testset = torchvision.datasets.FashionMNIST(root = "./data", train = False, download = True, transform = transforms.ToTensor())


# In[5]:


#loading the training data from trainset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle = True)
#loading the test data from testset
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)


# In[6]:


#features


# In[7]:


# Hyperparameters for our network
input_size = 784
hidden_sizes = [150,150,150,150]
output_size = 10
# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.BatchNorm1d(hidden_sizes[0]),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.BatchNorm1d(hidden_sizes[1]),
                      nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                      nn.ReLU(),
                      nn.BatchNorm1d(hidden_sizes[2]),
                      nn.Linear(hidden_sizes[2], hidden_sizes[3]),
                      nn.ReLU(),
                      nn.BatchNorm1d(hidden_sizes[3]),
                      nn.Linear(hidden_sizes[3], output_size)).to(device)


# In[8]:


#learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# ### Training 

# In[9]:


# ##### Model Training - New Version
# model=model.to(device)
# epochs = 100
# train_loss=[]
# for epoch in range(1,epochs):
#     model.train()
#     correct = 0
#     total = 0
#     running_loss=0
#     correct_test=0
#     total_test=0
#     #### Training Part
#     for data in trainloader:
#         inputs,labels=data
#         inputs, labels = inputs.to(device), labels.to(device)
#         inputs=inputs.view(-1,784)
#         #print(inputs)
#         out = model(inputs)
#         #print(out)
#         loss = criterion(out, labels) 
#         optimizer.zero_grad() 
#         loss.backward() 
#         optimizer.step()
#         running_loss+=loss 
#         _, predicted = torch.max(out.data, 1)
#         total = total + labels.size(0)
#         correct = correct + (predicted == labels).sum().item()
#     acc = 100*correct/total
#     #print(inputs.shape[0])
#     #print(running_loss)
#     loss_in_this_epoch=running_loss/len(trainloader)
#     train_loss.append(loss_in_this_epoch)
#     ### Validation Step
#     for test_data in testloader:
#         model.eval()
#         inputs_test,labels_test=test_data
#         inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
#         inputs_test=inputs_test.view(-1,784)
#         #print(inputs)
#         out_test = model(inputs_test)
#         #print(out)
#         loss = criterion(out_test, labels_test) 
#         _, predicted_test = torch.max(out_test.data, 1)
#         total_test = total_test + labels_test.size(0)
#         correct_test = correct_test + (predicted_test == labels_test).sum().item()
#     acc_test = 100*correct_test/total_test
#     print('Epoch : {:0>4d} | Loss : {:<6.4f} | Train Accuracy : {:<6.2f} | Test Accuracy : {:<6.2f}%'.format(epoch,loss_in_this_epoch,acc,acc_test))
#     if(epoch>40 and acc_test>89):
#         break


# In[10]:


# ## Visualize loss plot
# plt.plot(train_loss)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()


# In[11]:


#Save Model
# torch.save(model,'model/fnn_torch_model_v2.pth')


# In[12]:


#load model
#model=torch.load('model/fnn_torch_model.pth',map_location=torch.device('cpu')) # To run model on CPU Only system
#model=torch.load('model/fnn_torch_model_v2.pth') # To Run on GPU
model=torch.load('model/fnn_torch_model.pth',map_location='cpu') #To Run on CPU


# ### Testing Full Neural Network

# In[13]:


# function to do evaluation (calculate the accuracy) in gpu
from sklearn.metrics import confusion_matrix
def evaluation_fnn(dataloader):
    f=open("multi-layer-net.txt","w+")
    total, correct = 0, 0
    #keeping the network in evaluation mode 
    model.eval()
    prediction=[]
    actual=[]
    running_loss=0
    for data in dataloader:
        inputs, labels = data
        #moving the inputs and labels to gpu
        inputs, labels = inputs.to(device), labels.to(device)
        inputs=inputs.view(-1,784)
        outputs = model(inputs)
        loss=criterion(outputs, labels)
        #print(loss)
        _, pred = torch.max(outputs.data, 1)
        running_loss+=loss 
        total += labels.size(0)
        for i in range(len(pred.numpy())):
            prediction.append(pred[i].numpy())

        for j in range(len(labels.numpy())):
            actual.append(labels[j].numpy())
        #print(labels)
        correct += (pred == labels).sum().item()


    #print(actual)
    #print(prediction)
    accuracy=correct / total
    total_loss=running_loss/len(testloader)
    cm=confusion_matrix(prediction, actual)
    f.write("Loss on Test Data : ")
    f.write(str(total_loss.detach().numpy()))
    f.write("\n")
    f.write("Accuracy on Test Data : ")
    f.write(str(accuracy))
    f.write("\n")
    f.write("gt_label,pred_label")
    f.write("\n")
    for i in range(len(prediction)):
        f.write(str(actual[i]))
        f.write(',')
        f.write(str(prediction[i]))
        f.write("\n")
    f.close()
    print("multi-layer-net.txt file generated Successfully")
    return cm,accuracy


# In[14]:


confusion_matrix_test,test_accuracy=evaluation_fnn(testloader)
# print("Full Neural Network ::")
# confusion_matrix_test,test_accuracy=evaluation_fnn(testloader)
# confusion_matrix_train,train_accuracy=evaluation_fnn(trainloader)
# print('Test accuracy: %0.2f, Train accuracy: %0.2f' % (test_accuracy, train_accuracy))
# print("Confusion Matrix")
# print(confusion_matrix_test)


# ### CNN 

# In[15]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# In[16]:


model=CNN()


# In[17]:


#loss function and optimizer
#learning_rate=0.001
criterion = nn.CrossEntropyLoss();
optimizer = torch.optim.Adam(model.parameters());


# In[18]:


# ##### Model Training
# model=model.to(device)
# epochs = 50
# train_loss=[]
# epoch_arr=[]
# for epoch in range(1,epochs):
#     model.train()
#     correct = 0
#     total = 0
#     running_loss=0
#     correct_test=0
#     total_test=0
#     #### Training Part
#     for data in trainloader:
#         inputs,labels=data
#         inputs, labels = inputs.to(device), labels.to(device)
#         #inputs=inputs.view(-1,784)
#         #print(inputs)
#         out = model(inputs)
#         #print(out)
#         loss = criterion(out, labels) 
#         optimizer.zero_grad() 
#         loss.backward() 
#         optimizer.step()
#         running_loss+=loss 
#         _, predicted = torch.max(out.data, 1)
#         total = total + labels.size(0)
#         correct = correct + (predicted == labels).sum().item()
#     acc = 100*correct/total
#     #print(inputs.shape[0])
#     #print(running_loss)
#     loss_in_this_epoch=running_loss/len(trainloader)
#     train_loss.append(loss_in_this_epoch)
#     epoch_arr.append(epoch)
#     ### Validation Step
#     for data in testloader:
#         model.eval()
#         inputs_test,labels_test=data
#         inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
#         #inputs=inputs.view(-1,784)
#         #print(inputs)
#         out_test = model(inputs_test)
#         #print(out)
#         loss = criterion(out_test, labels_test) 
         
#         _, predicted_test = torch.max(out_test.data, 1)
#         total_test = total_test + labels_test.size(0)
#         correct_test = correct_test + (predicted_test == labels_test).sum().item()
#     acc_test = 100*correct_test/total_test
#     print('Epoch : {:0>4d} | Loss : {:<6.4f} | Train Accuracy : {:<6.2f} | Test Accuracy : {:<6.2f}%'.format(epoch,loss_in_this_epoch,acc,acc_test))
#     if(epoch>20 and acc_test>=91):
#         break


# In[19]:


#Save Model
# torch.save(model,'model/cnn_torch_model.pth')


# In[20]:


#model=torch.load('model/cnn_torch_model_colab_final_v2.pth.pth') #To Run on GPU
model=torch.load('model/cnn_torch_model.pth',map_location='cpu') #To Run on CPU


# ### Testing CNN Model 

# In[21]:


# function to do evaluation (calculate the accuracy) in gpu
from sklearn.metrics import confusion_matrix
def evaluation_cnn(dataloader):
    f=open("convolution-neural-net.txt","w+")
    total, correct = 0, 0
    #keeping the network in evaluation mode 
    model.eval()
    prediction=[]
    actual=[]
    running_loss=0
    for data in dataloader:
        inputs, labels = data
        #moving the inputs and labels to gpu
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss=criterion(outputs, labels)
        #print(loss)
        _, pred = torch.max(outputs.data, 1)
        running_loss+=loss 
        total += labels.size(0)
        for i in range(len(pred.numpy())):
            prediction.append(pred[i].numpy())

        for j in range(len(labels.numpy())):
            actual.append(labels[j].numpy())
        #print(labels)
        correct += (pred == labels).sum().item()


    #print(actual)
    #print(prediction)
    accuracy=correct / total
    total_loss=running_loss/len(testloader)
    cm=confusion_matrix(prediction, actual)
    f.write("Loss on Test Data : ")
    f.write(str(total_loss.detach().numpy()))
    f.write("\n")
    f.write("Accuracy on Test Data : ")
    f.write(str(accuracy))
    f.write("\n")
    f.write("gt_label,pred_label")
    f.write("\n")
    for i in range(len(prediction)):
        f.write(str(actual[i]))
        f.write(',')
        f.write(str(prediction[i]))
        f.write("\n")
    f.close()
    print("convolution-neural-net.txt file generated Successfully")
    return cm,accuracy


# In[22]:


confusion_matrix_test,test_accuracy=evaluation_cnn(testloader)
#confusion_matrix_train,train_accuracy=evaluation_cnn(trainloader)
#print('Test accuracy: %0.2f, Train accuracy: %0.2f' % (test_accuracy, train_accuracy))

print("Confusion Matrices are plotted in Report")

# f.write("Loss on Test Data : ")
# f.write("Accuracy on Test Data : : ",test_accuracy)
# f.write("gt_label,pred_label")
# for i in range(len(prediction_test)):
#     f.write(prediction_test[i],actual_test[i])
# f.close()


# In[23]:


# print(confusion_matrix_test)


# In[ ]:




