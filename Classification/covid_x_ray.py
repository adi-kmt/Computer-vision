import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import seaborn as sns
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
import cv2 
import shutil

#Cloning the GitHub profile with the COVID images are cloned to be accessed as a directory 
! git clone https://github.com/ieee8023/covid-chestxray-dataset.git

df=pd.read_csv('/content/covid-chestxray-dataset/metadata.csv')
df.head()

#Finding the column names of the data frame
print(df.columns.unique())

#Making a column for Image location
lis=[]
for i in range(len(df.filename)):
  stri='/content/covid-chestxray-dataset/'+ df.folder[i] + '/' + df.filename[i]
  lis.append(stri)
df['Image location']=lis

#Making sure that the column is added
print(df.columns.unique())
print(df.shape)

#Finding the types of views and folders to help clean the data further
print("Types of folders",df.folder.unique())
print("Types of x-ray views",df.view.unique())

#Now we will remove the unnecessary views like L and Coronal (for better efficiency as side views are very difficult to predict) and the volumes folder (for simplicity)
df = df[~df['view'].isin(['L', 'Coronal'])]
df = df[~df['folder'].isin(['volumes'])]
print(df.view.unique())
df.shape

#Forming a Seperate column for COVID Storing Truth value in list and then appending to dataframe
lis_findings=[]
for i in (df.finding):
  if ('COVID-19' in i.split('/')):
    lis_findings.append(1)
  else:
    lis_findings.append(0)

df['COVID-19']=lis_findings
df.drop(columns=['finding', 'view'], inplace=True)

#Shuffling the dataset and viewing

df = df.sample(frac = 1)
df=df.reset_index()
df.head()

#Using seaborn we can visualise the count of each class

sns.countplot(df['COVID-19'])
plt.show()

transforms=A.Compose([
    A.HorizontalFlip(),
    A.ShiftScaleRotate(),
    A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    A.Resize(256,256, always_apply=True),
    ToTensor()
])

from torch.utils.data import Dataset, DataLoader

#Dataset
class XrayDataset(Dataset):
    def __init__(self, df, transform):
        super().__init__()
        self.df=df
        self.transforms=transforms
        self.images=[df['Image location'][i] for i in range(self.df.shape[0])]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_loc=self.images[idx]
        image=cv2.imread(img_loc)
        img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        #Apply Albumentation tranforms
        transformed_img=self.transforms(image=img)['image']
            
        #Finding the label of the image
        label=self.df.iloc[idx, -1]
            
        #Convert to tensor
        target={}
        target['image']=transformed_img
        target['label']=torch.tensor(label, dtype=torch.float32)
        return target

#Stratified train-test split
from sklearn.model_selection import train_test_split
train,val=train_test_split(df, test_size=0.2, stratify=df['COVID-19'], random_state=35)
train.reset_index(drop=True, inplace=True)
val.reset_index(drop=True, inplace=True)

print(train.shape)
print(val.shape)

#Dataset
train_dataset=XrayDataset(train, transforms)
val_dataset=XrayDataset(val, transforms)

#DataLoader
train_loader=DataLoader(train_dataset, shuffle=True, batch_size=39)
val_loader=DataLoader(train_dataset, shuffle=True, batch_size=83)

tt=next(iter(train_loader))
tt['label']

model=torchvision.models.resnet50(pretrained=True)
model.fc=torch.nn.Sequential(
    torch.nn.Linear(2048, 1),
    torch.nn.Sigmoid())
print(model)

num_children=len([i for i in model.children()])
i=0
for child in model.modules():
    if i<7:
        for params in child.parameters():
            params.requires_grad=False
    else:
        for params in child.parameters():
            params.requires_grad=True
    i+=1

#Important hyperparameters
device=torch.device('cuda')
lr=3e-4
params=[i for i in model.parameters() if i.requires_grad==True]
optimizer=torch.optim.Adam(params=params, lr=lr)
criterion=torch.nn.BCELoss()
num_epochs=10
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)

model.to(device)
train_loss=[]
val_loss=[]

for epoch in range(num_epochs):
    loss = 0
    _loss = 0
    
    
    # The training loop
    model.train()
    for dic in train_loader:
        # Moving to device
        inputs, labels = dic['image'].to(device), dic['label'].to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        #Adding loss item
        loss += loss.item()
    train_loss.append(loss)
        
    # Evaluating the model
    model.eval()
    # Torch not calculating gradients for validatioon set, as it doesn't have to perform backprop
    with torch.no_grad():
        for dici in val_loader:
            # Move to device
            inputs, labels = dici['image'].to(device), dici['label'].to(device)
            outputs = model(inputs)
            valloss = criterion(outputs, labels.unsqueeze(1))
            _loss += valloss.item()
        val_loss.append(_loss)
    
    scheduler.step()
    # Print out the information
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch+1, train_loss[epoch], val_loss[epoch]))