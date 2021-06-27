import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import seaborn as sns
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2 as cv

df=pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
classes=['Cassava Bacterial Blight (CBB)', 'Cassava Brown Streak Disease (CBSD)',
        'Cassava Green Mottle (CGM)', 'Cassava Mosaic Disease (CMD)', 'Healthy']
df.head()

#Notice the imablanced data
sns.countplot(df['label'])

from torch.utils.data import Dataset, DataLoader

transforms=A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.ColorJitter(),
    A.ShiftScaleRotate(),
    A.RandomCrop(height=256, width=256),
    A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    A.GaussNoise(),
    ToTensorV2()
])

class CassavaDataset(Dataset):
    def __init__(self, df, transform):
        super().__init__()
        self.df=df
        self.transforms=transforms
        self.images=[df['image_id'][i] for i in range(self.df.shape[0])]
        self.image_path='../input/cassava-leaf-disease-classification/train_images'
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_loc=os.path.join(self.image_path,self.images[idx])
        img=cv.imread(img_loc)
        img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
        #Apply Albumentation tranforms
        transformed_img=self.transforms(image=img)['image']
            
        #Finding the label of the image
        label=self.df.iloc[idx, 1]
            
        #Convert to tensor
        target={}
        target['image']=transformed_img
        target['label']=torch.tensor(label, dtype=torch.float32)
        return target

#Stratified train-test split
from sklearn.model_selection import train_test_split
train,val=train_test_split(df, test_size=0.1, stratify=df['label'], random_state=35)
train.reset_index(drop=True, inplace=True)
val.reset_index(drop=True, inplace=True)

print(train.shape)
print(val.shape)

#Dataset
train_dataset=CassavaDataset(train, transforms)
val_dataset=CassavaDataset(val, transforms)

#DataLoader
train_loader=DataLoader(train_dataset, shuffle=True, batch_size=49)
val_loader=DataLoader(train_dataset, shuffle=True, batch_size=20)

#Model and pre-training
model=torchvision.models.resnet152(pretrained=True)
model.fc=torch.nn.Linear(2048, len(classes))
print(model)

num_children=len([i for i in model.children()])
i=0
for child in model.modules():
    if i<(0.9*num_children):
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
optimizer=torch.optim.Adam(params=params, lr=lr, )
criterion=torch.nn.CrossEntropyLoss()
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
        _,score=torch.max(output)
        loss = criterion(score, labels)
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
            _,val_score=torch.max(output)
            valloss = criterion(val_score, labels)
            _loss += valloss.item()
        val_loss.append(_loss)
    
    scheduler.step()
    # Print out the information
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss[epoch], valid_loss[epoch]))

plt.plot(train_loss)
plt.plot(val_loss)