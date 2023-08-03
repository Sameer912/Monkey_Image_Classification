import torch 
import torchvision 
import torchvision.transforms as transforms
import os 
import matplotlib.pyplot as plt
import numpy as np 
from Mean_and_std import data_loader, get_mean_and_std


train_dataset_path= "./data/training/training"
test_dataset_path= "./data/validation/validation"

loader= data_loader()
mean, std= get_mean_and_std(loader)
print(mean, std)

def Data_Loader(): 
  #Resizing the images sometimes help to train the model faster.
  train_transforms= transforms.Compose([ 
                                        transforms.Resize((224,224)), 
                                        transforms.RandomHorizontalFlip(), 
                                        transforms.RandomRotation(10), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)) 
                                      ])

  test_transforms= transforms.Compose([ 
                                      transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(), 
                                      transforms.RandomRotation(10), 
                                      transforms.ToTensor(), 
                                      transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
                                      ])

  #Create the train and test dataset. 
  train_dataset= torchvision.datasets.ImageFolder(root= train_dataset_path, transform= train_transforms)
  test_dataset= torchvision.datasets.ImageFolder(root= test_dataset_path, transform= test_transforms)

  train_loader= torch.utils.data.DataLoader(train_dataset, batch_size= 32, shuffle= True)
  test_loader= torch.utils.data.DataLoader(test_dataset, batch_size= 32, shuffle= True)

  return train_loader, test_loader

def show_transformed_images(dataset): 
  loaders= torch.utils.data.DataLoader(dataset, batch_size= 8, shuffle= True) 
  batch= next(iter(loaders))
  images, labels= batch 
  
  grid= torchvision.utils.make_grid(images, nrows= 4)
  plt.figure(figsize= (12,12))
  plt.imshow(np.transpose(grid, (1,2,0)))
  plt.show()
  print("labels:", labels)
  
# show_transformed_images(train_dataset)