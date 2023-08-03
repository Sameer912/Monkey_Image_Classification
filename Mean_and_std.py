import os 
import torch 
import torchvision 
import torchvision.transforms as transforms 

# print(os.listdir('data/training/training'))


def data_loader(): 
    training_dataset_path= "data/training/training"
    training_transforms= transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    train_dataset= torchvision.datasets.ImageFolder(root= training_dataset_path, transform= training_transforms)
    train_loader= torch.utils.data.DataLoader(dataset= train_dataset, batch_size= 32, shuffle= False)
    
    return train_loader

def get_mean_and_std(loader): 
    mean= 0.
    std= 0.
    total_images= 0
    for images, _ in loader: 
        images_count_in_batch= images.size(0) #The shape of the input image is (batch_size, channels, height, width)
        images= images.view(images_count_in_batch, images.size(1), -1) #The shape of image after reshaping using images.view is (batch_size, channels, height * width)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += images_count_in_batch
        
    mean /= total_images
    std /= total_images
    return mean, std
