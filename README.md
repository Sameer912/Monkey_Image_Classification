## Monkey classification: 
#### This is a tutorial for monkey image classification in PyTorch which uses residual network (ResNet) and has already been trained on a large set of image datasets. This tutorial is an example of transfer learning where we take the pre-trained model, and either retrain it using our own dataset or just change the fully connected layer with the features pre-trained model and num of classes we need to predict. The dataset contains images of monkey of 10 different species.

#### Since I am a beginner in this field, I have not gone through the hyperparameter tuning process, since the performance of the model is quite good even without hyperparameter tuning. If we want to increase the performance of the model, we can just set the ```pretrain= True``` but it would be like cheating because the model will use pre-trained weights, which definitely gives better performance. 

#### In order to evaluate the actual performance of the model on our dataset, it is better to use ```pretrain= False```. I have trained the model using ```pretrain= False''', and still achieved state-of-the-art performance in the training set and quite a good accuracy in the validation set. This is a basic image classification tutorial and might be helpful for those, who have just started learning PyTorch like me. 
