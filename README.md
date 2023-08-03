## Monkey classification: 
#### This is a tutorial for monkey image classication in PyTorch which uses residual network (ResNet) which has already been trained on a large set of image dataset. This tutorial is a example of transfer learning where we take the pretrained model, and either retrain it using our own dataset, or just change the fully connected layer with the features pretrained model and num of classes of we need to predict. 

#### Since I am a beginner in this field, I have not gone through hyperparameter tuning process, so the performance of the model is quite poor even when using the weights from the pretrained model. If we want to increase the performance of the model, we can just set the ```pretrain= True``` but it would be like cheating because the model will use pretrained weights, which definitely gives better performance. 

#### In order to evaluate the actual performance of the model on our dataset, it is better to use ```pretrain= False```. 