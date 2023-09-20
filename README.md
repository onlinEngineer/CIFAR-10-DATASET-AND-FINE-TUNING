# CIFAR-10-DATASET-AND-FINE-TUNING
Transfer Learning using CIFAR-10 data

# PART 1.
In this part, we used keras.dataset cifar-10 dataset. We split data into two parts which we will train a CNN on just the classes (airplane, automobile, bird, cat, deer). Then we will train just the last layer(s) of the network on the classes (dog, frog, horse, ship, truck). For each part, we split data into training and testing parts. We used Sequential library to create a model.

## Model
We have 949,029 parameters in total. 896 is belongs to first convolutional layer , and 9248 is belongs to second convolutional layer. Other parameters belong to first and second fully connected layer and 645 of them belongs to output(softmax) layer. We have 5 classes in this model. Normally we have 10 but for this case we will train only 5 of them which are airplane, automobile, bird, cat, deer and 2 x 2 max pooling layer with strides 2 x 2. All layers have relu activation.

![image](https://github.com/onlinEngineer/CIFAR-10-DATASET-AND-FINE-TUNING/assets/70773825/acaaab3f-e950-4acb-85eb-e0754ab65390)

### Epochs
After 10 epochs with a 64-batch size, our training loss reduced from 0.9643 to 0.2308, accuracy increased from 0.6202 to 0.9198. Also, our test loss (val_loss) reduced from 0.7029 to 0.4941, test accuracy increased from 0.7332 to 0.8614. Also, the test score is 0.4941. That’s not a good model since we have overfitting problem because of inconsistency on the val_accuracy. Additionally, the training model of this model took more than 9 and a half minutes.

![image](https://github.com/onlinEngineer/CIFAR-10-DATASET-AND-FINE-TUNING/assets/70773825/873147f0-ceb7-4f3d-9fef-080cb19f5150)

### Outputs of Initial Model

![image](https://github.com/onlinEngineer/CIFAR-10-DATASET-AND-FINE-TUNING/assets/70773825/6e991442-b63c-4ab7-afc9-ff881286daa6)


## PART-2 FINE-TUNING
## FIRST FINE-TUNING
We freeze all layers except the last layer which is output layer. Also, in this part, we will use (dog, frog, horse, ship, truck) second part of data.

![image](https://github.com/onlinEngineer/CIFAR-10-DATASET-AND-FINE-TUNING/assets/70773825/e64b8a8a-4040-4830-a138-8ee2a12c9c0e)

### Model
In this case, we have 645 trainable params, 948,384 non-trainable params.

![image](https://github.com/onlinEngineer/CIFAR-10-DATASET-AND-FINE-TUNING/assets/70773825/3027b99f-2dba-48a6-a832-2dd0eb9b9544)

### Epochs
After 10 epochs with a 64-batch size by freezing the all layer but output layer, our training loss reduced from 0.9894 to 0.2329, accuracy increased from 0.6148 to 0.9196. Also, our test loss (val_loss) reduced from 0.10493 to 0.3806, test accuracy increased from 0.5684 to 0.8762. Also, the test score is 0.3806. Seems like we solve the overfitting problem even if accuracy less than the initial model. Also, this model took 37 second which is less than initial model roughly 9 minutes.

![image](https://github.com/onlinEngineer/CIFAR-10-DATASET-AND-FINE-TUNING/assets/70773825/9c023b93-0ec4-471c-ba18-cbd21cc35bfb)

### Outputs of First Fine-Tuning Model
![image](https://github.com/onlinEngineer/CIFAR-10-DATASET-AND-FINE-TUNING/assets/70773825/044b599d-c220-486e-acc2-cfa3be891e1d)

## SECOND FINE-TUNING
We freeze all layers except the fully connected layers and last layer.
![image](https://github.com/onlinEngineer/CIFAR-10-DATASET-AND-FINE-TUNING/assets/70773825/df8e391b-e898-407c-a1f6-0408e9f92be6)

### Model
In this case, we have 938,885 trainable params, 10,144 non-trainable params.

![image](https://github.com/onlinEngineer/CIFAR-10-DATASET-AND-FINE-TUNING/assets/70773825/81ee971e-fbfc-4ca5-be24-a9db964441a4)

### Epochs
After 10 epochs with a 64-batch size by freezing all layers except the fully connected layers and last layer, our training loss reduced from 0.8779 to 0.2800, accuracy increased from 0.6579 to 0.9030. Also, our test loss (val_loss) increased from 0.7122 to 0.3806, test accuracy increased from 0.7292 to 0.7826. Also, the test score is 0.7665. Seems like we have huge overfitting problem. This model took 53 second which is less than initial model roughly 8 times.

![image](https://github.com/onlinEngineer/CIFAR-10-DATASET-AND-FINE-TUNING/assets/70773825/486e2bb1-8366-40e7-add3-9c1d7cbb2c77)

### Outputs of Second Fine-Tuning Model

![image](https://github.com/onlinEngineer/CIFAR-10-DATASET-AND-FINE-TUNING/assets/70773825/7e6a6299-0972-4497-8012-4bb69b49fb56)


# QUESTIONS
1. How many trainable parameters are there in each case?
- For the first fine-tuning, we freeze the last layer (output layer). Thus, we have 645 trainable parameters and 948,384 non-trainable parameters in total. For the second case, we trained all fully connected layers and output layer with dropouts. In this case, we have 938,885 trainable parameters and 10,144 non-trainable parameters. For both cases, the total parameters do not change.
2. Which fine-tuning performs better in terms of classification accuracy and why?
- Once we compare two fine-tuning, the second fine-tuning is performing better performance because of high accuracy than first fine-tuning. While the second fine-tuning accuracy is 0.78 and test score 0.76, the first fine-tuning accuracy is 0.61 and test score 0.97 which is lower than first fine-tuning as we said before. The second model is performing better performance since our model is trained by for the first 5 outputs, in the fine-tuning we applied on different outputs. So, in the first fine-tuning we just used softmax layer to predict the outputs. So, that is not a good idea for different outputs. However, in the second fine-tuning, we used two fully connected layer. Thus, we expanded our predict mechanism. Therefore, we got better result.
3. Why is fine-tuning much faster than the initial training of the network?
- Fine-tuning should be faster than the initial training as we skip the lots of time-consuming steps such as convolutional layer or max pooling layers. Besides, in the initial training of networks, we train our data for prediction but the other case, most of time we are using pre-trained data to prediction. So that, fine-tuning much faster than the initial training of the network.
