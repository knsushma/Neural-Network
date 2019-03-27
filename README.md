# Neural-Network

The goals of this assignment is to learn about stochastic gradient descent and backpropagation by:

 implementing logistic regression and training and testing it on several data sets.\
 implementing a neural network with one hidden layer and training and testing it on several data sets.\
 evaluating the effect of the amount of training on the accuracy of the learned models.


## Environment Setup:
```
Follow steps mentioned in python-setup-on-remote/python-setup.pdf
```

Use below commands to run the required code.

### Logistic Regression
```
./logistic 0.01 10 ../Resources/banknote_train.json ../Resources/banknote_test.json 
./logistic 0.01 10 ../Resources/magic_train.json ../Resources/magic_test.json 
./logistic 0.05 20 ../Resources/heart_train.json ../Resources/heart_test.json 
```

### Neural Network
```
./nnet 0.01 5 10 ../Resources/banknote_train.json ./Resources/banknote_test.json 
./nnet 0.01 10 5 ../Resources/magic_train.json ../Resources/magic_test.json 
./nnet 0.05 7 20 ../Resources/heart_train.json ../Resources/heart_test.json 
```
