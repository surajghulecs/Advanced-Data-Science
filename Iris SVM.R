# http://rischanlab.github.io/SVM.html

library("e1071")
library("caret")

# Using Iris data

head(iris,5)

##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
## 1          5.1         3.5          1.4         0.2  setosa
## 2          4.9         3.0          1.4         0.2  setosa
## 3          4.7         3.2          1.3         0.2  setosa
## 4          4.6         3.1          1.5         0.2  setosa
## 5          5.0         3.6          1.4         0.2  setosa

# Attach the Data

attach(iris)

# Divide Iris data to x (containt the all features) and y only the classes

x <- subset(iris, select=-Species)
y <- Species

# Create SVM Model and show summary

svm_model <- svm(Species ~ .,kernel = 'radial', data=iris, cost=100)
summary(svm_model)
#varImp(svm_model, scale=FALSE)

plot(svm_model, iris,
     Petal.Width ~ Petal.Length,
     slice = list(Sepal.Width = 3, Sepal.Length = 4))

pred <- predict(svm_model, head(iris), decision.values = TRUE)
pred

## 
## Call:
## svm(formula = Species ~ ., data = iris)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  radial 
##        cost:  1 
##       gamma:  0.25 
## 
## Number of Support Vectors:  51
## 
##  ( 8 22 21 )
## 
## 
## Number of Classes:  3 
## 
## Levels: 
##  setosa versicolor virginica

#Create SVM Model and show summary

svm_model1 <- svm(x,y)
summary(svm_model1)

## 
## Call:
## svm.default(x = x, y = y)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  radial 
##        cost:  1 
##       gamma:  0.25 
## 
## Number of Support Vectors:  51
## 
##  ( 8 22 21 )
## 
## 
## Number of Classes:  3 
## 
## Levels: 
##  setosa versicolor virginica

# Run Prediction and you can measuring the execution time in R

pred <- predict(svm_model1,x)
system.time(pred <- predict(svm_model1,x))

##    user  system elapsed 
##       0       0       0

# See the confusion matrix result of prediction, using command table to compare the result of SVM
# prediction and the class data in y variable.

table(pred,y)

##             y
## pred         setosa versicolor virginica
##   setosa         50          0         0
##   versicolor      0         48         2
##   virginica       0          2        48

# TUning SVM to find the best cost and gamma ..

svm_tune <- tune(svm, train.x=x, train.y=y, 
              kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))

print(svm_tune)

## 
## Parameter tuning of 'svm':
## 
## - sampling method: 10-fold cross validation 
## 
## - best parameters:
##  cost gamma
##     1   0.5
## 
## - best performance: 0.05333

# After you find the best cost and gamma, you can create svm model again and try to run again

svm_model_after_tune <- svm(Species ~ ., data=iris, kernel="radial", cost=100, gamma=0.5)
summary(svm_model_after_tune)

## 
## Call:
## svm(formula = Species ~ ., data = iris, kernel = "radial", cost = 1, 
##     gamma = 0.5)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  radial 
##        cost:  1 
##       gamma:  0.5 
## 
## Number of Support Vectors:  59
## 
##  ( 11 23 25 )
## 
## 
## Number of Classes:  3 
## 
## Levels: 
##  setosa versicolor virginica

# Run Prediction again with new model

pred <- predict(svm_model_after_tune,x)
system.time(predict(svm_model_after_tune,x))

##    user  system elapsed 
##       0       0       0

# See the confusion matrix result of prediction, using command table to compare the result of SVM
# prediction and the class data in y variable.

table(pred,y)

##             y
## pred         setosa versicolor virginica
##   setosa         50          0         0
##   versicolor      0         48         2
##   virginica       0          2        48

