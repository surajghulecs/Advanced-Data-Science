
#PRUDENTIAL LIFE INSURANCE:
#Decision Tree
#First time set up:
options('java.home')
options("java.home"="/Library/Java/JavaVirtualMachines/jdk1.8.0_101.jdk/Contents/Home/jre")
Sys.setenv("LD_LIBRARY_PATH"='$JAVA_HOME/jre/lib/server')
Sys.setenv(JAVA_HOME='/Library/Java/JavaVirtualMachines/jdk1.8.0_101.jdk/Contents/Home/jre')

#Empty workspace
rm(list=ls())

#Get Working Directory
getwd()

---------
  
#STEP 1: Read, display and analyze structure of train data from CSV sheet:
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("XLConnect")
#install.packages("rJava")
#install.packages("caret")
#install.packages("rattle")
#install.packages("RColorBrewer")

library(XLConnect)
library(rJava)


#Read and display Prudential dataset:
#Selecting 80% of data as sample from total 59,380 rows of the data  
library(XLConnect)
data <- read.csv("train.csv")
View(data)

dim(data) #Display dimension of dataset [59381 rows, 128 columns]
names(data) #Display column names of dataset
str(data) #Display structure of dataset
summary(data) #Display summary (summarizes values of columns) of dataset


#STEP 2: DATA CLEANING

#Segregate columns based on the variable type:
#Categorical columns:
catData <- c(paste("Product_Info_", c(1:3,5:7), sep=""), paste("Employment_Info_", c(2,3,5), sep=""),
               paste("InsuredInfo_", 1:7, sep=""), paste("Insurance_History_", c(1:4,7:9), sep=""), 
               "Family_Hist_1", paste("Medical_History_", c(2:9, 11:14, 16:23, 25:31, 33:41), sep=""),"Response")

#Discrete columns:
discData <- c("Medical_History_1", "Medical_History_10", "Medical_History_15", "Medical_History_24", "Medical_History_32", 
                paste("Medical_Keyword_", 1:48, sep=""))


#Continuous columns:
contData <- c("Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4", 
                "Employment_Info_6", "Insurance_History_5", "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", 
                "Family_Hist_5")

categoricalData <- data[, catData]
categoricalData
summary(categoricalData)

continuousData <- data[, contData]
continuousData
summary(continuousData)

discreteData <- data[, discData]
discreteData
summary(discreteData)

#By looking at the summary of the data:
#Remove columns with missing values (NA)
discreteData <-discreteData[, -c(2,3,4,5,6)] #All NA's omitting
summary(discreteData)

continuousData <-continuousData[, -c(9, 10, 11, 13)] #Columns with 20,000+ Null values are eliminated
summary(continuousData)


#Remove columns with outlier data
categoricalData <-categoricalData[, -c(2,3,7,12,25)]
summary(categoricalData)

#Replace missing values with the column's mean for Continuous variables
#Column: Employment_Info_1
continuousData$Employment_Info_1[is.na(continuousData$Employment_Info_1)] <- mean(continuousData$Employment_Info_1, na.rm = TRUE)
#Column: Employment_Info_4
continuousData$Employment_Info_4[is.na(continuousData$Employment_Info_4)] <- mean(continuousData$Employment_Info_4, na.rm = TRUE)
#Column: Employment_Info_6
continuousData$Employment_Info_6[is.na(continuousData$Employment_Info_6)] <- mean(continuousData$Employment_Info_6, na.rm = TRUE)
#Column: Insurance_History_5
continuousData$Insurance_History_5[is.na(continuousData$Insurance_History_5)] <- mean(continuousData$Insurance_History_5, na.rm = TRUE)
#Column: Family_Hist_2
continuousData$Family_Hist_2[is.na(continuousData$Family_Hist_2)] <- mean(continuousData$Family_Hist_2, na.rm = TRUE)
#Column: Family_Hist_4
continuousData$Family_Hist_4[is.na(continuousData$Family_Hist_4)] <- mean(continuousData$Family_Hist_4, na.rm = TRUE)

summary(continuousData)

#Replace missing values with the column's mean for Discrete variables
#Column: Medical_Hist_1
discreteData$Medical_History_1[is.na(discreteData$Medical_History_1)] <- mean(discreteData$Medical_History_1, na.rm = TRUE)

summary(discreteData)
----------

#install.packages("ade4") #PCA Computation
library(ade4)
df_cat <-acm.disjonctif(categoricalData[c(1:55)])
summary(df_cat)
dim(df_cat)

response <- categoricalData[, c(56)]
summary(response)

df_catresp <- cbind(df_cat,response)
dim(df_catresp)

finalData <- cbind(continuousData,discreteData,df_catresp)
finalData <- finalData[,order(colnames(finalData))] #Reordering the dataset in alphabetical order
dim(finalData)
names(finalData)

write.csv(finalData, file = "Desktop/finalData.csv")

df_catresp$response

reg = lm(finalData$response~., data = finalData)
summary(reg)
#plot(reg)

prud1 <- finalData[, c(1,2,3,4,5,10,11,13,15,16,19,22,25,28,29,31,32,33,35,38,39,40,42,44,46,47,50,51,52,54,56,59,62,64,66,68,71,72,77,81,83,85,87,89,92,99,101,111,113,121,131,134,136,139,146,160,162,171,177,181,189,198,200,202,205,208)]
prud_res=cbind(prud1,response)
reg1 = lm(prud_res$response~., data = prud_res)
summary(reg1)


prud2 <- prud1[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,24,25,26,27,28,29,33,35,36,37,41,42,43,44,45,46,47,49,50,53,54,55,57,58,59,62,64,65,66)]
prud_res1=cbind(prud2,response)
reg2 = lm(prud_res1$response~., data = prud_res1)
summary(reg2)
 

#STEP 3: MODEL BUILDING using training dataset
#predict (test dataset)
#cross-verification using plot/confusion matrix

#Dataset to be used for training and testing:
sample <- sample.int(n = nrow(finalData), size = floor(.80*nrow(finalData)), replace = F)
finalTrain <- finalData[sample,]
dim(finalTrain)
names(finalTrain)
finalTest <- finalData[-sample,]
dim(finalTest)


#XGBoost:
#finalTrain: will be used to build the model
#finalTest: will be used to assess the quality of our model

#Find out the structure of train data:
str(finalTrain)

#Fitting XGBoost to the Training Set
#install.packages("xgboost")
library(xgboost)

#binary:logistic: train a binary classification model
#max_depth = 2: the trees won't be deep, because our case is very simple ;
#nthread = 2: the number of cpu threads we are going to use;

bstSparse <- xgboost(data = finalTrain, label = finalTrain$response, max_depth = 2, eta = 1, nthread = 2, nrounds = 10, objective = "binary:logistic")
#or we can put the dataset in a dense matrix:
classifier <- xgboost(data = as.matrix(finalTrain[-209]), label = finalTrain$response, nrounds=10) #nround will run 10 iterations
#class(data)


#xgb.DMatrix:
#a way to group 
dtrain <- xgb.DMatrix(data = train$data, label = train$label)
bstDMatrix <- xgboost(data = dtrain, max_depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")

#verbose option:
#simplest way to see the training progress is to set the verbose option
#XGBoost has several features to help you to view how the learning progress internally. 
#The purpose is to help set the best parameters, which is the key of your model quality.

#Verbose = 0, no message
bst <- xgboost(data = dtrain, max_depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic", verbose = 0)
#Verbose = 1, printing evaluation metric
bst <- xgboost(data = dtrain, max_depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic", verbose = 1)
#Verbose = 2, also print information about tree
bst <- xgboost(data = dtrain, max_depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic", verbose = 2)

#The purpose of the model we have built is to classify new data. 
pred <- predict(finalTrain, finalTrain$data)
# size of the prediction vector
print(length(pred))
head(pred)


#model1<-lm(Response~ Product_Info_4+Ins_Age+Ht+Wt+InsuredInfo_2+InsuredInfo_5+InsuredInfo_6+Medical_History_4+Insurance_History_2_1+
# Medical_History_39_1+Medical_History_17.2+Medical_History_20.1+
#Medical_History_30_2+Medical_History_23_1+
#Medical_History_40_1+BMI,finalData)


#STEP 4: Submit data (Write in CSV)
result <- test[,c("Id", "Response")]
write.csv(result, "result.csv", row.names=FALSE)

#Reading the data into R
library(XLConnect)
finalData <- read.csv("finalData.csv")
View(finalData)
summary(finalData)
finalData$X <- NULL
View(finalData)
str(finalData)
 table(finalData$Product_Info_7.3)
 finalData$Product_Info_7.2<-NULL
 finalData$Product_Info_5.3<-NULL
 finalData$Product_Info_5.2<-NULL
 finalData$Medical_Keyword_9<-NULL
 finalData$Medical_Keyword_8<-NULL
 finalData$Medical_Keyword_7<-NULL
 finalData$Medical_Keyword_6<-NULL
 finalData$Medical_Keyword_5<-NULL
 finalData$Medical_Keyword_46<-NULL
 finalData$Medical_Keyword_45<-NULL
 finalData$Medical_Keyword_44<-NULL
 finalData$Medical_Keyword_43<-NULL
 finalData$Medical_Keyword_41<-NULL
 finalData$Medical_Keyword_4<-NULL
 finalData$Medical_Keyword_39<-NULL
 finalData$Medical_Keyword_38<-NULL
 finalData$Medical_Keyword_36<-NULL
 finalData$Medical_Keyword_35<-NULL
 finalData$Medical_Keyword_31<-NULL
 finalData$Medical_Keyword_29<-NULL
 finalData$Medical_Keyword_28<-NULL
 finalData$Medical_Keyword_27<-NULL
 finalData$Medical_Keyword_26<-NULL
 finalData$Medical_Keyword_21<-NULL
 finalData$Medical_Keyword_2<-NULL
 finalData$Medical_Keyword_19<-NULL
 finalData$Medical_Keyword_18<-NULL
 finalData$Medical_Keyword_17<-NULL
 finalData$Medical_Keyword_16<-NULL
 finalData$Medical_Keyword_14<-NULL
 finalData$Medical_Keyword_13<-NULL
 finalData$Medical_Keyword_12<-NULL
 finalData$Medical_History_9.3<-NULL
 finalData$Medical_History_7.1<-NULL
 finalData$Medical_History_6.2<-NULL
 finalData$Medical_History_5.3<-NULL
 finalData$Medical_History_5.2<-NULL
 finalData$Medical_History_5.1<-NULL
 finalData$Medical_History_41.2<-NULL
 finalData$Medical_History_40.3<-NULL
 finalData$Medical_History_40.2<-NULL
 finalData$Medical_History_40.1<-NULL
 finalData$Medical_History_39.2<-NULL
 finalData$Medical_History_38.2<-NULL
 finalData$Medical_History_38.1<-NULL
 finalData$Medical_History_37.3<-NULL
 finalData$Medical_History_36.1<-NULL
 finalData$Medical_History_35.3<-NULL
 finalData$Medical_History_35.2<-NULL
 finalData$Medical_History_35.1<-NULL
 finalData$Medical_History_34.2<-NULL
 finalData$Medical_History_31.3<-NULL
 finalData$Medical_History_31.2<-NULL
 finalData$Medical_History_31.1<-NULL
 finalData$Medical_History_30.1<-NULL
 finalData$Medical_History_3.1<-NULL
 finalData$Insurance_History_3.2<-NULL
 finalData$Insurance_History_2.2<-NULL
 finalData$Insurance_History_9.1<-NULL
 finalData$InsuredInfo_1.3<-NULL
 finalData$InsuredInfo_2.2<-NULL
 finalData$InsuredInfo_2.3<-NULL
 finalData$InsuredInfo_5.1<-NULL
 finalData$InsuredInfo_5.3<-NULL
 finalData$InsuredInfo_7.3<-NULL
 finalData$InsuredInfo_7.1<-NULL
 finalData$Medical_History_11.1<-NULL
 finalData$Medical_History_11.2<-NULL
 finalData$Medical_History_11.3<-NULL
 finalData$Medical_History_12.<-NULL
 finalData$Medical_History_13.2<-NULL
 finalData$Medical_History_14.1<-NULL
 finalData$Medical_History_16.2<-NULL
 finalData$Medical_History_17.1<-NULL
 finalData$Medical_History_18.3<-NULL
 finalData$Medical_History_19.3<-NULL
 finalData$Medical_History_20.1<-NULL
 finalData$Medical_History_20.2<-NULL
 finalData$Medical_History_20.3<-NULL
 finalData$Medical_History_21.3<-NULL
 finalData$Medical_History_23.2<-NULL
 finalData$Medical_History_25.3<-NULL
 finalData$Medical_History_26.1<-NULL
 finalData$Medical_History_27.1<-NULL
 finalData$Medical_History_27.2<-NULL
 finalData$Medical_History_27.3<-NULL
 finalData$Medical_History_28.3<-NULL
 finalData$Medical_History_29.2<-NULL
 
#divide the data into training data and testing data
set.seed(3)
id<-sample(2,nrow(finalData), prob = c(0.7,0.3),replace=TRUE)
finaldata_train <-finalData[id==1,]
finaldata_test <-finalData[id==2,]

#storing the value of respone of test data
response_test <- finaldata_test$response
View(response_test)

#model1<-lm(Response~ Product_Info_4+Ins_Age+Ht+Wt+InsuredInfo_2+InsuredInfo_5+InsuredInfo_6+Medical_History_4+Insurance_History_2_1+
# Medical_History_39_1+Medical_History_17.2+Medical_History_20.1+
#Medical_History_30_2+Medical_History_23_1+
#Medical_History_40_1+BMI,finalData)

df<-data.frame(finalData$Ins_Age,finalData$Ht,finalData$Wt,finalData$InsuredInfo_2.2,finalData$InsuredInfo_6.1
               ,finalData$InsuredInfo_5.1,finalData$Medical_History_4.1,finalData$Insurance_History_2.1,finalData$Medical_History_39.1,
                 finalData$Medical_History_17.2,finalData$Medical_History_20.1,finalData$Medical_History_30.2,finalData$Medical_History_23.1
               ,finalData$Medical_History_40.1,finalData$BMI)
summary(df)
View(df)
#setting the response column of the testing data to null
finaldata_test$response <- NULL
summary(finaldata_test)
View(finaldata_test)
colnames(finaldata_test)

----------------------
summary(finaldata_train)

#Building the decision tree
library(rpart)
View()
finaldata_model<-rpart(finaldata_train$response~ finaldata_train$Ins_Age+finaldata_train$Ht+finaldata_train$Wt+finaldata_train$InsuredInfo_2.2+finaldata_train$InsuredInfo_6.1
                       +finaldata_train$InsuredInfo_5.1+finaldata_train$Medical_History_4.1+finaldata_train$Insurance_History_2.1+finaldata_train$Medical_History_39.1+
                         finaldata_train$Medical_History_17.2+finaldata_train$Medical_History_20.1+finaldata_train$Medical_History_30.2+finaldata_train$Medical_History_23.1
                       +finaldata_train$Medical_History_40.1+finaldata_train$BMI, data=finaldata_train )

finaldata_model<-rpart(finaldata_train$response~., data=finaldata_train )
View(finaldata_model)
finaldata_model
#here we are using all columns for model
#plot(finaldata_model, margin=.1)
text(finaldata_model,use.n=TRUE, pretty=TRUE, cex=.8)

library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(finaldata_model)

# prune the tree by using cp value
printcp(finaldata_model)
plotcp(finaldata_model)
pfit<- prune(finaldata_model, cp=.0100)

# plot the pruned tree 
#plot(pfit, uniform=TRUE, main="Pruned Classification Tree for Prudential Life Insurance")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)
post(pfit, file = "ptree1.ps", title = "Pruned Classification Tree for Prudential Life Insurance")

#predicting the results
predict_finaldatamodel<-predict(finaldata_model, finaldata_test)
predict_finaldatamodel

#now we need to compare it with actual value
table(predict_finaldatamodel, response_test)

#Calculating the accuracy for the predicted model
library(forecast)
accuracy(predict_finaldatamodel,response_test)
