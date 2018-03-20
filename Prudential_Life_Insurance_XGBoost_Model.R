
#PRUDENTIAL LIFE INSURANCE:

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
#install.packages("XLConnectJars")
#install.packages("XLConnect")
#install.packages("rJava")
#install.packages("tidyverse")

library(XLConnectJars)
library(XLConnect)
library(rJava)


#Read and display Prudential dataset:
library(XLConnect)
data <- read.csv(file = "C:/Users/Suraj/Desktop/train.csv", header = TRUE, sep=",")

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

#install.packages("ade4") #PCA Computation for 1-to-c transformation:
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
summary(finalData)

write.csv(finalData, file = "Desktop/finalData.csv")

df_catresp$response

reg = lm(finalData$response~., data = finalData)
summary(reg)
#plot(reg)

#Using back-elimination, we drill down the dataframe to come up with features for building the model:
prud1 <- finalData[, c(1,2,3,4,5,10,11,13,15,16,19,22,25,28,29,31,32,33,35,38,39,40,42,44,46,47,50,51,52,54,56,59,62,64,66,68,71,72,77,81,83,85,87,89,92,99,101,111,113,121,131,134,136,139,146,160,162,171,177,181,189,198,200,202,205,208)]
prud_res=cbind(prud1,response)
reg1 = lm(prud_res$response~., data = prud_res)
summary(reg1)


prud2 <- prud1[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,24,25,26,27,28,29,33,35,36,37,41,42,43,44,45,46,47,49,50,53,54,55,57,58,59,62,64,65,66)]
prud_res1=cbind(prud2,response)
reg2 = lm(prud_res1$response~., data = prud_res1)
summary(reg2)
 #removed features based on pr value (>0.05 is removed)

#STEP 3: MODEL BUILDING using training dataset
#predict (test dataset)
#cross-verification using plot/confusion matrix

#Dataset to be used for training and testing:
sample <- sample.int(n = nrow(finalData), size = floor(.80*nrow(finalData)), replace = F)
finalTrain <- finalData[sample,]
dim(finalTrain)
names(finalTrain)
finalTest <- finalData[-sample,]
response_test <- finalTest$response
finalTest$response <- NULL #removing the column from finalTest dataset
View(finalTest)
dim(finalTest)

finaldata_model<-rpart(finaldata_train$response~ finaldata_train$Ins_Age+finalData_train$Ht+finalData_train$Wt+finalData_train$InsuredInfo_2.2+finalData_train$InsuredInfo_6.1
                                                                   +finalData_train_train$InsuredInfo_5.1+finalData_train$Medical_History_4.1+finalData_train$Insurance_History_2.1+finalData_train$Medical_History_39.1+
                                                                     finalData_train$Medical_History_17.2+finalData_train$Medical_History_20.1+finalData_train$Medical_History_30.2+finalData_train$Medical_History_23.1
                                                                   +finalData_train$Medical_History_40.1+finalData_train$BMI, data=finaldata_train )
View(finaldata_model)

#XGBoost:
#finalTrain: will be used to build the model
#finalTest: will be used to assess the quality of our model

#Fitting XGBoost to the Training Set
#install.packages("xgboost")
library(xgboost)

#XGBoost only supports numerical dense matrix input

classifier <- xgboost(data = as.matrix(finalTrain[-207]), label = finalTrain$response, booster = "gblinear", objective = "reg:linear", 
                      lambda = 5, alpha = 5, max_depth=2, nthread = 2, nrounds=50, verbose = 2)

#Perform prediction:  
pred <- predict(classifier, as.matrix(finalTest))
print(length(pred)) # size of the prediction vector
head(pred)

#Calculate accuracy:
#install.packages('forecast')
library(forecast)
accuracy(as.numeric(pred),as.numeric(response_test))
  
#Result

#STEP 4: Submit data (Write in CSV)
result <- test[,c("Id", "Response")]
write.csv(result, "result.csv", row.names=FALSE)

