#INFO7390 34575 Advances Data Sci/Architecture SEC 03 - Spring 2018

#STEP 1: Installing the required packages and loading our data sets.
#install.packages("XLConnectJars")
#install.packages("XLConnect")
#install.packages("rJava")
#install.packages("tidyverse")
#install.packages("caret")
library(XLConnectJars)
library(XLConnect)
library(rJava)
library(caret)

#Read and display Prudential dataset:
#Selecting 80% of data as sample from total 59,380 rows of the data  
library(XLConnect)
data <- read.csv(file = "C:/Users/Suraj/Desktop/Mid Term ADS/train.csv", header = TRUE, sep=",")

dim(data) #Display dimension of dataset [59381 rows, 128 columns]
names(data) #Display column names of dataset
str(data) #Display structure of dataset
summary(data) #Display summary (summarizes values of columns) of dataset


#STEP 2: CLEANING OUR DATASET

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
#PCA Computation
#install.packages("ade4")
library(ade4)
#transformation
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

write.csv(finalData, file = "C:/Users/Suraj/Desktop/Mid Term ADS/finalData.csv")

df_catresp$response

reg = lm(finalData$response~., data = finalData)
summary(reg)
plot(reg)

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
 na.omit(finalData)
 names(finalData)
finalTrain <- finalData[sample,]
dim(finalTrain)
names(finalTrain)
finalTest <- finalData[-sample,]
dim(finalTest)

dim(finalTest) #Display dimension of dataset [38003 rows, 211 columns]  
names(finalTest) #Display column names of dataset
str(finalTest) #Display structure of dataset
summary(finalTest) #Display summary (summarizes values of columns) of dataset

dim(finalTrain) #Display dimension of dataset [9501 rows, 211 columns]
names(finalTrain) #Display column names of dataset
str(finalTrain) #Display structure of dataset
summary(finalTrain) #Display summary (summarizes values of columns) of dataset
View(finalTrain)

#Step 4:
#Applying linear Regression

fit_data = lm(response~ Product_Info_4+Ins_Age+Ht+Wt+
                Medical_History_17.2+Medical_History_20.1+BMI, data=finalTrain)
dim(fit_data)
summary(fit_data)

#Plotting Linear Regression
plot(fit_data)

#removing response column from the test data
response_test <- finalTest$response
finalTest$response <- NULL
View(finalTest$response)
# predicting the response on test data for linear regression
pred <- predict(fit_data, finalTest)
pred <- round(pred)
#Plotting predicted dataset
plot(pred)


#Finding the response
mat <- table(pred,response_test)
accuracy(pred,response_test)

#sum of diagoanl values in the table div by sum of other values in the table *100
sum(diag(mat))/sum(mat) * 100
View(mat)
#including prediction column of linear Regression to the file
predictedData <- data.frame(response_test,pred)
write.csv(predictedData, file = "C:/Users/Suraj/Desktop/Mid Term ADS/Result_linearRegression.csv", row.names = FALSE)

