# install.packages("e1071")
# install.packages("caret")

library(e1071)
library(caret)

# Load Data

file <- "C:\\Document\\Code\\Dropbox\\R-INFO7390\\Mid Term Project\\train.csv"
fix_file <- "C:\\Document\\Code\\Dropbox\\R-INFO7390\\Mid Term Project\\fix_train.csv"
input_data <- read.csv(file)
summary(input_data)
# table(input_data$Product_Info_4)
# summary(input_data$Product_Info_4)

# list <- which(rowSums(is.na(input_data)) > 0)

# Clean Data
fix_data <- input_data


# Fix Employment_Info_1 use mean (Continues variable)
fix_data[which(is.na(fix_data$Employment_Info_1)), "Employment_Info_1"] <- as.numeric(mean(input_data$Employment_Info_1,na.rm = TRUE))
fix_data$Employment_Info_1 <- as.numeric(fix_data$Employment_Info_1)

# Fix Employment_Info_4 use median (Continues variable)
fix_data[which(is.na(fix_data$Employment_Info_4)), "Employment_Info_4"] <- as.numeric(median(input_data$Employment_Info_4,na.rm = TRUE))
fix_data$Employment_Info_4 <- as.numeric(fix_data$Employment_Info_4)

# Fix Employment_Info_6 use mean (Continues variable)
fix_data[which(is.na(fix_data$Employment_Info_6)), "Employment_Info_6"] <- as.numeric(mean(input_data$Employment_Info_6,na.rm = TRUE))
fix_data$Employment_Info_6 <- as.numeric(fix_data$Employment_Info_6)

# Fix Insurance_History_5 use mean (Continues variable)
fix_data[which(is.na(fix_data$Insurance_History_5)), "Insurance_History_5"] <- as.numeric(mean(input_data$Insurance_History_5,na.rm = TRUE))
fix_data$Employment_Info_5 <- as.numeric(fix_data$Insurance_History_5)

# Fix Family_Hist_2 use mean (Continues variable)
fix_data[which(is.na(fix_data$Family_Hist_2)), "Family_Hist_2"] <- as.numeric(mean(input_data$Family_Hist_2,na.rm = TRUE))
fix_data$Family_Hist_2 <- as.numeric(fix_data$Family_Hist_2)

# Fix Family_Hist_3 use mean (Continues variable)
fix_data[which(is.na(fix_data$Family_Hist_3)), "Family_Hist_3"] <- as.numeric(mean(input_data$Family_Hist_3,na.rm = TRUE))
fix_data$Family_Hist_3 <- as.numeric(fix_data$Family_Hist_3)

# Fix Family_Hist_4 use mean (Continues variable)
fix_data[which(is.na(fix_data$Family_Hist_4)), "Family_Hist_4"] <- as.numeric(mean(input_data$Family_Hist_4,na.rm = TRUE))
fix_data$Family_Hist_4 <- as.numeric(fix_data$Family_Hist_4)

# Fix Family_Hist_5 use mean (Continues variable)
fix_data[which(is.na(fix_data$Family_Hist_5)), "Family_Hist_5"] <- as.numeric(mean(input_data$Family_Hist_5,na.rm = TRUE))
fix_data$Family_Hist_5 <- as.numeric(fix_data$Family_Hist_5)

# Fix Medical_History_1 use mean (Discrete variable)
fix_data[which(is.na(fix_data$Medical_History_1)), "Medical_History_1"] <- as.integer(8)
fix_data$Medical_History_1 <- as.integer(fix_data$Medical_History_1)

# Fix Medical_History_10 use mean (Discrete variable)
fix_data[which(is.na(fix_data$Medical_History_10)), "Medical_History_10"] <- as.integer(141)
fix_data$Medical_History_10 <- as.integer(fix_data$Medical_History_10)

# Fix Medical_History_15 use mean (Discrete variable)
fix_data[which(is.na(fix_data$Medical_History_15)), "Medical_History_15"] <- as.integer(124)
fix_data$Medical_History_15 <- as.integer(fix_data$Medical_History_15)

# Fix Medical_History_24 use mean (Discrete variable)
fix_data[which(is.na(fix_data$Medical_History_24)), "Medical_History_24"] <- as.integer(51)
fix_data$Medical_History_24 <- as.integer(fix_data$Medical_History_24)

# Fix Medical_History_32 use mean (Discrete variable)
fix_data[which(is.na(fix_data$Medical_History_32)), "Medical_History_32"] <- as.integer(12)
fix_data$Medical_History_32 <- as.integer(fix_data$Medical_History_32)

# Select some data as train

set.seed(100)
sample_index <- sample(nrow(fix_data), 0.6*nrow(fix_data))

train_data <- fix_data[sample_index, ]
test_data <- fix_data[-sample_index, ]

# Find better parameter

fit1 <- tune(svm, Response~.,kernel ="radial", data= train_data, ranges=list(gamma=10^(-2:2), cost=10^(-2:2)))

summary(fit1)

fit2 <- tune(svm, Response~.,kernel ="linear", data= train_data, ranges=list(cost=10^(-2:2)))

summary(fit2)

# Use the better parameter to build model

svm_model = svm(Response ~ ., kernel = "radial", cost =1 ,gamma=0.01, data = train_data, scale = F)

predictions <-  predict(svm_model, test_data[-128])

predictions <- as.integer(predictions)

# table(predictions, test_data[,128])
confusionMatrix(predictions, test_data[,128])
