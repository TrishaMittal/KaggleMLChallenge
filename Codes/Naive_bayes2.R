# Stumbleupon Evergreen Classification Challenge Dataset
# Methodology: Naive Bayes in R 

library(caret)
library(pROC)
library(party)
library(randomForest)
library(ggplot2)
library(e1071)
library(mlbench)
library(nnet)
library(ROCR)

set.seed(100000000)

# Loading Train Data ----------------------------------------------------------------------------------------------------
train <- read.table("../Dataset/train.tsv", header = T, sep = "\t")

# Loading Test Data -----------------------------------------------------------------------------------------------------
test <- read.table("../Dataset/test.tsv", header = T, sep = "\t")


# Setting label as factor variable to specity it as a classification problem
train$label <- as.factor(train$label) 
levels(train$label) <- list(Y1 = "1", Y2 = "0")


# Selecting only the required columns from the dataset
URLID <- test$urlid

train <- train[, c("label","compression_ratio", "spelling_errors_ratio")]
test <- test[, c("compression_ratio", "spelling_errors_ratio")]

# ---------------------------------------------------------------------------------------------------------------------
# Training a model based on the train data

model <- naiveBayes(label ~ compression_ratio + spelling_errors_ratio , train)

# Predicting the labels for the test data -------------------------------------------------------------------------------
pred <- predict(model, newdata = test)	
submit <- data.frame(URLID, pred)
names(submit)[2] <- "label"

write.csv(submit, "../Run_Output_Files/OUTPUT_NB2.csv", row.names = F)
cat("model result:\n")
print (model)	
