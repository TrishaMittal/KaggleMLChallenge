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

train <- train[, c("label","alchemy_category", "alchemy_category_score", "news_front_page")]
test <- test[, c("alchemy_category", "alchemy_category_score", "news_front_page")]


# Pre Processing for train data --------------------------------------------------------------------------------------

# A character object is used to represent string values in R. Declaring it as character so that we can replace the
# missing values
train$news_front_page  <- as.character(train$news_front_page )
train$alchemy_category_score = as.character(train$alchemy_category_score)	
train$alchemy_category = as.character(train$alchemy_category)

# news_front_page takes value either 0 or 1. For the missing value we are replacing it by 0.5
train$news_front_page [train$news_front_page == "?"] = "0.5"
train$news_front_page  <- as.numeric(train$news_front_page ) # Making it a numeric variable from character

# For alchemy_category with unknown as the value the respect alchemy_category_score is 0.400001. So replacing all the
# missing values by 0.400001
train$alchemy_category_score [train$alchemy_category_score == "?"] <- "0.400001"	
train$alchemy_category_score <- as.numeric(train$alchemy_category_score) # Making it a numeric variable from character	


train$alchemy_category[ train$alchemy_category == "?" ] <- "unknown"

# The as.numeric function directly assigns random numbers to the levels present in the attribute
train$alchemy_category = as.factor(train$alchemy_category)
train$alchemy_category <- as.numeric(train$alchemy_category)  	

# Pre Processing for test data ---------------------------------------------------------------------------------------
# A character object is used to represent string values in R. Declaring it as character so that we can replace the
# missing values
test$news_front_page  <- as.character(test$news_front_page )
test$alchemy_category_score = as.character(test$alchemy_category_score)	
test$alchemy_category = as.character(test$alchemy_category)

# news_front_page takes value either 0 or 1. For the missing value we are replacing it by 0.5
test$news_front_page [test$news_front_page == "?"] = "0.5"
test$news_front_page  <- as.numeric(test$news_front_page ) # Making it a numeric variable from character

# For alchemy_category with unknown as the value the respect alchemy_category_score is 0.400001. So replacing all the
# missing values by 0.400001
test$alchemy_category_score [test$alchemy_category_score == "?"] <- "0.400001"	
test$alchemy_category_score <- as.numeric(test$alchemy_category_score) # Making it a numeric variable from character	

test$alchemy_category[ test$alchemy_category == "?" ] <- "unknown"

# The as.numeric function directly assigns random numbers to the levels present in the attribute
test$alchemy_category <- as.factor(test$alchemy_category)  
test$alchemy_category <- as.numeric(test$alchemy_category)  

# ---------------------------------------------------------------------------------------------------------------------
# Training a model based on the train data

model <- naiveBayes(label ~ alchemy_category_score + alchemy_category + news_front_page , train)

# Predicting the labels for the test data -------------------------------------------------------------------------------
pred <- predict(model, newdata = test)	
submit <- data.frame(URLID, pred)
names(submit)[2] <- "label"

write.csv(submit, "../Run_Output_Files/OUTPUT_NB1.csv", row.names = F)
cat("model result:\n")
print (model)	
