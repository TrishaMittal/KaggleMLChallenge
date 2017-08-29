# Stumbleupon Evergreen Classification Challenge Dataset

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
train <- train[, c("label","alchemy_category", "alchemy_category_score", "news_front_page", "compression_ratio", "spelling_errors_ratio")]
URLID <- test$urlid
test <- test[, c("alchemy_category", "alchemy_category_score", "news_front_page", "compression_ratio", "spelling_errors_ratio")]


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
NO_OF_TREES <- 500
NO_OF_VARIABLES_TO_TRY <- 8
NO_OF_CV <- 10
REPEAT <- 3

rfGrid <- expand.grid(.mtry = c(1:8) )

fitControl <- trainControl(
  method = "repeatedcv",
  number = NO_OF_CV, 
  repeats = REPEAT,
  classProb = TRUE,
  summaryFunction = twoClassSummary)

model1 <- train( 
  label ~ alchemy_category + alchemy_category_score + news_front_page + compression_ratio + spelling_errors_ratio,
  data = train,
  method ="rf",
  trControl = fitControl,
  ntree = NO_OF_TREES,
  importance = TRUE,	
  tuneGrid = rfGrid,
  metric = "Accuracy")

plot(model1)

# Predicting the labels for the test data -------------------------------------------------------------------------------
pred1 <- predict(model1, newdata = test, type = "prob")	
submit1 <- data.frame(URLID, pred1$Y1)
names(submit1)[2] <- "label"

write.csv(submit1, "OUTPUT_RF2.csv", row.names = F)
cat("model result:\n")
print (model1)	

#-------------------------------------------------------------------------------------------------------------------------
model2 <- glm(label ~ alchemy_category_score + alchemy_category + news_front_page +  compression_ratio + spelling_errors_ratio , train, family= binomial(link=logit))

# Predicting the labels for the test data -------------------------------------------------------------------------------
pred2 <- predict(model2, newdata = test)	
submit2 <- data.frame(URLID, pred2)
names(submit2)[2] <- "label"

write.csv(submit2, "OUTPUT_GLR2.csv", row.names = F)
cat("model result:\n")
print (model2)	
plot(model2)
#--------------------------------------------------------------------------------------------------------------------------
# Training a model based on the train data

model3 <- naiveBayes(label ~ alchemy_category_score + alchemy_category + news_front_page + compression_ratio + spelling_errors_ratio, train)

# Predicting the labels for the test data -------------------------------------------------------------------------------
pred3 <- predict(model3, newdata = test)	
submit3 <- data.frame(URLID, pred3)
names(submit3)[2] <- "label"

write.csv(submit3, "../Run_Output_Files/OUTPUT_NB3.csv", row.names = F)
cat("model result:\n")
print (model3)	
