# Stumbleupon Evergreen Classification Challenge Dataset
# Methodology: Random Forest in R
 
library(caret)
library(pROC)
library(randomForest)
library(ggplot2)
library(party)



set.seed(100000000)

# Loading Train Data ----------------------------------------------------------------------------------------------------
train <- read.table("../Dataset/train_mod.csv", header = T, sep = ",")

# Loading Test Data -----------------------------------------------------------------------------------------------------
test <- read.table("../Dataset/test_mod.csv", header = T, sep = ",")


# Setting label as factor variable to specity it as a classification problem
train$label <- as.factor(train$label) 
levels(train$label) <- list(Y1 = "1", Y2 = "0")


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

varnames <- names(train)
varnames <- varnames[!varnames %in% c("label")]
varnames1 <- paste(varnames, collapse = "+")

rf.form <- as.formula(paste("label", varnames1, sep = " ~ "))
modell <- randomForest( rf.form, train,ntree = 500, importance = TRUE)

print(modell)

# Predicting the labels for the test data -------------------------------------------------------------------------------
pred <- predict(modell, newdata = test)	
submit <- data.frame(test$urlid, pred)
names(submit)[2] <- "label"

write.csv(submit, "../Run_Output_Files/OUTPUT_RF_new.csv", row.names = F)
cat("model result:\n")
print (modell)	
