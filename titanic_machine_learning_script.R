# ----------------------------------- 

# TITANIC MACHINE LEARNING CODE

# ----------------------------------- 

# Load the libraries

library(tidyverse)
library(dplyr)
library(tidyr)
library(readr)
library(tibble)
library(stats)
library(ggplot2)
library(data.table)
library(caret)
library(randomForest)
library(e1071)
library(grid)
library(gridExtra)
library(mice)
library(pscl)
library(knncat)
library(aod)
library(ROCR)
library(pROC)
library(InformationValue)
library(stringr)
library(rowr)

# Load the CSV
data <- read.csv("ready_for_ML_titanic_data.csv", header = TRUE)

# View header
head(data, n=10)

# Determine the structure of data set
str(data)

# Examine the statistical distribution
summary(data)
table(data$title)

colnames(data)
subset(data, select = -c(X))

data <- data[,c("survive","pclass","sex","age","sibsp","parch","title","deck","family_size","child")]
list <- c("survive","pclass","sex","title","deck")
data[list] <- lapply(data[list],function(x) as.factor(x))
str(data)

# First split the data into train and test sets using the caret package
# Partition the data into train and test sets (70/30)

'%ni%' <- Negate('%in%')  # define 'not in' func
options(scipen=999)  # prevents printing scientific notations.

# Prep Training and Test data.
trainDataIndex <- createDataPartition(data$survive, p=0.7, list = F)  # 70% training data
trainData <- data[trainDataIndex, ]
testData <- data[-trainDataIndex, ]

# ----------------------------------- 

# LOGISTIC REGRESSION

# ----------------------------------- 

glm_model <- glm(survive ~ ., family = binomial(link = 'logit'), data = trainData)
anova(glm_model, test = 'Chisq')

# Predicted scores
predicted <- plogis(predict(glm_model, testData)) 

# Determine the optimal cut-off point
optCutOff <- optimalCutoff(testData$survive, predicted)[1]
# .95

# Summarize the findings
summary(glm_model)

# Let's adjust the fit of the model.
pR2(glm_model)

glm_fit <- predict(glm_model, newdata = testData, type = 'response')
glm_fit <- ifelse(glm_fit > 0.5,1,0)
misclassificationerror <- mean(glm_fit != testData$survive)
print(paste('Accuracy of Logistic Regression Model = ', 1-misclassificationerror))

# Accuracy of Logistic Regression Model =  0.829516539440204

# ----------------------------------- 

# ROC AND AUC

# ----------------------------------- 

# Plot the ROC and AUC
plotROC(testData$survive, predicted)
plotAUC(testData$survive, predicted)

# Of all combinations of 1-0 pairs (actuals), concordance is the percentage
# of pairs, whose scores of positives are greater than the scores of negatives. 
# For a perfect model, this will be 100%. 

# The higher the concordance, the better is the quality of model.
Concordance(testData$survive, predicted)

# Concordance = [1] 0.7566393 = ~75%
# Not that great of a fit 

sensitivity(testData$survive, predicted)
# 76%

specificity(testData$survive, predicted)
# [1] 0.6575342 = 65.75%

confusionMatrix(testData$survive, predicted)
# The columns are actuals, while rows are predicted values

#    0    1
# 0 48  450
# 1 25 1425

# Plot the ROC and AUC
plotROC(testData$survive, predicted)
plotAUC(testData$survive, predicted)

# Another way of calculating ROC and AUC

ROC1 <- roc(testData$survive, predicted)

plot(ROC1, col = "blue")

AUC1 <- auc(ROC1)

# ----------------------------------- 

# RANDOM FOREST

# ----------------------------------- 

rf_model <- randomForest(survive ~ ., data = trainData, type = classification, ntrees = 3000, importance = TRUE, proximity = TRUE, na.action = na.exclude)
rf_model

# OOB estimate of  error rate: 21.51%

# Confusion matrix:
#   0   1 class.error
# 0 495  71   0.1254417
# 1 126 224   0.3600000

varImpPlot(rf_model)

rf_fit <- predict(rf_model, newdata = testData)
confusionMatrix(rf_fit, testData$survive)

# Confusion Matrix and Statistics
# 
# Prediction   
#    0   1
# 0 225  41
# 1  18 109
# 
# Accuracy : 0.8499               
# 95% CI : (0.8107, 0.8837)     
# No Information Rate : 0.6183               
# P-Value [Acc > NIR] : < 0.00000000000000022
# 
# Kappa : 0.6723               
# Mcnemar's Test P-Value : 0.004181             
# 
# Sensitivity : 0.9259               
# Specificity : 0.7267               
# Pos Pred Value : 0.8459               
# Neg Pred Value : 0.8583               
# Prevalence : 0.6183               
# Detection Rate : 0.5725               
# Detection Prevalence : 0.6768               
# Balanced Accuracy : 0.8263               
# 
# 'Positive' Class : 0

a <- confusionMatrix(rf_fit, testData$survive)$overall[1]
print(paste('Accuracy of Random Forest Model = ', a))

# Accuracy of Random Forest Model =  0.849872773536896

# ----------------------------------- 

# SVM MODEL

# ----------------------------------- 

# Use the library(e1071)
svm_model <- svm(survived ~ ., data = trainData)
svm_model

# Parameters:
#   SVM-Type:  C-classification 
# SVM-Kernel:  radial 
# cost:  1 
# gamma:  0.04347826 
# 
# Number of Support Vectors:  450

# Assess the fit of the model against the testing data
svm_fit <- predict(svm_model, newdata = testData)
confusionMatrix(svm_fit, testData$survive)

b <- confusionMatrix(svm_fit, testData$survive)$overall[1]
print(paste('Accuracy of Support Vector Machine Model = ', b))

# "Accuracy of Support Vector Machine Model =  0.819338422391858"

# ----------------------------------- 

# LINEAR DISCRIMINANT ANALYSIS

# ----------------------------------- 

lda_model <- train(survive ~ ., data = trainData, method = "lda", na.action = na.exclude)
lda_model

lda_fit <- predict(lda_model, newdata = testData)
confusionMatrix(lda_fit, testData$survive)

l <- confusionMatrix(lda_fit, testData$survive)$overall[1]
print(paste('Accuracy of Linear Discriminent Accuracy Model = ', l))

# "Accuracy of Linear Discriminent Accuracy Model =  0.819338422391858"

# ----------------------------------- 

# K NEAREST NEIGHBORS USING CARET

# ----------------------------------- 

ctrl <- trainControl(method="repeatedcv", repeats = 3) 
knn_model <- train(survive ~ ., data = trainData, method = "knn", trControl = ctrl, preProcess = c("center", "scale"), tuneLength = 20, na.action = na.exclude)
knn_model

plot(knn_model)

knn_fit <- predict(knn_model, newdata = testData)
confusionMatrix(knn_fit, testData$survive)

misclassificationerror <- mean(knn_fit != testData$survive)
print(paste('Accuracy of KNN Model = ', 1-misclassificationerror))

# "Accuracy of KNN Model =  0.801526717557252"

# ----------------------------------- 

# REVISED SOLUTION FOR BETTER ANALYSIS

# ----------------------------------- 

# Use ensembling to improve predictive accuracy
# We will ensemble all the above models and average the results. 

cat('Difference ratio between ridge and conditional random forest:', sum(glm_fit!=rf_fit)/nrow(testData))
cat('Difference ratio between ridge and conditional random forest:', sum(glm_fit!=svm_fit)/nrow(testData))
cat('Difference ratio between ridge and conditional random forest:', sum(rf_fit!=svm_fit)/nrow(testData))
cat('Difference ratio between ridge and conditional random forest:', sum(glm_fit!=lda_fit)/nrow(testData))
cat('Difference ratio between ridge and conditional random forest:', sum(glm_fit!=knn_fit)/nrow(testData))
cat('Difference ratio between ridge and conditional random forest:', sum(rf_fit!=lda_fit)/nrow(testData))
cat('Difference ratio between ridge and conditional random forest:', sum(rf_fit!=knn_fit)/nrow(testData))
cat('Difference ratio between ridge and conditional random forest:', sum(svm_fit!=lda_fit)/nrow(testData))
cat('Difference ratio between ridge and conditional random forest:', sum(svm_fit!=knn_fit)/nrow(testData))
cat('Difference ratio between ridge and conditional random forest:', sum(lda_fit!=knn_fit)/nrow(testData))

# Time to ensemble everything 

ensemble_model <- 0.2*(as.numeric(glm_fit)-1) + 0.4*(as.numeric(rf_fit)-1) + 0.4*(as.numeric(svm_fit)-1)
ensemble_model <- sapply(ensemble_model, round)
confusionMatrix(table(ensemble_model, testData$survived)) 

c <- confusionMatrix(table(ensemble_model, testData$survived))$overall[1]
print(paste('Accuracy of Ensembled Model = ', c))

# "Accuracy of Ensembled Model =  0.849872773536896"

# ----------------------------------- 

# Time to Test Ensemble Model for the Lusitania Dataset

# ----------------------------------- 

lusitania_data <- read.csv("cleaned_lusitania_data.csv", header = TRUE)

str(lusitania_data)
str(data)

# Convert appropriate factors
lusitania_data <- lusitania_data[,c("survived","pclass","sex","age","sibsp","parch","title","deck","family_size","child")]
list <- c("survived","pclass","sex","title","deck","child")
lusitania_data[list] <- lapply(lusitania_data[list],function(x) as.factor(x))

list2 <- c("age")
lusitania_data[list2] <- lapply(lusitania_data[list2],function(x) as.numeric(x))
str(lusitania_data)

# Time to fit the lusitania data to the models 

glm_fit <- predict(glm_model, newdata = lusitania_data, type = 'response')
glm_fit <- ifelse(glm_fit > 0.5,1,0)

rf_fit <- predict(rf_model, newdata = lusitania_data)
confusionMatrix(rf_fit, lusitania_data$survived)

svm_fit <- predict(svm_model, newdata = lusitania_data)
confusionMatrix(svm_fit, lusitania_data$survived)

lda_fit <- predict(lda_model, newdata = lusitania_data)
confusionMatrix(lda_fit, lusitania_data$survived)

knn_fit <- predict(knn_model, newdata = lusitania_data)
confusionMatrix(knn_fit, lusitania_data$survived)

# Create the ensemble model

cat('Difference ratio between ridge and conditional random forest:', sum(glm_fit!=rf_fit)/nrow(lusitania_data))
cat('Difference ratio between ridge and conditional random forest:', sum(glm_fit!=svm_fit)/nrow(lusitania_data))
cat('Difference ratio between ridge and conditional random forest:', sum(rf_fit!=svm_fit)/nrow(lusitania_data))
cat('Difference ratio between ridge and conditional random forest:', sum(glm_fit!=lda_fit)/nrow(lusitania_data))
cat('Difference ratio between ridge and conditional random forest:', sum(glm_fit!=knn_fit)/nrow(lusitania_data))
cat('Difference ratio between ridge and conditional random forest:', sum(rf_fit!=lda_fit)/nrow(lusitania_data))
cat('Difference ratio between ridge and conditional random forest:', sum(rf_fit!=knn_fit)/nrow(lusitania_data))
cat('Difference ratio between ridge and conditional random forest:', sum(svm_fit!=lda_fit)/nrow(lusitania_data))
cat('Difference ratio between ridge and conditional random forest:', sum(svm_fit!=knn_fit)/nrow(lusitania_data))
cat('Difference ratio between ridge and conditional random forest:', sum(lda_fit!=knn_fit)/nrow(lusitania_data))

# Time to test the ensemble model 
ensemble_model <- 0.2*(as.numeric(glm_fit)-1) + 0.4*(as.numeric(rf_fit)-1) + 0.4*(as.numeric(svm_fit)-1)
ensemble_model <- sapply(ensemble_model, round)
confusionMatrix(table(ensemble_model, lusitania_data$survived)) 

lusitania_cm <- confusionMatrix(table(ensemble_model, lusitania_data$survived))$overall[1]
print(paste('Accuracy of Lusitania Ensembled Model = ', lusitania_cm))

# [1] "Accuracy of Lusitania Ensembled Model =  0.529597474348856"

