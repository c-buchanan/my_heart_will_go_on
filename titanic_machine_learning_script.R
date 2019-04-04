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
library(stringr)
library(rowr)
library(randomForest)
library(e1071)
library(grid)
library(gridExtra)
library(mice)
library(dplyr)
library(pscl)
library(knncat)
library(aod)
library(ROCR)
library(pROC)
library(InformationValue)


# Load the CSV
data <- read.csv("ready_for_ML_titanic_data.csv", header = TRUE)

# View header
head(data, n=10)

# Determine the structure of data set
str(data)

# Examine the statistical distribution
summary(data)
table(data$title)

# First split the data into train and test sets using the caret package
# Partition the data into train and test sets (70/30)

'%ni%' <- Negate('%in%')  # define 'not in' func
options(scipen=999)  # prevents printing scientific notations.

# Prep Training and Test data.
trainDataIndex <- createDataPartition(data$survived, p=0.7, list = F)  # 70% training data
trainData <- data[trainDataIndex, ]
testData <- data[-trainDataIndex, ]


# ----------------------------------- 

# LOGISTIC REGRESSION

# ----------------------------------- 

model1 <- glm(survived ~ ., family = binomial(link = 'logit'), data = trainData)
anova(model1, test = 'Chisq')

# Predicted scores
predicted <- plogis(predict(model1, testData)) 

# Determine the optimal cut-off point
optCutOff <- optimalCutoff(test$survived, predicted)[1]

# Summarize the findings
summary(model1)

# Let's adjust the fit of the model.
pR2(model1)

fit1 <- predict(model1, newdata = testData, type = 'response')
fit1 <- ifelse(fit1 > 0.5,1,0)
misclassificationerror <- mean(fit1 != testData$survived)
print(paste('Accuracy of Logistic Regression Model = ', 1-misclassificationerror))

# Accuracy of Logistic Regression Model =  0.829516539440204

# ----------------------------------- 

# ROC AND AUC

# ----------------------------------- 

# Plot the ROC and AUC
plotROC(testData$survived, predicted)
plotAUC(testData$survived, predicted)

# Of all combinations of 1-0 pairs (actuals), concordance is the percentage
# of pairs, whose scores of positives are greater than the scores of negatives. 
# For a perfect model, this will be 100%. 
# The higher the concordance, the better is the quality of model.
Concordance(testData$survived, predicted)

# Concordance = [1] 0.7566393 = ~75%
# Not that great of a fit 

sensitivity(testData$survived, predicted)
# 76%

specificity(testData$survived, predicted)
# [1] 0.6575342 = 65.75%

confusionMatrix(testData$survived, predicted)
# The columns are actuals, while rows are predicted values

#    0    1
# 0 48  450
# 1 25 1425

# Plot the ROC and AUC
plotROC(testData$survived, predicted)
plotAUC(testData$survived, predicted)

# Another way of calculating ROC and AUC

ROC1 <- roc(testData$survived, predicted)

plot(ROC1, col = "blue")

AUC1 <- auc(ROC1)

# Area under the curve: 0.7566

# ----------------------------------- 

# RANDOM FOREST

# ----------------------------------- 