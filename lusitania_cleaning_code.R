# ----------------------------------- 

# CLEANING THE LUSITANIA DATA SET

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

# Load the CSV
raw_data <- read.csv("lusitania_passenger_crew_manifest_crossing202.csv", header = TRUE)

# View header
head(raw_data, n=10)

# Determine the structure of data set
str(raw_data)

# Examine the statistical distribution
summary(raw_data)

# Drop the columns we no longer need
colnames(raw_data)
raw_data <- subset(raw_data, select = -c(Family.name, Position, Personal.name, Citizenship, City, State, Country, County, Lifeboat, Rescue.Vessel, Body.No., Ticket.No.))
raw_data <- subset(raw_data, select = -c(County))
raw_data <- subset(raw_data, select = -c(Position))

# Finish up in Excel to save time 
write.csv(raw_data, file = "semi_cleaned_lusitania_data.csv")

# ----------------------------------- 

# DATA MUNGING PART II

# ----------------------------------- 

raw_data <- read.csv("wip_cleaned_lusitania_data.csv", header = TRUE)

# Determine the structure of data set
str(raw_data)

# Examine the statistical distribution
summary(raw_data)

# ----------------------------------- 

# FEATURE ENGINEERING 

# ----------------------------------- 

# Let's get the titles separated from the names to see if that impacted survivability.
# Cast the names as a string 
raw_data$title[1]
raw_data$title <- as.character(raw_data$title)

# Clean up the extra spaces
raw_data$title <- sub(' ', '', raw_data$title)
table(raw_data$title)

raw_data$title[raw_data$title %in% c('Sir', 'Nobleman')] <- 'Nobleman'
raw_data$title[raw_data$title %in% c('Rev', 'Sister', 'Father')] <- 'Rev'

table(raw_data$title)

# Let's get the deck number, if applicable. 
raw_data$cabin <- as.character(raw_data$cabin)
raw_data$deck <- substr(raw_data$cabin, 1, 1)
raw_data$deck[raw_data$deck==""] <- NA
paste("Lusitania has", nlevels(factor(raw_data$deck)),"decks on the ship.")

# Finish up in Excel to save time 
write.csv(raw_data, file = "semi_wip_cleaned_lusitania_data.csv")

# ----------------------------------- 

# DATA MUNGING PART III
# USING RANDOM FOREST TO IMPUTATE MISSING VALUES 

# ----------------------------------- 

# Load the data back 
raw_data <- read.csv("semi_wip_cleaned_lusitania_data.csv", header = TRUE)
colnames(raw_data)
head(raw_data, n=10)

# Time to covert everything to a factor variable. 
raw_data_engineered <- raw_data[,c("title", "survived", "age", "pclass", "sibsp",      
                                    "parch", "family_size", "child", "sex", "deck")]

list <- c("survived","pclass","sibsp","parch","family_size")
raw_data_engineered[list] <- lapply(raw_data_engineered[list],function(x) as.factor(x))
str(raw_data_engineered)

# Run randomForest

set.seed(2476)
imp = mice(raw_data_engineered, method = "rf", m = 5)

# Once finished... 
imputed_data_engineered = complete(imp)
summary(imp)

# Apply the randomForest data to imputate
apply(apply(imputed_data_engineered,2,is.na),2,sum)

# ----------------------------------- 

# FINISHING TOUCHES 

# ----------------------------------- 

# Shuffle the rows to randomize the data, as the original dataset is ordered by class
randomized_data <- imputed_data_engineered[sample(nrow(imputed_data_engineered)),]

# Write the cleaned data to a new CSV
write.csv(randomized_data, file = "cleaned_lusitania_data.csv")
