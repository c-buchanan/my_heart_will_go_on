# ----------------------------------- 

# CLEANING THE TITANIC DATA SET

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
raw_data <- read.csv("titanic_passenger_manifest.csv", header = TRUE)

# View header
head(raw_data, n=10)

# Determine the structure of data set
str(raw_data)

# Examine the statistical distribution
summary(raw_data)

# Let's see the distribution of male and females that survived. 
prop.table(table(raw_data$sex, raw_data$survived), 1)

# Earlier, we saw that age, Cabin, and Fare had missing values. 
len(list(raw_data$cabin=="")) 
class1_no_cabin <-which(raw_data$pclass==1 & raw_data$cabin=="")
len(class1_no_cabin)

# Replace the values with NA
raw_data$cabin[class1_no_cabin] <- NA
len(is.na(raw_data$cabin))

# ----------------------------------- 

# FEATURE ENGINEERING 

# ----------------------------------- 

# Let's see if a passenger traveling alone had a better survival rate than
# a passenger traveling with a family by combining the parch and sibsp values.

# Convert to characters.
raw_data$sibsp <- as.numeric(as.character(raw_data$sibsp))
raw_data$parch <- as.numeric(as.character(raw_data$parch))

# Convert to numeric.
raw_data$sibsp <- as.numeric(raw_data$sibsp)
raw_data$parch <- as.numeric(raw_data$parch)

# Add values together.
raw_data$family_size <- raw_data$sibsp + raw_data$parch

# Let's create a new variable, “child”, to indicate whether the 
# passenger is below the age of 18 or not. 

raw_data$child <- 0
raw_data$child[raw_data$age < 18] <- 1

# Let's check the number of survivors by sub-set.
aggregate(survived ~ child + sex, data=raw_data, FUN=function(x) {sum(x)/length(x)})

# Let's get the titles separated from the names to see if that impacted survivability.
# Cast the names as a string 
raw_data$name[1]
raw_data$name <- as.character(raw_data$name)

strsplit(raw_data$name[1], split='[,.]')
strsplit(raw_data$name[1], split='[,.]')
strsplit(raw_data$name[1], split='[,.]')[[1]][2]

# Now create a new column
raw_data$title <- sapply(raw_data$name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})

# Clean up the extra spaces
raw_data$title <- sub(' ', '', raw_data$title)
table(raw_data$title)

# Now let's combine some of the titles to limit the factor variance 
# Since Mademoiselle and Madame are similar, let's combine them. 
raw_data$title[raw_data$title %in% c('Mme', 'Mlle')] <- 'Mlle'

# Now let's look at the military titles: Captain, Colonel, and Major
raw_data$title[raw_data$title %in% c('Capt', 'Major', 'Col')] <- 'Military'

# Now for the lower male peerage titles. While "Master" just denotes the heir
# of an estate, we should combine it with the other titles of nobility to create a peerage class.
raw_data$title[raw_data$title %in% c('Don', 'Jonkheer', 'Master', 'Sir')] <- 'Nobleman'

# Let's do the same for the female peerage titles. 
raw_data$title[raw_data$title %in% c('Dona', 'Lady', 'the Countess')] <- 'Noblewoman'

# Let's combine Miss and Ms because they are similar enough, along with Mademoiselle. 
raw_data$title[raw_data$title %in% c('Miss', 'Ms', 'Mlle')] <- 'Miss'

# Let's see how it looks now. 
table(raw_data$title)

# Awesome! Let's turn the title into a factor. 
raw_data$title <- factor(raw_data$title)

# Let's get the deck number, if applicable. 
raw_data$cabin <- as.character(raw_data$cabin)
raw_data$deck <- substr(raw_data$cabin, 1, 1)
raw_data$deck[raw_data$deck==""] <- NA
paste("Titanic has", nlevels(factor(raw_data$deck)),"decks on the ship.")

# Time to covert everything to a factor variable. 
raw_data_engineered <- raw_data[,c("survived","pclass","sex","age","sibsp","parch","fare","embarked","title","deck","family_size")]
list <- c("survived","pclass","sex","embarked","title","deck")
raw_data_engineered[list] <- lapply(raw_data_engineered[list],function(x) as.factor(x))
str(raw_data_engineered)

# ---------------------------------

# Imputing Missing Data using Random Forest

# ----------------------------------- 

md.pattern(raw_data_engineered)

set.seed(2476)
imp = mice(raw_data_engineered, method = "rf", m = 5)

# Once finished... 
imputed_data_engineered = complete(imp)
summary(imp)

# Apply the randomForest data to imputate
apply(apply(imputed_data_engineered,2,is.na),2,sum)

# Visualize the data before and after imputation
par(mfrow=c(1,2))
hist(raw_data_engineered$fare, main = "Before Imputation", col = "violet")
hist(imputed_data_engineered$fare, main = "Post Imputation", col = "blue")

par(mfrow=c(1,2))
hist(raw_data_engineered$age, main = "Before Imputation", col = "violet")
hist(imputed_data_engineered$age, main = "Post Imputation", col = "blue")

# ----------------------------------- 

# EXPLORATORY DATA ANALYSIS

# ----------------------------------- 

# Let's see how the size of one's family impacts the survival rate.
ggplot(imputed_data_engineered, aes(x=family_size, fill = factor(survived))) + geom_bar(stat = "count", position = "dodge") + scale_x_continuous(breaks = c(1:11)) + labs(x= "family_size")

# Let's see how age and gender impact survival rate.
ggplot(imputed_data_engineered, aes(x = age, fill = factor(survived))) + geom_histogram()+ facet_grid(.~sex)

prop.table(table(imputed_data_engineered$sex, imputed_data_engineered$survived), 1)

# Let's see how adult versus child impacts the survival rate.
imputed_data_engineered$child[imputed_data_engineered$age < 18] <- 'child'
imputed_data_engineered$child[imputed_data_engineered$age>= 18] <- 'adult'
prop.table(table(imputed_data_engineered$child, imputed_data_engineered$survived), 1)

ggplot(imputed_data_engineered, aes(x = age, fill = factor(survived))) + geom_bar(stat = "count")+ facet_grid(.~child)

# Let's look at class 
p1 <- ggplot(imputed_data_engineered,aes(x = pclass,fill = factor(survived)))+geom_bar(stat = "count", position = "stack")
p2 <- ggplot(imputed_data_engineered,aes(x = pclass,fill = factor(survived)))+geom_bar(position = "fill")+labs(y = "Proportion")
grid.arrange(p1,p2,ncol=2)

# ----------------------------------- 

# STATISTICAL INFERENCE

# ----------------------------------- 

# H0 : There is no difference between the two population means.
# H1 : There is difference between the two population means.

t.test(fare~survived, data = imputed_data_engineered)

# As the p-value <0.05, we reject H0 since there is difference between 
# the two population means. Further, average fare paid by those who survived 
# is higher by $25 than those who did not survive.

# H0 : There is no difference between the two population means.
# H1 : There is difference between the two population means.

t.test(age~survived,data = imputed_data_engineered)

# As the p-value is relatively higher but less than 0.05, we can tentatively 
# reject the H0 as there is a minor difference between the two population means. 

# Run a chi-squared test 

# H0 : Passenger Class and Survivals are independent.
# H1 : Passenger Class and Survivals are not independent.

chisq.test(imputed_data_engineered$pclass, imputed_data_engineered$survived)

# As the p-value <0.05, we reject H0, as pclass and survival are not independent. 

# If needed, drop the columns we no longer need
colnames(imputed_data_engineered)
cleaned_data <- subset(imputed_data_engineered, select = -c(name, boat, body, home.dest, family_sizeC))


# ----------------------------------- 

# LAST MINUTE ADJUSTMENTS

# ----------------------------------- 

# Read back in the previous CSV to shuffle the variables 
imputed_data_engineered <- read.csv("cleaned_titanic_data.csv", header = TRUE)
head(imputed_data_engineered, n=10)
colnames(imputed_data_engineered)

imputed_data_engineered <- subset(imputed_data_engineered, select = -c(fare, family_sizeC, embarked))

# Shuffle the rows to randomize the data, as the original dataset is ordered by class
randomized_data <- imputed_data_engineered[sample(nrow(imputed_data_engineered)),]

# Write the cleaned data to a new CSV
write.csv(randomized_data, file = "ready_for_ML_titanic_data.csv")