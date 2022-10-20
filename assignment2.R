
## Data Mining Assignment 2 ##
# Classification for the Detection of Opinion Spam #

library(tm)
library(glmnet)
library(naivebayes)
library(rpart)
library(randomForest)

# Data pre-processing -----------------------------------------------------
data_path <- "C:/Users/chw/OneDrive/Utrecht/sem03/data_mining/assignment2/data/"


files_decep <- list.files(path = paste0(data_path, "deceptive_from_MTurk"),
                          recursive = TRUE,
                          pattern = ".txt",
                          full.names = TRUE)
files_true <- list.files(path = paste0(data_path, "truthful_from_Web"),
                         recursive = TRUE,
                         pattern = ".txt",
                         full.names = TRUE)

revs_decep <- sapply(files_decep, function(x) readLines(x))
revs_decep <- VCorpus(VectorSource(revs_decep))

revs_true <- sapply(files_true, function(x) readLines(x))
revs_true <- VCorpus(VectorSource(revs_true))

revs_all <- c(revs_decep, revs_true)

# Remove punctuation marks (comma’s, etc.)
revs_all <- tm_map(revs_all, removePunctuation)
# Make all letters lower case
revs_all <- tm_map(revs_all, content_transformer(tolower)) 
# Remove stopwords
revs_all <- tm_map(revs_all, removeWords, stopwords("english"))
# Remove numbers
revs_all <- tm_map(revs_all, removeNumbers)
# Remove excess whitespace
revs_all <- tm_map(revs_all, stripWhitespace)

# more editing???
#- stemming?

# make dtm, split into train and test

# remove sparse terms (if enough left after stemming), find percentage that works

# Modeling ----------------------------------------------------------------
# Try 4 different approaches below, each with and without bigram features
# -> 8 final models to be compared

## Multinomial naive Bayes (generative linear classifier)------------------


## LASSO logistic regression (discriminative linear classifier)------------


## Classification trees (non-linear classifier)----------------------------


## Random forest (ensemble of non-linear classifiers)----------------------



# Model comparison --------------------------------------------------------


## Performance measures ---------------------------------------------------
# compare accuracy, precision, recall and F1 score
# conduct statistical test for accuracy


## Most important features ------------------------------------------------
# list top 5 features for fake and genuine reviews (for best performing model)


