
## Data Mining Assignment 2 ##
# Classification for the Detection of Opinion Spam #

# Packages and functions --------------------------------------------------

library(tm)            # basic text mining operations 
library(SnowballC)     # dependency for stemming
library(naivebayes)    # multinomial naive bayes
library(glmnet)        # lasso regression
library(rpart)         # class. trees
library(randomForest)  # random forests

# function to return unigrams and bigrams of a document
get_unibigrams <- function(x) {
  ngrams <- ngrams(words(x), 1:2)
  return(unlist(lapply(ngrams, paste, collapse = " "), 
                use.names = FALSE))
}

# Data pre-processing -----------------------------------------------------

# Read data and make it a VCorpus object
files_all <- list.files(path = "data",
                        recursive = TRUE,
                        pattern = ".txt",
                        full.names = TRUE)

revs_all <- sapply(files_all, function(x) readLines(x))
revs_all <- VCorpus(VectorSource(revs_all))

# Pre-processing of text
revs_all <- tm_map(revs_all, removePunctuation)
revs_all <- tm_map(revs_all, content_transformer(tolower)) 
revs_all <- tm_map(revs_all, removeWords, stopwords("english"))
revs_all <- tm_map(revs_all, removeNumbers)
revs_all <- tm_map(revs_all, stripWhitespace)
revs_all <- tm_map(revs_all, stemDocument)

# Create label vector (0=deceptive, 1=truthful)
labels <- c(rep(0, 400), rep(1, 400))

# Define training corpus: folds 1-4 of both deceptive and true reviews
index_train <- c(1:320, 401:720)
index_test <- c(1:800)[-index_train]

# Create document term matrix (dtm)
dtm_train <- DocumentTermMatrix(revs_all[index_train])
dtm_train # -> 4.9k features, 99% sparsity

# Remove sparse terms
dtm_train <- removeSparseTerms(dtm_train, .95)
dtm_train # -> 321 features, 88% sparsity

# Create dtm for test set 
dtm_test <- DocumentTermMatrix(revs_all[index_test],
                               # (has same features as dtm_train)
                               list(dictionary = dimnames(dtm_train)[[2]]))

# Repeat steps above including bigrams 
dtm2_train <- DocumentTermMatrix(revs_all[index_train],
                                 control = list(tokenize = get_unibigrams))
dtm2_train # -> 45k features, 100% sparsity

dtm2_train <- removeSparseTerms(dtm2_train, .95)
dtm2_train # -> 338 features, 88% sparsity

dtm2_test <- DocumentTermMatrix(revs_all[index_test],
                                list(dictionary = dimnames(dtm2_train)[[2]]))

# Modeling ----------------------------------------------------------------
# Try 4 different approaches below, each with and without bigram features
# -> 8 final models to be compared

## Multinomial naive Bayes (generative linear classifier)------------------
### Only unigrams

mnb_train <- multinomial_naive_bayes(x = as.matrix(dtm_train), 
                                     y = as.factor(labels[index_train]))

mnb_predict <- predict(mnb_train, as.matrix(dtm_test))

conf_mat_mnb <- table(mnb_predict, labels[index_test])
sum(diag(conf_mat_mnb))/length(index_test) # -> 80% accuracy

# get top five features (based on Mutual Information, entropy..., slide 42 ff)

### With bigrams

mnb2_train <- multinomial_naive_bayes(x = as.matrix(dtm2_train), 
                                      y = as.factor(labels[index_train]))

mnb2_predict <- predict(mnb2_train, as.matrix(dtm2_test))

conf_mat_mnb2 <- table(mnb2_predict, labels[index_test])
sum(diag(conf_mat_mnb2))/length(index_test) # -> 80% accuracy, same conf. mat.

# get top five features (based on Mutual Information, entropy..., slide 42 ff)


## LASSO logistic regression (discriminative linear classifier)------------
### Unigrams

lasso_train <- cv.glmnet(x = as.matrix(dtm_train),
                         y = labels[index_train],
                         family = "binomial", type.measure = "class")
#plot(lasso_train)

lasso_predict <- predict(lasso_train, newx = as.matrix(dtm_test), 
                         s = "lambda.1se", type = "class")

conf_mat_lasso <- table(lasso_predict, labels[index_test])
sum(diag(conf_mat_lasso))/length(index_test) # -> 77.5% accuracy

### Bigrams

lasso2_train <- cv.glmnet(x = as.matrix(dtm2_train),
                          y = labels[index_train],
                          family = "binomial", type.measure = "class")

lasso2_predict <- predict(lasso2_train, newx = as.matrix(dtm2_test), 
                          s = "lambda.1se", type = "class")

conf_mat_lasso2 <- table(lasso2_predict, labels[index_test])
sum(diag(conf_mat_lasso2))/length(index_test) # -> 76.25% accuracy


## Classification trees (non-linear classifier)----------------------------
### Unigrams
tree_full <- rpart(label ~.,
                   data = data.frame(as.matrix(dtm_train),
                                     label = labels[index_train]),
                   cp = 0, method = "class")

# plot CV error of pruning sequence
plotcp(tree_full)
printcp(tree_full)

# prune tree at lowest CV error (11 splits)
tree_pruned <- prune(tree_full, cp = 0.009375)
plotcp(tree_pruned)

tree_predict <- predict(tree_pruned, newdata = data.frame(as.matrix(dtm_test)), 
                        type = "class")

conf_mat_tree <- table(tree_predict, labels[index_test])
sum(diag(conf_mat_tree))/length(index_test) # -> 66.88% accuracy

### Bigrams
tree2_full <- rpart(label ~.,
                    data = data.frame(as.matrix(dtm2_train),
                                      label = labels[index_train]),
                    cp = 0, method = "class")

printcp(tree2_full)

# prune tree at lowest CV error (1 split)
tree2_pruned <- prune(tree2_full, cp = 0.0250)

tree2_predict <- predict(tree2_pruned, newdata = data.frame(as.matrix(dtm2_test)), 
                         type = "class")

conf_mat_tree2 <- table(tree2_predict, labels[index_test])
sum(diag(conf_mat_tree2))/length(index_test) # -> 64.38% accuracy

## Random forest (ensemble of non-linear classifiers)----------------------
### Unigrams

rf_train <- randomForest(as.factor(label) ~.,
                         data = data.frame(as.matrix(dtm_train),
                                           label = labels[index_train]))

rf_predict <- predict(rf_train, newdata = data.frame(as.matrix(dtm_test)))

conf_mat_rf <- table(rf_predict, labels[index_test])
sum(diag(conf_mat_rf))/length(index_test) # -> 80% accuracy

### Bigrams

rf2_train <- randomForest(as.factor(label) ~.,
                          data = data.frame(as.matrix(dtm2_train),
                                            label = labels[index_train]))

rf2_predict <- predict(rf2_train, newdata = data.frame(as.matrix(dtm2_test)))

conf_mat_rf2 <- table(rf2_predict, labels[index_test])
sum(diag(conf_mat_rf2))/length(index_test) # -> 78.75% accuracy


# Model comparison --------------------------------------------------------


## Performance measures ---------------------------------------------------
# compare accuracy, precision, recall and F1 score
# conduct statistical test for accuracy


## Most important features ------------------------------------------------
# list top 5 features for fake and genuine reviews (for best performing model)


