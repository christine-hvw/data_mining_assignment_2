
## Data Mining Assignment 2 ##
# Classification for the Detection of Opinion Spam #

# Packages and functions --------------------------------------------------

library(tm)            # basic text mining operations 
library(udpipe)        # lemmatization
library(SnowballC)     # dependency for stemming
library(caret)         # modeling workflow
library(modelr)        # modelling outside of caret
library(naivebayes)    # multinomial naive bayes
library(glmnet)        # lasso regression
library(rpart)         # class. trees
library(randomForest)  # random forests

# function to perform lemmatization with tm
get_lemmas <- function(x) {
  lemmas <- udpipe(words(x), "english")[["lemma"]]
  paste(lemmas, collapse = " ")
}

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
revs_all <- tm_map(revs_all, content_transformer(get_lemmas)) # can take a minute
revs_all <- tm_map(revs_all, stemDocument)

# Create label vector (0=deceptive, 1=truthful)
labels <- c(rep(0, 400), rep(1, 400))

# Data partitioning
set.seed(123)
train_partition <- createDataPartition(labels, p = 0.8, list = FALSE)

# Create document term matrix (dtm)
dtm_train <- DocumentTermMatrix(revs_all[train_partition])
dtm_train # -> 4.9k features, 99% sparsity

# Remove sparse terms
dtm_train <- removeSparseTerms(dtm_train, .99)
dtm_train # -> 1042 features, 95% sparsity

# Create dtm for test set 
dtm_test <- DocumentTermMatrix(revs_all[-train_partition],
                               # (has same features as dtm_train)
                               list(dictionary = dimnames(dtm_train)[[2]]))

# Repeat steps above including bigrams 
# remove sparse terms before creating bigrams
dtm2_train <- DocumentTermMatrix(revs_all[train_partition],
                                 control = list(tokenize = get_unibigrams))
dtm2_train # -> 44k features, 100% sparsity

# try to set to .99 percent -> make extra training set
dtm2_train <- removeSparseTerms(dtm2_train, .99)
dtm2_train # -> 1530 features, 96% sparsity

dtm2_test <- DocumentTermMatrix(revs_all[-train_partition],
                                list(dictionary = dimnames(dtm2_train)[[2]]))

# Modeling ----------------------------------------------------------------
# Try 4 different approaches below, each with and without bigram features
# -> 8 final models to be compared

## Multinomial naive Bayes (generative linear classifier)------------------
# hyperparameters: - laplace smoothing
# - number of features used (if feature selection is performed)


### Hyperparameter tuning -------------------------------------------------

set.seed(123)
cv_folds <- crossv_kfold(data.frame(as.matrix(dtm_train)), k = 10)

laplace_vals <- seq(0.1, 1, by = 0.1)

# Training
models <- list()

models <- lapply(cv_folds$train, function(fold) {
  lapply(laplace_vals, function(l) {
    multinomial_naive_bayes(x = as.matrix(fold$data[fold$idx,]), 
                            y = as.factor(labels[fold$idx]),
                            laplace = l)
  })
})

# Testing
predictions <- list()

predictions <- lapply(models, function(m) {
  lapply(1:length(laplace_vals), function(l) {
    predict(m[[l]], as.matrix(as.data.frame(cv_folds$test[[l]])))
  })
})

accuracies <- list()

accuracies <- lapply(1:length(predictions), function(fold) {
  lapply(1:length(laplace_vals), function(l) {
    mean(predictions[[fold]][[l]] == labels[cv_folds$test[[fold]][["idx"]]])
  })
})

# Average accuracy per Laplace value over folds 
ave_accuracies <- rowMeans(
  # bind sub-lists into one data frame
  do.call("rbind", 
          # make sub-lists into data frame columns
          lapply(accuracies, function(x) do.call("cbind", x))
  )
)

# identify best value for Laplace smoothing parameter 
best_laplace <- laplace_vals[ave_accuracies == max(ave_accuracies)]


### Model training and validation -----------------------------------------

#### Only unigrams

mnb_train <- multinomial_naive_bayes(x = as.matrix(dtm_train), 
                                     y = as.factor(labels[train_partition]),
                                     laplace = best_laplace)

mnb_predict <- predict(mnb_train, as.matrix(dtm_test))

confusionMatrix(as.factor(mnb_predict), as.factor(labels[-train_partition]),
                mode = "everything")

# get top five features (based on Mutual Information, entropy..., slide 42 ff)

#### With bigrams

mnb2_train <- multinomial_naive_bayes(x = as.matrix(dtm2_train), 
                                      y = as.factor(labels[train_partition]),
                                      laplace = best_laplace)

mnb2_predict <- predict(mnb2_train, as.matrix(dtm2_test))

confusionMatrix(as.factor(mnb2_predict), as.factor(labels[-train_partition]),
                mode = "everything")

# get top five features (based on Mutual Information, entropy..., slide 42 ff)


## LASSO logistic regression (discriminative linear classifier)------------
# hyperparameters: - alpha = 1 (lasso penalty), - lambda

### Unigrams

set.seed(123)
lasso_train <- cv.glmnet(x = as.matrix(dtm_train),
                         y = labels[train_partition],
                         family = "binomial", type.measure = "class", nfolds = 10)
#plot(lasso_train)

lasso_predict <- predict(lasso_train, newx = as.matrix(dtm_test), 
                         s = "lambda.1se", type = "class")

confusionMatrix(as.factor(lasso_predict), as.factor(labels[-train_partition]),
                mode = "everything")

### Bigrams

lasso2_train <- cv.glmnet(x = as.matrix(dtm2_train),
                          y = labels[train_partition],
                          family = "binomial", type.measure = "class", nfolds = 10)

lasso2_predict <- predict(lasso2_train, newx = as.matrix(dtm2_test), 
                          s = "lambda.1se", type = "class")

confusionMatrix(as.factor(lasso2_predict), as.factor(labels[-train_partition]),
                mode = "everything")

## Classification trees (non-linear classifier)----------------------------
# hyperparameters: - complexity (cp)

### Unigrams

set.seed(123)
tree_train <- train(x = as.matrix(dtm_train),
                    y = as.factor(labels[train_partition]),
                    method = "rpart",
                    trControl = trainControl(method = "cv", number = 10, search = "random"),
                    tuneLength = 100)

tree_predict <- predict(tree_train, as.matrix(dtm_test))

confusionMatrix(as.factor(tree_predict), as.factor(labels[-train_partition]),
                mode = "everything")

### Bigrams

set.seed(123)
tree2_train <- train(x = as.matrix(dtm2_train),
                     y = as.factor(labels[train_partition]),
                     method = "rpart",
                     trControl = trainControl(method = "cv", number = 10, search = "random"),
                     tuneLength = 100)

tree2_predict <- predict(tree2_train, as.matrix(dtm2_test))

confusionMatrix(as.factor(tree2_predict), as.factor(labels[-train_partition]),
                mode = "everything")


## Random forest (ensemble of non-linear classifiers)----------------------
# hyperparameters: - numbers of parameters to sample from (mtry), 
# - number of trees (ntree) (not in caret, need to loop over ntree)
# 
### Unigrams

# try adaptive random sampling for ntree
mtry_grid <- data.frame(mtry = seq(from = 5, to = 20, by = 5))

set.seed(123)
rf_train <- train(x = as.matrix(dtm_train),
                  y = as.factor(labels[train_partition]),
                  method = "rf",
                  trControl = trainControl(method = "cv", number = 10),
                  tuneGrid = mtry_grid)

rf_predict <- predict(rf_train, as.matrix(dtm_test))

confusionMatrix(as.factor(rf_predict), as.factor(labels[-train_partition]),
                mode = "everything")

### Bigrams

set.seed(123)
rf2_train <- train(x = as.matrix(dtm2_train),
                   y = as.factor(labels[train_partition]),
                   method = "rf",
                   trControl = trainControl(method = "cv", number = 10),
                   tuneGrid = mtry_grid)

rf2_predict <- predict(rf2_train, as.matrix(dtm2_test))

confusionMatrix(as.factor(rf2_predict), as.factor(labels[-train_partition]),
                mode = "everything")

# Model comparison --------------------------------------------------------


## Performance measures ---------------------------------------------------
# compare accuracy, precision, recall and F1 score
# conduct statistical test for accuracy


## Most important features ------------------------------------------------
# list top 5 features for fake and genuine reviews (for best performing model)


