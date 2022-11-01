
## Data Mining Assignment 2 ##
# Classification for the Detection of Opinion Spam #

# Packages and functions --------------------------------------------------

library(tm)            # basic text mining operations 
library(udpipe)        # lemmatization
library(SnowballC)     # dependency for stemming
library(entropy)       # computing mutual information
library(caret)         # modeling workflow
library(modelr)        # modelling outside of caret
library(naivebayes)    # multinomial naive bayes
library(glmnet)        # lasso regression
library(rpart)         # class. trees
library(doParallel)    # parallel processing for RFs
library(randomForest)  # random forests

source("utils.R")


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

# Data partitioning (folds 1-4 out of 5 for both classes)
train_partition <- c(1:320, 401:720)

# Create document term matrix (dtm)
dtm_train <- DocumentTermMatrix(revs_all[train_partition])
dtm_train # -> 4.9k features, 99% sparsity

# Remove sparse terms
dtm_train <- removeSparseTerms(dtm_train, .99)
dtm_train # -> 1078 features, 95% sparsity

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
dtm2_train # -> 1568 features, 96% sparsity

dtm2_test <- DocumentTermMatrix(revs_all[-train_partition],
                                list(dictionary = dimnames(dtm2_train)[[2]]))


# Feature selection -------------------------------------------------------
# compute mutual information of features and class labels and generate ranking
# (maybe reduce feature set to k most important for modeling)

# convert document term matrix to binary
dtm_train_bin <- as.matrix(dtm_train) > 0

# compute mutual information of each term with class label
mi_train <- apply(as.matrix(dtm_train_bin), 2,
                  function(x,y) {
                    mi.plugin(table(x, y)/length(y), unit = "log2")
                    },
                  labels[train_partition])

# sort the indices from high to low mutual information
mi_train_ord <- order(mi_train, decreasing = TRUE)

# get top five features
cat("Top 5 features (mutual information)",
    capture.output(mi_train[mi_train_ord[1:5]]), "\n",
    file = "results/results.txt", sep = "\n", append = TRUE)


# Modeling ----------------------------------------------------------------
# Try 4 different approaches below, each with and without bigram features
# -> 8 final models to be compared

## Multinomial naive Bayes (generative linear classifier)------------------
# hyperparameters: - laplace smoothing
# - number of features used (if feature selection is performed)


### Hyperparameter tuning -------------------------------------------------

best_laplace <- tune_mnb(dtm_train)

best_laplace2 <- tune_mnb(dtm2_train)

### Model training and validation -----------------------------------------

#### Only unigrams

mnb_train <- multinomial_naive_bayes(x = as.matrix(dtm_train), 
                                     y = as.factor(labels[train_partition]),
                                     laplace = best_laplace)

mnb_predict <- predict(mnb_train, as.matrix(dtm_test), type = "class")

mnb_5features <- get_features_mnb(mnb_train)
  
cat("MNB Unigrams",
    paste("Laplace smoothing param. = ", best_laplace), "\n",
    capture.output(
      confusionMatrix(as.factor(mnb_predict), as.factor(labels[-train_partition]),
                      mode = "everything")),
    paste("Deceptive features:", mnb_5features[1]),
    paste("Truthful features:", mnb_5features[2]), "\n",
    file = "results/results.txt", sep = "\n", append = TRUE)


#### With bigrams

mnb2_train <- multinomial_naive_bayes(x = as.matrix(dtm2_train), 
                                      y = as.factor(labels[train_partition]),
                                      laplace = best_laplace2)

mnb2_predict <- predict(mnb2_train, as.matrix(dtm2_test), type = "class")

mnb2_5features <- get_features_mnb(mnb2_train)

cat("MNB Bigrams",
    paste("Laplace smoothing param. = ", best_laplace2), "\n",
    capture.output(
      confusionMatrix(as.factor(mnb2_predict), as.factor(labels[-train_partition]),
                      mode = "everything")),
    paste("Deceptive features:", mnb2_5features[1]),
    paste("Truthful features:", mnb2_5features[2]), "\n",
    file = "results/results.txt", sep = "\n", append = TRUE)


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

lasso_5features <- get_features_lasso(lasso_train, dtm_test)

cat("LASSO Unigrams",
    paste("lambda (1 SE) =", lasso_train[["lambda.1se"]]), "\n",
    capture.output(
      confusionMatrix(as.factor(lasso_predict), as.factor(labels[-train_partition]),
                      mode = "everything")),
    paste("Deceptive features:", lasso_5features[1]),
    paste("Truthful features:", lasso_5features[2]), "\n",
    file = "results/results.txt", sep = "\n", append = TRUE)

### Bigrams

lasso2_train <- cv.glmnet(x = as.matrix(dtm2_train),
                          y = labels[train_partition],
                          family = "binomial", type.measure = "class", nfolds = 10)

lasso2_predict <- predict(lasso2_train, newx = as.matrix(dtm2_test), 
                          s = "lambda.1se", type = "class")

lasso2_5features <- get_features_lasso(lasso2_train, dtm2_test)

cat("LASSO Bigrams",
    paste("lambda (1 SE) =", lasso2_train[["lambda.1se"]]), "\n",
    capture.output(
      confusionMatrix(as.factor(lasso2_predict), as.factor(labels[-train_partition]),
                      mode = "everything")),
    paste("Deceptive features:", lasso2_5features[1]),
    paste("Truthful features:", lasso2_5features[2]), "\n",
    file = "results/results.txt", sep = "\n", append = TRUE)



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

tree_5features <- get_features_tree(tree_train)

cat("Tree Unigrams",
    paste("cp (random search, 100 values) =", tree_train[["bestTune"]][["cp"]]), "\n",
    capture.output(
      confusionMatrix(as.factor(tree_predict), as.factor(labels[-train_partition]),
                      mode = "everything")),
    paste("Most important features:", tree_5features), "\n",
    file = "results/results.txt", sep = "\n", append = TRUE)

### Bigrams

set.seed(123)
tree2_train <- train(x = as.matrix(dtm2_train),
                     y = as.factor(labels[train_partition]),
                     method = "rpart",
                     trControl = trainControl(method = "cv", number = 10, search = "random"),
                     tuneLength = 100)

tree2_predict <- predict(tree2_train, as.matrix(dtm2_test))

tree2_5features <- get_features_tree(tree2_train)

cat("Tree Bigrams",
    paste("cp (random search, 100 values) =", tree2_train[["bestTune"]][["cp"]]), "\n",
    capture.output(
      confusionMatrix(as.factor(tree2_predict), as.factor(labels[-train_partition]),
                      mode = "everything")),
    paste("Most important features:", tree2_5features), "\n",
    file = "results/results.txt", sep = "\n", append = TRUE)


## Random forest (ensemble of non-linear classifiers)----------------------
# hyperparameters: - numbers of parameters to sample from (mtry), 
# - number of trees (ntree) (not in caret, need to loop over ntree)

### Unigrams

# mtry grid centered around sqrt(nfeat) ~ 32
mtry_grid <- data.frame(mtry = seq(from = 22, to = 42, by = 5))

rf_list <- list()
ntrees <- c(500, 1000, 1500, 2000)

start.time <- proc.time()
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

rf_list <- lapply(ntrees, function(ntree) {
  cat("Computing for ntree = ", ntree)
  
  set.seed(123)
  
  rf_train <- train(x = as.matrix(dtm_train),
                    y = as.factor(labels[train_partition]),
                    method = "rf",
                    trControl = trainControl(method = "cv", number = 10),
                    tuneGrid = mtry_grid,
                    ntree = ntree)
})

stop.time <- proc.time()
print(stop.time - start.time)
stopCluster(cl)
env <- foreach:::.foreachGlobals
rm(list=ls(name=env), pos=env)


rf_predict_ls <- list()

cat("Forest Unigrams",
    capture.output(
      for (i in 1:4) {
        cat("ntrees =", ntrees[i], "; ",
            "mtry =", rf_list[[i]][["bestTune"]][["mtry"]], "\n")
        rf_predict_ls[[i]] <- predict(rf_list[[i]], as.matrix(dtm_test))
        print(
          confusionMatrix(as.factor(rf_predict_ls[[i]]), as.factor(labels[-train_partition]),
                          mode = "everything")
        )
      }
    ), file = "results/results.txt", sep = "\n", append = TRUE
)


### Bigrams

# mtry grid centered around sqrt(nfeat) ~ 40
mtry_grid2 <- data.frame(mtry = seq(from = 30, to = 50, by = 5))

rf2_list <- list()

start.time <- proc.time()
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

rf2_list <- lapply(ntrees, function(ntree) {
  cat("Computing for ntree = ", ntree)
  
  set.seed(123)
  
  rf2_train <- train(x = as.matrix(dtm2_train),
                     y = as.factor(labels[train_partition]),
                     method = "rf",
                     trControl = trainControl(method = "cv", number = 10),
                     tuneGrid = mtry_grid2)
})

stop.time <- proc.time()
print(stop.time - start.time)
stopCluster(cl)
env <- foreach:::.foreachGlobals
rm(list=ls(name=env), pos=env)


rf2_predict_ls <- list()

cat("Forest Bigrams",
    capture.output(
      for (i in 1:4) {
        cat("ntrees =", ntrees[i], "; ",
            "mtry =", rf2_list[[i]][["bestTune"]][["mtry"]], "\n")
        rf2_predict_ls[[i]] <- predict(rf2_list[[i]], as.matrix(dtm2_test))
        print(
          confusionMatrix(as.factor(rf2_predict_ls[[i]]), as.factor(labels[-train_partition]),
                          mode = "everything")
        )
      }
    ), file = "results/results.txt", sep = "\n", append = TRUE
)


# Model comparison --------------------------------------------------------


## Performance measures ---------------------------------------------------
# compare accuracy, precision, recall and F1 score
# conduct statistical test for accuracy


## Most important features ------------------------------------------------
# list top 5 features for fake and genuine reviews (for best performing model)


