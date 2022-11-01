
## Auxiliary functions for Data Mining Assignment 2 ## 


# Lemmatization with tm ---------------------------------------------------

get_lemmas <- function(x) {
  lemmas <- udpipe(words(x), "english")[["lemma"]]
  paste(lemmas, collapse = " ")
}


# Return unigrams and bigrams of a document -------------------------------

get_unibigrams <- function(x) {
  ngrams <- ngrams(words(x), 1:2)
  return(unlist(lapply(ngrams, paste, collapse = " "), 
                use.names = FALSE))
}


# Hyperparameter tuning of multinomial naive bayes ------------------------

tune_mnb <- function(train_data){
  
  set.seed(123)
  cv_folds <- crossv_kfold(data.frame(as.matrix(train_data)), k = 10)
  
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
  return(best_laplace = laplace_vals[ave_accuracies == max(ave_accuracies)])
  
}


# Feature extraction ------------------------------------------------------

## MNB

get_features_mnb <- function(train_model) {
  
  top5 <- lapply(1:2, function(class) {
    coefs <- coef(train_model)
    coefs["feature"] <- rownames(coefs)
    order <- order(coefs[class], decreasing = TRUE)
    top5_df <- coefs[order[1:5], c(class, 3)]
    top5 <- top5_df[[1]]
    names(top5) <- top5_df[["feature"]]
    top5
  })
  
  names(top5) <- c("deceptive", "truthful")
  top5
}


## LASSO

get_features_lasso <- function(train_model, test_data) {
  
  coefs <- predict(train_model, newx = as.matrix(test_data), 
                   s = "lambda.1se", type = "coefficient")
  coef_names <- coefs@Dimnames[[1]]
  coef_mat <- summary(coefs)
  order <- order(coef_mat[,"x"], decreasing = TRUE)
  
  top5_true_mat <- coef_mat[order[1:5],]
  top5_true_vals <- top5_true_mat[,"x"]
  names(top5_true_vals) <- coef_names[top5_true_mat[,"i"]]
  
  top5_decep_mat <- coef_mat[rev(order)[1:5],]
  top5_decep_vals <- top5_decep_mat[,"x"]
  names(top5_decep_vals) <- coef_names[top5_decep_mat[,"i"]]
  
  list(deceptive = top5_decep_vals, truthful = top5_true_vals)
}

## TREE

get_features_tree <- function(train_model) {
  
  imps <- varImp(train_model, scale = FALSE)$importance
  imps["feature"] <- rownames(imps)
  imps_ord <- imps[order(imps$Overall, decreasing = TRUE),]
  
  top5_vals <- imps_ord[1:5, "Overall"]
  names(top5_vals) <- imps_ord[1:5, "feature"]
  list(top5 = top5_vals)
}