##########################################################
# MovieLens Rating Prediction Project
# HarvardX: PH125.9x Data Science Capstone
# Name: Yiqing Qu
##########################################################

##########################################################
# Setup and Data Loading
##########################################################

# Install and load required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(lubridate)

# MovieLens 10M dataset
options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Create edx and final_holdout_test sets
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# Data Exploration
##########################################################

# Dataset dimensions
cat("edx dataset rows:", nrow(edx), "\n")
cat("edx dataset columns:", ncol(edx), "\n")
cat("Number of unique users:", n_distinct(edx$userId), "\n")
cat("Number of unique movies:", n_distinct(edx$movieId), "\n")

# Rating distribution
edx %>% 
  group_by(rating) %>% 
  summarize(count = n()) %>% 
  arrange(desc(count))

##########################################################
# Create Train and Test Sets from edx
##########################################################

# Split edx for model development (90% train, 10% test)
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Ensure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

##########################################################
# RMSE Function
##########################################################

# Function to calculate RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

##########################################################
# Model 1: Baseline - Just the Average
##########################################################

# Calculate the overall average rating
mu <- mean(train_set$rating)
cat("Overall average rating:", mu, "\n")

# Predict all ratings as the average
baseline_rmse <- RMSE(test_set$rating, mu)
cat("Baseline RMSE:", baseline_rmse, "\n")

# Store results
rmse_results <- data.frame(Method = "Just the average", RMSE = baseline_rmse)

##########################################################
# Model 2: Movie Effect Model
##########################################################

# Calculate movie bias (b_i)
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Predict ratings with movie effect
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

movie_effect_rmse <- RMSE(test_set$rating, predicted_ratings)
cat("Movie Effect Model RMSE:", movie_effect_rmse, "\n")

rmse_results <- bind_rows(rmse_results,
                          data.frame(Method = "Movie Effect Model",
                                     RMSE = movie_effect_rmse))

##########################################################
# Model 3: Movie + User Effect Model
##########################################################

# Calculate user bias (b_u)
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict ratings with movie and user effects
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

movie_user_rmse <- RMSE(test_set$rating, predicted_ratings)
cat("Movie + User Effect Model RMSE:", movie_user_rmse, "\n")

rmse_results <- bind_rows(rmse_results,
                          data.frame(Method = "Movie + User Effect Model",
                                     RMSE = movie_user_rmse))

##########################################################
# Model 4: Regularized Movie + User Effect Model
##########################################################

# Find optimal lambda using cross-validation
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  # Calculate regularized movie effect
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # Calculate regularized user effect
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  # Predict and calculate RMSE
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(test_set$rating, predicted_ratings))
})

# Find optimal lambda
lambda <- lambdas[which.min(rmses)]
cat("Optimal lambda:", lambda, "\n")

# Plot lambda vs RMSE
qplot(lambdas, rmses, 
      main = "Regularization",
      xlab = "Lambda",
      ylab = "RMSE")

regularized_rmse <- min(rmses)
cat("Regularized Movie + User Effect Model RMSE:", regularized_rmse, "\n")

rmse_results <- bind_rows(rmse_results,
                          data.frame(Method = "Regularized Movie + User Effect",
                                     RMSE = regularized_rmse))

##########################################################
# Final Model: Train on Full edx Set
##########################################################

# Calculate overall average on full edx set
mu_edx <- mean(edx$rating)

# Calculate regularized movie effect on full edx set
b_i_edx <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_edx)/(n()+lambda))

# Calculate regularized user effect on full edx set
b_u_edx <- edx %>% 
  left_join(b_i_edx, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu_edx)/(n()+lambda))

##########################################################
# Final Prediction on final_holdout_test Set
##########################################################

# Generate predictions for final holdout test set
final_predictions <- final_holdout_test %>% 
  left_join(b_i_edx, by = "movieId") %>%
  left_join(b_u_edx, by = "userId") %>%
  mutate(pred = mu_edx + b_i + b_u) %>%
  pull(pred)

# Calculate final RMSE
final_rmse <- RMSE(final_holdout_test$rating, final_predictions)

# Add to results
rmse_results <- bind_rows(rmse_results,
                          data.frame(Method = "Final Model on Validation Set",
                                     RMSE = final_rmse))

##########################################################
# Results Summary
##########################################################

cat("\n=== RMSE Results Summary ===\n")
print(rmse_results)

cat("\n=== FINAL RMSE ON HOLDOUT TEST SET ===\n")
cat("Final RMSE:", final_rmse, "\n")

# Save results
write.csv(rmse_results, "rmse_results.csv", row.names = FALSE)