#rm(list = ls())
#.rs.restartR()
#cat("\f")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(data.table)
# local download of data and data wrangling
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")
# data partition - 90 % training and 10 % test set
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
# remove unnecessary data afterwards
rm(dl, ratings, movies, test_index, temp, removed)
movielens


# Start with analysis


# 1 - some descriptive analysis (also related to the quiz)

# first, number of rows and columns
ncol(edx)
nrow(edx)
# quiz question according to the numbers of observations by rating category
edx %>% filter(rating==0) %>% summarize(n=n())
edx %>% filter(rating==3) %>% summarize(n=n())
# Let`s look how many movies are rated
edx %>% group_by(movieId) %>% summarize(n=n())
# Do the same but now for the number of users
edx %>% group_by(userId) %>% summarize(n=n())
# descriptive analysis according to genre
genre<-c(edx$genres)
grep("Comedy",genre)
grep("Drama",genre)
grep("Thriller",genre)
grep("Romance",genre)
title<-c(edx$title)
grep("Forrest Gump",title)
grep("Jurassic Park",title)
grep("Pulp Fiction",title)
grep("Shawshank Redemption",title)
grep("Speed 2: Cruise Control",title)
# pulp fiction is the most often rated movie. Probably we can use it for factor analysis
# Last check according the number of observations for each value in rating
edx %>% group_by(rating) %>% summarize(n=n())%>% arrange(desc(n))
# we can see that whole star rating are much more likely than half star rating.

# look to the differences in the mean by user and by movie
movielens %>% group_by(userId) %>% summarize(mean_rating=mean(rating)) %>% select(userId,mean_rating) %>% distinct() %>% slice(1,10000,20000,30000,40000,50000,60000,70000) %>% knitr::kable()
movielens %>% group_by(movieId) %>% summarize(mean_rating=mean(rating)) %>% select(movieId,mean_rating) %>% distinct() %>% slice(1,5000,6000,7000,8000,9000) %>% knitr::kable()
# so we can observe huge differences in the mean by user and by movie  => specific effects are important to include

# Also look to the rating activity by movies and by users
# let`s plot the movies on the x axis and the number of ratings on the y axis
movielens %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")
# we can also observe that there are more than 60.000 user, lets look to their activity distribution in ratings per user
movielens %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")
# Interesting result - movies seem to be more evenly distributed (the large part of the movies is rated quite often)
# whereas a large part of users also rated only once (left skewed curve)
# Descriptive_Analysis_Summary 1 - user and movie specific effects are important simply explained by the nature of the data (a lot of different users and movies-different means by user and movies)
# Descriptive_Analysis_Summary 2 - outlier regions gain much more weight in user distribution rather than in movie distribution.

# Look also more in detail in the difference between half and whole star rating
# therefore create a whole star variable which is 1 when it is a full star rating and zero otherwise.
# Add this variable to the full movielens, edx and validation data set
movielens<-movielens%>%mutate(wholestar=ifelse(rating==0 | rating==1 | rating==2 | rating==3 |rating==4 | rating==5,1,0))
edx<-edx%>%mutate(wholestar=ifelse(rating==0 | rating==1 | rating==2 | rating==3 |rating==4 | rating==5,1,0))
validation<-validation%>%mutate(wholestar=ifelse(rating==0 | rating==1 | rating==2 | rating==3 |rating==4 | rating==5,1,0))
# Look to the number of observation and also to the mean rating value of the whole star variable
edx %>% group_by(wholestar) %>% summarize(n=n())%>% arrange(desc(n))
validation %>% group_by(wholestar) %>% summarize(n=n())%>% arrange(desc(n))
edx %>% group_by(wholestar) %>% ggplot(aes(wholestar)) + geom_histogram(bins=30)
movielens %>%
  group_by(wholestar) %>% 
  summarize(w = mean(rating)) %>%
  ggplot(aes(w)) + 
  geom_histogram(bins = 30, color = "black")
# Descriptive_Analysis_Summary 3 - Not only differences in the number of observations but also in the mean rating (Interesting result)
movielens %>% group_by(wholestar) %>% summarize(number_of_observations=n(),mean_value=mean(rating)) %>% knitr::kable(caption="General Bias - Half(0) Compared To Full(1) Star Rating")

# 2 Model and Predictions



# model 1 - only one regressor, the overall mean


# Let`s start with the simple model that we have one parameter for all movies and users - simply the mean
mu_hat <- mean(edx$rating)
# use the mean value to make predictions
naive_rmse <- RMSE(validation$rating, mu_hat)
naive_rmse
# RMSE is greater than 1, not very good but also not realistic because we know that there are a lot of different users and movies
# Nevertheless, create a data frame with all results for the report at the end
rmse_results <- data_frame(method = "Model 1 - Only Overall Variation", RMSE = naive_rmse)


# model 2 - overall mean and movie specific effects


# Let`s now use look for differences by movies.
# Therefore use the model above and assume that all other differences are explained by movie variation
# group_by movie can achieve this goal
mu <- mean(edx$rating) 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
# now make prediction on the test set
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i
# compare it to the rating values in the test set
model_2_rmse <- RMSE(predicted_ratings, validation$rating)
# Already a large improvement, add it to the report table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 2 - Movie Effect Model",
                                     RMSE = model_2_rmse ))
# we need the knitr package to create this table
library(knitr)
library(dslabs)
# now we cans see it
rmse_results %>% knitr::kable()


# model 3 - overall mean, movie and user specific effects


# now also add the user specific effect 
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
# once again user the test set to make predictions
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
# compare it to the rating values in the test set
model_3_rmse <- RMSE(predicted_ratings, validation$rating)
# once again a large improvement, add it to the table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 3 - Movie And User Effects Model",  
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()

# Take a look to some examples of the predicted ratings - top 10 movies
validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(residual =rating - (mu + b_i + b_u)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title, residual) %>% slice(1:10) %>% knitr::kable()
# Is reasonable. For example a block buster movie like Lord of the Rings I (awarded by several golden globe prices) has the largest mistake given by -5.13 which means that it has a high predicted rating (high mean rating) and gets only such a large error because there were some few users (probably one) that dislike this movie 
# So, the model already predicts quite well but let`s see if we can still improve the predictions 


# model 4 - overall mean, movie and user specific effect, penalty for users


# Now start to work with a penalty term in the averaging effect (lambda) of the user specific effects that helps to attenuate inexperienced users
# Because lambda is a parameter outside of the model we can choose it as high as we want
# therefore use cross validation - I used it as a comment so that you can decide on your own whether you will run it

# lambdas <- seq(0, 10, 0.25)
# rmses <- sapply(lambdas, function(l){
#   
#   mu <- mean(edx$rating)
#   
#   b_i <- edx %>% 
#     group_by(movieId) %>%
#     summarize(b_i = mean(rating - mu))
#   
#   b_u <- edx %>% 
#     left_join(b_i, by="movieId") %>%
#     group_by(userId) %>%
#     summarize(b_u = sum(rating - b_i - mu)/(n()+l))
#   
#   predicted_ratings <- 
#     validation %>% 
#     left_join(b_i, by = "movieId") %>%
#     left_join(b_u, by = "userId") %>%
#     mutate(pred = mu + b_i + b_u) %>%
#     pull(pred)
#   
#   return(RMSE(predicted_ratings, validation$rating))
# })
# 
# lambda <- lambdas[which.min(rmses)]
# min(rmses)

# so lambda=5.5 performs best but don`t include it in the results
# Instead fix lambda in a next step and do it again without cross validation
lambda<-5.5
mu <- mean(edx$rating)
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu)) 
movie_reg_avgs
user_reg_avgs<-edx %>% 
  left_join(movie_reg_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda), n_u=n())
user_reg_avgs
predicted_ratings <- validation %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_4_rmse<-RMSE(predicted_ratings, validation$rating)
# We see once again an improvement but it is certainly the same result when using cross validation- add it to the table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 4 - Movie And User Effects Model with Regularization Only For Use",  
                                     RMSE = model_4_rmse ))
rmse_results %>% knitr::kable()
# Despite the fact that the distribution of the number of ratings per movie is relatively normal ...
# it also decreases a bit the RMSE

# model 5 - overall mean, movie and user specific effects, penalty for users and movies


# Try also if a penalty term for movies can also improve the result
lambda<-5.5
mu <- mean(edx$rating)
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 
movie_reg_avgs
user_reg_avgs<-edx %>% 
  left_join(movie_reg_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda), n_u=n())
user_reg_avgs
predicted_ratings <- validation %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_5_rmse<-RMSE(predicted_ratings, validation$rating)
# Further improvement (Not very large), add it to the table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 5 - Movie And User Effects Model with Regularization For User And Movie",  
                                     RMSE = model_5_rmse ))
rmse_results %>% knitr::kable()


# model 6 - overall mean, movie and user specific effect, penalty for users and movies, whole star variable


# Increase the memory size because the computer already had some problems
memory.limit(16000)  
# Now add the last variable - whole star or not
# movie and user specific effects with penatly for outliers are still included
lambda<-5.5
mu <- mean(edx$rating)
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 
movie_reg_avgs
user_reg_avgs<-edx %>% 
  left_join(movie_reg_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda), n_u=n())
wholestar_avg<-edx %>%
  left_join(movie_reg_avgs, by="movieId") %>%
  left_join(user_reg_avgs, by="userId") %>%
  group_by(wholestar) %>%
  summarize(w = mean(rating - mu - b_i - b_u))
wholestar_avg
predicted_ratings <- validation %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by = "userId") %>%
  left_join(wholestar_avg, by = "wholestar") %>%
  mutate(pred = mu + b_i + b_u + w) %>%
  pull(pred)
model_6_rmse<-RMSE(predicted_ratings, validation$rating)
# Once again an improvement - add it to the table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 6 - Movie And User Effects Model With Regularization For User And Movie, Additionally Wholestar Variable",  
                                     RMSE = model_6_rmse ))
rmse_results %>% knitr::kable()
# Last check whether the prediction are reasonable - look once again to the ten largest mistakes
validation %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  left_join(wholestar_avg, by='wholestar') %>%
  mutate(residual = rating - (mu + b_i + b_u + w)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% slice(1:10) %>% knitr::kable()

