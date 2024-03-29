---
title: "Project 1-Movielens"
author: "Haas Nicolas"
date: "30 11 2020"
output:
  pdf_document: default
  html_document: default
---
Introduction
============

Companies like Youtube or Netflix are quite famous because of their well performing movie/video recommendation systems.If a user only gives a small peace of information about his preferences (one movie that he likes), applied algorithms can well predict which movie he wants to look next. Nevertheless, the main challenge in building such algorithms/recommendations systems lies in finding an efficient way which is still consistent over time. In other words, it is important to find the main sources that account for the variation in the data so that at the end a still "general" model can be used to different people over different years. Hence, this analysis is interested in the question how fast can we get a useful model? Or in other words, how many predictors do we need to have to recommend/predict well?

This analysis therefore uses a data set which is a subset of the MovieLens data and has been published by the grouplens researcher group (https://grouplens.org/datasets/movielens/10m/). This data set contains about 10 million movie ratings given by different people all over the world and over different years but it is still highly unbalanced (Some users only rate one movie and vice versa). The dependent variable is the continuous rating variable (0.5,1,...,4.5,5.0). The main idea is to work with movie ID and user ID that can be used as predictors (independent variables) because we know that not all users have the same preferences and not all movies represent the same genre, have the same actors,etc. Thus, it is important to include user and movie specific preferences/effects. Though, not only movie and user specific effects will be used to look for variation in the data but also rating patterns which are more general and are not specifically related to users and/or movies. In this context, a difference between half star and full star ratings can also improve the algorithms at the end. The performance of the algorithms itself will be measured by the root mean squared error (RMSE). At the end, the algorithms will lead to a RMSE of about 0.8639 (data split - 10 % test and 90 % train set).  

```{r include=FALSE}
chooseCRANmirror(graphics=FALSE, ind=1)
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")
cat("\f")
install.packages("data.table")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(data.table)
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
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1,list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, removed)
```

Raw Data
========

To understand the main challenge in building such a recommendation system it is a good idea to explain first why it is important to include movie and user specific effects. Below you can find some examples of the mean rating of some movies. There are huge differences in their mean rating. It lays therefore in the nature of the problem that not all movies have the same properties (different genres, different actors, different quality/budget, different landscapes,etc.). Hence, different movies will be rated differently simply because of the reason that they are different. Thus, it is important to account for differences in movies. 

\newpage

```{r include=FALSE}
m<-movielens %>% group_by(movieId) %>% summarize(mean_rating=mean(rating)) %>% select(movieId,mean_rating) %>% distinct() %>% slice(1,5000,6000,7000,8000,9000)
```
```{r echo=FALSE}
m %>% knitr::kable(caption="Some Examples - Mean Of Rating By Movie X")
```
The same can also be observed for users when looking to their mean rating patterns. User differences can occur because of their different preferences, different nationality, different intelligence levels, etc. All this leads to the reason that not all users are the same. Some prefer comedies for example more than thrillers or some dislike french movies, etc. Therefore, it is also important to account for these differences in the mean rating pattern in terms of user specific effects.
```{r include=FALSE}
u<-movielens %>% group_by(userId) %>% summarize(mean_rating=mean(rating)) %>% select(userId,mean_rating) %>% distinct() %>% slice(1,10000,20000,30000,40000,50000,60000,70000)
```
```{r echo=FALSE}
u %>% knitr::kable(caption="Some Examples - Mean Of Rating By User X")
```
Furthermore, users do not only differ because of their preferences and movies do not only depend on their natural properties but there are also differences in the rating activities within movies and within users. Thus, Figure "Movies" and "Users" show the distribution of the rating activity by movies and users specifically. For example in "Movies" we can see that most movies are rated between 10 and 3000 times (Still a normal distribution). Only a small part of movies are rated less than 5 times or by more than 10.000 (more outliers). A bit more differently looks the distribution of "Users". The part of users with only a few of ratings seems to be evenly distributed to the part of users which already have some experience (close to 100 rated movies). Only after reaching this critical point of 100 rated movies, users begin rapidly to decrease.
In general, these two figures already show that there is specific rating activity for every user and for every movies. Furthermore, the number of users with very little experience (very few observations) is much larger compared to the movie effect. In other words, users with very few ratings gain much more weight in the data set compared to movies with very few ratings. The question then is, shall we trust these inexperienced users? Probably not and this analysis will account for this problem in terms of penalty term that we will see later. 

\newpage

```{r include=FALSE}
movielens<-movielens%>%mutate(wholestar=ifelse(rating==0 | rating==1 | rating==2 | rating==3 |rating==4 | rating==5,1,0))
edx<-edx%>%mutate(wholestar=ifelse(rating==0 | rating==1 | rating==2 | rating==3 |rating==4 | rating==5,1,0))
validation<-validation%>%mutate(wholestar=ifelse(rating==0 | rating==1 | rating==2 | rating==3 |rating==4 | rating==5,1,0))
edx %>% group_by(wholestar) %>% summarize(n=n())%>% arrange(desc(n))
validation %>% group_by(wholestar) %>% summarize(n=n())%>% arrange(desc(n))
```
```{r echo=FALSE}
movielens %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")
movielens %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")
```

\newpage

As already mentioned in the introduction, it is also useful to look to some general effects that might "bias" the results. The word "bias" is often a sample property or a sample problem (Sample does not entirely represent the true population). In the context of this analysis, we look to the difference between half and full star ratings. Table "General Bias - Half(0) Compared To Full(1) Star Rating" shows that there is a clear discrepancy in the number of observations and their mean value between half and full star ratings in the data. Why can this be? First, we can once again get the impression that it depends on users because they probably don`t know that they can also give a half star rating. But we will see later, that after including user specific effects, the difference still remains so that it is more a general problem. A second reason therefore could be that it is for example related to general rating terms/conditions and sometimes they apply and sometimes they do not apply. We can here think about a yearly cut in the rating terms/conditions. For example, in year XXXX a half star rating has been introduced and therefore it was not possible to give a half star rating before. This could explain the huge gap in observations between half and whole star ratings. Despite the fact that it is not clear why there is a discrepancy in observations and the mean value between half and full star ratings, it seems to be a sample issue (for example related to year of the rating). Therefore it is important to take this variation in this analysis specifically into account.
```{r include=FALSE}
w<-movielens %>% group_by(wholestar) %>% summarize(number_of_observations=n(),mean_value=mean(rating))
```
```{r echo=FALSE}
w %>% knitr::kable(caption="General Bias - Half(0) Compared To Full(1) Star Rating")
```


\newpage

Model/Method
============
With all this insights in mind gained by the data exploration, the analysis looks for the following model (See also Irizarry, 2019):

rating(i,u) = a + b(i) + b(u) + c + Error_term(i,u)

where a:= the overall mean of the general rating pattern

and   b(i):= Different coefficient/Mean value for every movie

and   b(u):= Different coefficient/Mean value for every user 

and   c:= Different coefficient/Mean value for half and whole star ratings  

and   Error_term(i,u):=independent error term for every movie and every user

The idea is to measure the user (movie) specific effects by the mean of all ratings given by each user (for each movie). Later, the model is adapted by a penalty term for specifically user effects because of the mentioned large amount of inexperienced raters that has already been mentioned in the data visualization. The penalty term takes place in the averaging effect of every users overall ratings in terms of parameter lambda. Because of this effect, a single rating will not only be out weighted by the total number of ratings of this user but also by a parameter lambda. Hence, this effect becomes precisely then large when a user only rates a couple of movies (or only just one). In general we can use the following adaptation:

b(u)_hat = 1/(n(u) + lambda) * sum(rating - (a + b(i))

where n(u):= user specific sample size 

and lambda:= penalty parameter

b(u)_hat;= estimated values of user specific effects

All other components have already been explained above. The choice of the value of lambda will be calculated by cross validation and leads to an optimal choice of lambda equal to 5.5.

NB: We will later see that the analysis also includes a penalty term for movies but the decrease in RMSE is only small (b(i)_hat = 1/(n(u) + lambda)*sum(rating - a).

\newpage

Results
=======

Despite the rough elaboration of the final model in the last chapter, the analysis itself takes a more step-wise process. You can see below the results of a model that neither includes a penalty term nor a a variable for whole star ratings. The first model includes only the overall mean as regressor, the second one adds a movie specific effect and the third model also adds the user specific effect. Because the accuracy (measured by the RMSE) is increasing by each adaptation (RMSE goes down), the steps chosen are reasonable. 
```{r include=FALSE}
mu_hat <- mean(edx$rating)
naive_rmse <- RMSE(validation$rating, mu_hat)
rmse_results <- data_frame(method = "Model 1 - Only Overall Variation", RMSE = naive_rmse)
mu <- mean(edx$rating) 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i
model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 2 - Movie Effect Model",
                                     RMSE = model_2_rmse ))
library(knitr)
library(dslabs)
rmse_results %>% knitr::kable()
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
model_3_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 3 - Movie And User Effects Model",  
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()
```
```{r echo=FALSE}
rmse_results %>% knitr::kable()
```
Nevertheless, to see if the model is also doing well in detail we can look to the largest errors/residuals when movie and user specific effects are included. Table "Ten Largest Mistakes When Using User And Movie Specific Effects Without Penalty Term" clearly shows that the largest mistakes are reasonable. For example a block buster movie like Lord of the Rings I (awarded by several golden globe prices) has a mistake given by -4.70 which means that it has a high predicted rating (high mean rating) and gets only such a large error because there were some few users (probably one) who dislike this movie. All other movies of this table can be explained in the same way. Therefore our model already performs well without penalty term incorporation or without the use of a whole star variable. 
```{r echo=FALSE}
validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(residual = rating - (mu + b_i + b_u)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% slice(1:10) %>% knitr::kable(caption="Ten Largest Mistakes When Using User And Movie Specific Effects Without Penalty Term")
```

\newpage

Nevertheless, let`s look whether the model can perform better in including the mentioned penalty term for user and movie effects and also the whole star variable. Looking to the table below, we can always see an improvement although the impact of including a penalty for movies is the smallest one (Improvement only by 0.00016 units from model 4 to model 5). This stays in accordance with the insights gained by the raw data elaboration. Whereas movies only has a small number of outliers (few movies were rated seldom or very often), the impact of inexperienced raters on the sight of users is much larger. This explains the only small win in accuracy from model 4 (RMSE: 0.86497) to model 5 (RMSE: 0.86481). Furthermore, the incorporation of the whole star variable also has its desired effect, the RMSE goes down to 0.8639. 
```{r include=FALSE}
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
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 4 - Movie And User Effects Model with Regularization Only For User",  
                                     RMSE = model_4_rmse ))
rmse_results %>% knitr::kable()
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
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 5 - Movie And User Effects Model with Regularization For User And Movie",  
                                     RMSE = model_5_rmse ))
rmse_results %>% knitr::kable()
memory.limit(16000)  
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
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 6 - Movie And User Effects Model With Regularization For User And Movie, Additionally Wholestar Variable",  
                                     RMSE = model_6_rmse ))
```
```{r echo=FALSE}
rmse_results %>% knitr::kable()
```
A last verification of the residuals shows that the predicted mistakes are still reasonable (Almost the same list as seen before but the values are slightly changing-the errors get smaller). The final model (Model 6) performs best and is therefore the most preferred model of this analysis.
```{r echo=FALSE}
validation %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  left_join(wholestar_avg, by='wholestar') %>%
  mutate(residual = rating - (mu + b_i + b_u + w)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% slice(1:10) %>% knitr::kable(caption="Ten Largest Mistakes When Using User and Movie Specific Effects + Penalty Term + Whole Star Variable")
```

\newpage

Conclusion
==========

Building a recommendation system for movies is far from being easy because we have different users (different preferences) and different movies (different genres, different actors, etc.). Furthermore, not all movies and users represent the same rating activity. Although the data are very handy in this data set, we could directly observe that there are different patterns in the rating activity between movies and users. Whereas movies tend to be more normally distributed around a certain amount of ratings, the same is not true for users. The shape of the user curve was much more left skewed and therefore also users with few ratings (inexperienced users) gain much more weight. The solution to this was the incorporation of a penalty term that has the desired effect only when the number of ratings per user are very small. The result directly improved in accuracy (measured by the RMSE) even when this penalty term was used for both, movies and users. The last step then added a whole star variable which differed between half and whole star raters. The large amount of whole star rater (observed in the data visualization) already let assume that there is an abnormality in the data and we should control for this effect in this analysis (as a control variable). Fortunately, it decreased the RMSE still by some units. Despite the fact, that the reason for this difference between half and whole star raters are not clear for the moment (Different rating conditions over the years?), the incorporation of this variable certainly decreased the variation in the model. The final result of the RMSE was close to 0.8639.

When we talk about limitations, we can still keep going with the last mentioned point. The weakness of the preferred result is that we still do not know why there is a difference between half and whole star rater? Because this was a general bias (user and movie specific effects already incorporated), it seems to depend more on years or another factor. Probably there was a cut in the rating terms in one year. This could explain that rating patterns in general are a bit higher or lower compared to another year, simply because of rating condition changes over the years. A second weakness, is that this is still a robust model but not really precise. We only included specific effects for users and movies and the whole star variable but there are still differences by genre, by an interaction between movies and users, by years of ratings, etc. Therefore the model can be applied in general but has to be adapted in cases where the population is quite specific. For example, when you want to apply this model to a country specific recommendation system (for example for television), it has to be adapted once again (Probably because of national movie and actor bias). A third weakness is the data set itself. For example we don`t have many user information. At least the address of the user could give some important information or the year when the rating was given. Therefore, future research can not only be done in the model improvement (more factor analysis) but also in data collection. More user information is an important point in this aspect.

Finally, to answer the question of the introduction how fast we can get a relatively precise model, this analysis could give robust results in only accounting for an overall, a user specific and a movie specific effect, adapted for a penalty term for outliers in the user and movie rating activity. In the end, the whole star variable also helped to reduce the RMSE quite efficiently (few computation time with large precision effect) but one can assume that this was only a sample problem specifically related to the data set. Hence, a well performing recommendation system certainly depends on the sample variation (national movie recommendation system for television would certainly look differently). Though, the variables chosen to perform the final model in this analysis already help to explain a large proportion of the variation in a large movie recommendation model (with a large set of movies). 

References
==========

Rafael A. Irizarry (2019), Introduction to Data Science: Data Analysis and Prediction Algorithms with R

https://www.edx.org/professional-certificate/harvardx-data-science

https://movielens.org/

https://grouplens.org/datasets/movielens/10m/
