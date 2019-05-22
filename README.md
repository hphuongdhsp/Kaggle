# Kaggle Santa

This repository include all of codes in the Santa Competition https://www.kaggle.com/c/santander-value-prediction-challenge

The main interesting I found in this competition is a leak hunting problem. More precisely, you can find exactly some targets of the test set from certain columns of the train set. It made a huge boosted in the leaderboard. For this comp, I used 2 similar models, ( coded by lightgbm and xgboost libraries). We are easy to get 1.37 (LB) after 5 mins training. Then I blended 2 models together and fill the leaky columns. With the simple code, I get 0.51 in LB and got silver medal (top 5% of competitors).  
You can find the leaky hunting algorithm in  https://www.kaggle.com/nulldata/jiazhen-to-armamut-via-gurchetan1000-0-56

## Some descritions of my solution:

The data of Santa comp has more than 5000 columns and has 5000 rows in train set (tabular). I seem that we need to find what is the good feature from 5000 columns given. Using the clustering algorithm (t-NES), we selected 46 important columns  https://www.kaggle.com/ogrellier/santander-46-features. We then take the mean, std, min, max, ... of the 46 above columns. I also took the mean, std, min, max of all of 5000 columns. It gave a little improvement.  From the fact that almost of tabula is zeros, then I split the data into 2 categories,  one has more than 70% 0 in each column and another one is less than 70% in each column. By this trick, I improved a little bit, 0,001 scores. Finally, I split k-fold to train my model. 



The solution is simple. Further, we can improve the score with some tricks.


[1] Using the data leaky. As above, some data in the test set is found exactly without ML model, then we can put these data into the train set. It makes us have more data to train. 
 

