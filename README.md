# Kaggle Santa

This repository include all of codes in the Santa Competition https://www.kaggle.com/c/santander-value-prediction-challenge

The main intersting I found in this competition is a leak hunting problem. More presisely, you can find exactly some targets of test set from certain columns of train set. It made a huge boosted in the leaderboard.
For this comp, I used 2 similar models, ( coded by lgbm and xgboost libraries). We are easy to get 1.37 (LB) after 5 mins training. Then I blended 2 models togethrer and fill the leakly columns. With the simple code, I get 0.51 in LB and got silve medal (top 5% of compititors).  
You can find yhe leaky hunting algorithm in  https://www.kaggle.com/nulldata/jiazhen-to-armamut-via-gurchetan1000-0-56

## Some descritions of my solution:

The data of Santa comp has more than 5000 columns and has 5000 rows in train set (tabular). I seem that we need to find what is the good fearture from 5000 columns given. Using the clustering algorithm (t-NES), we selected 46 important columns  https://www.kaggle.com/ogrellier/santander-46-features. We then take the mean, std, min, max, ... of the 46 above columns. I also took the mean, std, min, max of all of 5000 colums. It gave a little improvement.  From the fact that almost of tarbulr is zeros, the I split the data into 2 categories,  once has more than 70% 0 in each column and another one  is less than 70% in each column. By this trick, I improved a little bit, 0,001 score. Finally, I split k-fold to train my model. 

The solution is simple. Further, we can improve the score by some tricks.

 [1] Using the data leaky. As above, some datas in the test set is found exactly without ML model, then we can put these datas into the train set. It make us have more datas to train. 
 

