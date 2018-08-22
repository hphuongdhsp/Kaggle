# Kaggle Santa

This repositories include all of codes in the Santa Competition https://www.kaggle.com/c/santander-value-prediction-challenge

This competition is just a hunting leak problem, that means you can find directly some targets of test set from certan columns of train set. 
My solution include 2 similar models, lgbm and xgboost. It easy to get 1.37 in LP without leakm 5 mins trainng. After that I blending 2 solutions  without leak and fill the leakly columns. With the simple code, I get 0.51 in LB and got silve.  
With leaky teachnic hunting you can find in https://www.kaggle.com/nulldata/jiazhen-to-armamut-via-gurchetan1000-0-56

Some descritions of my solution

The Santa data have more than 5000 columns and have only about 5000 rows. It is not too much train by NN, other size, finding the good fearture is very difficult. From the kernel https://www.kaggle.com/ogrellier/santander-46-featuresm I have selected 46 model and use some mean, std, min, max, ... from 46 famous columns. I also took mean, std, min, max of all of colums. To do it I split our data into 2 categories, more than 70% 0 in each column and less than 70% in each column. By this trick, I can improve a little bit, 0,001 score. Because of the smallness of the data, I used K-fold to train my modelm also change seed parameter in LGBM,XGBM model.

The solution is very basic. We can improve our score by some trick.
[1] Because the data is leaky, then we can use all leaky rows in test set  into train set.
[2] 46 colums is similar to 40 columns in the leaky hunting, then maybe it is not good for ML, we can use some reduce dimension technique (FE) to get better score. 

