# All algos
| Day | Date | Algorithm | No. of Entries | Highest Accuracy |
| --- | ---- | --------- | -------------- | ---------------- |
| 1 | Monday 21st October 2024 | Decision Tree | 10 | 0.88788 |
| 2 | Tuesday 22nd October 2024 | Decision Tree | 10 | 0.89330 | 
| 3 | Wednesday 23rd October 2024 | Decision Tree | 10 | 0.89522 |
| 4 | Thursday 24th October 2024 | NaiveBayes | 10 | 0.87148 |
| 5 | Friday 25th October 2024 | NaiveBayes + K-Nearest Neighbour | - | - |
| 6 | Saturday 26th October 2024 | Random Forest | - | - |
| 7 | Sunday 27th October 2024 | Random Forest | - | - |
| 8 | Monday 28th October 2024 | Gradient Boosting | - | - |
| 9 | Tuesday 29th October 2024 | Adaptive Boosting | - | - |
| 10 | Wednesday 30th October 2024 | Light GBM | - | - |
| 11 | Thursday 31st October 2024 | XGBoost | - | - |
| 12 | Friday 1st November 2024 | XGBoost | - | - |
| 13 | Saturday 2nd November 2024 | CatBoost | - | - |
| 14 | Sunday 3rd November 2024 | BaggingClassifier | - | - |
| 15 | Monday 4th November 2024 | ExtraTree Classifier | - | - |
| 16 | Tuesday 5th November 2024 | Voting | - | - |
| 17 | Wednesday 6th November 2024 | Stacking | - | - |
| 18 | Thursday 7th November 2024 | - | - | - |
| 19 | Friday 8th November 2024 | - | - | - |
| 20 | Saturday 9th November 2024 | - | - | - |
| 21 | Sunday 10th November 2024 | - | - | - |

o	Decision Tree
o	Naive Bayes
o	K-Nearest Neighbor
o	Random Forest
o	Gradient Boosting
o	Adaptive Boosting
o	Light GBM
o	XGBoost
o	CatBoost
o	BaggingClassifier
o	ExtraTree Classifier (Extremely Randomized Tree)
o	Voting
o	Stacking

# DecisionTrees

### Analyzing decision trees
| case number | scaler | imputer | splitting | criteria | max depth | min samples split | max features | min samples leaf | roc | accuracy | analysis |
| ----------- | ------ | ------- | --------- | -------- | --------- | ----------------- | ------------ | ---------------- | --- | -------- | -------- |
| 1 | minmax | row removal | holdout 70-30 | gini | 7 | 20 | - | - | 0.5230564082198742 | 0.72913 | - |
| 2 | minmax | row removal | holdout 70-30 | entropy | 7 | 20 | - | - | 0.5229797007820421 | 0.83323 | improved on this criteria |
| 3 | minmax | row removal | holdout 70-30 | entropy | 7 | 15 | - | - | 0.5203814609840954 | 0.83327 | improved minutely of 5dp on lesser samples per split | 
| 4 | minmax | row removal | holdout 70-30 | entropy | 8 | 15 | - | - | 0.5306000850899 | 0.78883 | deteriorated, longer trees resulted in overfit; test had bad performance |
| 5 | minmax | row removal | holdout 70-30 | entropy | 6 | 15 | - | - | 0.5204372482116096  | 0.85532 | lowering depth shot the improvement up |
| 6 | minmax | row removal | holdout 70-30 | entropy | 5 | 15 | - | - | 0.5230843018336313 | 0.87815 | lesser depth improved tree | 
| 7 | minmax | row removal | holdout 70-30 | entropy | 4 | 15 | - | - | 0.5025494259738718 | 0.87296 | depth is too low, accuracy deteriorated at 3dp | 
| 8 | minmax | row removal | holdout 70-30 | entropy | 5 | 15 | 10 | - | 0.5154151037016982 | 0.83207 | need to use more features to improve |
| 9 | minmax | row removal | holdout 70-30 | entropy | 5 | 15 | 50 | - | 0.5128726511312657 | 0.86817 | more features improved tree but to the full extent |
| 10 | minmax | row removal | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.5205418492631988 | 0.88788 | increasing features improved highly | 
| 11 | minmax | simple | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.5184642260176702 | 0.89330 | - | 
| 12 | minmax | knn5 | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.5138345831202585 | 0.87791 | simple performs better on minmax | 
| 13 | standard | simple | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.5086297947750887 | 0.77620 | - | 
| 14 | standard | knn3 | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.5151673183025117 | 0.80450 | knn performs better on standard | 
| 15 | maxabs | simple | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.5071707936463162 | 0.88454 | - | 
| 16 | maxabs | knn7 | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.523861491669948 | 0.88389 | both simple and knn perform well with maxabs | 
| 17 | robust | simple | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.5181612733948899 | 0.88517 | - | 
| 18 | robust | knn5 | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.5173654042244641 | 0.87303 | simple performs better on robust | 
| 19 | normalizer | simple | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.512349538904424 | 0.64969 | - | 
| 20 | normalizer | knn7 | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.5204856620977856 | 0.65308 | normalizer is not a good scaler | 
| 21 | minmax | knn3 | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.510941492151186 | 0.89142 | - | 
| 22 | minmax | knn7 | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.5100979659586905 | 0.88134 | in minmax, simple performed best | 
| 23 | maxabs | knn3 | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.5200733426222888 | 0.88155 | - | 
| 24 | maxabs | knn5 | holdout 70-30 | entropy | 5 | 15 | 60 | - | 0.5133797612483052 | 0.84670 | in maxabs, simple performed just 3dp better then knn=7 | 
| 25 | maxabs | knn7 | holdout 70-30 | entropy | 5 | 15 | 60 | 50 | 0.5150632467068051 | 0.85877 | - | 
| 26 | maxabs | knn7 | holdout 70-30 | entropy | 5 | 15 | 60 | 80 | 0.5 | 0.89522 | BEST CASE: highest accuracy | 
| 27 | maxabs | knn7 | holdout 70-30 | entropy | 5 | 15 | 60 | 100 | 0.5 | 0.85276 | worser performance when too many samples on leaf | 
| 28 | maxabs | knn7 | crossfold k=10 | entropy | 5 | 15 | 60 | 80 | 0.8964 | 0.89174 | near to best accuracy | 
| 29 | maxabs | knn7 | crossfold k=5 | entropy | 5 | 15 | 60 | 80 | 0.8812 | 0.88935 | decreasing k didnt change accuracy too much | 
| 30 | maxabs | knn7 | crossfold k=15 | entropy | 5 | 15 | 60 | 80 | 0.8994 | 0.87891 | increasing k resulted in overfit | 

highest accuracy achieved: 0.89522   
started accuracy: 0.72913   
best parameters: (case 26)    
- max absolute scaler
- knn=7 imputation
- entropy criteria 
- holdout method, at 70-30 ratio
- maximum 5 depth tree
- minimum 15 samples split
- maximum 60 features used
- minimum samples per leaf is 80

analyzed best:    
- minmax and maxabs is the best scalers
- normalizer is the worst scaler
- simple and knn=7 perform best, knn=3 performs a bit lesser
- depth of tree is good at 5, 6, while 4, 8 leads to underfit/overfit
- too less features used like 10 and 50 is bad
- too many samples on a leaf like 100 result in underfit. 80 is the breakpoint. smaller then 80 result in overfit
- cross fold performs best at k=10
- entropy is better then gini

DAY 1: Monday 21st October 2024

## Case 1 - started
- DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=20), 
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal, 
- minmax scaling, 
- no grid

roc score = 0.5230564082198742   
Accuracy: 0.72913 

### ERROR SOLUTION
one error approached was that we had first splitted the data into features and target variables (X, Y) and then dropped all the NaN rows from X which made X and Y have differnet number of rows. after trying to "index" and match these rows, nothing was working. So a solution was found to restart the ipynb and first remove all the NaN rows from the train_data_processed and THEN split it into X, Y which resulted in same rows and this code line worked. 

## Case 2 - changed criteria from gini to entropy
- DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=20)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal,
- minmax scaling,
- no grid

roc score = 0.5229797007820421   
Accuracy: 0.83323  

### Analyzing
entropy performed better then gini

## Case 3 - decreased no.of samples per split
- DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=15)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5203814609840954    
accuracy: 0.83327

### Analyzing
splitting on smaller samples is better

## Case 4 - max_depth increased
- DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=15)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5306000850899    
accuracy: 0.78883

### Analyzing
higher depth trees arent good

## Case 5 - max_depth decreased
- DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=15)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5204372482116096    
accuracy: 0.85532

## Case 6 - max_depth decreased
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5230843018336313    
accuracy: 0.87815

## Case 7 - max_depth decreased
- DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=15)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5025494259738718    
accuracy: 0.87296

### Analyzing
smaller depth trees are good, but too small arent. keep till 5 depth

## Case 8 - max_features introduced
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=10)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5154151037016982    
accuracy: 0.83207

## Case 9 - max_features increased
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=50)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5128726511312657    
accuracy: 0.86817

## Case 10 - max_features increased
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5205418492631988    
accuracy: 0.88788

### Analyzing
using less features isnt good. around 60 features improves accuracy

DAY 2: Tuesday 22nd October 2024

## Case 11 - simple, minmax
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- SimpleImputer(strategy='mean')
- minmax scaling
- no grid

roc score =  0.5184642260176702   
accuracy: 0.89330

## Case 12 - knn5, minmax 
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=5)
- minmax scaling
- no grid

roc score =  0.5138345831202585      
accuracy: 0.87791

### Analyzing
MinMaxScaler average: 0.885605

## Case 13 - simple, standard
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- SimpleImputer(strategy='mean')
- StandardScaler()
- no grid

roc score =  0.5086297947750887    
accuracy: 0.77620   

## Case 14 - knn3, standard
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=3)
- StandardScaler()
- no grid

roc score =  0.5151673183025117     
accuracy: 0.80450

### Analyzing
StandardScaler average: 0.79035

## Case 15 - simple, maxabse
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- SimpleImputer(strategy='mean')
- MaxAbsScaler()
- no grid

roc score =  0.5071707936463162    
accuracy: 0.88454

## Case 16 - knn7, maxabs
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=7)
- MaxAbsScaler()
- no grid

roc score =  0.523861491669948   
accuracy: 0.88389

### Analyzing
MaxAbsScaler average: 0.884215

## Case 17 - simple, robust
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- SimpleImputer(strategy='mean')
- RobustScaler
- no grid

roc score =  0.5181612733948899   
accuracy: 0.88517

## Case 18 - knn5, robust
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=5)
- RobustScaler
- no grid

roc score =  0.5173654042244641   
accuracy: 0.87303

### Analyzing
RobustScaler average: 0.87910

## Case 19 - simple, normalizer
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- SimpleImputer(strategy='mean')
- Normalizer
- no grid

roc score =  0.512349538904424   
accuracy: 0.64969

## Case 20 - knn7, normalizer
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=7)
- Normalizer
- no grid

roc score =  0.5204856620977856   
accuracy: 0.65308

### Analyzing
NormalizerScaler average: 0.651385

### Analyzing
| Scalers / Imputers | SimpleImputer | KNN = 3 | KNN = 5 | KNN = 7 | Average  | New Average |
| ------------------ | ------------- | ------- | ------- | ------- | -------- | ----------- |
| MinMaxScaler       | 0.89330       | 0.89142 | 0.87791 | 0.88134 | 0.885605 | 0.883556667 |
| StandardScaler     | 0.77620       | 0.80450 | -       | -       | 0.79035  | -           |
| MaxAbsScaler       | 0.88454       | 0.88155 | 0.84670 | 0.88389 | 0.884215 | 0.870713333 |
| RobustScaler       | 0.88517       | -       | 0.87303 | -       | 0.87910  | -           |
| Normalizer         | 0.64969       | -       | -       | 0.65308 | 0.651385 | -           |

the best among all 5 scalers is MinMaxScaler and MaxAbsScaler. the third best is RobustScaler, after that StandardScaler is lower significantly and NormalizerScaler is very very low. Hence we shall be alternating between MinMaxScaler and MaxAbsScaler as they are only differnet in the third decimal point.    

out of KNN and SimpleImputers, we can see that both are good however simple imputer performs better on average. thus we will work with both. in the next 4 cases, lets test knn=3, 5, 7 for MinMaxScaler and MaxAbsScaler to find the best KNN going forward.

DAY 3: Wednesday 23rd October 2024

## Case 21 - knn3, minmax
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=3)
- minmax scaling
- no grid

roc score =  0.510941492151186      
accuracy: 0.89142

## Case 22 - knn7, minmax
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=7)
- minmax scaling
- no grid

roc score =  0.5100979659586905   
accuracy: 0.88134

## Case 23 - knn3, maxabs
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=3)
- MaxAbsScaler
- no grid

roc score =  0.5200733426222888    
accuracy: 0.88155

## Case 24 - knn5, maxabs
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=5)
- MaxAbsScaler
- no grid

roc score =  0.5133797612483052   
accuracy: 0.84670

### analyzing
so according to updated table, knn=7 is best on average and we can alternate between MaxAbs and MinMax however MinMax will be prioritised.

## Case 25 - min_samples_leaf introduced
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=50)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=7)
- MaxAbsScaler
- no grid

roc score =  0.5150632467068051   
accuracy: 0.85877

## Case 26 - min_samples_leaf increased
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=7)
- MaxAbsScaler
- no grid

roc score = 0.5
accuracy: 0.89522

## Case 27 - min_samples_leaf increased
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=100)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=7)
- MaxAbsScaler
- no grid

roc score = 0.5
accuracy: 0.85276

### analyzing
min_samples_leaf is fine at 80

## Case 28 - crossfold at k=10
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80)
- crossfold = RepeatedKFold(n_splits=10, n_repeats=1)#, random_state=1)
- KNNImputer(n_neighbors=7)
- MaxAbsScaler
- no grid

'0.8964'   
accuracy: 0.89174

## Case 29 - crossfold at k=5
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80)
- crossfold = RepeatedKFold(n_splits=5, n_repeats=1)#, random_state=1)
- KNNImputer(n_neighbors=7)
- MaxAbsScaler
- no grid

'0.8812'
accuracy: 0.88935

## Case 30 - crossfold at k=15
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80)
- crossfold = RepeatedKFold(n_splits=15, n_repeats=1)#, random_state=1)
- KNNImputer(n_neighbors=7)
- MaxAbsScaler
- no grid

'0.8994'
accuracy: 0.87891

### analyzing
k-fold is best at k = 10. it does give good accuracy like holdout, we can use both of them alternatively.

DAY 4: Thursday 24th October 2024

# NaiveBayes

### Analyzing Naive Bayes
| case number | imputer | scaler | feature selection | no. of features selected | model accuracy | roc | kaggle accuracy | analysis |
| ----------- | ------- | ------ | ----------------- | ------------------------ | -------------- | --- | --------------- | -------- |
| 31 | simple | maxabs | - | 78 | 0.8670720641412841 | 0.8015837569508879 | 0.83725 | need to test a bit more to see what was lacking and what to improve |
| 32 | simple | minmax | - | 78 | 0.9109254167964571 | 0.7956827745161095 | 0.83350 | minmax and maxabs have negligible difference in 3 dp |
| 33 | knn7 | minmax | - | 78 | 0.9121307745439279 | 0.7969697685450362 | 0.83350 | knn7 and simple is same! we must reduce the features to try and improve accuracy |
| 34 | simple | minmax | forward | 5 | 0.9838834188821323 | 0.5964048120149146 | 0.82386 | will need to test it a bit more to deduce | 
| 35a | simple | minmax | forward | 10 | 0.9630943835746306 | 0.6591343574667629 | not submitted | high model accuracy means overfit. not submitted |
| 35b | simple | minmax | forward | 20 | 0.9770711161071007 | 0.6887415307743486 | 0.87148 | imrpoved with more features |
| 36 | simple | minmax | variance=0.1, correlation=0.9 filter | 7 | 0.9972642442136056 | 0.5 | 0.70704 | even though ROC was good, model overfit | 
| 37a | simple | minmax | variance=0.5, correlation=0.9 filter | 0 | - | - | - | error, no columns exist with that variance limit | 
| 37b | simple | minmax | variance=0.3, correlation=0.9 filter | 0 | - | - | - | error, no columns exist with that variance limit | 
| 37c | simple | minmax | variance=0.01, correlation=0.9 filter | 49 | 0.9121849479258366 | 0.7996191732926649 | not submitted | looking for lower ROC |
| 37d | simple | minmax | variance=0.05, correlation=0.9 filter | 14 | 0.9577176754201823 | 0.6956493745514011 | not submitted | too high accuracy, overfit chance |
| 37e | simple | minmax | variance=0.03, correlation=0.8 filter | 19 | 0.9457318146728605 | 0.6916013795163546 | not submitted | too high accuracy, overfit chance |
| 37f | simple | minmax | variance=0.03, correlation=0.9 filter | 20 | 0.9441201565610737 | 0.6819374512237548 | 0.75852 | accuracy improved from last time but we need more features |
| 38 | simple | minmax | variance=0.01, correlation=0.9 filter | 49 | 0.9129975486544686 | 0.7505591957401304 | 0.81494 | accuracy improved due to number of features, will use more in next round |
| 39a | simple | minmax | variance=0.005, correlation=0.9 filter | 57 | 0.9132142421821038 | 0.7612309576110358 | not submitted | model accuracy is too high, overfit chance | 
| 39b | simple | minmax | variance=0.001, correlation=0.9 filter | 61 | 0.9134715657461706 | 0.758662810781139 | 0.82598 | accuracy improved, need to increase more rows |
| 40a | simple | minmax | variance=0.0001, correlation=0.9 filter | 66 | 0.9162479515689965 | 0.7761983869690179 | not submitted | - | 
| 40b | simple | minmax | variance=0.0001, correlation=0.8 filter | 57 | 0.9183200834270081 | 0.7965337080939787 | not submitted | - | 
| 40c | simple | minmax | variance=0.0001, correlation=0.85 filter | 62 | 0.9187128404458469 | 0.7588877586887783 | not submitted | - | 
| 40d | simple | minmax | variance=0.0001, correlation=0.87 filter | 65 | 0.9158958245865894 | 0.7754906950864966 | 0.83187 | good accuracy achieved when more features are used | 
| 41a | simple | minmax | forward | 30 | 0.964340371358533 | 0.7356985869936561 | 0.86669 | forward has been put too high, should try =25 next time |
| 41b | simple | minmax | kbest | 30 | 0.9212183593591289 | 0.7565948275603417 | not submitted | dont know if its good, dont want to waste an entry |
| 42 | simple | minmax | forward | 25 | 0.8100274929913187 | 0.8185403319589406 | 0.86875 | not highest but near |
| 45 | simple | minmax | forward | 15 | 0.9776399366171432 | 0.7000642467303131 | 0.87413 | BEST CASE: improved with lesser features | 

highest accuracy achieved: 0.87413 (case 45)  
started accuracy: 0.83725
- forward is best at 15
- simple, minmax worked best with NB
- NB performed better with more features then lesser   

## Case 31
- naive bayes
- simple imputer
- maxabs scaler
- 78 features

model accuracy =  0.8670720641412841    
roc score =  0.8015837569508879   
accuracy: 0.83725    

## Case 32 - scaler changed to minmax
- naive bayes
- simple imputer
- minmax scaler
- 78 features

model accuracy =  0.9109254167964571    
roc score =  0.7956827745161095    
accuracy: 0.83350

## Case 33 - imputer changed to knn7
- naive bayes
- knn7 imputer
- minmax scaler
- 78 features 

model accuracy =  0.9121307745439279       
roc score =  0.7969697685450362  
accuracy: 0.83350

### analyzing
NO DIFFERENCE! knn7 == simple. we will have to use feature selections now to try and improve the accuracy

## Case 34 - simple, forward selection=5
- naive bayes
- simple imputer
- minmax scaler
- forward=5

model accuracy =  0.9838834188821323    
roc score =  0.5964048120149146   
accuracy: 0.82386

## Case 35a - forward=10
- naive bayes
- simple imputer
- minmax scaler
- forward=10

model accuracy =  0.9630943835746306    
roc score =  0.6591343574667629 

## Case 35b - forward=20
- naive bayes
- simple imputer
- minmax scaler 
- forward=20

model accuracy =  0.9770711161071007    
roc score =  0.6887415307743486   
accuracy: 0.87148

### analyzer 
accuracy improved! meaning forward is doing good but at more features. lets try and implement filters now

## Case 36 - variance and correlation filter
- naive bayes
- simple imputer
- minmax scaler
- variance=0.1, correlation=0.9 filter = 7

model accuracy =  0.9972642442136056    
roc score =  0.5   
accuracy: 0.70704

### analyzer
too much features reduced.. accuracy shot down drastically even though ROC was good

## Case 37a - variance=0.5
- naive bayes
- simple imputer
- minmax scaler
- variance=0.5, correlation=0.9 filter

ERROR occured saying no columns in X exist with variance of 0.5 criteria (correlation has not run yet)

## Case 37b - variance=0.3
same error as variance=0.5

## Case 37c - variance=0.01
61 columns extracted through variance filter and then 49 columns left with correlation filter   
model accuracy =  0.9121849479258366    
roc score =  0.7996191732926649   

## Case 37d - variance=0.05
14 columns extracted through variance and correlation didnt do more   
model accuracy =  0.9577176754201823    
roc score =  0.6956493745514011   

## Case 37e - variance=0.3, correlation=0.8
20 columns extracted through variance and correlation made it 19    
model accuracy =  0.9457318146728605    
roc score =  0.6916013795163546   

## Case 37f - variance=0.3, correlation=0.9
20 columns extracted through variance and correlation didnt do more   
- naive bayes
- simple imputer
- minmax scaler
- variance=0.3, correlation=0.9 filter

model accuracy =  0.9441201565610737    
roc score =  0.6819374512237548    
accuracy: 0.75852

### analyzer
need to increase features, the variance=0.01 seems better now, will try that next

## Case 38 - variance=0.01
- naive bayes
- simple imputer
- minmax scaler
- variance=0.01, correlation=0.9 
- rows = 49 (61 by variance, 49 by correlation)

model accuracy =  0.9129975486544686    
roc score =  0.7505591957401304  
accuracy: 0.81494

### analyzer
wow! accuracy shot up. lets use more features now (we used approx 50 right now)

## Case 39a - variance=0.005
69 by variance, 57 by correlation   
model accuracy =  0.9132142421821038    
roc score =  0.7612309576110358 

## Case 39b - variance=0.001
- naive bayes
- simple imputer
- minmax scaler
- variance=0.001, correlation=0.9 
- rows = 61 (73 by variance, 61 by correlation)

model accuracy =  0.9134715657461706    
roc score =  0.758662810781139  
accuracy: 0.82598

### analyzer
wow! lets increase the rows even further

## Case 40a - variance=0.0001
78 by variance, 66 by correlation  
model accuracy =  0.9162479515689965     
roc score =  0.7761983869690179 

## Case 40b - variance=0.0001, correlation=0.8
78 by variance, 57 by correlation     
model accuracy =  0.9183200834270081    
roc score =  0.7965337080939787  

## Case 40c - variance=0.0001, correlation=0.85
78 by variance, 62 by correlation    
model accuracy =  0.9187128404458469    
roc score =  0.7588877586887783  

## Case 40d - variance=0.0001, correlation=0.87
- naive bayes
- simple imputer
- minmax scaler
- variance=0.0001, correlation=0.87
- rows = 65 (78 by variance, 65 by correlation)

model accuracy =  0.9158958245865894    
roc score =  0.7754906950864966    
accuracy: 0.83187

### analyzer
accuracy improved, but not even to the starting point.. submissions have ended otherwise we could have tested for 70s features.

DAY: Friday 25th October 2024

## Case 41a - forward=30
- naive bayes
- simple imputer
- minmax scaler
- forward=30 rows

model accuracy =  0.9638392675758766    
roc score =  0.7566986273209292    

model accuracy =  0.9562820807995991    
roc score =  0.7497710932217062 

model accuracy =  0.964340371358533    
roc score =  0.7356985869936561    
accuracy: 0.86669   

## Case 41b - kbest=30
kbest=30 rows    
model accuracy =  0.9212183593591289    
roc score =  0.7565948275603417    
not submitted   

## Case 42 - forward=25
-  naive bayes
- simple imputer
- minmax scaler
- forward=25 rows

model accuracy =  0.9617942224088194    
roc score =  0.7343364208492563    
-- samplesubmission.csv   

model accuracy =  0.8100274929913187    
roc score =  0.8185403319589406   
accuracy: 0.86875   
-- samplesubmission2.csv    

model accuracy =  0.9550090063247424    
roc score =  0.7018435381296126    
-- nb1.csv   

# K Nearest Neighbours

### Analyzing K Nearest Neighbours
| case number | K used | imputer | scaler | feature selector | features used | validation accuracy | roc | kaggle accuracy | analysis | 
| ----------- | ------ | ------- | ------ | ---------------- | ------------- | ------------------- | --- | --------------- | -------- |
| 1 | 5 | simple | minmax | - | 78 | 0.9972642442136056 | 0.5 | 0.53003 | so low, lets try k=7 and k=11 to improve |

## Case 43 - k=5
- KNeighborsClassifier(k=5)
- simple imputer
- minmax scaling
- no feature selection, 78 features used

model accuracy =  0.9972642442136056    
roc score =  0.5    
accuracy: 0.53003

## Case 44a - k=7
-- running for 360 minutes
-- knn1.csv

## Case 44b - k=7, forward=10
-- running for 316 minutes
-- knn2.csv

## Case 44d - k=7, maxabs, knn=7, variance=0.0001, corr=0.87
78 by variance, 65 by correlation
model accuracy =  0.9973455042864688    
roc score =  0.5
-- knn4.csv

## Case 44e - simple, variance=0.1, corr=0.9
5 by variance, none by coorelation
model accuracy =  0.9974267643593321    
roc score =  0.5  
-- knn4.csv

### Analyzing
KNN + forward is running since past 6 hours, we are shifting to RandomForest.

# Random Forest

### Analyzing RandomForest
| case number | imputer | scaler | max depth | n estimators | features used | criteria | min samples split | max features | min samples leaf | validation accuracy | roc | kaggle accuracy | analyzing |
| ----------- | ------- | ------ | --------- | ------------ | ------------- | -------- | ----------------- | ------------ | ------------ | ------------------- | --- | --------------- | --------- |
| 44 | simple | maxabs | 10 | 200 | 78 | default = gini | - | - | - | 0.9973319609409916 | 0.5050251256281407 | 0.90507 | ok good, now lets used the best parameters that we found from decision trees |

## Case 44 - simple, maxabs, max_depth=10, n_estimators=200
- RandomForestClassifier(max_depth=10, n_estimators=200)
- simple imputer
- maxabs scaler

model accuracy =  0.9973319609409916    
roc score =  0.5050251256281407   
accuracy: 0.90507

## Case 45 - naivebayes, forward=15
- naive bayes
- simple imputer
- minmax scaler
- forward=15 rows

model accuracy =  0.9776399366171432    
roc score =  0.7000642467303131 
accuracy: 0.87413

## Case 46 - knearestneighbours=7, simple, variance=0.001, corr=0.9
- KNeighborsClassifier(k=7)
- simple imputer
- minmax scaling
- variance=0.001, corr=0.9
- 70 by variance, 58 by correlation

model accuracy =  0.9973319609409916    
roc score =  0.5  
accuracy: 0.54796    
-- knn4.csv

## Case 47 - knearest neighbours, k=7, kbest=30
- KNeighborsClassifier(k=7)
- simple imputer
- minmax scaling
- kbest=30
- 30

model accuracy =  0.9975892845050585    
roc score =  0.5028112828678414   
accuracy: 0.57056
-- knn3.csv

## Case 48 - naivebayes, forward = 17
- naive bayes
- simple imputer
- minmax scaler
- forward=17 rows

model accuracy =  0.968064791364763    
roc score =  0.71144752877173 
accuracy: 0.87278

## Case 47 - best decision tree parameters
- RandomForestClassifier(max_depth=5, n_estimators=200, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80)
- maxabs scaler
- knn=7 imputer   

-- waiting for its result....