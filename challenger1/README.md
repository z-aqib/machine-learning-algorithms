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