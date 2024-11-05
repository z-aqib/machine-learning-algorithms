# All algos

Work done daily: 

| Day | Date | Algorithm | No. of Entries | Highest Accuracy | Case of Highest |
| --- | ---- | --------- | -------------- | ---------------- | --------------- |
| 1 | Monday 21st October 2024 | Decision Tree | 10 | 0.88788 | 10 |
| 2 | Tuesday 22nd October 2024 | Decision Tree | 10 | 0.89330 | 11 | 
| 3 | Wednesday 23rd October 2024 | Decision Tree | 10 | 0.89522 | 26 | 
| 4 | Thursday 24th October 2024 | NaiveBayes | 10 | 0.87148 | 35b | 
| 5 | Friday 25th October 2024 | NaiveBayes + K-Nearest Neighbours + Random Forest | 10 | 0.90507 | 44 |
| 6 | Saturday 26th October 2024 | K-Nearest Neighbours + Random Forest | 10 | 0.91889 | 58 | 
| 7 | Sunday 27th October 2024 | Gradient Boosting + Random Forest + K-Nearest Neighbours + Adaptive Boosting | 10 | 0.94475 | 69 |
| 8 | Monday 28th October 2024 | Adaptive Boosting + Random Forest + K-Nearest Neighbours + Gradient Boosting | 10 | 0.94966 | 76 | 
| 9 | Tuesday 29th October 2024 | Adaptive Boosting + K-Nearest Neighbours + Random Forest + Gradient Boosting | 10 | 0.94949 |  81 | 
| 10 | Wednesday 30th October 2024 | Light GBM | - | - |  - |  
| 11 | Thursday 31st October 2024 | XGBoost | - | - | - | 
| 12 | Friday 1st November 2024 | XGBoost | - | - | - | 
| 13 | Saturday 2nd November 2024 | CatBoost | - | - | - | 
| 14 | Sunday 3rd November 2024 | BaggingClassifier | - | - | - | 
| 15 | Monday 4th November 2024 | ExtraTree Classifier | - | - | - | 
| 16 | Tuesday 5th November 2024 | Voting | - | - | - | 
| 17 | Wednesday 6th November 2024 | Stacking | - | - | - | 
| 18 | Thursday 7th November 2024 | Stacking | - | - | - | 
| 19 | Friday 8th November 2024 | PCA | - | - | - | 
| 20 | Saturday 9th November 2024 | - | - | - | - | 
| 21 | Sunday 10th November 2024 | - | - | - | - | 

Algorithms worked on: 

| Algorithm Name | No. of Tries | No. of Submissions | Best Accuracy | Case Number | Imputer | Scaler | Feature Selector | No. of Features | Properties |
| - | - | - | - | - | - | - | - | - | - |
| Decision Tree | 31 | 31 | 0.89522 | 26 | knn=7 | maxabs | - | 78 | DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80), train_test_split(X, Y, test_size=0.3) | 
| Naive Bayes | 31 | 16 | 0.87413 | 45 | simple | minmax | forward | 15 | GaussianNB() |
| K-Nearest Neighbor | 20 | 17 | 0.85212 | 88 | knn=3 | minmax | kbest | 5 | KNeighborsClassifier(n_neighbors=1500, weights="distance") |
| Random Forest | 13 | 12 | 0.93546 | 79 | knn=7 | maxabs | - | 78 | RandomForestClassifier(max_depth=11, n_estimators=400, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80) | 
| Gradient Boosting | 12 | 10 | 0.90485 | 155 | simple | minmax | - | 78 | GradientBoostingClassifier(max_depth=3, criterion='friedman_mse', n_estimators=200), BaggingClassifier(estimator=model, n_estimators=10, verbose=2) | 
| Adaptive Boosting | 17 | 15 | 0.94966 | 76 | simple | minmax | - | 78 | AdaBoostClassifier(n_estimators=170) |
| Light GBM | 22 | 20 | 0.95323 | 126c | simple | maxabs | algorithm feature importance | 20 | lgb.LGBMClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000), BaggingClassifier(estimator=model, n_estimators=50, verbose=2) |
| XGBoost | 18 | 18 | 0.95979 | 138 | simple | maxabs | algorithm feature importance | 35 | xgb.XGBClassifier(), BaggingClassifier(estimator=model, n_estimators=100, verbose=2) |
| CatBoost | 13 | 11 | 0.95270 | 144 | simple | maxabs | algorithm feature importance | 14 | CatBoostClassifier(max_depth=1, n_estimators=2000, learning_rate=0.1), BaggingClassifier(estimator=model, n_estimators=50, verbose=2) |
| BaggingClassifier | - | - | - | - | - | - |
| ExtraTree Classifier (Extremely Randomized Tree) | - | - | - | - | - | - |
| Voting | - | - | - | - | - | - |
| Stacking | - | - | - | - | - | - |

# DecisionTrees

code cleaned and commented, done

## Analyzing Decision Trees
| case number | scaler | imputer | splitting | criteria | max depth | min samples split | max features | min samples leaf | PCA | roc | accuracy | analysis |
| ----------- | ------ | ------- | --------- | -------- | --------- | ----------------- | ------------ | ---------------- | --- | -------- | -------- | - |
| 1 | minmax | row removal | holdout 70-30 | gini | 7 | 20 | - | - | - | 0.5230564082198742 | 0.72913 | - |
| 2 | minmax | row removal | holdout 70-30 | entropy | 7 | 20 | - | - | - | 0.5229797007820421 | 0.83323 | improved on this criteria |
| 3 | minmax | row removal | holdout 70-30 | entropy | 7 | 15 | - | - | - | 0.5203814609840954 | 0.83327 | improved minutely of 5dp on lesser samples per split | 
| 4 | minmax | row removal | holdout 70-30 | entropy | 8 | 15 | - | - | - | 0.5306000850899 | 0.78883 | deteriorated, longer trees resulted in overfit; test had bad performance |
| 5 | minmax | row removal | holdout 70-30 | entropy | 6 | 15 | - | - | - | 0.5204372482116096  | 0.85532 | lowering depth shot the improvement up |
| 6 | minmax | row removal | holdout 70-30 | entropy | 5 | 15 | - | - | - | 0.5230843018336313 | 0.87815 | lesser depth improved tree | 
| 7 | minmax | row removal | holdout 70-30 | entropy | 4 | 15 | - | - | - | 0.5025494259738718 | 0.87296 | depth is too low, accuracy deteriorated at 3dp | 
| 8 | minmax | row removal | holdout 70-30 | entropy | 5 | 15 | 10 | - | - | 0.5154151037016982 | 0.83207 | need to use more features to improve |
| 9 | minmax | row removal | holdout 70-30 | entropy | 5 | 15 | 50 | - | - | 0.5128726511312657 | 0.86817 | more features improved tree but to the full extent |
| 10 | minmax | row removal | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.5205418492631988 | 0.88788 | increasing features improved highly | 
| 11 | minmax | simple | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.5184642260176702 | 0.89330 | - | 
| 12 | minmax | knn5 | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.5138345831202585 | 0.87791 | simple performs better on minmax | 
| 13 | standard | simple | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.5086297947750887 | 0.77620 | - | 
| 14 | standard | knn3 | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.5151673183025117 | 0.80450 | knn performs better on standard | 
| 15 | maxabs | simple | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.5071707936463162 | 0.88454 | - | 
| 16 | maxabs | knn7 | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.523861491669948 | 0.88389 | both simple and knn perform well with maxabs | 
| 17 | robust | simple | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.5181612733948899 | 0.88517 | - | 
| 18 | robust | knn5 | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.5173654042244641 | 0.87303 | simple performs better on robust | 
| 19 | normalizer | simple | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.512349538904424 | 0.64969 | - | 
| 20 | normalizer | knn7 | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.5204856620977856 | 0.65308 | normalizer is not a good scaler | 
| 21 | minmax | knn3 | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.510941492151186 | 0.89142 | - | 
| 22 | minmax | knn7 | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.5100979659586905 | 0.88134 | in minmax, simple performed best | 
| 23 | maxabs | knn3 | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.5200733426222888 | 0.88155 | - | 
| 24 | maxabs | knn5 | holdout 70-30 | entropy | 5 | 15 | 60 | - | - | 0.5133797612483052 | 0.84670 | in maxabs, simple performed just 3dp better then knn=7 | 
| 25 | maxabs | knn7 | holdout 70-30 | entropy | 5 | 15 | 60 | 50 | - | 0.5150632467068051 | 0.85877 | - | 
| 26 | maxabs | knn7 | holdout 70-30 | entropy | 5 | 15 | 60 | 80 | - | 0.5 | 0.89522 | BEST CASE: highest accuracy | 
| 27 | maxabs | knn7 | holdout 70-30 | entropy | 5 | 15 | 60 | 100 | - | 0.5 | 0.85276 | worser performance when too many samples on leaf | 
| 28 | maxabs | knn7 | crossfold k=10 | entropy | 5 | 15 | 60 | 80 | - | 0.8964 | 0.89174 | near to best accuracy | 
| 29 | maxabs | knn7 | crossfold k=5 | entropy | 5 | 15 | 60 | 80 | - | 0.8812 | 0.88935 | decreasing k didnt change accuracy too much | 
| 30 | maxabs | knn7 | crossfold k=15 | entropy | 5 | 15 | 60 | 80 | - | 0.8994 | 0.87891 | increasing k resulted in overfit | 
| 156 | maxabs | knn7 | holdout 70-30 | entropy | 5 | 15 | 60 | 80 | 15 | 0.5 | 0.76281 | wow PCA is bad. lets analyse its graph plot to find the best |

total test cases done: 31   
started accuracy: 0.72913  
highest accuracy achieved: 0.89522    
best parameters: (case 26)    
- scaler = maxabs
- imputer = knn=7
- feature selection = none
- data splitting = holdout, 70-30 ratio
- criteria: entropy
- maximum depth tree: 5
- minimum samples split: 15
- maximum features used: 60
- minimum samples per leaf: 80

analyzed best:    
- minmax and maxabs is the best scalers
- normalizer is the worst scaler
- simple and knn=7 perform best, knn=3 performs a bit lesser
- depth of tree is good at 5, 6, while 4, 8 leads to underfit/overfit
- too less features used like 10 and 50 is bad
- too many samples on a leaf like 100 result in underfit. 80 is the breakpoint. smaller then 80 result in overfit
- cross fold performs best at k=10
- entropy is better then gini

## Analyzing Scalers and Imputers
| Scalers / Imputers | SimpleImputer | KNN = 3 | KNN = 5 | KNN = 7 | Average  | New Average |
| ------------------ | ------------- | ------- | ------- | ------- | -------- | ----------- |
| MinMaxScaler       | 0.89330       | 0.89142 | 0.87791 | 0.88134 | 0.885605 | 0.883556667 |
| StandardScaler     | 0.77620       | 0.80450 | -       | -       | 0.79035  | -           |
| MaxAbsScaler       | 0.88454       | 0.88155 | 0.84670 | 0.88389 | 0.884215 | 0.870713333 |
| RobustScaler       | 0.88517       | -       | 0.87303 | -       | 0.87910  | -           |
| Normalizer         | 0.64969       | -       | -       | 0.65308 | 0.651385 | -           |

the best among all 5 scalers is MinMaxScaler and MaxAbsScaler. the third best is RobustScaler, after that StandardScaler is lower significantly and NormalizerScaler is very very low. Hence we shall be alternating between MinMaxScaler and MaxAbsScaler as they are only differnet in the third decimal point.    

out of KNN and SimpleImputers, we can see that both are good however simple imputer performs better on average. thus we will work with both. in the next 4 cases, lets test knn=3, 5, 7 for MinMaxScaler and MaxAbsScaler to find the best KNN going forward.

# NaiveBayes

code cleaned and commented, done

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
| 42a | simple | minmax | forward | 25 | 0.9617942224088194 | 0.7343364208492563 | not submitted | waiting for a good roc |
| 42b | simple | minmax | forward | 25 | 0.9550090063247424 | 0.7018435381296126 | not submitted | waiting for a good roc |
| 42c | simple | minmax | forward | 25 | 0.8100274929913187 | 0.8185403319589406 | 0.86875 | not highest but near |
| 45 | simple | minmax | forward | 15 | 0.9776399366171432 | 0.7000642467303131 | 0.87413 | BEST CASE: improved with lesser features | 
| 48 | simple | minmax | forward | 17 | 0.968064791364763 | 0.71144752877173 | 0.87278 | accuracy decreased with more features, lets decrease them |
| 49 | simple | minmax | forward | 13 | 0.9724663786448529 | 0.657890605723844 | 0.87353 | accuracy improved but not to the full. lets try 14 | 
| 50 | simple | minmax | forward | 14 | 0.97654292563349 | 0.6687466615541444 | 0.87271 | forward=15 was the highest breakpoint for naivebayes |
| 61a | row removal | minmax | forward | 15 | - | - | - | error, NaN values in test cannot perform forward |
| 61b | row removal | minmax | - | 78 | - | - | - | error, NaN values in test cannot train GNB model |

total test cases done: 31    
total submissions: 16   
starting accuracy: 0.83725 (case 31)    
highest accuracy achieved: 0.87413 (case 45)   
highest parameters: 
- imputer = simple
- scaler = minmax 
- forward selection with 15 rows
    
analysis:    
- forward selection is best at 15
- knn=7 and simple imputer have no difference in performance with naivebayes
- simple, minmax worked best with NB
- NB performed better with more features then lesser   
- row removal does not work in naivebayes    

### Analyzing Forward Feature Selection
we have run naivebayes multiple times with forward selection, lets analyse its accuracies (while keeping all other parameters like imputer and scaler constant):

| case number | feature selector | no. of features | kaggle accuracy |
| ----------- | ---------------- | --------------- | --------------- |
| 34 | forward | 5 | 0.82386 |
| 49 | forward | 13 | 0.87353 | 
| 50 | forward | 14 | 0.87271 | 
| 45 | forward | 15 | 0.87413 |
| 48 | forward | 17 | 0.87278 |
| 35b | forward | 20 | 0.87148 |
| 42 | forward | 25 | 0.86875 |
| 41a | forward | 30 | 0.86669 |

we can see that forward=14 might be an outlier, but forward=15 (case = 45) is the best case. 

# K Nearest Neighbours

code cleaned and commented, done

### Analyzing K Nearest Neighbours
| case number | K used | imputer | scaler | feature selector | no. of features | validation accuracy | roc | kaggle accuracy | analysis | 
| ----------- | ------ | ------- | ------ | ---------------- | ------------- | ------------------- | --- | --------------- | -------- |
| 43 | 5 | simple | minmax | - | 78 | 0.9972642442136056 | 0.5 | 0.53003 | so low, lets try k=7 and k=11 to improve |
| 44a | 7 | simple | minmax | forward | 10 | - | - | - | ran for 1000 minutes, didnt work |
| 44b | 7 | knn-7 | maxabs | variance=0.0001, correlationr=0.87 | 68 | 0.9973455042864688 | 0.5 | not submitted | waiting for forward to run, didnt want to waste a submission |
| 44c | 7 | knn-7 | maxabs | variance=0.1, correlationr=0.9 | 5 | 0.9974267643593321 | 0.5 | not submitted | waiting for forward to run, didnt want to waste a submission |
| 46 | 7 | simple | minmax | variance=0.001, correlation=0.9 | 58 | 0.9973319609409916 | 0.5 | 0.54796 | accuracy improved with knn=7 and some features removed | 
| 47 | 7 | simple | minmax | kbest | 30 | 0.9975892845050585 | 0.5028112828678414 | 0.57056 | works better with lesser features | 
| 52 | 3 | simple | minmax | forward | 5 | 0.9965735335942685 | 0.5254314763954754 | 0.55037 | accuracy decreased with too low features |
| 53 | 11 | knn=7 | minmax | kbest | 20 | 0.9970340073404933 | 0.5 | 0.59883 | imporvement, lets decrease kbest further |
| 54 | 11 | knn=7 | minmax | kbest | 15 | 0.9976434578869673 | 0.5 | 0.60509 | improved! lets decrease further |
| 55 | 11 | knn=7 | minmax | kbest | 10 | 0.9975486544686268 | 0.5055112852519732 | 0.61709 | improved, lets change the imputer to knn=3 |
| 56 | 11 | knn=3 | minmax | kbest | 10 | 0.9972507008681284 | 0.5024509803921569 | 0.61709 | file values remained same, entry wasted |
| 57 | 11 | knn=3 | minmax | kbest | 5 | 0.9974267643593321 | 0.5053136492515911 | 0.62622 | lets try kbest=3 next |
| 59 | 11 | knn=3 | minmax | kbest | 3 | 0.9971152674133564 | 0.507075049343145 | 0.62207 | deteroiration, kbest=5 was best |
| 63 | 9 | knn=3 | minmax | kbest | 5 | 0.9972236141771741 | 0.5048989470087107 | 0.63158 | increased, lets decrease k-nearest to 7 | 
| 64 | 7 | knn=3 | minmax | kbest | 5 | 0.9974538510502864 | 0.5157623171129992 | 0.61114 | accuracy decreased. breakdown is best at 9 |
| 79 | 300 | knn=3 | minmax | kbest | 5 | 0.9973861343229005 | 0.5 | 0.81121 | k-neighbours increases accuracy | 
| 82 | 500 | knn=3 | minmax | kbest | 5 | 0.9972507008681284 | 0.5 | 0.82533 | lets increase to 1000 |
| 85 | 1000 | knn=3 | minmax | kbest | 5 | 0.9973048742500372 | 0.5 | 0.83911 | lets add weights and increase k |
| 86b | 2000 | knn=3 | minmax | kbest | 5 | 0.9970746373769248 | 0.4999185147963549 | 0.82641 | too many k |
| 88 | 1500 | knn=3 | minmax | kbest | 5 | 0.997088180722402 | 0.49995246438224067 | 0.85212 | BEST CASE: wow! i think thats enough though |

total tries: 20    
total submissions: 17    
started accuracy: 0.53003   
highest accuracy: 0.85212 (case 88)   
highest case parameters:
- scaler = minmax
- imputer = knn=3
- feature selector = kbest, at k = 5
- k-nearest neighbours at k = 1500

analysis:
- kbest works at lower number of features
- knn=3, knn=7, simple imputers have no difference on accuracy
- forward selector + k-nearest is very time taking, even after 17 hours it didnt work. at smaller forward selection and smaller k-nearest-neighbours, it runs after 2 hours but accuracy is too low due to less features
- performs best on k-nearest neighbours = 1500. the more the k, the higher the accuracy

### Analyzing K in K-nearest neighbours
| case number | K=3 | K=5 | K=7 | K=9 | K=11 | K=300 | K=500 | K=1000 | K=1500 | K=2000 |
| ----------- | --- | --- | --- | --- | ---- | ----- | ----- | ------ | ------ | ------ |
| 43 | - | 0.53003 | - | - | - | - | - | - | - | - |
| 46 | - | - | 0.54796 | - | - | - | - | - | - | - |
| 47 | - | - | 0.57056 | - | - | - | - | - | - | - |
| 52 | 0.55037 | - | - | - | - | - | - | - | - | - |
| 53 | - | - | - | - | 0.59883 | - | - | - | - | - |
| 54 | - | - | - | - | 0.60509 | - | - | - | - | - |
| 55 | - | - | - | - | 0.61709 | - | - | - | - | - |
| 56 | - | - | - | - | 0.61709 | - | - | - | - | - |
| 57 | - | - | - | - | 0.62622 | - | - | - | - | - |
| 59 | - | - | - | - | 0.62207 | - | - | - | - | - |
| 63 | - | - | - | 0.63158 | - | - | - | - | - | - |
| 64 | - | - | 0.61114 | - | - | - | - | - | - | - |
| 79 | - | - | - | - | - | 0.81121 | - | - | - | - |
| 82 | - | - | - | - | - | - | 0.82533 | - | - | - |
| 85 | - | - | - | - | - | - | - | 0.83911 | - | - |
| 88 | - | - | - | - | - | - | - | - | 0.85212 | - |
|86b | - | - | - | - | - | - | - | - | - | 0.82641 |

BEFORE USING LARGE K:   
- we can have mixed analysis to this table. k=9 was only tested once but it had the highest, although k=11 has a continous record of being high   
- k=3 was only tested once because a lower k means very few variables are taken into consideration whereas the dataset is very large. 

AFTER USING LARGE K:
- we see that there is a linear relationship or exponential relationship between k-nearest neighbours and accuracy. the more the k, the better the accuracy - but there is a breakpoint where increasing k decreases the accuarcy. this breakpoint was found between k=1500 and k=2000 where k=1500 has a very high accuracy but k=2000 has low.
- mistakenly, odd k's should have been used. this was realized after all submissions were made. 

### Analyzing KBest Feature Selection
| case number | algo used | kbest features | kaggle accuracy |
| ----------- | --------- | -------------- | --------------- |
| 59 | k nearest neighbours, 11 | 3 | 0.62207 |
| 57 | k nearest neighbours, 11 | 5 | 0.62622 |
| 55 | k nearest neighbours, 11 | 10 | 0.61709 |
| 54 | k nearest neighbours, 11 | 15 | 0.60509 | 
| 53 | k nearest neighbours, 11 | 20 | 0.59883 |
| 47 | k nearest neighbours, 7 | 30 | 0.57056 |

kbest works better with lower number of features. as according to this table, kbest=5 is the best breakpoint. 

# Random Forest

### Analyzing RandomForest
| case number | imputer | scaler | grid | max depth | n estimators | feature selector | no. of features | criteria | min samples split | max features | min samples leaf | validation accuracy | roc | kaggle accuracy | analyzing |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 44 | simple | maxabs | - | 10 | 200 | - | 78 | default = gini | - | - | - | 0.9973319609409916 | 0.5050251256281407 | 0.90507 | ok good, now lets used the best parameters that we found from decision trees |
| 51 | knn=7 | maxabs | - | 5 | 200 | - | 78 | entropy | 15 | 60 | 80 | 0.9970610940314476 | 0.5 | 0.91554 | accuracy imrpoved, lets increase no. of trees |
| 58 | knn=7 | maxabs | - | 5 | 300 | - | 78 | entropy | 15 | 60 | 80 | 0.997359047631946 | 0.5 | 0.91889 | very slight imporvement, lets try and increase depth |
| 60 | knn=7 | maxabs | - | 6 | 10 | - | 78 | entropy | 15 | 60 | 80 | 0.9972642442136056 | 0.5 | 0.91309 | slight difference even though trees are 30 times less. interesting. |
| 62 | knn=7 | maxabs | - | 7 | 400 | - | 78 | entropy | 15 | 60 | 80 | 0.9974132210138549 | 0.5 | 0.92693 | depth increased trees, lets increase it |
| 66 | knn=7 | maxabs | - | 8 | 400 | - | 78 | entropy | 15 | 60 | 80 | 0.9974403077048093 | 0.5 | 0.93079 | depth increased trees, lets increase further |
| 68 | knn=7 | maxabs | - | 10 | 400 | - | 78 | entropy | 15 | 60 | 80 | 0.9972777875590828 | 0.5 | 0.93256 | depth is increasing accuracy |
| 78 | knn=7 | maxabs | - | 11 | 400 | - | 78 | entropy | 15 | 60 | 80 | 0.99729133090456 | 0.5 | 0.93452 | improved, we can increase further |
| 83 | knn=7 | maxabs | - | 11 | 400 | kbest | 30 | entropy | 15 | 60 | 80 | 0.9972371575226513 | 0.5 | 0.92633 | deterioration, could be too many features or too less features |
| 118 | knn=7 | maxabs | - | 11 | 400 | algorithm feature importance | 35 | entropy | 15 | 60 | 80 | 0.997359047631946 | 0.5 | 0.93546 | BEST CASE: improved! lets use some grid on RF to find best depth + estimators |
| 123b | knn=7 | maxabs | param_grid = { 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  'n_estimators': [50, 100, 200, 300, 400, 500, 1000] } | - | - | algorithm feature importance | 35 | entropy | - | - | - | - | - | - | error, code stopped after 9hours of running |
| 125 | knn=7 | maxabs | param_grid = { 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'n_estimators': [50, 100, 200] } | 9 | 200 | algorithm feature importance | 35 | entropy | - | - | - | 0.9973319609409916 | 0.5050251256281407 | 0.92154 | deterioration, lets increase estimators in depth to find the best estimators |
| 133 | knn=7 | maxabs | param_grid = { 'n_estimators': [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700] } | 9 | 650 | algorithm feature importance | 35 | entropy | - | - | - | 0.9976570012324444 | 0.5167597765363129 | 0.92877 | deterioration, slight imporvement but overall less. lets introduce min samples split in grid |

total tests: 13   
total submissions: 12   
started accuracy: 0.90507   
highest accuracy: 0.93546 (case 118)    
highest parameters: 
- imputer: knn=7
- scaler: maxabs
- max depth: 11
- estimators: 400
- criterion: entropy
- min samples split: 15
- max features: 60
- min samples leaf: 80
- algorithm feature importance: 35 features

analysis:
- higher depth of trees allows greater accuracy while lower depth moves to underfitting
- gini underperforms while entropy performs way better
- kbest feature selection doesnt work well with random forest (wasnt very tested heavily to say this)
- algorithm feature importance improves the accuracy
- grid search takes alot of time in random forest

### Analyzing Depth of Trees
| case number | max_depth | kaggle accuracy |
| ----------- | --------- | --------------- |
| 51, 58 | 5 | 0.91554, 0.91889 |
| 60 | 6 | 0.91309 |
| 62 | 7 | 0.92693 |
| 66 | 8 | 0.93079 |
| 125, 133 | 9 | 0.92154, 0.92877 |
| 44, 68 | 10 | 0.90507, 0.93256 |
| 78, 83, 118 | 11 | 0.93452, 0.92633, 0.93546 |

from here we can see that best accuracy is on depth=10 and depth=11 and depth=8 to some extent. thus, the larger the tree, the more the near the breakpoint. we have not found the breakpoint yet. 

# Gradient Boosting

### Analyzing GradientBoosting
| case number | imputer | scaler | grid | max depth | n estimators | criterion | max features | feature selection | no. of features | bagging | validation accuracy | roc | kaggle accuracy | analyzing |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 61c | simple | minmax | - | 6 | 300 | default = friedman_mse | - | - | 78 | - | 0.9955442393380013 | 0.560056823582126 | 0.88298 | improved, lets add some features we used in decision trees |
| 65 | simple | minmax | - | 6 | 300 | squared-error | 60 | - | 78 | - | 0.9955984127199101 | 0.5545980654020854 | 0.88297 | accuracy remained same, lets increase the depth |
| 67 | simple | minmax | - | 10 | 300 | squared error | 60 | - | 78 | - | 0.9953140024648889 | 0.5702606327046124 | 0.79753 | too high depth ruined the accuracy |
| 80 | simple | minmax | - | 8 | 300 | squared error | 60 | - | 78 | - | 0.9951514823191625 | 0.5543197973296156 | 0.83659 | too low, lets try feature selection |
| 86a | simple | minmax | - | 6 | 300 | squared error | 60 | forward | 10 | - | - | - | - | error, ran for 756 min, didnt work |
| 86b | simple | minmax | - | 6 | 300 | squared error | 60 | forward | 10 | - | - | - | - | error, again ran for 256 min with n_jobs = -1, didnt work |
| 89 | simple | minmax | - | 6 | 300 | squared error | 60 | kbest | 30 | - | 0.9961807765754297 | 0.5478679502290538 | 0.85929 | low, lets try bagging next |
| 102 | simple | minmax | - | 6 | 100 | squared error | 60 | - | 78 | estimators = 50 | 0.9975351111231496 | 0.5317120864929359 | 0.90158 | 24hour running: improved but not efficient |
| 116 | simple | minmax | param_grid = { 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] } | 3 | - | default = friedman_mse | - | - | 78 | - | 0.9968308571583353 | 0.5404496829489119 | 0.84769 | very low. lets use this depth and repeat grid with estimators + learning rate |
| 122 | simple | minmax | param_grid = { 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] } | 3 | - | default = friedman_mse | - | - | 78 | estimators = 10 | 0.9974809377412408 | 0.5458640497792794 | 0.88551 | bagging improved the same model, even at less estimators |
| 131 | simple | minmax | param_grid = { 'criterion': ['friedman_mse', 'squared_error'] } | 3 | - | friedman_mse | - | - | 78 | estimators = 10 | 0.9974132210138549 | 0.5332722268995048 | 0.90057 | 7hour running: improved and more effecient. lets do grid on estimators next |
| 155 | simple | minmax | param_grid = { 'n_estimators': [50, 100, 200] } | 3 | 200 | friedman_mse | - | - | 78 | estimators = 10 | 0.9974267643593321 | 0.5358363294636074 | 0.90485 | BEST CASE: ok good improved, lets grid again with more estimators |

total tests: 12  
total submissions: 10   
started accuracy: 0.88298   
highest accuracy: 0.90485 (case 155)   
highest case parameters:
- imputer: simple
- scaler: minmax
- max depth: 3
- estimators: 200
- criterion: friedman_mse
- bagging of 10 estimators
- no feature selection

analysis:
- each submission took AT LEAST three hours and more. some submissions took exceptional time like 12+ hours. 
- boosting is itself a very slow algorithm. using any forward or backword feature selector takes more than 12 hours and still doesnt even run on train data let alone full data and prediction
- the max_depth breakpoint was 6. too high depth leads to overfitting thus low accuracy
- kbest could not be rigourously tested as each submission took over 3 hours. 
- bagging does work but it takes extremely long, **over 24 hours of running** and laptop use. its not very feasible. accuracy improved by 2 percent but very inefficient. better to use 10 estimators instead of 50, whereas 50 gives better results
- friedman_mse criterion is much better than squared_error
- uptil now, more estimators means better accuracy

### Analyzing depth
| case number | depth | accuracy |
| ----------- | ----- | -------- | 
| 116, 122, 131 | 3 | 0.84769, 0.88551, 0.90057 |
| 61c, 65, 89, 102 | 6 | 0.88298, 0.88297, 0.85929, 0.90158 |
| 80 | 8 | 0.83659 |
| 67 | 10 | 0.79753 |

according to this table, higher depth doesnt have much power, but differnet values such as bagging and other parameters matter. on average, smaller depths performed better. 

# Adaptive Boosting

### Analyzing AdaptiveBoosting
| case number | imputer | scaler | grid | n estimators | learning rate | bagging | feature selector | no. of features | validation accuracy | roc | kaggle accuracy | analysis |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| 69 | simple | minmax | - | 100 | default = 0.5 | - | - | 78 | 0.9969527472676301 | 0.5430550209247345 | 0.94475 | - |
| 70 | simple | minmax | - | 200 | default = 0.5 | - | - | 78 | 0.996885030540244 | 0.5717622628834583 | 0.93379 | deterioration, too high estimators | 
| 71 | simple | minmax | - | 50  | default = 0.5 | - | - | 78 | 0.9970746373769248 | 0.5522487676032429 | 0.93853 | deterioration, too low estimators |
| 72 | simple | minmax | - | 75  | default = 0.5 | - | - | 78 | 0.9966141636307001 | 0.5660048884094492 | 0.94053 | improvement but not to the highest |
| 73 | simple | minmax | - | 110 | default = 0.5 | - | - | 78 | 0.9965464469033141 | 0.5644500047530453 | 0.94521 | improved! best estimators are between 100 and 200 |
| 74 | simple | minmax | - | 150 | default = 0.5 | - | - | 78 | 0.9971288107588336 | 0.5607445471728567 | 0.94780 | more improvement, we are closer to the breakpoint |
| 75 | simple | minmax | - | 160 | default = 0.5 | - | - | 78 | 0.9967360537399949 | 0.5648649150311703 | 0.94948 | lets increase 10 further |
| 76 | simple | minmax | - | 170 | default = 0.5 | - | - | 78 | 0.997088180722402 | 0.563818101949167 | 0.94966 | BEST CASE: highest, lets increase 10 further |
| 77 | simple | minmax | - | 180 | default = 0.5 | - | - | 78 | 0.9969933773040617 | 0.5571944470850252 | 0.93516 | deterioration, breakpoint found! if possible, can try 175 to see if its highest |
| 81 | simple | minmax | - | 175 | default = 0.5 | - | - | 78 | 0.9970204639950161 | 0.5683960928977767 | 0.94949 | near to highest but not highest |
| 84 | simple | minmax | - | 175 | 0.1 | - | - | 78 | 0.9974538510502864 | 0.5026527855096471 | 0.93301 | too low, increase learning rate |
| 87 | simple | minmax | - | 170 | 0.75 | - | - | 78 | 0.9972371575226513 | 0.5632272905179162 | 0.93369 | not too good |
| 90 | simple | minmax | - | 170 | 0.6 | - | - | 78 | 0.9971152674133564 | 0.5492075935795596 | 0.94896 | not the highest, i guess learning rate=0.5 was the best |
| 114 | simple | minmax | - | 170 | default = 0.5 | estimators = 50 | - | 78 | 0.9972371575226513 | 0.5406533738276343 | 0.93386 | deterioration, lets use grid to find best params, bagging didnt do so well |
| 123a | simple | minmax | param_grid = { 'n_estimators': [50, 100, 140, 160, 170, 180, 200, 300, 400, 500, 1000, 3000], 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.6, 0.9] } | - | - | estimators = 50 | - | 78 | - | - | - | error: after 10hours of running, it would take 3 days to run, stopped as parameters too many |
| 126b | simple | minmax | param_grid = { 'n_estimators': [50, 100, 140, 160, 170, 180, 200, 300, 400, 500], 'learning_rate': [0.001, 0.005, 0.01, 0.05] } | - | - | estimators = 50 | - | 78 | - | - | - | error, crashed after 5hours |
| 142 | simple | minmax | param_grid = { 'n_estimators': [50, 100, 140, 160, 170, 180, 200, 300], 'learning_rate': [0.001, 0.005] } | 300 | 0.005 | estimators = 50 | - | 78 | 0.9971965274862197 | 0.5 | 0.86318 | too long and very low. lets do some other grid with bagging at 10 because bagging at 50 takes 22 hours+ |

total tests: 17    
total submissions: 15    
starting accuracy: 0.94475   
highest accuracy: 0.94966 (case 76)
highest case parameters:
- imputer: simple
- scaler: minmax
- n estimators: 170
- no feature selector, no bagging

analysis:
- best estimator value is 170
- too many estimators and too less estimators can be wrong
- the default learning rate of 0.5 performs best
- grid search is very slow on adaboost and takes alot of time. especially with bagging
- bagging of 50 estimators is too slow
- greater estimators == lower learning rate

### Analyzing Estimators with AdaBoost
| case number | n estimators | kaggle accuracy |
| ----------- | ------------ | --------------- | 
| 71 | 50 | 0.93853 |
| 72 | 75 | 0.94053 |
| 69 | 100 | 0.94475 |
| 73 | 110 | 0.94521 |
| 74 | 150 | 0.94780 |
| 75 | 160 | 0.94948 |
| 76, 87, 90, 114 | 170 | 0.94966, 0.93369, 0.94896, 0.93386 |
| 81, 84 | 175 | 0.94949, 0.93301 |
| 77 | 180 | 0.93516 |
| 70 | 200 | 0.93379 |
| 142 | 300 | 0.86318 |

from this table we can see that roughly 170 estimators is the breakpoint with the highest accuracy. lesser than 100 is too less estimators and more than 200 is too many estimators. in our search to find the breakpoint, we tested 10 different estimator values and found 170 as the best. 
after more testing we can also see that too high estimators is bad. 

### Analyzing Learning Rate with AdaBoost
| case number | learning rate | kaggle accuracy |
| ----------- | ------------- | --------------- |
| 142 | 0.005 | 0.86318 --estimators=300 |
| 84 | 0.1 | 0.93301 |
| 76, 114 | default=0.5 | 0.94966, 0.93386 |
| 90 | 0.6 | 0.94896 |
| 87 | 0.75 | 0.93369 |

from this we can analyse that learning rate is best at default of 0.5, even though l.rate is good at 0.6, however best is at 0.5. having a too high or too low learning rate depreciates the accuracy performance.    
this table analyses with constant estimators of 170. 

# LightGBM

### Analyzing LightGBM
| case number | imputer | scaler | grid | max depth | n estimators | learning rate | min child samples | bagging params | feature selector | no. of features | validation accuracy | roc | kaggle accuracy | analysis |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 91 | simple | minmax | - | 10 | 100 | 0.9 | default = 20 | - | - | 78 | 0.9934450207890353 | 0.5815248594933158 | 0.75561 | lets try and increase estimators |
| 92 | simple | minmax | - | 10 | 400 | 0.9 | default = 20 | - | - | 78 | 0.9963297533756789 | 0.5020350035153998 | 0.50533 | very very low. lets normalize the learning rate |
| 93 | simple | minmax | - | 10 | 400 | 0.5 | default = 20 | - | - | 78 | 0.9897206007828053 | 0.5109621532460307 | 0.50408 | still low, estimators is the problem |
| 94 | simple | minmax | - | 9 | 100 | 0.5 | default = 20 | - | - | 78 | 0.9382288012784918 | 0.5318142882431025 | 0.37172 | wow. so low. lets add bagging because its not controlling |
| 95 | simple | maxabs | - | 10 | 100 | 0.9 | default = 20 | estimators = 50 | - | 78 | 0.9972777875590828 | 0.5 | 0.77939 | improved, but v low, lets increase estimators in bagging | 
| 96 | simple | maxabs | - | 10 | 100 | 0.9 | default = 20 | estimators = 100 | - | 78 | 0.997494481086718 | 0.5 | 0.77699 | reduced, lets go back and try feature selection | 
| 97 | knn=7 | maxabs | - | 10 | 100 | 0.9 | default = 20 | estimators = 50 | kbest | 5 | 0.9974403077048093 | 0.5 | 0.49841 | very low, we dont know if the issue is imputer or kbest selector. lets decrease the learning rate |
| 98 | simple | maxabs | - | 10 | 200 | 0.1 | default = 20 | estimators = 50 | - | 78 | 0.9972777875590828 | 0.5024807720320663 | 0.87767 | relationship b/w estimators and learning rate is found |
| 99 | simple | maxabs | -| 10 | 300 | 0.01 | default = 20 | estimators = 50 | - | 78 | 0.9973048742500372 | 0.517066380205849 | 0.94165 | shotup! lets decrease depth now | 
| 100 | simple | maxabs | - | 8 | 300 | 0.01 | default = 20 | estimators = 50 | - | 78 | 0.9973455042864688 | 0.5174796298056683 | 0.94321 | increased, lets decrease depth further |
| 101 | simple | maxabs | - | 7 | 300 | 0.01 | default = 20 | estimators = 100 | - | 78 | 0.997359047631946 | 0.512612682865538 | 0.94351 | negligible increase, lets decrease bagging |
| 103 | simple | maxabs | - | 6 | 300 | 0.01 | default = 20 | estimators = 50 | - | 78 | 0.9976705445779216 | 0.5302129918143766 | 0.94395 | negligible increase, lets perform grid to find better parameters |
| 104a | simple | maxabs | param_grid = { 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9], 'n_estimators': [50, 100, 200, 300, 400, 500, 1000, 2000, 3000] } | 6 | 400 | 0.001 | default = 20 | estimators = 50 | - | 78 | - | - | - | error, failed after 271min, too many parameters == 3100 fits |
| 104b | simple | maxabs | param_grid = { 'max_depth': [1, 2, 3, 4, 5], 'learning_rate': [0.001, 0.005, 0.01, 0.05], 'n_estimators': [50, 100, 200, 300] } | 2 | 300 | 0.05 | default = 20 | - | - | 78 | 0.9973725909774233 | 0.5521437437723113 | 0.94948 | increased FINALLY. lets run a second grid search with diff parameters | 
| 111 | simple | maxabs | param_grid = { 'max_depth': [2, 3, 6, 7, 8, 9, 10], 'learning_rate': [0.001, 0.005, 0.01, 0.05], 'n_estimators': [400, 500, 1000, 2000, 3000] } | 3 | 1000 | 0.01 | default = 20 | estimators = 50 | - | 78 | 0.9974403077048093 | 0.5307284931466785 | 0.95106 | improved! lets grid with min child weight |
| 119 | simple | maxabs | param_grid = { 'min_child_samples': [1, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000] } | 3 | 1000 | 0.01 | 40 | estimators = 50 | - | 78 | 0.9976028278505357 | 0.5320719837648076 | 0.95087 | deterioration, seems like the default=40 was better. now lets try lgbm feature importance |
| 123c | simple | maxabs | - | 3 | 1000 | 0.01 | default = 20 | estimators = 50 | algorithm feature importance | 35 | 0.9975757411595813 | 0.5317324536447935 | 0.95070 | nice, but a bit low. lets decrease features and try again | 
| 126c | simple | maxabs | - | 3 | 1000 | 0.01 | default = 20 | estimators = 50 | algorithm feature importance | 20 | 0.9974267643593321 | 0.5321714275725801 | 0.95323 | BEST CASE: improved! lets decrease them further |
| 128 | simple | maxabs | - | 3 | 1000 | 0.01 | default = 20 | estimators = 50 | algorithm feature importance | 15 | 0.9972777875590828 | 0.5286809703922187 | 0.94903 | deterioration. would they work at features = 25? |
| 132 | simple | maxabs | - | 3 | 1000 | 0.01 | default = 20 | estimators = 50 | algorithm feature importance | 25 | 0.9974538510502864 | 0.5331293168263355 | 0.95242 | wow, nice, but not to the fullest. seems like features = 20 was the breakpoint. lets increase estimators |
| 136 | simple | maxabs | - | 3 | 3000 | 0.01 | default = 20 | estimators = 50 | algorithm feature importance | 20 | 0.9974809377412408 | 0.5184913623160419 | 0.94173 | so low. i think this is enough testing on lgbm |

total tests: 22    
total submissions: 20     
starting accuracy: 0.75561         
highest accuracy: 0.95323 (case 126c)        
highest parameters:     
- imputer: simple
- scaler: maxabs
- max depth: 3
- estimators: 1000
- learning rate: 0.01
- bagging estimators: 50
- algorithm feature importance: 20 features

analysis:     
- relationship found between number of estimators and learning rate. less estimators == high learning rate. more estimators == low learning rate
- 20 features are the breakpoint, too less decreases accuracy whether it be algorithm feature importance or kbest feature selector, and too many is also bad
- too much depth can be long and ineffective, smaller depth is better
- default min child weight is better, 20 has a higher roc however accuracy decreased
- increasing estimators without altering learning rate makes no difference

### Analyzing Estimators and Learning Rate
| case number | estimators | learning rate | accuracy |
| ----------- | ---------- | ------------- | -------- | 
| 94 | 100 | 0.5 | 0.37172 |
| 91, 95, 96, 97 | 100 | 0.9 | 0.75561, 0.77939, 0.77699, 0.49841 |
| 98 | 200 | 0.1 | 0.87767 |
| 99, 100, 101, 103 | 300 | 0.01 | 0.94165, 0.94321, 0.94351, 0.94395 |
| 104b | 300 | 0.05 | 0.94948 |
| 92 | 400 | 0.9 | 0.50533 |
| 93 | 400 | 0.5 | 0.50408 | 
| 111, 119, 123c, 126c, 128, 132 | 1000 | 0.01 | 0.95106, 0.95087, 0.95070, 0.95323, 0.94903, 0.95242 |
| 136 | 3000 | 0.01 | 0.94173 |

based on this table, we can see that if estimators are small <=100, then a higher learning rate i preferred however if the learning rate is high >=300, ==1000, then a lower learning rate is preferred. too much high requires too much low, otherwise accuracy decreases. 

### Analyzing Algorithm Feature Importance
| case number | features | accuracy |
| ----------- | -------- | -------- |
| 128 | 15 | 0.94903 |
| 126c, 136 | 20 | 0.95323, 0.94173 |
| 132 | 25 | 0.95242 |
| 123c | 35 | 0.95070 |

according to this table, 20, 25, 35 are good accuracies however 20 is the highest. it does not perform well on lower number of features. 

# XGBoost

### Analyzing XGBoost
| case number | imputer | scaler | grid | n estimators | max depth | learning rate | bagging | feature selector | no. of features | validation accuracy | roc | kaggle accuracy | analysis |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 105 | simple | maxabs | - | - | - | - | estimators = 50 | - | 78 | 0.9972100708316969 | 0.5164319248826291 | 0.95474 | lets add some estimators |
| 106 | simple | maxabs | - | 100 | - | - | estimators = 50 | - | 78 | 0.9975215677776724 | 0.5210458424361445 | 0.95347 | deterioration, lets increase the estimators |
| 108 | simple | maxabs | - | 500 | - | - | estimators = 50 | - | 78 | 0.9973725909774233 | 0.5150685869118535 | 0.94900 | deterioration, too high estimators. lets decrease them and move to learning rate |
| 110 | simple | maxabs | - | 100 | - | 0.1 | estimators = 50 | - | 78 | 0.9975892845050585 | 0.5137985780696528 | 0.95063 | improved but not so much. lets try shifting depth next |
| 112 | simple | maxabs | param_grid = { 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] } | - | 2 | - | estimators = 50 | - | 78 | 0.9975215677776724 | 0.5390217640369339 | 0.95332 | good but overall not highest. lets put grid search for estimators now |
| 117 | simple | maxabs | param_grid = { 'n_estimators': [50, 100, 200, 300, 400, 500, 1000, 2000] } | 100 | 2 | - | estimators = 50 | - | 78 | 0.9974267643593321 | 0.5485165268480601 | 0.95236 | good, but not best. lets add in learning rate |
| 120 | simple | maxabs | param_grid = { 'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 0.9] } | 100 | 2 | 0.2 | estimators = 50 | - | 78 | 0.9974403077048093 | 0.5430861000283999 | 0.94989 | deterioration, default was fine. lets add feature importance |
| 124 | simple | maxabs | - | - | - | - | estimators = 50 | algorithm feature importance | 35 | 0.9973725909774233 | 0.5274660496761139 | 0.95846 | wow! we can increase features in the next round |
| 127 | simple | maxabs | - | - | - | - | estimators = 50 | algorithm feature importance | 40 | 0.9975080244321952 | 0.5184142634887761 | 0.95371 | deterioration. lets decrease features in next round |
| 129 | simple | maxabs | - | - | - | - | estimators = 50 | algorithm feature importance | 30 | 0.9974403077048093 | 0.52283584980575 | 0.94984 | deterioration. would 36 be better? |
| 130 | simple | maxabs | - | - | - | - | estimators = 50 | algorithm feature importance | 36 | 0.9971694407952653 | 0.5118279679945126 | 0.95522 | deterioration. 35 was the breakpoint. |
| 135 | simple | maxabs | param_grid = { 'learning_rate': [0.01, 0.05, 0.1, 0.5, 0.9], 'n_estimators': [100, 200, 300, 400], 'max_depth': [1, 2, 3, 4] } | 200 | 2 | 0.1 | estimators = 50 | algorithm feature importance | 35 | 0.997359047631946 | 0.5319857527264058 | 0.94971 | deterioration, alot. looks like xgboost only works best with default parameters |
| 138 | simple | maxabs | - | - | - | - | estimators = 100 | algorithm feature importance | 35 | 0.9976163711960129 | 0.5325815412932204 | 0.95979 | BEST CASE: wow, lets increase bagging more |
| 139 | simple | maxabs | - | - | - | - | estimators = 150 | algorithm feature importance | 35 | 0.9973725909774233 | 0.5249728397408912 | 0.95586 | uh oh. lets try 99 baggers lol |
| 140 | simple | maxabs | - | - | - | - | estimators = 99 | algorithm feature importance | 35 | 0.997697631268876 | 0.5251396648044693 | 0.94394 | YAAR WHAT. this is an outlier. lets make a new file which analyses all the roc's at all bagging estimators 1 to 150 and selects the best one |
| 141 | simple | maxabs | forloop for bagging, highest roc 1 to 100 estimators | - | - | - | estimators = 4 | algorithm feature importance | 35 | 0.9975351111231496 | 0.5287822330483466 | 0.94413 | so lets try lowest ROC |
| 145 | simple | maxabs | forloop for bagging, lowest roc 1 to 100 estimators | - | - | - | estimators = 3 | algorithm feature importance | 35 | 0.997494481086718 | 0.5107323218497934 | 0.94978 | decreased like alot. so its not roc dependent |
| 146 | simple | maxabs | - | - | - | - | estimators = 100 | correlation filter = 0.9, algorithm feature importance | 66 then 35 | 0.9973048742500372 | 0.5 | 0.50000 | file was faulty and all values submitted in each row were same |

total tries: 18    
total submissions: 18   
starting accuracy: 0.95474   
highest accuracy: 0.95979 (case 138)
highest parameters:
- imputer: simple
- scaler: maxabs
- no parameters in xgboost brackets
- bagging of 100 estimators
- algorithm feature importance, 35 features

analysis:
- after singular grids the best params were 100 estimators, 0.2 learning rate, 2 max depth but still accuracy is low
- after multiple grids the best params were 200 estimators, 0.1 learning rate, 2 max depth but still accuracy is low
- best accuracy was achieved at "None" parameters, when brackets are empty
- best algorithm feature importance is at 35 features. 40 is good as well, but 30 has alot of deterioration
- bagging was best at 100. i tested all bagging from 1 to 100 and the lowest roc was at bagging=3 and highest roc at bagging=4 but still accuracy was low on both. bagging=99 was an outlier, but bagging=100 and bagging=150 were highest. 

### Analyzing Algorithm Feature Importance
| case number | features | accuracy | 
| - | - | - | 
| 129 | 30 | 0.94984 |
| 124 | 35 | 0.95846 |
| 130 | 36 | 0.95522 |
| 127 | 40 | 0.95371 |

35 features was breakpoint, more were good but less were bad 

### Analyzing Bagging
| case number | bagging estimators | accuracy |
| - | - | - |
| 145 | 3 | 0.94978 |
| 141 | 4 | 0.94413 |
| 135 | 50 | 0.94971 |
| 140 | 99 | 0.94394 |
| 138 | 100 | 0.95979 |
| 139 | 150 | 0.95586 |

bagging=4 is probably an outlier, as well as bagging=99, but bagging=100 is the best. too low baggers can be bad and too many baggers is very time consuming and ineffective. 

# CatBoost

### Analyzing CatBoost
| case number | imputer | scaler | grid | iterations | depth | learning rate | loss function | bagging | feature selector | no. of features | validation accuracy | roc | kaggle accuracy | analysis |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 107 | simple | maxabs | - | 1000 | default = 6 | 0.108132 | default = Logloss | - | - | 78 | 0.99729133090456 | 0.5215311004784688 | 0.93798 | lets introduce bagging |
| 109 | simple | maxabs | - | 100 | default = 6 | 0.5 | default = Logloss | estimators = 50 | - | 78 | 0.9975892845050585 | 0.5165542206956393 | 0.93612 | bagging didnt do so well. could be because estimators decreased and learning rate increased. lets change depth |
| 111 | simple | maxabs | - | default = 1000 | 10 | 0.108132 | Logloss | - | - | 78 | 0.99729133090456 | 0.504950495049505 | 0.92756 | deterioration, lets try grid to find the perfect depth |
| 115a | simple | maxabs | param_grid = { 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] } | default = 1000 | 1 | 0.108132 | default = Logloss | - | - | 78 | 0.9975351111231496 | 0.5523504215805054 | not submitted | wanted to do bagging first |
| 115b | simple | maxabs | param_grid = { 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] } | default = 1000 | 1 | 0.108132 | default = Logloss | estimators = 50 | - | 78 | 0.9976028278505357 | 0.5496287205207183 | 0.94063 | improved! lets do grid on iterations and learning rate |
| 121 | simple | maxabs | param_grid = { 'iterations': [50, 100, 200, 300, 500, 600, 700, 900, 1000, 2000], 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9] } | 2000 | 1 | 0.1 | default = Logloss | estimators = 50 | - | 78 | 0.5287075514634675 | 0.9973861343229005 | 0.94631 | improved! lets analyse if there are any other features otherwise we can apply feature importance |
| 126a | simple | maxabs | param_grid = { 'iterations': [2000, 2500, 3000, 3500], 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9] } | - | - | - | default = Logloss | estimators = 50 | - | 78 | - | - | - | error, crashed after 3hours |
| 134 | simple | maxabs | param_grid = { 'iterations': [2000, 2200, 2500], 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9] } | 2000 | 1 | 0.1 | default = Logloss | estimators = 50 | - | 78 | 0.9975892845050585 | 0.5304944615658465 | 0.94664 | wow, improved further (same thing was run as case 121 lol). lets add some feature importance now |
| 137 | simple | maxabs | - | 2000 | 1 | 0.1 | default = Logloss | estimators = 50 | algorithm feature importance | 20 | 0.9975892845050585 | 0.519719707742552 | 0.95006 | omg wow. should we increase features or decrease? |
| 142 | simple | maxabs | - | 2000 | 1 | 0.1 | default = Logloss | estimators = 50 | algorithm feature importance | 15 | 0.997494481086718 | 0.5264142921513115 | 0.95191 | wow. lets decrease to 14 |
| 144 | simple | maxabs | - | 2000 | 1 | 0.1 | default = Logloss | estimators = 50 | algorithm feature importance | 14 | 0.9975215677776724 | 0.5188849755093714 | 0.95270 | BEST CASE: lets decrease to 13 |
| 147 | simple | maxabs | - | 2000 | 1 | 0.1 | default = Logloss | estimators = 50 | algorithm feature importance | 13 | 0.9976028278505357 | 0.5190217391304348 | 0.94950 | breakpoint found. lets increase bagging now |
| 152 | simple | maxabs | - | 2000 | 1 | 0.1 | default = Logloss | estimators = 100 | algorithm feature importance | 14 | 0.9974267643593321 | 0.5186486888775136 | 0.95263 | decreased. lets do bagging=75 |
| 157 | simple | maxabs | - | 2000 | 1 | 0.1 | default = Logloss | estimators = 75 | algorithm feature importance | 14 | 0.9976299145414901 | 0.5493758769911102 | 0.95106 | decreased. bagging=50 was fine |

no. of tries: 13    
no. of submissions: 11     
starting accuracy: 0.93798     
highest accuracy: 0.95270 (case 144)      
highest case parameters:
- imputer: simple
- scaler: maxabs
- no of estimators: 2000
- depth: 1
- learning rate: 0.1
- bagging, 50 estimators
- algorithm feature importance of 14 features

analyse:      
- bagging improves the performance on average
- higher depth is not good. smaller depth works better
- default iterations are 1000, smaller iterations underperform, 2000 is the best. 2200, 2500, 3000 were also checked but they had lower roc in grid
- the more the iterations, the relatively smaller learning rate
- feature importance is AMAZING. accuracy immediately went into 95s
- the smaller number of features, the better. breakpoint of best features number was found at 14
- USES ALOT OF MEMORY/RAM. cannot be done with other boosting algorithms or the laptop terminates the process due to >95% usage of RAM

### Analyze Algorithm Feature Importance
| case number | features | accuracy |
| - | - | - |
| 147 | 13 | 0.94950 |
| 144 | 14 | 0.95270 |
| 142 | 15 | 0.95191 |
| 137 | 20 | 0.95006 |

hence we can see the breakpoint at features=14, which has the highest accuracy

### Analyze Bagging Estimators
| case number | estimators | accuracy |
| - | - | - |
| 144 | 50 | 0.95270 |
| 157 | 75 | 0.95106 |
| 152 | 100 | 0.95263 |

best bagging is seen at 100 estimators, while 75 may be an outlier. 

# Extremely Randomized Tree

### Analyzing ERT
| case number | time | imputer | scaler | grid | estimators | bootstrap | bagging | feature selector | no. of features | validation accuracy | roc | kaggle accuracy | analysis |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 148 | 181min | simple | maxabs | - | default = 100 | default = False | estimators = 50 | - | 78 | 0.9976028278505357 | 0.5028089887640449 | 0.91809 | ok good, lets do without bagging, that was very long |
| 149 | 7min | simple | maxabs | - | default = 100 | default = False | - | - | 78 | 0.9973861343229005 | 0.5051478496636952 | 0.82590 | so bagging matters alot, lets check, does estimators mean alot too? |
| 150 | 3min | simple | maxabs | - | 10 | default = False | - | - | 78 | 0.9976299145414901 | 0.5138888888888888 | 0.66440 | wow. low. so we need to do bagging + high estimators |
| 151 | 163min | simple | maxabs | - | default = 100 | default = False | estimators = 100 | - | 78 | 0.9974267643593321 | 0.5186486888775136 | 0.91783 | good but higher bagging doesnt give better result |
| 153 | 73min | simple | maxabs | - | default = 100 | True | estimators = 100 | - | 78 | 0.9976705445779216 | 0.5028901734104047 | 0.91258 | decreased, lets do bagging=50 with bootstrap true |
| 154 | 38min | simple | maxabs | - | default = 100 | True | estimators = 50 | - | 78 | 0.9972371575226513 | 0.5 | 0.91049 | decreased further. lets increase estimators |

total tries: 6   
total submissions: 6   
starting accuracy: 0.91809   
highest accuracy: X (case Y)
highest case parameters:
- imputer: simple
- scaler: maxabs

analysis:
- doesnot have a difference with bootstrap=True
- estimators matter alot - the lower they are the worse
- bagging improves results but on average is very slow and requires alot of time
- performs better on bagging=50 instead of bagging=100