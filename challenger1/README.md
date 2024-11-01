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
| 18 | Thursday 7th November 2024 | - | - | - | - | 
| 19 | Friday 8th November 2024 | - | - | - | - | 
| 20 | Saturday 9th November 2024 | - | - | - | - | 
| 21 | Sunday 10th November 2024 | - | - | - | - | 

Algorithms worked on: 

| Algorithm Name | No. of Tries | No. of Submissions | Best Accuracy | Case Number | Imputer | Scaler | Feature Selector | No. of Features | Properties |
| - | - | - | - | - | - | - | - | - | - |
| Decision Tree | 30 | 30 | 0.89522 | 26 | knn=7 | maxabs | - | 78 | DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80), train_test_split(X, Y, test_size=0.3) | 
| Naive Bayes | 31 | 16 | 0.87413 | 45 | simple | minmax | forward | 15 | GaussianNB() |
| K-Nearest Neighbor | 20 | 17 | 0.85212 | 88 | knn=3 | minmax | kbest | 5 | KNeighborsClassifier(n_neighbors=1500, weights="distance") |
| Random Forest | 9 | 9 | 0.93452 | 78 | knn=7 | maxabs | - | 78 | RandomForestClassifier(max_depth=11, n_estimators=400, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80) | 
| Gradient Boosting | 7 | 5 | 0.88298 | 61c | simple | minmax | - | 78 | GradientBoostingClassifier(max_depth=6, n_estimators=300) |
| Adaptive Boosting | 0.94966 | 76 | simple | minmax | - | 78 | AdaBoostClassifier(n_estimators=170) |
| Light GBM | 0.77939 | 95 | simple | minmax | - | 78 | lgb.LGBMClassifier(max_depth=10, n_estimators=100, learning_rate=0.9), BaggingClassifier(estimator=model, n_estimators=50) |
| XGBoost | - | - | - | - | - |
| CatBoost | - | - | - | - | - |
| BaggingClassifier | - | - | - | - | - |
| ExtraTree Classifier (Extremely Randomized Tree) | - | - | - | - | - |
| Voting | - | - | - | - | - |
| Stacking | - | - | - | - | - |

# DecisionTrees

code cleaned and commented, done

## Analyzing Decision Trees
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

total test cases done: 30   
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

| case number | imputer | scaler | max depth | n estimators | feature selector | no. of features | criteria | min samples split | max features | min samples leaf | validation accuracy | roc | kaggle accuracy | analyzing |
| - | - | - | - | - | - | - | -------- | --------------- | ------------ | ---------------- | -- | - | - | - |
| 44 | simple | maxabs | 10 | 200 | - | 78 | default = gini | - | - | - | 0.9973319609409916 | 0.5050251256281407 | 0.90507 | ok good, now lets used the best parameters that we found from decision trees |
| 51 | knn=7 | maxabs | 5 | 200 | - | 78 | entropy | 15 | 60 | 80 | 0.9970610940314476 | 0.5 | 0.91554 | accuracy imrpoved, lets increase no. of trees |
| 58 | knn=7 | maxabs | 5 | 300 | - | 78 | entropy | 15 | 60 | 80 | 0.997359047631946 | 0.5 | 0.91889 | very slight imporvement, lets try and increase depth |
| 60 | knn=7 | maxabs | 6 | 10 | - | 78 | entropy | 15 | 60 | 80 | 0.9972642442136056 | 0.5 | 0.91309 | slight difference even though trees are 30 times less. interesting. |
| 62 | knn=7 | maxabs | 7 | 400 | - | 78 | entropy | 15 | 60 | 80 | 0.9974132210138549 | 0.5 | 0.92693 | depth increased trees, lets increase it |
| 66 | knn=7 | maxabs | 8 | 400 | - | 78 | entropy | 15 | 60 | 80 | 0.9974403077048093 | 0.5 | 0.93079 | depth increased trees, lets increase further |
| 68 | knn=7 | maxabs | 10 | 400 | - | 78 | entropy | 15 | 60 | 80 | 0.9972777875590828 | 0.5 | 0.93256 | depth is increasing accuracy |
| 78 | knn=7 | maxabs | 11 | 400 | - | 78 | entropy | 15 | 60 | 80 | 0.99729133090456 | 0.5 | 0.93452 | improved, we can increase further |
| 83 | knn=7 | maxabs | 11 | 400 | kbest | 30 | entropy | 15 | 60 | 80 | 0.9972371575226513 | 0.5 | 0.92633 | deterioration, could be too many features or too less features |

total tests: 9   
total submissions: 9   
started accuracy: 0.90507   
highest accuracy: X (case Y)    
highest parameters: 
- params

analysis:
- higher depth of trees allows greater accuracy while lower depth moves to underfitting

### Analyzing Depth of Trees
| case number | max_depth | kaggle accuracy |
| ----------- | --------- | --------------- |
| 51, 58 | 5 | 0.91554, 0.91889 |
| 60 | 6 | 0.91309 |
| 62 | 7 | 0.92693 |
| 66 | 8 | 0.93079 |
| 44, 68 | 10 | 0.90507, 0.93256 |
| 78, 83 | 11 | 0.93452, 0.92633 |

from here we can see that best accuracy is on depth=10 and depth=11 and depth=8 to some extent. thus, the larger the tree, the more the near the breakpoint. we have not found the breakpoint yet. 

# Gradient Boosting

### Analyzing GradientBoosting
| case number | imputer | scaler | grid | max depth | n estimators | criterion | max features | feature selection | no. of features | bagging | validation accuracy | roc | kaggle accuracy | analyzing |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 61c | simple | minmax | - | 6 | 300 | - | - | - | 78 | - | 0.9955442393380013 | 0.560056823582126 | 0.88298 | improved, lets add some features we used in decision trees |
| 65 | simple | minmax | - | 6 | 300 | squared-error | 60 | - | 78 | - | 0.9955984127199101 | 0.5545980654020854 | 0.88297 | accuracy remained same, lets increase the depth |
| 67 | simple | minmax | - | 10 | 300 | squared error | 60 | - | 78 | - | 0.9953140024648889 | 0.5702606327046124 | 0.79753 | too high depth ruined the accuracy |
| 80 | simple | minmax | - | 8 | 300 | squared error | 60 | - | 78 | - | 0.9951514823191625 | 0.5543197973296156 | 0.83659 | too low, lets try decreasing depth now to 5 |
| 86a | simple | minmax | - | 6 | 300 | squared error | 60 | forward | 10 | - | - | - | - | error, ran for 756 min, didnt work |
| 86b | simple | minmax | - | 6 | 300 | squared error | 60 | forward | 10 | - | - | - | - | error, again ran for 256 min with n_jobs = -1, didnt work |
| 89 | simple | minmax | - | 6 | 300 | squared error | 60 | kbest | 30 | - | 0.9961807765754297 | 0.5478679502290538 | 0.85929 | low, lets try bagging next |
| 102 | simple | minmax | - | 6 | 100 | squared error | 60 | - | 78 | estimators = 50 | 0.9975351111231496 | 0.5317120864929359 | 0.90158 | 24hour running: improved but not efficient |
| 116 | simple | minmax | param_grid = { 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] } | 3 | - | - | - | - | 78 | estimators = 50 | 0.9968308571583353 | 0.5404496829489119 | 0.84769 | very low. lets use this depth and repeat grid with estimators + learning rate |

total tests: 7  
total submissions: 5   
started accuracy: 0.88298   
highest accuracy: X (case Y)   
highest case parameters:
- params

analysis:
- each submission took AT LEAST three hours and more. some submissions took exceptional time like 12+ hours. 
- boosting is itself a very slow algorithm. using any forward or backword feature selector takes more than 12 hours and still doesnt even run on train data let alone full data and prediction
- the max_depth breakpoint was 6. too high depth leads to overfitting thus low accuracy
- kbest could not be rigourously tested as each submission took over 3 hours. 
- bagging does work but it takes extremely long, over 24 hours of running and laptop use. its not very feasible. accuracy improved by 2 percent but very inefficient. 

# Adaptive Boosting

### Analyzing AdaptiveBoosting
| case number | imputer | scaler | n estimators | learning rate | bagging | feature selector | no. of features | validation accuracy | roc | kaggle accuracy | analysis |
| - | - | - | - | - | - | - | - | - | - | - | - |
| 69 | simple | minmax | 100 | default = 0.5 | - | - | 78 | 0.9969527472676301 | 0.5430550209247345 | 0.94475 | - |
| 70 | simple | minmax | 200 | default = 0.5 | - | - | 78 | 0.996885030540244 | 0.5717622628834583 | 0.93379 | deterioration, too high estimators | 
| 71 | simple | minmax | 50 | default = 0.5 | - | - | 78 | 0.9970746373769248 | 0.5522487676032429 | 0.93853 | deterioration, too low estimators |
| 72 | simple | minmax | 75 | default = 0.5 | - | - | 78 | 0.9966141636307001 | 0.5660048884094492 | 0.94053 | improvement but not to the highest |
| 73 | simple | minmax | 110 | default = 0.5 | - | - | 78 | 0.9965464469033141 | 0.5644500047530453 | 0.94521 | improved! best estimators are between 100 and 200 |
| 74 | simple | minmax | 150 | default = 0.5 | - | - | 78 | 0.9971288107588336 | 0.5607445471728567 | 0.94780 | more improvement, we are closer to the breakpoint |
| 75 | simple | minmax | 160 | default = 0.5 | - | - | 78 | 0.9967360537399949 | 0.5648649150311703 | 0.94948 | lets increase 10 further |
| 76 | simple | minmax | 170 | default = 0.5 | - | - | 78 | 0.997088180722402 | 0.563818101949167 | 0.94966 | highest, lets increase 10 further |
| 77 | simple | minmax | 180 | default = 0.5 | - | - | 78 | 0.9969933773040617 | 0.5571944470850252 | 0.93516 | deterioration, breakpoint found! if possible, can try 175 to see if its highest |
| 81 | simple | minmax | 175 | default = 0.5 | - | - | 78 | 0.9970204639950161 | 0.5683960928977767 | 0.94949 | near to highest but not highest |
| 84 | simple | minmax | 175 | 0.1 | - | - | 78 | 0.9974538510502864 | 0.5026527855096471 | 0.93301 | too low, increase learning rate |
| 87 | simple | minmax | 170 | 0.75 | - | - | 78 | 0.9972371575226513 | 0.5632272905179162 | 0.93369 | not too good |
| 90 | simple | minmax | 170 | 0.6 | - | - | 78 | 0.9971152674133564 | 0.5492075935795596 | 0.94896 | not the highest, i guess learning rate=0.5 was the best |
| 114 | simple | minmax | 170 | default = 0.5 | estimators = 50 | - | 78 | 0.9972371575226513 | 0.5406533738276343 | 0.93386 | deterioration, lets use grid to find best params, bagging didnt do so well |

total tests: 13    
total submissions: 13    
starting accuracy: 0.94475   
highest accuracy: X (case Y)
highest case parameters:
- params

analysis:
- best estimator value is 170
- too many estimators and too less estimators can be wrong

### Analyzing Estimators with AdaBoost
| case number | n estimators | kaggle accuracy |
| ----------- | ------------ | --------------- | 
| 71 | 50 | 0.93853 |
| 72 | 75 | 0.94053 |
| 69 | 100 | 0.94475 |
| 73 | 110 | 0.94521 |
| 74 | 150 | 0.94780 |
| 75 | 160 | 0.94948 |
| 76, 87, 90 | 170 | 0.94966, 0.93369, 0.94896 |
| 81, 84 | 175 | 0.94949, 0.93301 |
| 77 | 180 | 0.93516 |
| 70 | 200 | 0.93379 |

from this table we can see that roughly 170 estimators is the breakpoint with the highest accuracy. lesser than 100 is too less estimators and more than 200 is too many estimators. in our search to find the breakpoint, we tested 10 different estimator values and found 170 as the best. 

### Analyzing Learning Rate with AdaBoost
| case number | learning rate | kaggle accuracy |
| ----------- | ------------- | --------------- |
| 84 | 0.1 | 0.93301 |
| 76 | default=0.5 | 0.94966 |
| 90 | 0.6 | 0.94896 |
| 87 | 0.75 | 0.93369 |

from this we can analyse that learning rate is best at default of 0.5, even though l.rate is good at 0.6, however best is at 0.5. having a too high or too low learning rate depreciates the accuracy performance. 

# LightGBM

### Analyzing LightGBM
| case number | imputer | scaler | grid |  max depth | n estimators | learning rate | min child weight | bagging params | feature selector | no. of features | validation accuracy | roc | kaggle accuracy | analysis |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 91 | simple | minmax | - | 10 | 100 | 0.9 | default = 0.001 | - | - | 78 | 0.9934450207890353 | 0.5815248594933158 | 0.75561 | lets try and increase estimators |
| 92 | simple | minmax | - | 10 | 400 | 0.9 | default = 0.001 | - | - | 78 | 0.9963297533756789 | 0.5020350035153998 | 0.50533 | very very low. lets normalize the learning rate |
| 93 | simple | minmax | - | 10 | 400 | 0.5 | default = 0.001 | - | - | 78 | 0.9897206007828053 | 0.5109621532460307 | 0.50408 | still low, estimators is the problem |
| 94 | simple | minmax | - | 9 | 100 | 0.5 | default = 0.001 | - | - | 78 | 0.9382288012784918 | 0.5318142882431025 | 0.37172 | wow. so low. lets add bagging because its not controlling |
| 95 | simple | maxabs | - | 10 | 100 | 0.9 | default = 0.001 | estimators = 50 | - | 78 | 0.9972777875590828 | 0.5 | 0.77939 | improved, but v low, lets increase estimators in bagging | 
| 96 | simple | maxabs | - | 10 | 100 | 0.9 | default = 0.001 | estimators = 100 | - | 78 | 0.997494481086718 | 0.5 | 0.77699 | reduced, lets go back and try feature selection | 
| 97 | knn=7 | maxabs | - | 10 | 100 | 0.9 | default = 0.001 | estimators = 50 | kbest | 5 | 0.9974403077048093 | 0.5 | 0.49841 | very low, we dont know if the issue is imputer or kbest selector |
| 98 | simple | maxabs | - | 10 | 200 | 0.1 | default = 0.001 | estimators = 50 | - | 78 | 0.9972777875590828 | 0.5024807720320663 | 0.87767 | relationship b/w estimators and learning rate is found |
| 99 | simple | maxabs | -| 10 | 300 | 0.01 | default = 0.001 | estimators = 50 | - | 78 | 0.9973048742500372 | 0.517066380205849 | 0.94165 | shotup! lets decrease depth now | 
| 100 | simple | maxabs | - | 8 | 300 | 0.01 | default = 0.001 | estimators = 50 | - | 78 | 0.9973455042864688 | 0.5174796298056683 | 0.94321 | increased, lets decrease depth further |
| 101 | simple | maxabs | - | 7 | 300 | 0.01 | default = 0.001 | estimators = 100 | - | 78 | 0.997359047631946 | 0.512612682865538 | 0.94351 | negligible increase, lets decrease bagging |
| 103 | simple | maxabs | - | 6 | 300 | 0.01 | default = 0.001 | estimators = 50 | - | 78 | 0.9976705445779216 | 0.5302129918143766 | 0.94395 | negligible increase, lets increase estimators + learning rate combo |
| 104 | simple | maxabs | param_grid = { 'max_depth': [1, 2, 3, 4, 5], 'learning_rate': [0.001, 0.005, 0.01, 0.05], 'n_estimators': [50, 100, 200, 300] } | 2 | 300 | 0.05 | default = 0.001 | - | - | 78 | 0.9973725909774233 | 0.5521437437723113 | 0.94948 | increased FINALLY. lets run a second grid search with diff parameters | 
| 111 | simple | maxabs | param_grid = { 'max_depth': [2, 3, 6, 7, 8, 9, 10], 'learning_rate': [0.001, 0.005, 0.01, 0.05], 'n_estimators': [400, 500, 1000, 2000, 3000] } | 3 | 1000 | 0.01 | default = 0.001 | estimators = 50 | - | 78 | 0.9974403077048093 | 0.5307284931466785 | 0.95106 | improved! lets grid with min child weight

starting accuracy: 0.75561     
highest accuracy:    
highest parameters:
- params

analysis: 
- relationship found between number of estimators and learning rate. less estimators == high learning rate. more estimators == low learning rate

# XGBoost

### Analyzing XGBoost
| case number | imputer | scaler | grid | n estimators | max depth | learning rate | bagging | feature selector | no. of features | validation accuracy | roc | kaggle accuracy | analysis |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 105 | simple | maxabs | - | - | - | - | estimators = 50 | - | 78 | 0.9972100708316969 | 0.5164319248826291 | 0.95474 | lets add some estimators |
| 106 | simple | maxabs | - | 100 | - | - | estimators = 50 | - | 78 | 0.9975215677776724 | 0.5210458424361445 | 0.95347 | deterioration, lets increase the estimators |
| 108 | simple | maxabs | - | 500 | - | - | estimators = 50 | - | 78 | 0.9973725909774233 | 0.5150685869118535 | 0.94900 | deterioration, too high estimators. lets decrease them and move to learning rate |
| 110 | simple | maxabs | - | 100 | - | 0.1 | estimators = 50 | - | 78 | 0.9975892845050585 | 0.5137985780696528 | 0.95063 | improved but not so much. lets try shifting depth next |
| 112 | simple | maxabs | param_grid = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]} | - | 2 | - | estimators = 50 | - | 78 | 0.9975215677776724 | 0.5390217640369339 | 0.95332 | good but overall not highest. lets put grid search for estimators now |

# CatBoost

### Analyzing CatBoost
| case number | imputer | scaler | grid | iterations | depth | learning rate | loss function | bagging | feature selector | no. of features | validation accuracy | roc | kaggle accuracy | analysis |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 107 | simple | maxabs | - | 1000 | default = 6 | 0.108132 | default = Logloss | - | - | 78 | 0.99729133090456 | 0.5215311004784688 | 0.93798 | lets introduce bagging |
| 109 | simple | maxabs | - | 100 | default = 6 | 0.5 | default = Logloss | estimators = 50 | - | 78 | 0.9975892845050585 | 0.5165542206956393 | 0.93612 | bagging didnt do so well. could be because estimators decreased and learning rate increased. lets change depth |
| 111 | simple | maxabs | - | default = 1000 | 10 | 0.108132 | Logloss | - | - | 78 | 0.99729133090456 | 0.504950495049505 | 0.92756 | deterioration, lets try grid to find the perfect depth |
| 115 | simple | maxabs | param_grid = { 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] } | default = 1000 | 1 | 0.108132 | default = Logloss | estimators = 50 | - | 78 | 0.9976028278505357 | 0.5496287205207183 | 0.94063 | improved! lets do grid on iterations and learning rate |