# DAY 1: Monday 21st October 2024

## * Case 1 - started
- DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=20), 
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal, 
- minmax scaling, 
- no grid

roc score = 0.5230564082198742   
Accuracy: 0.72913 

### ERROR SOLUTION
one error approached was that we had first splitted the data into features and target variables (X, Y) and then dropped all the NaN rows from X which made X and Y have differnet number of rows. after trying to "index" and match these rows, nothing was working. So a solution was found to restart the ipynb and first remove all the NaN rows from the train_data_processed and THEN split it into X, Y which resulted in same rows and this code line worked. 

## * Case 2 - changed criteria from gini to entropy
- DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=20)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal,
- minmax scaling,
- no grid

roc score = 0.5229797007820421   
Accuracy: 0.83323  

### Analyzing
entropy performed better then gini

## * Case 3 - decreased no.of samples per split
- DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=15)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5203814609840954    
accuracy: 0.83327

### Analyzing
splitting on smaller samples is better

## * Case 4 - max_depth increased
- DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=15)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5306000850899    
accuracy: 0.78883

### Analyzing
higher depth trees arent good

## * Case 5 - max_depth decreased
- DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=15)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5204372482116096    
accuracy: 0.85532

## * Case 6 - max_depth decreased
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5230843018336313    
accuracy: 0.87815

## * Case 7 - max_depth decreased
- DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=15)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5025494259738718    
accuracy: 0.87296

### Analyzing
smaller depth trees are good, but too small arent. keep till 5 depth

## * Case 8 - max_features introduced
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=10)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5154151037016982    
accuracy: 0.83207

## * Case 9 - max_features increased
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=50)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5128726511312657    
accuracy: 0.86817

## * Case 10 - max_features increased
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- rows removal
- minmax scaling
- no grid

roc score =  0.5205418492631988    
accuracy: 0.88788

### Analyzing
using less features isnt good. around 60 features improves accuracy

# DAY 2: Tuesday 22nd October 2024

## * Case 11 - simple, minmax
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- SimpleImputer(strategy='mean')
- minmax scaling
- no grid

roc score =  0.5184642260176702   
accuracy: 0.89330

## * Case 12 - knn5, minmax 
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=5)
- minmax scaling
- no grid

roc score =  0.5138345831202585      
accuracy: 0.87791

### Analyzing
MinMaxScaler average: 0.885605

## * Case 13 - simple, standard
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- SimpleImputer(strategy='mean')
- StandardScaler()
- no grid

roc score =  0.5086297947750887    
accuracy: 0.77620   

## * Case 14 - knn3, standard
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=3)
- StandardScaler()
- no grid

roc score =  0.5151673183025117     
accuracy: 0.80450

### Analyzing
StandardScaler average: 0.79035

## * Case 15 - simple, maxabse
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

# DAY 3: Wednesday 23rd October 2024

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

# DAY 4: Thursday 24th October 2024

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

# DAY 5: Friday 25th October 2024

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

## Case 43 - k=5
- KNeighborsClassifier(k=5)
- simple imputer
- minmax scaling
- no feature selection, 78 features used

model accuracy =  0.9972642442136056    
roc score =  0.5    
accuracy: 0.53003

## Case 44a - k=7, forward=10
-- ran for 1000 minutes + and failed
-- knn1.csv

## Case 44b - k=7, maxabs, knn=7, variance=0.0001, corr=0.87
78 by variance, 65 by correlation
model accuracy =  0.9973455042864688    
roc score =  0.5
-- knn4.csv

## Case 44c - simple, variance=0.1, corr=0.9
5 by variance, none by coorelation
model accuracy =  0.9974267643593321    
roc score =  0.5  
-- knn4.csv

### Analyzing
KNN + forward is running since past 6 hours, we are shifting to RandomForest.

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

## Case 49 - naivebayes, forward = 13
- naive bayes
- simple imputer
- minmax scaler
- forward=13 rows

model accuracy =  0.9724663786448529    
roc score =  0.657890605723844
accuracy: 0.87353

## Case 50 - naivebayes, forward = 14
- naive bayes
- simple imputer
- minmax scaler
- forward=14 rows

model accuracy =  0.97654292563349    
roc score =  0.6687466615541444  
accuracy: 0.87271  

# DAY 6: Saturday 26th October 2024

## Case 51 - best decision tree parameters
- RandomForestClassifier(max_depth=5, n_estimators=200, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80)
- maxabs scaler
- knn=7 imputer   

-- waiting for its result.... (45 minutes to fit and transform, 50 minutes to fit)   
model accuracy =  0.9970610940314476    
roc score =  0.5   
accuracy: 0.91554

## Case 52 - knearestneighbours=3, forward=5
- KNeighborsClassifier(n_neighbors=3)
- selection = SequentialFeatureSelector(model, direction='forward',n_features_to_select=5, scoring='roc_auc', n_jobs=-1)
- minmax scaler
- simple imputer
-- knn2.csv

model accuracy =  0.9965735335942685    
roc score =  0.5254314763954754 
accuracy: 0.55037

## Case 53 - knearest=11, kbest=20, knn=7
- KNeighborsClassifier(n_neighbors=11)
- SelectKBest(score_func=f_classif, k=20)
- knn=7 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9970340073404933    
roc score =  0.5   
accuracy: 0.59883

## Case 54 - kbest=15
- KNeighborsClassifier(n_neighbors=11)
- SelectKBest(score_func=f_classif, k=15)
- knn=7 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9976434578869673    
roc score =  0.5   
accuracy: 0.60509

## Case 55 - kbest=10
- KNeighborsClassifier(n_neighbors=11)
- SelectKBest(score_func=f_classif, k=10)
- knn=7 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9975486544686268    
roc score =  0.5055112852519732    
accuracy: 0.61709

## Case 56 - knn=3
- KNeighborsClassifier(n_neighbors=11)
- SelectKBest(score_func=f_classif, k=10)
- knn=3 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9972507008681284    
roc score =  0.5024509803921569
accuracy: 0.61709 [ file had not changed. unfortunately my entry is wasted ]

## Case 57 - kbest=5
- KNeighborsClassifier(n_neighbors=11)
- SelectKBest(score_func=f_classif, k=5)
- knn=3 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9974267643593321    
roc score =  0.5053136492515911 
accuracy: 0.62622

## Case 58 - random forest, n_estimators = 300
- RandomForestClassifier(max_depth=5, n_estimators=300, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80)
- maxabs scaler
- knn=7 imputer
... 39 minutes + 70 minutes
--rf1.csv

model accuracy =  0.997359047631946    
roc score =  0.5    
accuracy: 0.91889

## Case 59 - kbest=3
- KNeighborsClassifier(n_neighbors=11)
- SelectKBest(score_func=f_classif, k=3)
- knn=3 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9971152674133564    
roc score =  0.507075049343145    
accuracy: 0.62207

## Case 60 - random forest, depth = 6, trees=10
- RandomForestClassifier(max_depth=6, n_estimators=10, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80)
- maxabs scaler
- knn=7 imputer
-- rf1.csv

model accuracy =  0.9972642442136056    
roc score =  0.5    
accuracy: 0.91309

# DAY 7: Sunday 27th October 2024

## Case 61a - GNaiveBayes, row dropping
- gaussian naive bayes
- row removal
- minmax scaler
- forward=15 rows

// ERROR: does not work, as rows removal works only on train_data_set and then not for test_data_set or it would be rejected by Kaggle. But, NaiveBayes forward selection does not run on NaN values. 

## Case 61b - GNaiveBayes, row dropping, no feature selector
- gaussian naive bayes
- row removal
- minmax scaler
- no feature selector

// ERROR: does not work, if forward selection is not done, again it gives error upon training the model. Thus row-removal cannot be done. 

## Case 61c - GradientBoosting
- GradientBoostingClassifier(max_depth=6, n_estimators=300)
- no feature selector
- minmax scaler
- simple imputer

model accuracy =  0.9955442393380013    
roc score =  0.560056823582126    
accuracy: 0.88298