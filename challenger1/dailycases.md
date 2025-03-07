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

## * Case 16 - knn7, maxabs
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=7)
- MaxAbsScaler()
- no grid

roc score =  0.523861491669948   
accuracy: 0.88389

### Analyzing
MaxAbsScaler average: 0.884215

## * Case 17 - simple, robust
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- SimpleImputer(strategy='mean')
- RobustScaler
- no grid

roc score =  0.5181612733948899   
accuracy: 0.88517

## * Case 18 - knn5, robust
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=5)
- RobustScaler
- no grid

roc score =  0.5173654042244641   
accuracy: 0.87303

### Analyzing
RobustScaler average: 0.87910

## * Case 19 - simple, normalizer
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- SimpleImputer(strategy='mean')
- Normalizer
- no grid

roc score =  0.512349538904424   
accuracy: 0.64969

## * Case 20 - knn7, normalizer
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

## * Case 21 - knn3, minmax
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=3)
- minmax scaling
- no grid

roc score =  0.510941492151186      
accuracy: 0.89142

## * Case 22 - knn7, minmax
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=7)
- minmax scaling
- no grid

roc score =  0.5100979659586905   
accuracy: 0.88134

## * Case 23 - knn3, maxabs
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=3)
- MaxAbsScaler
- no grid

roc score =  0.5200733426222888    
accuracy: 0.88155

## * Case 24 - knn5, maxabs
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=5)
- MaxAbsScaler
- no grid

roc score =  0.5133797612483052   
accuracy: 0.84670

### analyzing
so according to updated table, knn=7 is best on average and we can alternate between MaxAbs and MinMax however MinMax will be prioritised.

## * Case 25 - min_samples_leaf introduced
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=50)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=7)
- MaxAbsScaler
- no grid

roc score =  0.5150632467068051   
accuracy: 0.85877

## * Case 26 - min_samples_leaf increased
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=7)
- MaxAbsScaler
- no grid

roc score = 0.5     
accuracy: 0.89522

## * Case 27 - min_samples_leaf increased
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=100)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- KNNImputer(n_neighbors=7)
- MaxAbsScaler
- no grid

roc score = 0.5    
accuracy: 0.85276

### analyzing
min_samples_leaf is fine at 80

## * Case 28 - crossfold at k=10
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80)
- crossfold = RepeatedKFold(n_splits=10, n_repeats=1)#, random_state=1)
- KNNImputer(n_neighbors=7)
- MaxAbsScaler
- no grid

'0.8964'   
accuracy: 0.89174

## * Case 29 - crossfold at k=5
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80)
- crossfold = RepeatedKFold(n_splits=5, n_repeats=1)#, random_state=1)
- KNNImputer(n_neighbors=7)
- MaxAbsScaler
- no grid

'0.8812'    
accuracy: 0.88935

## * Case 30 - crossfold at k=15
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

## * Case 31
- naive bayes
- simple imputer
- maxabs scaler
- 78 features

model accuracy =  0.8670720641412841    
roc score =  0.8015837569508879   
accuracy: 0.83725    

## * Case 32 - scaler changed to minmax
- naive bayes
- simple imputer
- minmax scaler
- 78 features

model accuracy =  0.9109254167964571    
roc score =  0.7956827745161095    
accuracy: 0.83350

## * Case 33 - imputer changed to knn7
- naive bayes
- knn7 imputer
- minmax scaler
- 78 features 

model accuracy =  0.9121307745439279       
roc score =  0.7969697685450362  
accuracy: 0.83350

### analyzing
NO DIFFERENCE! knn7 == simple. we will have to use feature selections now to try and improve the accuracy

## * Case 34 - simple, forward selection=5
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
not submitted

## * Case 35b - forward=20
- naive bayes
- simple imputer
- minmax scaler 
- forward=20

model accuracy =  0.9770711161071007    
roc score =  0.6887415307743486   
accuracy: 0.87148

### analyzer 
accuracy improved! meaning forward is doing good but at more features. lets try and implement filters now

## * Case 36 - variance and correlation filter
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
not submitted

## Case 37d - variance=0.05
14 columns extracted through variance and correlation didnt do more   
model accuracy =  0.9577176754201823    
roc score =  0.6956493745514011     
not submitted   

## Case 37e - variance=0.3, correlation=0.8
20 columns extracted through variance and correlation made it 19    
model accuracy =  0.9457318146728605    
roc score =  0.6916013795163546     
not submitted   

## * Case 37f - variance=0.3, correlation=0.9
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

## * Case 38 - variance=0.01
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
not submitted 

## * Case 39b - variance=0.001
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
not submitted

## Case 40b - variance=0.0001, correlation=0.8
78 by variance, 57 by correlation     
model accuracy =  0.9183200834270081    
roc score =  0.7965337080939787     
not submitted  

## Case 40c - variance=0.0001, correlation=0.85
78 by variance, 62 by correlation    
model accuracy =  0.9187128404458469    
roc score =  0.7588877586887783     
not submitted  

## * Case 40d - variance=0.0001, correlation=0.87
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

## * Case 41a - forward=30
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

## Case 42a - forward=25
-  naive bayes
- simple imputer
- minmax scaler
- forward=25 rows

model accuracy =  0.9617942224088194    
roc score =  0.7343364208492563    
not submitted
-- samplesubmission.csv     

## Case 42b - forward=25
-  naive bayes
- simple imputer
- minmax scaler
- forward=25 rows
model accuracy =  0.9550090063247424    
roc score =  0.7018435381296126    
not submitted    
-- nb1.csv   

## * Case 42c - forward=25
-  naive bayes
- simple imputer
- minmax scaler
- forward=25 rows

model accuracy =  0.8100274929913187    
roc score =  0.8185403319589406   
accuracy: 0.86875   
-- samplesubmission2.csv  

## * Case 43 - k=5
- KNeighborsClassifier(k=5)
- simple imputer
- minmax scaling
- no feature selection, 78 features used

model accuracy =  0.9972642442136056    
roc score =  0.5    
accuracy: 0.53003

## Case 44a - k=7, forward=10
- KNeighborsClassifier(k=7)
- simple imputer
- minmax scaling
- no feature selection, 78 features used
- ERROR: ran for 1000 minutes + and failed
-- knn1.csv

## Case 44b - k=7, maxabs, knn=7, variance=0.0001, corr=0.87
- KNeighborsClassifier(k=7)
- simple imputer
- maxabs scaling
- no feature selection, 78 features used
- 78 by variance, 65 by correlation

model accuracy =  0.9973455042864688    
roc score =  0.5     
not submitted
-- knn4.csv

## Case 44c - simple, variance=0.1, corr=0.9
- KNeighborsClassifier(k=7)
- simple imputer
- minmax scaling
- no feature selection, 78 features used
- 5 by variance, none by coorelation

model accuracy =  0.9974267643593321    
roc score =  0.5    
not submitted
-- knn4.csv

### Analyzing
KNN + forward is running since past 6 hours, we are shifting to RandomForest.

## * Case 44d - randomforest, simple, maxabs, max_depth=10, n_estimators=200
- RandomForestClassifier(max_depth=10, n_estimators=200)
- simple imputer
- maxabs scaler
- no feature importance or selection

model accuracy =  0.9973319609409916    
roc score =  0.5050251256281407   
accuracy: 0.90507

## * Case 45 - naivebayes, forward=15
- naive bayes
- simple imputer
- minmax scaler
- forward=15 rows

model accuracy =  0.9776399366171432    
roc score =  0.7000642467303131    
accuracy: 0.87413

## * Case 46 - knearestneighbours=7, simple, variance=0.001, corr=0.9
- KNeighborsClassifier(k=7)
- simple imputer
- minmax scaling
- variance=0.001, corr=0.9
- 70 by variance, 58 by correlation

model accuracy =  0.9973319609409916    
roc score =  0.5  
accuracy: 0.54796    
-- knn4.csv

## * Case 47 - knearest neighbours, k=7, kbest=30
- KNeighborsClassifier(k=7)
- simple imputer
- minmax scaling
- kbest=30
- 30

model accuracy =  0.9975892845050585    
roc score =  0.5028112828678414   
accuracy: 0.57056
-- knn3.csv

## * Case 48 - naivebayes, forward = 17
- naive bayes
- simple imputer
- minmax scaler
- forward=17 rows

model accuracy =  0.968064791364763    
roc score =  0.71144752877173    
accuracy: 0.87278

## * Case 49 - naivebayes, forward = 13
- naive bayes
- simple imputer
- minmax scaler
- forward=13 rows

model accuracy =  0.9724663786448529    
roc score =  0.657890605723844   
accuracy: 0.87353

## * Case 50 - naivebayes, forward = 14
- naive bayes
- simple imputer
- minmax scaler
- forward=14 rows

model accuracy =  0.97654292563349    
roc score =  0.6687466615541444  
accuracy: 0.87271  

# DAY 6: Saturday 26th October 2024

## * Case 51 - best decision tree parameters
- RandomForestClassifier(max_depth=5, n_estimators=200, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80)
- maxabs scaler
- knn=7 imputer   
- 45 minutes to fit and transform, 50 minutes to fit = 95min

model accuracy =  0.9970610940314476    
roc score =  0.5   
accuracy: 0.91554

## * Case 52 - knearestneighbours=3, forward=5
- KNeighborsClassifier(n_neighbors=3)
- selection = SequentialFeatureSelector(model, direction='forward',n_features_to_select=5, scoring='roc_auc', n_jobs=-1)
- minmax scaler
- simple imputer
-- knn2.csv

model accuracy =  0.9965735335942685    
roc score =  0.5254314763954754   
accuracy: 0.55037

## * Case 53 - knearest=11, kbest=20, knn=7
- KNeighborsClassifier(n_neighbors=11)
- SelectKBest(score_func=f_classif, k=20)
- knn=7 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9970340073404933    
roc score =  0.5   
accuracy: 0.59883

## * Case 54 - kbest=15
- KNeighborsClassifier(n_neighbors=11)
- SelectKBest(score_func=f_classif, k=15)
- knn=7 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9976434578869673    
roc score =  0.5   
accuracy: 0.60509

## * Case 55 - kbest=10
- KNeighborsClassifier(n_neighbors=11)
- SelectKBest(score_func=f_classif, k=10)
- knn=7 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9975486544686268    
roc score =  0.5055112852519732    
accuracy: 0.61709

## * Case 56 - knn=3
- KNeighborsClassifier(n_neighbors=11)
- SelectKBest(score_func=f_classif, k=10)
- knn=3 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9972507008681284    
roc score =  0.5024509803921569    
accuracy: 0.61709 [ file had not changed. unfortunately my entry is wasted ]

## * Case 57 - kbest=5
- KNeighborsClassifier(n_neighbors=11)
- SelectKBest(score_func=f_classif, k=5)
- knn=3 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9974267643593321    
roc score =  0.5053136492515911    
accuracy: 0.62622

## * Case 58 - random forest, n_estimators = 300
- RandomForestClassifier(max_depth=5, n_estimators=300, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80)
- maxabs scaler
- knn=7 imputer
- 39 minutes + 70 minutes = 109min
--rf1.csv

model accuracy =  0.997359047631946    
roc score =  0.5    
accuracy: 0.91889

## * Case 59 - kbest=3
- KNeighborsClassifier(n_neighbors=11)
- SelectKBest(score_func=f_classif, k=3)
- knn=3 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9971152674133564    
roc score =  0.507075049343145    
accuracy: 0.62207

## * Case 60 - random forest, depth = 6, trees=10
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

## * Case 61c - GradientBoosting
- GradientBoostingClassifier(max_depth=6, n_estimators=300)
- no feature selector
- minmax scaler
- simple imputer

model accuracy =  0.9955442393380013    
roc score =  0.560056823582126    
accuracy: 0.88298

## * Case 62 - randomforest, depth increased, trees increased
- RandomForestClassifier(max_depth=7, n_estimators=400, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80)
- maxabs scaler
- knn=7 imputer
- no feature selection
--rf1.csv
- 5min + 63min + 90min = 158min

model accuracy =  0.9974132210138549    
roc score =  0.5    
accuracy: 0.92693

## * Case 63 - knearest=9, kbest=5
- KNeighborsClassifier(n_neighbors=9)
- SelectKBest(score_func=f_classif, k=5)
- knn=3 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9972236141771741    
roc score =  0.5048989470087107   
accuracy: 0.63158

## * Case 64 - knearest=7, kbest=5
- KNeighborsClassifier(n_neighbors=7)
- SelectKBest(score_func=f_classif, k=5)
- knn=3 imputer
- minmax scaler
--knn2.csv

odel accuracy =  0.9974538510502864    
roc score =  0.5157623171129992    
accuracy: 0.61114

## * Case 65 - gradient boosting, criteria + max_depth added
- GradientBoostingClassifier(max_depth=6, n_estimators=300, criterion='squared_error', max_features=60)
- simple imputer
- minmax scaler
- no feature selection
-- gb1.csv

model accuracy =  0.9955984127199101    
roc score =  0.5545980654020854   
accuracy: 0.88297

## * Case 66 - random forest, depth increased
- RandomForestClassifier(max_depth=8, n_estimators=400, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80)
- maxabs scaler
- knn=7 imputer
- no feature selector
--rf1.csv

model accuracy =  0.9974403077048093    
roc score =  0.5    
accuracy: 0.93079  

## * Case 67 - gradient boosting, depth increased
- GradientBoostingClassifier(max_depth=10, n_estimators=300, criterion='squared_error', max_features=60)
- minmax scaler
- simple imputer
- no feature selector
--gb1.csv

model accuracy =  0.9953140024648889    
roc score =  0.5702606327046124  
accuracy: 0.79753

## * Case 68 - random forest, depth increased
- RandomForestClassifier(max_depth=10, n_estimators=400, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80)
- maxabs scaler
- knn=7 imputer
- no feature selector
--rf1.csv

model accuracy =  0.9972777875590828    
roc score =  0.5    
accuracy: 0.93256

## * Case 69 - adaptive boosting
- AdaBoostClassifier(n_estimators=100)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv

model accuracy =  0.9969527472676301    
roc score =  0.5430550209247345    
accuracy: 0.94475

## * Case 70 - adaptive boosting, estimators increased
- AdaBoostClassifier(n_estimators=200)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv

model accuracy =  0.996885030540244    
roc score =  0.5717622628834583    
accuracy: 0.93379

# DAY 8: Monday 28th October 2024

## * Case 71 - adaptive boosting, estimators decreased
- AdaBoostClassifier(n_estimators=50)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv

model accuracy =  0.9970746373769248    
roc score =  0.5522487676032429    
accuracy: 0.93853

## * Case 72 - adaptive boosting, estimators increased
- AdaBoostClassifier(n_estimators=75)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv

model accuracy =  0.9966141636307001    
roc score =  0.5660048884094492  
accuracy: 0.94053  

## * Case 73 - adaptive boosting, estimators increased
- AdaBoostClassifier(n_estimators=110)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv

model accuracy =  0.9965464469033141    
roc score =  0.5644500047530453    
accuracy: 0.94521

## * Case 74 - adaptive boosting, estimators increased
- AdaBoostClassifier(n_estimators=150)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv

model accuracy =  0.9971288107588336    
roc score =  0.5607445471728567    
accuracy: 0.94780

## * Case 75 - adaptive boosting, estimators increased
- AdaBoostClassifier(n_estimators=160)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv

model accuracy =  0.9967360537399949    
roc score =  0.5648649150311703    
accuracy: 0.94948

## * Case 76 - adaptive boosting, estimators increased
- AdaBoostClassifier(n_estimators=170)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv

model accuracy =  0.997088180722402    
roc score =  0.563818101949167    
accuracy: 0.94966

## * Case 77 - adaptive boosting, estimators increased
- AdaBoostClassifier(n_estimators=180)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv

model accuracy =  0.9969933773040617    
roc score =  0.5571944470850252    
accuracy: 0.93516

## * Case 78 - random forest, depth increased
- RandomForestClassifier(max_depth=11, n_estimators=400, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80)
- maxabs scaler
- knn=7 imputer
- no feature selector
--rf1.csv

model accuracy =  0.99729133090456    
roc score =  0.5    
accuracy: 0.93452

## * Case 79 - k-nearest neighbours, increased k
- KNeighborsClassifier(n_neighbors=300)
- SelectKBest(score_func=f_classif, k=5)
- knn=3 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9973861343229005    
roc score =  0.5    
accuracy: 0.81121

## * Case 80 - gradient boosting, depth decreased
- GradientBoostingClassifier(max_depth=8, n_estimators=300, criterion='squared_error', max_features=60)
- minmax scaler
- simple imputer
- no feature selector
--gb1.csv
- 92min + 66min = 158min

model accuracy =  0.9951514823191625    
roc score =  0.5543197973296156   
accuracy: 0.83659

# DAY 9: Tuesday 29th October 2024

## * Case 81 - adaptive boosting, estimators decreased
- AdaBoostClassifier(n_estimators=175)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv

model accuracy =  0.9970204639950161    
roc score =  0.5683960928977767    
accuracy: 0.94949

## * Case 82 - k-nearest neighbours, increased k
- KNeighborsClassifier(n_neighbors=500)
- SelectKBest(score_func=f_classif, k=5)
- knn=3 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9972507008681284    
roc score =  0.5    
accuracy: 0.82533

## * Case 83 - random forest, kbest=30
- RandomForestClassifier(max_depth=11, n_estimators=400, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80)
- maxabs scaler
- knn=7 imputer
- kbest feature selector, 30 features
--rf1.csv
- 51 min + 72 min = 123min

model accuracy =  0.9972371575226513    
roc score =  0.5   
accuracy: 0.92633

## * Case 84 - adaboost, learning rate introduced 
- AdaBoostClassifier(n_estimators=175, learning_rate=0.1)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv

model accuracy =  0.9974538510502864    
roc score =  0.5026527855096471  
accuracy: 0.93301

## * Case 85 - k-nearest neighbours, increased k
- KNeighborsClassifier(n_neighbors=1000)
- SelectKBest(score_func=f_classif, k=5)
- knn=3 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9973048742500372    
roc score =  0.5   
accuracy: 0.83911

## Case 86a - gradboost, forward=10
- GradientBoostingClassifier(max_depth=6, n_estimators=300, criterion='squared_error', max_features=60)
- simple imputer
- minmax scaler
- forward selection, 10 features
--gb1.csv
- ran it for 12.5 hours (756min) and then stopped, then turned on n_job=-1 and started re-running 
- ran it again for 256 min but laptop stopped working

## * Case 86b - k-nearest neighbours, increased k and weights
- KNeighborsClassifier(n_neighbors=2000, weights="distance")
- SelectKBest(score_func=f_classif, k=5)
- knn=3 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.9970746373769248    
roc score =  0.4999185147963549    
accuracy: 0.82641

## * Case 87 - adaboost, learning rate increased
- AdaBoostClassifier(n_estimators=170, learning_rate=0.75)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv

model accuracy =  0.9972371575226513    
roc score =  0.5632272905179162    
accuracy: 0.93369

## * Case 88 - k-nearest neighbours, decreased k
- KNeighborsClassifier(n_neighbors=1500, weights="distance")
- SelectKBest(score_func=f_classif, k=5)
- knn=3 imputer
- minmax scaler
--knn2.csv

model accuracy =  0.997088180722402    
roc score =  0.49995246438224067     
accuracy: 0.85212

## * Case 89 - gradboost, kbest=30
- GradientBoostingClassifier(max_depth=6, n_estimators=300, criterion='squared_error', max_features=60)
- simple imputer
- minmax scaler
- kbest, 30 features
--gb1.csv
- 27min + 35min = 62min

model accuracy =  0.9961807765754297    
roc score =  0.5478679502290538    
accuracy: 0.85929   

## * Case 90 - adaboost, learning rate decreased
- AdaBoostClassifier(n_estimators=170, learning_rate=0.6)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv

model accuracy =  0.9971152674133564    
roc score =  0.5492075935795596   
accuracy: 0.94896

# DAY 10: Wednesday 30th October 2024

## * Case 91 - lightgbm
- lgb.LGBMClassifier(max_depth=10, n_estimators=100, learning_rate=0.9)
- no bagging
- minmax scaler
- simple imputer
- no feature selector
--lgbm.csv

model accuracy =  0.9934450207890353    
roc score =  0.5815248594933158    
accuracy: 0.75561

## * Case 92 - lightgbm, increased estimators
- lgb.LGBMClassifier(max_depth=10, n_estimators=400, learning_rate=0.9)
- no bagging
- minmax scaler
- simple imputer
- no feature selector
--lgbm.csv

model accuracy =  0.9963297533756789    
roc score =  0.5020350035153998    
accuracy: 0.50533

## * Case 93 - lightgbm, default learning rate
- lgb.LGBMClassifier(max_depth=10, n_estimators=400, learning_rate=0.5)
- no bagging
- minmax scaler
- simple imputer
- no feature selector
--lgbm.csv

model accuracy =  0.9897206007828053    
roc score =  0.5109621532460307    
accuracy: 0.50408

## * Case 94 - lightgbm, decreased depth
- lgb.LGBMClassifier(max_depth=9, n_estimators=100, learning_rate=0.5)
- no bagging
- minmax scaler
- simple imputer
- no feature selector
--lgbm.csv

model accuracy =  0.9382288012784918    
roc score =  0.5318142882431025    
accuracy: 0.37172

## * Case 95 - lightgbm, depth10, higher learning rate, bagging, maxabs scaler
- lgb.LGBMClassifier(max_depth=10, n_estimators=100, learning_rate=0.9)
- BaggingClassifier(estimator=model, n_estimators=50)
- maxabs scaler
- simple imputer
- no feature selector
--lgbm.csv

model accuracy =  0.9972777875590828    
roc score =  0.5    
accuracy: 0.77939

## * Case 96 - lightgbm, higher bagging estimators
- lgb.LGBMClassifier(max_depth=10, n_estimators=100, learning_rate=0.9)
- BaggingClassifier(estimator=model, n_estimators=100)
- maxabs scaler
- simple imputer
- no feature selector
--lgbm.csv

model accuracy =  0.997494481086718    
roc score =  0.5    
accuracy: 0.77699

## * Case 97 - lightgbm, kbest feature selector added
- lgb.LGBMClassifier(max_depth=10, n_estimators=100, learning_rate=0.9)
- BaggingClassifier(estimator=model, n_estimators=50)
- maxabs scaler
- knn=7 imputer 
- SelectKBest(score_func=f_classif, k=5)
--lgbm.csv
 
model accuracy =  0.9974403077048093    
roc score =  0.5    
accuracy: 0.49841

### Analyzing
i remember hearing once that the more the estimators, the lesser the learning rate. so lets try that. lets increase estimators but decrease learning rate at the same time

## * Case 98 - lightgbm, learning rate decreased, estimators increased
- lgb.LGBMClassifier(max_depth=10, n_estimators=200, learning_rate=0.1)
- BaggingClassifier(estimator=model, n_estimators=50)
- maxabs scaler
- simple imputer
- no feature selection
--lgbm.csv

model accuracy =  0.9972777875590828    
roc score =  0.5024807720320663    
accuracy: 0.87767    

### Analyzing
this case just proved my theory. lets repeat - increase estimators and decrease learning rate

## * Case 99 - lightgbm, learning rate decreased, estimators increased
- lgb.LGBMClassifier(max_depth=10, n_estimators=300, learning_rate=0.01)
- BaggingClassifier(estimator=model, n_estimators=50)
- maxabs scaler
- simple imputer
- no feature selection
--lgbm.csv

model accuracy =  0.9973048742500372    
roc score =  0.517066380205849   
accuracy: 0.94165

### Analyzing
omg wow! i dont even know what to change to get higher. im thinking what to do - how do i use my 10th entry of the day?

## * Case 100 - lightgbm, max_depth decreased
- lgb.LGBMClassifier(max_depth=8, n_estimators=300, learning_rate=0.01)
- BaggingClassifier(estimator=model, n_estimators=50)
- maxabs scaler
- simple imputer
- no feature selection
--lgbm.csv

model accuracy =  0.9973455042864688    
roc score =  0.5174796298056683    
accuracy: 0.94321

# DAY 11: Thursday 31st October 2024

## * Case 101 - lightgbm, max_depth decreased, bagging estimators increased
- lgb.LGBMClassifier(max_depth=7, n_estimators=300, learning_rate=0.01)
- BaggingClassifier(estimator=model, n_estimators=100)
- maxabs scaler
- simple imputer
- no feature selection
--lgbm.csv

model accuracy =  0.997359047631946    
roc score =  0.512612682865538
accuracy: 0.94351

## * Case 102 - gradientboosting, bagging added
- GradientBoostingClassifier(max_depth=6, n_estimators=100, criterion='squared_error', max_features=60)
- BaggingClassifier(estimator=model, n_estimators=50)
- simple imputer
- minmax scaler
- no feature selector
--gb1.csv
- 453 min for training fitting + 1002 min

model accuracy =  0.9975351111231496    
roc score =  0.5317120864929359
accuracy: 0.90148

## * Case 103 - lightgbm, bagging estimators decreased
- lgb.LGBMClassifier(max_depth=7, n_estimators=300, learning_rate=0.01)
- BaggingClassifier(estimator=model, n_estimators=50)
- maxabs scaler
- simple imputer
- no feature selection
--lgbm.csv

model accuracy =  0.9976705445779216    
roc score =  0.5302129918143766
accuracy: 0.94395

## Case 104a - lightgbm, grid search
- lgb.LGBMClassifier(max_depth=6, n_estimators=400, learning_rate=0.001)
- param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9],
    'n_estimators': [50, 100, 200, 300, 400, 500, 1000, 2000, 3000]
}
- BaggingClassifier(estimator=model, n_estimators=50)
- maxabs scaler
- simple imputer
- no feature selection
--lgbm.csv
- failed after 271min, need to reduce grid search parameters as it was over 3100 fits

## * Case 104b - lightgbm, grid search
- lgb.LGBMClassifier(max_depth=6, n_estimators=400, learning_rate=0.001)
- param_grid = {
    'max_depth': [1, 2, 3, 4, 5],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'n_estimators': [50, 100, 200, 300]
}
- no bagging
- best params: {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300}
- maxabs scaler
- simple imputer
- no feature selection
--lgbm.csv

model accuracy =  0.9973725909774233    
roc score =  0.5521437437723113
accuracy: 0.94948

## * Case 105 - xgboost
- xgb.XGBClassifier() 
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- no feature selection
--xgb1.csv
 
model accuracy =  0.9972100708316969    
roc score =  0.5164319248826291   
accuracy: 0.95474

## * Case 106 - xgboost, n_estimators introduced
- xgb.XGBClassifier(n_estimators=100) 
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- no feature selection
--xgb1.csv

model accuracy =  0.9975215677776724    
roc score =  0.5210458424361445
accuracy: 0.95347

## * Case 107 - catboost
- CatBoostClassifier()
- no bagging
- Learning rate set to 0.108132
- simple imputer
- maxabs scaler
- no feature selection
--cat1.csv

model accuracy =  0.99729133090456    
roc score =  0.5215311004784688
accuracy: 0.93798

## * Case 108 - xgboost, n_estimators increased
- xgb.XGBClassifier(n_estimators=500) 
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- no feature selection
--xgb1.csv
- 33min + 50min

model accuracy =  0.9973725909774233    
roc score =  0.5150685869118535
accuracy: 0.94900

## * Case 109 - catboost, bagging introduced
- CatBoostClassifier(n_estimators=100)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- Learning rate set to 0.5
- simple imputer
- maxabs scaler
- no feature selection
--cat1.csv
- 22min + 10min

model accuracy =  0.9975892845050585    
roc score =  0.5165542206956393
accuracy: 0.93612

## * Case 110 - xgboost, estimators decreased + learning rate introduced
- xgb.XGBClassifier(n_estimators=100, learning_rate=0.1) 
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- no feature selection
--xgb1.csv

model accuracy =  0.9975892845050585    
roc score =  0.5137985780696528
accuracy: 0.95063

# DAY 12: Friday 1st November 2024

## * Case 111 - catboost, bagging removed, depth increased
- CatBoostClassifier(loss_function='Logloss', depth=10)
- no bagging
- Learning rate set to 0.108132
- simple imputer
- maxabs scaler
- no feature selection
--cat1.csv

model accuracy =  0.99729133090456    
roc score =  0.504950495049505
accuracy: 0.92756

## * Case 112 - xgboost, grid search for depth
- xgb.XGBClassifier()
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
- best depth found: 2
- XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=2, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, random_state=None, ...)
- simple imputer
- maxabs scaler
- no feature selection
--xgb1.csv

model accuracy =  0.9975215677776724    
roc score =  0.5390217640369339
accuracy: 0.95332

## * Case 113 - lightgbm, grid search
- lgb.LGBMClassifier()
- param_grid = {
    'max_depth': [2, 3, 6, 7, 8, 9, 10],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'n_estimators': [400, 500, 1000, 2000, 3000]
}
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- best params: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1000}
- maxabs scaler
- simple imputer
- no feature selection
--lgbm.csv
- 393min + 15min + 21min

model accuracy =  0.9974403077048093    
roc score =  0.5307284931466785
accuracy: 0.95106

## * Case 114 - adaboost, bagging
- AdaBoostClassifier(n_estimators=170)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv
- stopped at 216min
- restarted, 330min + 11min + 73 min, failed
- restarted, bismillah, 353min + 302min, completed

model accuracy =  0.9972371575226513    
roc score =  0.5406533738276343
accuracy: 0.93386

## Case 115a - catboost, grid search for depth, no bagging
- CatBoostClassifier()
- no bagging
- param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
- best depth: 1
- simple imputer
- maxabs scaler
- Learning rate set to 0.092856
- no feature selection
--cat1.csv
- 69min + 10min

model accuracy =  0.9975351111231496    
roc score =  0.5523504215805054
accuracy: not submitted

## * Case 115b - catboost, grid search for depth, bagging
- CatBoostClassifier()
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
- best depth: 1
- Learning rate set to 0.108132
- simple imputer
- maxabs scaler
- no feature selection
--cat1.csv
- 69min + 30min + 46 min

model accuracy =  0.9976028278505357    
roc score =  0.5496287205207183   
accuracy: 0.94063

## * Case 116 - gradboost, grid search for depth
- GradientBoostingClassifier()
- no bagging as with bagging takes 24 hours+
- param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
- best depth: 3
- minmax scaler
- simple imputer
- no feature selector
--gb1.csv 
- 512min + 12min + 15min

model accuracy =  0.9968308571583353    
roc score =  0.5404496829489119
accuracy: 0.84769

## * Case 117 - xgb, grid search for estimators
- xgb.XGBClassifier(max_depth=2)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- param_grid = {
    'n_estimators': [50, 100, 200, 300, 400, 500, 1000, 2000]
}
- best estimators found: 100
- XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=2, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=100, n_jobs=None,
              num_parallel_tree=None, random_state=None, ...)
- simple imputer
- maxabs scaler
- no feature selection
--xgb1.csv

model accuracy =  0.9974267643593321    
roc score =  0.5485165268480601
accuracy: 0.95236

## * Case 118 - randomforest, algo feature importance
- RandomForestClassifier(max_depth=11, n_estimators=400, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80)
- feature_importance_df['Feature'].head(35).values
- knn=7 imputer
- maxabs scaler
--rf1.csv
- ran it, 63min later, error
- fixed error, ran it, 72min later, error
- fixed error, ran it, 198min + 52min + 129min

model accuracy =  0.997359047631946    
roc score =  0.5
accuracy: 0.93546

## * Case 119 - lgbm, grid search on min_child_samples
- lgb.LGBMClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000) 
- param_grid = {
    'min_child_samples': [1, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000]
}
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- best params: {'min_child_samples': 40}
- maxabs scaler
- simple imputer
- no feature selection
--lgbm.csv

model accuracy =  0.9976028278505357    
roc score =  0.5320719837648076
accuracy: 

## * Case 120 - xgb, grid search for learning rate
- xgb.XGBClassifier(max_depth=2, n_estimators=100)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- param_grid = {
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 0.9]
}
- best learning_rate found: 0.2
- XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.2, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=2, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=100, n_jobs=None,
              num_parallel_tree=None, random_state=None, ...)
- simple imputer
- maxabs scaler
- no feature selection
--xgb1.csv

model accuracy =  0.9974403077048093    
roc score =  0.5430861000283999 
accuracy: 0.94989

# DAY 13: Saturday 2nd November 2024

## * Case 121 - catboost, grid search for estimators and learning rate
- CatBoostClassifier(max_depth=1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- param_grid = {
    'iterations': [50, 100, 200, 300, 500, 600, 700, 900, 1000, 2000],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9]
}
- best params: {'iterations': 2000, 'learning_rate': 0.1}
- simple imputer
- maxabs scaler
- no feature selection
--cat1.csv
- 147min + 58min + 93min

model accuracy =  0.9973861343229005    
roc score =  0.5287075514634675
accuracy: 0.94631

## * Case 122 - gradboost, grid search for depth + bagging
- same model trained as in case 116 is now being bagged, low bagging estimators though as 50 estimators take over 24 hours of running time (1 estimator takes approx 20-30 minutes on average)
- GradientBoostingClassifier()
- BaggingClassifier(estimator=model, n_estimators=10, verbose=2)
- param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
- best depth: 3
- minmax scaler
- simple imputer
- no feature selector
--gb1.csv 
- 512min + 229min + 186min

model accuracy =  0.9974809377412408    
roc score =  0.5458640497792794 
accuracy: 0.88551

## Case 123a - adaboost, grid search for best estimators and learning rate
- AdaBoostClassifier()
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- param_grid = {
    'n_estimators': [50, 100, 140, 160, 170, 180, 200, 300, 400, 500, 1000, 3000],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.6, 0.9]
}
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv
- after 605min of running, i calculated that it would take estimated 4105mins to run it all which is equivalent to 3 days of running. this is extremely inefficient. so i interrupted it and shortened the parameters

## Case 123b - randomforest, grid search for depth and estimators
- RandomForestClassifier(criterion='entropy', verbose=2)
- param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
    'n_estimators': [50, 100, 200, 300, 400, 500, 1000]
}
- best depth and estimators: 
- feature_importance_df['Feature'].head(35).values
- knn=7 imputer
- maxabs scaler
--rf1.csv
- code stopped running after 512min, so i manually stopped it. too many parameters in grid made it very long. 

## * Case 123c - lgbm, feature importance
- lgb.LGBMClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000) 
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- algorithm feature importance of top 35 features
--lgbm1.csv
- 16min + 19min

model accuracy =  0.9975757411595813    
roc score =  0.5317324536447935
accuracy: 0.95070

## * Case 124 - xgboost, feature importance
- xgb.XGBClassifier()
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- feature importance on top 35 features
- simple imputer
- maxabs scaler
--xgb1.csv
- 5min + 4min

model accuracy =  0.9973725909774233    
roc score =  0.5274660496761139  
accuracy: 0.95846

## * Case 125 - randomforest, grid search for depth and estimators
- RandomForestClassifier(criterion='entropy', verbose=2)
- param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
    'n_estimators': [50, 100, 200]
}
- best depth and estimators: {'max_depth': 9, 'n_estimators': 200}
- feature_importance_df['Feature'].head(35).values
- knn=7 imputer
- maxabs scaler
--rf1.csv
- 134min + 7min + 6min

model accuracy =  0.9973319609409916    
roc score =  0.5050251256281407
accuracy: 0.92154

## Case 126a - catboost, grid search for estimators and learning rate
- case 121 is same but now we have increased iterations to find is 2000 the breakpoint? or was it 2000 because it was the highest parameter in case 121
- CatBoostClassifier(max_depth=1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- param_grid = {
    'iterations': [2000, 2500, 3000, 3500],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9]
}
- best params: 
- simple imputer
- maxabs scaler
- no feature selection
--cat1.csv
- crashed after 170mins (VS code shut down)

## Case 126b - adaboost, grid search for best estimators and learning rate
// started at 2.Nov.24 1pm: acccording to calculations this will be completed at 3.Nov.24 6am. so this would most probably be entered as Day 14 entry
- AdaBoostClassifier()
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- param_grid = {
    'n_estimators': [50, 100, 140, 160, 170, 180, 200, 300, 400, 500],
    'learning_rate': [0.001, 0.005, 0.01, 0.05]
}
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv
- crashed after 300mins (VS code shut down)

## * Case 126c - lgbm, feature importance decreased featuree
- lgb.LGBMClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000) 
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- algorithm feature importance of top 20 features
--lgbm1.csv
- 6min + 11min

model accuracy =  0.9974267643593321    
roc score =  0.5321714275725801    
accuracy: 0.95323

## * Case 127 - xgboost, feature importance increased
- xgb.XGBClassifier()
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- feature importance on top 40 features
- simple imputer
- maxabs scaler
--xgb1.csv

model accuracy =  0.9975080244321952    
roc score =  0.5184142634887761    
accuracy: 0.95371

## * Case 128 - lgbm, feature importance decreased features
- lgb.LGBMClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000) 
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- algorithm feature importance of top 15 features
--lgbm1.csv

model accuracy =  0.9972777875590828    
roc score =  0.5286809703922187    
accuracy: 0.94903

## * Case 129 - xgboost, feature importance decreased
- xgb.XGBClassifier()
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- feature importance on top 30 features
- simple imputer
- maxabs scaler
--xgb1.csv

model accuracy =  0.9974403077048093    
roc score =  0.52283584980575    
accuracy: 0.94984

## * Case 130 - xgboost, feature importance increased
- xgb.XGBClassifier()
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- feature importance on top 36 features
- simple imputer
- maxabs scaler
--xgb1.csv

model accuracy =  0.9971694407952653    
roc score =  0.5118279679945126 
accuracy: 0.95522

# DAY 14: Sunday 3rd November 2024

## Case 131 - gboost, grid search for criterion
- GradientBoostingClassifier(max_depth=3)
- BaggingClassifier(estimator=model, n_estimators=10, verbose=2)
- param_grid = {
    'criterion': ['friedman_mse', 'squared_error']
}
- best criteria: {'criterion': 'friedman_mse'}
- minmax scaler
- simple imputer
- no feature selector
--gb1.csv 
- 92min + 156min + 163min

model accuracy =  0.9974132210138549    
roc score =  0.5332722268995048  
accuracy: 0.90057

## Case 132 - lgbm, feature importance increased features
- lgb.LGBMClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000) 
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- algorithm feature importance of top 25 features
--lgbm1.csv
- 16min + 18min + 2min

model accuracy =  0.9974538510502864    
roc score =  0.5331293168263355 
accuracy: 0.95242

## Case 133 - randomforest, grid search for depth and estimators
- RandomForestClassifier(criterion='entropy', max_depth=9, verbose=2)
- param_grid = {
    'n_estimators': [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
}
- best estimators: {'n_estimators': 650}
- feature_importance_df['Feature'].head(35).values
- knn=7 imputer
- maxabs scaler
--rf1.csv
- 463min + 12min + 145min + 13min

model accuracy =  0.9976570012324444    
roc score =  0.5167597765363129 
accuracy: 0.92877

## Case 134 - catboost, grid search for estimators and learning rate
- case 121 is same but now we have increased iterations to find is 2000 the breakpoint? or was it 2000 because it was the highest parameter in case 121
- CatBoostClassifier(max_depth=1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- param_grid = {
    'iterations': [2000, 2200, 2500],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9]
}
- best params: {'iterations': 2000, 'learning_rate': 0.1}
- simple imputer
- maxabs scaler
- no feature selection
--cat1.csv
- 91min + 49min + 57min

model accuracy =  0.9975892845050585    
roc score =  0.5304944615658465    
accuracy: 0.94664

## Case 135 - xgboost, grid search on all values
- xgb.XGBClassifier()
- param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 0.9],
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [1, 2, 3, 4]
}
- best parameters: {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 200}
- XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.1, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=2, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=200, n_jobs=None,
              num_parallel_tree=None, random_state=None, ...)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- feature importance on top 35 features
- simple imputer
- maxabs scaler
--xgb1.csv
- 15min + 2min + 3min

model accuracy =  0.997359047631946    
roc score =  0.5319857527264058    
accuracy: 0.94971  

## Case 136 - lgbm, estimators increased
- lgb.LGBMClassifier(learning_rate=0.01, max_depth=3, n_estimators=3000) 
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- algorithm feature importance of top 20 features
--lgbm1.csv
- 31min + 3min + 30min + 5min

model accuracy =  0.9974809377412408    
roc score =  0.5184913623160419
accuracy: 0.94173

## Case 137 - catboost, feature importance introduced
- CatBoostClassifier(max_depth=1, n_estimators=2000, learning_rate=0.1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- algorithm feature importance on top 20 features
--cat1.csv
- 45min + 45min

model accuracy =  0.9975892845050585    
roc score =  0.519719707742552  
accuracy: 0.95006

## Case 138 - xgboost, bagging increased
- xgb.XGBClassifier()
- BaggingClassifier(estimator=model, n_estimators=100, verbose=2)
- feature importance on top 35 features
- simple imputer
- maxabs scaler
--xgb1.csv
- 5min + 7min

model accuracy =  0.9976163711960129    
roc score =  0.5325815412932204
accuracy: 0.95979

## Case 139 - xgboost, bagging increased
- xgb.XGBClassifier()
- BaggingClassifier(estimator=model, n_estimators=150, verbose=2)
- feature importance on top 35 features
- simple imputer
- maxabs scaler
--xgb1.csv
- 7min + 9min

model accuracy =  0.9973725909774233    
roc score =  0.5249728397408912 
accuracy: 

## Case 140 - xgboost, bagging decreased
- xgb.XGBClassifier()
- BaggingClassifier(estimator=model, n_estimators=99, verbose=2)
- feature importance on top 35 features
- simple imputer
- maxabs scaler
--xgb1.csv
- 5min + 8min

model accuracy =  0.997697631268876    
roc score =  0.5251396648044693    
accuracy: 

# DAY 15: Monday 4th November 2024

## Case 141 - xgboost, bagging for loop
- xgb.XGBClassifier()
- xgboostANDbagging.ipynb is used, in which a for-loop is used and each bagging estimator is analysed to find the best bagging estimators. the HIGHEST ROC one is chosen which is 4
- BaggingClassifier(estimator=model, n_estimators=4, verbose=2)
- feature importance on top 35 features
- simple imputer
- maxabs scaler
--xgb1.csv

model accuracy = 0.9975351111231496
roc = 0.5287822330483466
accuracy: 0.94413

## Case 142 - catboost, feature importance decreased
- CatBoostClassifier(max_depth=1, n_estimators=2000, learning_rate=0.1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- algorithm feature importance on top 15 features
--cat1.csv
- 82min + 112min

model accuracy =  0.997494481086718    
roc score =  0.5264142921513115     
accuracy: 0.95191

## Case 143 - adaboost, grid search for best estimators and learning rate
- AdaBoostClassifier()
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- param_grid = {
    'n_estimators': [50, 100, 140, 160, 170, 180, 200, 300],
    'learning_rate': [0.001, 0.005]
}
- best params: {'learning_rate': 0.005, 'n_estimators': 300}
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv
- 210min + 747min + 15min + 644min + 15min

model accuracy =  0.9971965274862197    
roc score =  0.5    
accuracy: 

## Case 144 - catboost, feature importance decreased
- CatBoostClassifier(max_depth=1, n_estimators=2000, learning_rate=0.1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- algorithm feature importance on top 14 features
--cat1.csv
- 71min + 39min

model accuracy =  0.9975215677776724    
roc score =  0.5188849755093714    
accuracy: 

## Case 145 - xgboost, bagging for loop
- xgb.XGBClassifier()
- xgboostANDbagging.ipynb is used, in which a for-loop is used and each bagging estimator is analysed to find the best bagging estimators. the LOWEST ROC one is chosen which is
- BaggingClassifier(estimator=model, n_estimators=3, verbose=2)
- feature importance on top 35 features
- simple imputer
- maxabs scaler
--xgbBagging1.csv
- 148min

model accuracy = 0.997494481086718
roc = 0.5107323218497934
accuracy: 0.94978

## Case 146 - xgboost, correlation filter introduced
- xgb.XGBClassifier()
- BaggingClassifier(estimator=model, n_estimators=100, verbose=2)
- feature importance on top 35 features
- filter: correlation > 0.9, features left = 66, then feature importance to 35
- simple imputer
- maxabs scaler
--xgb1.csv
- MODEL IS INCORRECTLY TRAINED, ALL VALUES IN SUBMISSION ARE SAME

model accuracy =  0.9973048742500372    
roc score =  0.5    
accuracy: 0.50000

## Case 147 - catboost, feature importance decreased
- CatBoostClassifier(max_depth=1, n_estimators=2000, learning_rate=0.1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- algorithm feature importance on top 13 features
--cat1.csv
- 26min + 45min

model accuracy =  0.9976028278505357    
roc score =  0.5190217391304348    
accuracy: 0.94950

## Case 148 - ert, bagging
- ExtraTreesClassifier()
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- no feature selection
--ert1.csv
- 71min + 4min + 100min + 6min = 181min

model accuracy =  0.9976028278505357    
roc score =  0.5028089887640449   
accuracy: 0.91809

## Case 149 - ert, no bagging
- ExtraTreesClassifier(verbose=2) # verbose helps display the tree output to know status of the work going on
- no bagging
- simple imputer
- maxabs scaler
- no feature selection
--ert1.csv
- 4min + 3min = 7min

model accuracy =  0.9973861343229005    
roc score =  0.5051478496636952    
accuracy: 0.82590

## Case 150 - ert, estimators decreased
- ExtraTreesClassifier(n_estimators=10, verbose=2)
- no bagging
- simple imputer
- maxabs scaler
- no feature selection
--ert1.csv
- 1min + 2min = 3min

model accuracy =  0.9976299145414901    
roc score =  0.5138888888888888    
accuracy: 0.66440

# DAY 16: Tuesday 5th November 2024

## Case 151 - ert, bagging increased
- ExtraTreesClassifier(verbose=2)
- BaggingClassifier(estimator=model, n_estimators=100, verbose=2)
- simple imputer
- maxabs scaler
- no feature selection
--ert1.csv
- 121min + 42min

model accuracy =  0.9975215677776724    
roc score =  0.5    
accuracy: 0.91783

## Case 152 - catboost, bagging increased
- CatBoostClassifier(max_depth=1, n_estimators=2000, learning_rate=0.1)
- BaggingClassifier(estimator=model, n_estimators=100, verbose=2)
- simple imputer
- maxabs scaler
- algorithm feature importance on top 14 features
--cat1.csv
- 121min + 174min

model accuracy =  0.9974267643593321    
roc score =  0.5186486888775136    
accuracy: 0.95263

## Case 153 - ert, bootstrap inserted
- model = ExtraTreesClassifier(bootstrap=True)
- BaggingClassifier(estimator=model, n_estimators=100, verbose=2)
- simple imputer
- maxabs scaler
- no feature selection
--ert1.csv
- 27min + 2min + 41min + 3min = 73min

model accuracy =  0.9976705445779216    
roc score =  0.5028901734104047    
accuracy: 0.91258

## Case 154 - ert, bagging decreased
- model = ExtraTreesClassifier(bootstrap=True)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- no feature selection
--ert1.csv
- 17min + 2min + 17min + 2min = 38min

model accuracy =  0.9972371575226513    
roc score =  0.5    
accuracy: 0.91049

## Case 155 - gboost, grid search for estimators
- GradientBoostingClassifier(max_depth=3, criterion='friedman_mse')
- BaggingClassifier(estimator=model, n_estimators=10, verbose=2)
- param_grid = {
    'n_estimators': [50, 100, 200]
}
- best estimators: {'n_estimators': 200}
- minmax scaler
- simple imputer
- no feature selector
--gb1.csv 
- crashed after 200mins
- restarted, 93min + 154min + 200min = 447min

model accuracy =  0.9974267643593321    
roc score =  0.5358363294636074    
accuracy: 0.90485

## Case 156 - decision trees, best, PCA 15
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- no grid, no bagging
- knn=7 imputer
- maxabs scaler
- PCA(n_components=15), represents 80% features
--dt1.csv
- 10min max

roc score =  0.5
accuracy: 0.76281

## Case 157 - catboost, bagging decreased
- CatBoostClassifier(max_depth=1, n_estimators=2000, learning_rate=0.1)
- BaggingClassifier(estimator=model, n_estimators=75, verbose=2)
- simple imputer
- maxabs scaler
- algorithm feature importance on top 14 features
--cat1.csv
- 3min + 110min + 121min = 235min

model accuracy =  0.9976299145414901    
roc score =  0.5493758769911102    
accuracy: 0.95106

## Case 158 - decision trees, best, PCA graph plot
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- no grid, no bagging
- knn=7 imputer
- maxabs scaler
- PCA(n_components=24), captures 90% features
--dt1.csv

roc score =  0.5
accuracy: 0.76001

## Case 159 - decision trees, best, bagging 50
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- no grid
- knn=7 imputer
- maxabs scaler
--dt1.csv
- 34min

roc score =  0.5
accuracy: 0.91215

## Case 160 - ert, grid search on estimators
- ExtraTreesClassifier(verbose=2)
- param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
}
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- best estimators: {'n_estimators': 800}
- simple imputer
- maxabs scaler
- no feature selection
--ert1.csv
- 85min + 193min + 10min + 400min = 

model accuracy =  0.9974132210138549    
roc score =  0.5026041666666666    
accuracy: 0.92081

# DAY 17: Wednesday 6th November 2024

## Case 161 - voting, xgb + lgbm + xgb features
- model_1 = xgb.XGBClassifier()
- model_1 = featureImportance(model_1, 35, X, trainX, trainY, testX, test_data_processed)
- model_1 = BaggingClassifier(estimator=model_1, n_estimators=100, verbose=2)
- model_2 = lgb.LGBMClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000)
- model_2 = BaggingClassifier(estimator=model_2, n_estimators=50, verbose=2)
- VotingClassifier(estimators=[['xgb_best_fimp', model_1], ['lgb_best', model_2]], voting='soft', verbose=True)
- simple imputer
- maxabs scaler
--voting1.csv
- 14.8mins to train XGB model + code crashed. amazing
- running again, 45min (20 + 2 + 24 + 2)

model accuracy =  0.9974538510502864    
roc score =  0.5324932099352228   
accuracy: 0.95250

## Case 162 - voting, lgbm + xgb features
- xgb = xgb.XGBClassifier()
- xgb = featureImportance(xgb, 35, X, trainX, trainY, testX, test_data_processed)
- model_2 = lgb.LGBMClassifier(learning_rate=0.02, max_depth=2, n_estimators=4000)
- model_1 = BaggingClassifier(estimator=model_2, n_estimators=50, verbose=2)
- model = VotingClassifier(estimators=[['lgb_bagged', model_1], ['lgb', model_2]], voting='soft', verbose=True)
- simple imputer
- maxabs scaler
--voting1.csv
- 60min + 2min + 67min + 3min

model accuracy =  0.997765347996262    
roc score =  0.5406501620314466    
accuracy: 0.95703

## Case 163 - decision trees, best, algo feature importance 35
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- no grid
- knn=7 imputer
- maxabs scaler
- algorithm feature importance, 35 features
--dt1.csv
- 15min

roc score =  0.5
accuracy: 0.91039

## Case 164 - naive bayes, best, bagging 50
- GaussianNB()
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- minmax scaler
- SequentialFeatureSelector(model, direction='forward',n_features_to_select=15, scoring='roc_auc')
--nb1.csv
- 10mins

model accuracy =  0.9773148963256904    
roc score =  0.6661261228757619    
accuracy: 0.87014

## Case 165a - naive bayes, best, algorithm feature importance
- GaussianNB()
- no bagging
- simple imputer
- minmax scaler
- algorithm feature importance, 20 features
--nb1.csv
ERR0R, NaiveBayes does not have feature importance

## Case 165b - naive bayes, best, PCA
- GaussianNB()
- no bagging
- simple imputer
- minmax scaler
- PCA(n_components=24) covers 90%
--nb1.csv
- 2min

model accuracy =  0.9702181832956377    
roc score =  0.6879233888954087    
accuracy: 0.85555

## Case 166a - lgbm, best, forward=10
- lgb.LGBMClassifier(learning_rate=0.01, max_depth=3, n_estimators=3000) 
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- forward of 10 features
--lgbm1.csv
- ERROR, crashed after 45mins of running

## Case 166b - lgbm, best, forward=5
- lgb.LGBMClassifier(learning_rate=0.01, max_depth=3, n_estimators=3000) 
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- forward of 5 features
--lgbm1.csv
- ERROR, crashed after 30mins of running

## Case 166c - lgbm, best, PCA 24
- lgb.LGBMClassifier(learning_rate=0.01, max_depth=3, n_estimators=3000) 
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- PCA(n_components=24) 
- no feature importance or selection
--lgb1.csv
- 45min + 6min + 60min + 15min

model accuracy =  0.9973996776683777    
roc score =  0.5201884404412956    
accuracy: 0.89981

## Case 167 - xgb, best, PCA
- xgb.XGBClassifier()
- BaggingClassifier(estimator=model, n_estimators=100, verbose=2)
- PCA at 24 features
- simple imputer
- maxabs scaler
--xgb1.csv
- 5min + 6min

model accuracy =  0.9975486544686268    
roc score =  0.5027472527472527    
accuracy: 0.88955

## Case 168 - xgb, best, kbest=35
- xgb.XGBClassifier()
- BaggingClassifier(estimator=model, n_estimators=100, verbose=2)
- SelectKBest(score_func=f_classif, k=35) 
- simple imputer
- maxabs scaler
--xgb1.csv
- 5min + 9min 

model accuracy =  0.9972642442136056    
roc score =  0.5144859629128801    
accuracy: 0.95000

## Case 169 - k-nearest neighbours, PCA=25
- KNeighborsClassifier(n_neighbors=1500, weights="distance")
- PCA(n_components=25)    
- knn=3 imputer
- minmax scaler
--knn2.csv
- 17min

model accuracy =  0.9971152674133564    
roc score =  0.5    
accuracy: 0.79966

## Case 170a - xgb, best, forward=10
- xgb.XGBClassifier()
- BaggingClassifier(estimator=model, n_estimators=100, verbose=2)
- SequentialFeatureSelector(model, direction='forward',n_features_to_select=10, scoring='roc_auc')
- simple imputer
- maxabs scaler
--xgb1.csv
- ERROR, failed after 20 minutes of running

## Case 170b - xgb, best, forward=5
- xgb.XGBClassifier()
- BaggingClassifier(estimator=model, n_estimators=100, verbose=2)
- SequentialFeatureSelector(model, direction='forward',n_features_to_select=5, scoring='roc_auc')
- simple imputer
- maxabs scaler
--xgb1.csv
- 40min + 16min + idunno = 61min total from file

model accuracy =  0.9976163711960129    
roc score =  0.5029042822218859    
accuracy: 0.93188

# DAY 18: Thursday 7th November 2024

## Case 171 - ert, kbest features
- ExtraTreesClassifier(n_estimators=800, verbose=1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- SelectKBest(score_func=f_classif, k=15) 
--ert1.csv
- 66min + 10min + 261min + 17min = 360min approx

model accuracy =  0.9971829841407425    
roc score =  0.5023923444976076    
accuracy: 0.87757

## Case 172 - k-nearest neighbours, PCA=25
- KNeighborsClassifier(n_neighbors=1500, weights="distance")
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2) 
- SelectKBest(score_func=f_classif, k=5) 
- knn=3 imputer
- minmax scaler
--knn2.csv
- 450min (15min imputer + 132min pred + 298min pred)

model accuracy =  0.9971829841407425    
roc score =  0.5023923444976076    
accuracy: 0.83150

## Case 173 - catboost, best, kbest 14
- CatBoostClassifier(max_depth=1, n_estimators=2000, learning_rate=0.1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- SelectKBest(score_func=f_classif, k=14)
- simple imputer
- maxabs scaler
--cat1.csv
- 87min

model accuracy =  0.997155897449788    
roc score =  0.5141237787103181    
accuracy: 0.93659

## Case 174a - catboost, best, forward=10
- CatBoostClassifier(max_depth=1, n_estimators=2000, learning_rate=0.1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- SequentialFeatureSelector(model, direction='forward',n_features_to_select=10, scoring='roc_auc')
- simple imputer
- maxabs scaler
--cat1.csv
- ERROR, crashed after 110mins

## Case 174b - catboost, best, forward=5
- CatBoostClassifier(max_depth=1, n_estimators=2000, learning_rate=0.1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- SequentialFeatureSelector(model, direction='forward',n_features_to_select=5, scoring='roc_auc')
- simple imputer
- maxabs scaler
--cat1.csv
- ERROR, crashed after 10mins, took too much memory RAM

## Case 174c - ert, algorithm feature importance
- ExtraTreesClassifier(n_estimators=800, verbose=1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- algorithm feature importance, 20 features
--ert1.csv
- ERROR, crashed after 100mins, will retry this at night when laptop is not in use

## Case 174d - catboost, best, PCA
- CatBoostClassifier(max_depth=1, n_estimators=2000, learning_rate=0.1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- PCA(n_components=33) 
- simple imputer
- maxabs scaler
--cat1.csv
- 52min + 72min

model accuracy =  0.9972777875590828    
roc score =  0.5074121620758532    
accuracy: 0.90736

## Case 175 - adaboost, best, bagged = 10
- AdaBoostClassifier(n_estimator=170)
- BaggingClassifier(estimator=model, n_estimators=10, verbose=2)
- minmax scaler
- simple imputer
- no feature selector
--ab1.csv
- 180min

model accuracy =  0.9975486544686268    
roc score =  0.5478800520332386    
accuracy: 0.93704

## Case 176 - adaboost, best, PCA
- AdaBoostClassifier(n_estimator=170)
- no bagging
- minmax scaler
- simple imputer
- PCA(n_components=25)
--ab1.csv
- 20min

model accuracy =  0.9968579438492896    
roc score =  0.5312123336570682    
accuracy: 0.88640

## Case 177 - adaboost, best, algorithm feature importance
- AdaBoostClassifier(n_estimator=170)
- no bagging
- minmax scaler
- simple imputer
- algorithm feature importance, 20 features
--ab1.csv
- 30min

model accuracy =  0.9975757411595813    
roc score =  0.5437035391127617    
accuracy: 0.94906

## Case 178 - adaboost, best, kbest
- AdaBoostClassifier(n_estimator=170)
- no bagging
- minmax scaler
- simple imputer
- SelectKBest(score_func=f_classif, k=30)
--ab1.csv
- 15min

model accuracy =  0.9970340073404933    
roc score =  0.5412529083325384    
accuracy: 0.94456

## Case 179 - voting, lgbm, lgbm+bagging // done on kaggle
- in previous 2 voting cases, feature importance did not work and was still 78 features. i have fixed the error and re ran the second case with tuned parmeters
- xgb_m = xgb.XGBClassifier(n_estimators=2000 ,learning_rate= 0.03, max_depth = 4, random_state  = 42, device = "cuda")
- xgb_m, X, trainX, trainY, testX, test_data_processed = featureImportance(xgb_m, 40, X, trainX, trainY, testX, test_data_processed)
- model_2 = lgb.LGBMClassifier(learning_rate=0.02, max_depth=2, n_estimators=4000 , device='gpu')
- model_1 = BaggingClassifier(estimator= model_2, n_estimators=100, verbose=2, n_jobs=-1)
- model = VotingClassifier(estimators=[('bg_c', model_1), ('lgb2', model_2)], voting='soft', verbose=True)
- simple imputer
- minmax scaler
--voting1.csv

model accuracy =  0.9972236141771741    
roc score =  0.5271326592834472    
accuracy: 0.96165

## Case 180 - gradient boosting, best, algo feature imp
- GradientBoostingClassifier(max_depth=3, criterion='friedman_mse', n_estimators=200) 
- no bagging
- algo feature importance of best 20 features
- minmax scaler
- simpe imputer
--gb1.csv

model accuracy =  0.9972642442136056    
roc score =  0.5331975412581588    
accuracy: 

# DAY 19: Friday 8th November 2024

## Case 181 - ert, algorithm feature importance
- ExtraTreesClassifier(n_estimators=800, verbose=1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- algorithm feature importance, 20 features
--ert1.csv
- 240min

model accuracy =  0.9975892845050585    
roc score =  0.5055555555555555    
accuracy: 0.88873

## Case 182a - random forest + bagging
- RandomForestClassifier(max_depth=11, n_estimators=400, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80, verbose=1)
- BaggingClassifier(estimator=model, n_estimators=10, verbose=2)
- feature importances of 35 features
- knn=7 imputer
- maxabs scaler
--rf1.csv
- ran it for 10 hours+ and then an error came, and realized code was wrong and wil have to re-run (bagged first and then tried to extract feature importances which is wrong)
- running for 414mins and this comes: 'The Kernel crashed while executing code in the current cell or a previous cell.' and it says 'Python Kernel is dead'

## Case 182b - ert, PCA
- ExtraTreesClassifier(n_estimators=800, verbose=1)
- BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- maxabs scaler
- PCA(n_components=25) 
--ert1.csv
- ERROR: after 8h running: The Kernel crashed while executing code in the current cell or a previous cell. Python kernel is dead. 

## Case 182c - ert, PCA
- ExtraTreesClassifier(n_estimators=800, verbose=1)
- no bagging
- simple imputer
- maxabs scaler
- PCA(n_components=25) 
--ert1.csv

model accuracy =  0.9974809377412408    
roc score =  0.5    
accuracy: 0.85415

## Case 183 - gradient boosting, best PCA
- GradientBoostingClassifier(max_depth=3, criterion='friedman_mse', n_estimators=200) 
- no bagging
- PCA(n_components=40) 
- minmax scaler
- simpe imputer
--gb1.csv
- 25min + 41min = 66min

model accuracy =  0.9962484933028156    
roc score =  0.5144364246234909    
accuracy: 0.86502

## Case 184 - random forest, algo feature imp
- RandomForestClassifier(max_depth=11, n_estimators=400, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80, verbose=1)
- feature importances of 20 features
- knn=7 imputer
- maxabs scaler
--rf1.csv
- 127min + 33min + 100min = 260min

model accuracy =  0.9972371575226513    
roc score =  0.5    
accuracy: 0.93581

## Case 185 - random forest, kbest
- RandomForestClassifier(max_depth=11, n_estimators=400, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80, verbose=1)
- SelectKBest(score_func=f_classif, k=15)    
- knn=7 imputer
- maxabs scaler
--rf1.csv
- 45min

model accuracy =  0.9974403077048093    
roc score =  0.5    
accuracy: 0.92622

## Case 186 - categorical NB
- CategoricalNB()
- minmax scaler
- simple imputer
- categorical cols: ['X5','X8','X11','X4', 'X6', 'X10', 'X16']
- no bagging
- no feature selection
--nb1.csv
- 2min

model accuracy =  0.9489686742419112    
roc score =  0.6666960854865647    
accuracy: 0.81229

## Case 187 - categorical NB, bagged
- CategoricalNB()
- minmax scaler
- simple imputer
- categorical cols: ['X5','X8','X11','X4', 'X6', 'X10', 'X16']
- BaggingClassifier(estimator=model, n_estimators=10, verbose=2)
- no feature selection
--nb1.csv
- 4min

model accuracy =  0.9513523030458984    
roc score =  0.6672982355307021    
accuracy: 0.81199 

### Anayzing
bagging doesnt have a good response, lets not do it

## Case 188 - random forest, PCA
- RandomForestClassifier(max_depth=11, n_estimators=400, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80, verbose=1)
- PCA(n_components=25)    
- knn=7 imputer
- maxabs scaler
--rf1.csv
- 210min

model accuracy =  0.9974673943957636    
roc score =  0.5    
accuracy: 0.88214

## Case 189 - categorical NB, alpha params
- CategoricalNB(alpha=0.1, fit_prior=False)
- minmax scaler
- simple imputer
- categorical cols: ['X5','X8','X11','X4', 'X6', 'X10', 'X16']
- no bagging
- no feature selection
--nb1.csv
- 4min

model accuracy =  0.7837534027655512    
roc score =  0.7802110762469083    
accuracy: 0.81257

## Case 190 - categorical NB, alpha params
- CategoricalNB(alpha=0.001, fit_prior=False)
- minmax scaler
- simple imputer
- categorical cols: ['X5','X8','X11','X4', 'X6', 'X10', 'X16']
- no bagging
- no feature selection
--nb1.csv
- 4min

model accuracy =  0.7860828581876295    
roc score =  0.7687108687644401    
accuracy: 0.81312

# DAY 20: Saturday 9th November 2024

## Case 191a - voting, lgbm, lgbm+bagging
- xgb_m = xgb.XGBClassifier(n_estimators=2000 ,learning_rate= 0.03, max_depth = 4, random_state  = 42, device = "cuda")
- xgb_m = featureImportance(xgb_m, 45)
- model_2 = lgb.LGBMClassifier(learning_rate=0.02, max_depth=2, n_estimators=4000 , device='gpu')
- model_1 = BaggingClassifier(estimator= model_2, n_estimators=150, verbose=2, n_jobs=-1)
- model = VotingClassifier(estimators=[('bg_c', model_1), ('lgb2', model_2)], voting='soft', verbose=True)
- simple imputer
- minmax scaler
--voting1.csv
- ran it 4 times on kaggle, 2 times on google colab, har dafa koi naya masla. we have switched to VSCode now
- error, failed execution at 1150min

## Case 191b - voting, lgbm, lgbm+bagging, xgb
- model_3 = xgb.XGBClassifier( n_estimators=2000, learning_rate=0.03, max_depth=4, random_state=42,tree_method='hist', n_jobs=-1, eval_metric='auc', verbosity=2 )
- model_3, X, trainX, trainY, testX, test_data_processed = featureImportance( model_3, 45, X, trainX, trainY, testX, test_data_processed )
- model_2 = lgb.LGBMClassifier( learning_rate=0.02, max_depth=2, n_estimators=2000, device='gpu', n_jobs=-1, verbose=1, min_child_samples=20, min_split_gain=0.1 )
- model_2.fit(trainX, trainY)
- model_1 = BaggingClassifier( estimator=model_2, n_estimators=100, verbose=2, n_jobs=-1 )
- model_1.fit(trainX, trainY)
- model = VotingClassifier( estimators=[('bg_c', model_1), ('lgb2', model_2), ('xgb', model_3)], voting='soft', verbose=True, n_jobs=-1 )
- simple imputer
- minmax scaler
--voting1.csv
error, failed too many times

### Analyzing
voting is failing on too many parameters on VS code. i am running stacking on kaggle.
lets use smaller and lesser parameters on VS code. the goal is to complete the entries. 

## Case 191c - stacking, xgb, lgb, lgb+bagged
- simple imputer
- minmax scaler
- model_3 = xgb.XGBClassifier(n_estimators=2000 ,learning_rate= 0.03, max_depth = 4, random_state  = 42, device = "cuda")
- model_3 = featureImportance(model_3, 40)
- model_2 = lgb.LGBMClassifier(learning_rate=0.02, max_depth=2, n_estimators=4000 , device='gpu')
- model_1 = BaggingClassifier(estimator= model_2, n_estimators=100, verbose=2, n_jobs=-1)
- estimators = [('model_1', model_1), ('model_2', model_2), ('model_3', model_3)]
- model = StackingClassifier(estimators=estimators, final_estimator=model_2)
--stacking1.csv
- ERROR: Failed after running twice (5h each)

## Case 191d - stacking, decision trees
- simple imputer
- minmax scaler
- model_1 = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80)
- model_2 = BaggingClassifier(estimator=model_1, n_estimators=50, verbose=2)
- model = StackingClassifier(estimators=estimators, final_estimator=DecisionTreeClassifier(), verbose=2)
- model_4 = featureImportance( xgb.XGBClassifier(), 45 )
--stacking1.csv
- 30min

model accuracy =  0.9948941587550957    
roc score =  0.5383070227984298    
accuracy: 0.52617

## Case 192 - categorical NB, alpha params decreased
- CategoricalNB(alpha=0.0000000000001, fit_prior=False)
- minmax scaler
- simple imputer
- categorical cols: ['X5','X8','X11','X4', 'X6', 'X10', 'X16']
- no bagging
- no feature selection
--nb1.csv
- 2min

model accuracy =  0.7884935736825711    
roc score =  0.7787090047089726    
accuracy: 0.81346

## Case 193 - categorical NB, knn=7
- CategoricalNB(alpha=0.0000000000001, fit_prior=False)
- minmax scaler
- knn=7 imputer
- categorical cols: ['X5','X8','X11','X4', 'X6', 'X10', 'X16']
- no bagging
- no feature selection
--nb1.csv
- 2min

model accuracy =  0.7848097837127728    
roc score =  0.7976348770414773   
accuracy: 0.81346

## Case 194a - stacking, decision trees
- simple imputer
- minmax scaler
- model_1 = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, random_state=42)
- model_2 = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=10, random_state=42)
- model_3 = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=5, random_state=42)
- model = StackingClassifier(estimators=estimators, final_estimator=DecisionTreeClassifier(max_depth=8, random_state=42))
- model_4 = featureImportance( xgb.XGBClassifier(), 45 )
--stacking1.csv
- 5min

model accuracy =  0.9971152674133564    
roc score =  0.5320152518409307    
accuracy: didnt submit yet

## Case 194b - stacking, decision trees, stacking bagged
- simple imputer
- minmax scaler
- model_1 = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, random_state=42)
- model_2 = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=10, random_state=42)
- model_3 = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=5, random_state=42)
- model = StackingClassifier(estimators=estimators, final_estimator=DecisionTreeClassifier(max_depth=8, random_state=42))
- model_4 = featureImportance( xgb.XGBClassifier(), 45 )
- model = BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
--stacking2.csv
- 120min

model accuracy =  0.9972642442136056    
roc score =  0.5           
accuracy: 0.91600

## Case 195 - voting, lgb, randomforest+bagged, ert, xgb
- model_1 = lgb.LGBMClassifier(n_estimators=150, max_depth=7, learning_rate=0.07)
- model_2 = BaggingClassifier(estimator=RandomForestClassifier(), n_estimators=70, max_samples=0.8, max_features=0.8, verbose=2)
- model_3 = ExtraTreesClassifier(n_estimators=50, max_depth=8, max_features='sqrt', verbose=2)
- model_4 = xgb.XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
- model = VotingClassifier( estimators=[ ('lgb', model_1), ('bagging', model_2), ('extra_trees', model_3), ('xgb', model_4) ], voting='soft',  verbose=True )
- model_4 = featureImportance( model_4, 45 )
- simple imputer
- minmax scaler
--voting1.csv
- 166min + 2min + 265min + 3min

model accuracy =  0.9973184175955144    
roc score =  0.5148379046501106        
accuracy: 0.93356

## Case 196 - voting, lgb, xgb, ert, cat
- model_1 = lgb.LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.05, num_leaves=50)
- model_2 = xgb.XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', use_label_encoder=False)
- model_3 = ExtraTreesClassifier(n_estimators=100, max_depth=12, max_features='sqrt', verbose=2)
- model_4 = CatBoostClassifier(iterations=300, depth=10, learning_rate=0.03, verbose=2)
- model_4 = featureImportance( model_4, 45 )
- 10min

model accuracy =  0.99729133090456    
roc score =  0.5098874096006056    
accuracy: 0.92521

## Case 197 - stacking, nb
- model_1 = GaussianNB(var_smoothing=1e-9)
- model_2 = GaussianNB(var_smoothing=1e-8)
- model_3 = GaussianNB(var_smoothing=1e-7)
- estimators = [('nb1', model_1), ('nb2', model_2), ('nb3', model_3)]
- model = StackingClassifier(estimators=estimators, final_estimator=GaussianNB(), verbose=2)
- model_1 = fbselection('forward', GaussianNB(), 15 )
- model = BaggingClassifier(estimator=model, n_estimators=50, verbose=2)
- simple imputer
- minmax scaler
- 20min

model accuracy =  0.9559164104717147    
roc score =  0.7310520112919827    
accuracy: 0.86283

## Case 198 - categorical NB, forward=15
- CategoricalNB(alpha=0, fit_prior=False)
- no bagging
- knn=7 imputer
- minmax scaler
- SequentialFeatureSelector(model, direction='forward',n_features_to_select=15, scoring='roc_auc')
--cnb1.csv
- 15min
 
model accuracy =  0.8133591559787099    
roc score =  0.8185694365485746    
accuracy: 0.83356

## Case 199a - categorical NB, kbest=20
- CategoricalNB(alpha=0, fit_prior=False)
- no bagging
- knn=7 imputer
- minmax scaler
- SelectKBest(score_func=f_classif, k=20) 
--cnb1.csv

model accuracy =  0.7630185408399582    
roc score =  0.7819689023611237    
not submitted

## Case 199b - stacking, KNN
- model_1 = KNeighborsClassifier(n_neighbors=1500, weights="distance")
- model_2 = KNeighborsClassifier(n_neighbors=1000, weights="distance")
- model_3 = KNeighborsClassifier(n_neighbors=1200, weights="uniform")
- estimators = [('knn1', model_1), ('knn2', model_2), ('knn3', model_3)]
- model = StackingClassifier(estimators=estimators, final_estimator=KNeighborsClassifier(n_neighbors=500, weights="distance"), verbose=2)
- model_1 = kbest(model_1, 5)
-- stacking1.csv
- 14min + 12min

model accuracy =  0.9972507008681284    
roc score =  0.49999320975079786    
accuracy: 0.83304

## Case 200 - lgbm, xgb features
- model = lgb.LGBMClassifier( learning_rate=0.02, max_depth=2, n_estimators=3500 )
- model_2 = xgb.XGBClassifier( max_depth=5, n_estimators=250, learning_rate=0.1, eval_metric='auc', random_state=42 )
- model_2 = featureImportance( model_2, 45 )
- BaggingClassifier( estimator=model, n_estimators=50, max_features=0.8, max_samples=0.8, bootstrap=True, random_state=42, verbose = 2 )
- simple imputer
- minmax scaler
--mixed1.csv
- 2min + 65min + 9min + 86min + 6min

model accuracy =  0.9973319609409916    
roc score =  0.5271869812770643    
accuracy: 0.96244   

# DAY 21: Sunday 10th November 2024

## Case 201 - voting, lgbm, lgbm+bagged
- this case was done before in case 179 except now we have trained both models and then put in voting. 
- model = lgb.LGBMClassifier( learning_rate=0.02, max_depth=2, n_estimators=3500 )
- model_2 = xgb.XGBClassifier( max_depth=5, n_estimators=250, learning_rate=0.1, eval_metric='auc', random_state=42 )
- model_2 = featureImportance( model_2, 45 )
- model_1 = BaggingClassifier( estimator=model, n_estimators=50, max_features=0.8, max_samples=0.8, bootstrap=True, random_state=42, verbose = 2 )
- model = VotingClassifier(estimators=[('one', model), ('two', model_1)], voting='soft', verbose=True)
- simple imputer
- minmax scaler
--mixed1.csv

model accuracy =  0.9972777875590828    
roc score =  0.5296282775558061    
accuracy: 0.96178

## Case 202a - voting, lgbm, lgbm+bagged, xgb
- this case was done before in case 179 except now we have trained both models and then put in voting. 
- model = lgb.LGBMClassifier( learning_rate=0.02, max_depth=2, n_estimators=3500 )
- model_2 = xgb.XGBClassifier( max_depth=5, n_estimators=250, learning_rate=0.1, eval_metric='auc', random_state=42 )
- model_2 = featureImportance( model_2, 45 )
- model_1 = BaggingClassifier( estimator=model, n_estimators=50, max_features=0.8, max_samples=0.8, bootstrap=True, random_state=42, verbose = 2 )
- model_3 = VotingClassifier(estimators=[('one', model), ('two', model_1), ('three', model_2)], voting='soft', verbose=True)
- simple imputer
- minmax scaler
--mixed1.csv
- ERROR: Failed after 360min, VSCode crashed

## Case 202b - stacking, lgbm, xgb
- lgbm = LGBMClassifier(learning_rate=0.02, max_depth=3, n_estimators=3000, random_state=42)
- xgb = XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.07, random_state=42)
- stacked_model = StackingClassifier(
    estimators=[('lgbm', lgbm), ('xgb', xgb)],
    final_estimator=XGBClassifier(n_estimators=150, learning_rate=0.05, random_state=42),
    cv=5
)
- xgb feature importances of 50 features
- simple imputer
- minmax scaler

roc = 0.96045
accuracy: 0.95049

## Case 203 - stacking, adaboost, rf, lgb
- estimators = [ ('adaboost', AdaBoostClassifier( n_estimators=1000, learning_rate=0.07, algorithm='SAMME.R', random_state=42 )), ('rf', RandomForestClassifier( n_estimators=512, criterion='gini', max_depth=930, min_samples_split=29, min_samples_leaf=40, max_features='sqrt', bootstrap=True, n_jobs=-1 )), ('LightGBM', lgb.LGBMClassifier( boosting_type='dart', n_estimators=1000, learning_rate=0.07, num_leaves=40, max_depth=7, min_child_samples=25, subsample=0.85, colsample_bytree=0.75, random_state=42, n_jobs=-1 )) ]
- model = StackingClassifier( estimators=estimators, final_estimator=xgb.XGBClassifier(), cv=5, n_jobs=-1 )
- simple imputer
- standard scaler
--stacking2.csv

AUC-ROC Score: 0.7730
accuracy: 0.94613

## Case 204 - stacking, xgb, lgb, rf, nb
- XGBoost = xgb.XGBClassifier(scale_pos_weight=99, use_label_encoder=False, eval_metric='logloss', random_state=42)
LightGBM = lgb.LGBMClassifier(is_unbalance=True, random_state=42)
Random_Forest = RandomForestClassifier(class_weight='balanced', random_state=42)
stacking_clf = StackingClassifier(estimators=[('rf', Random_Forest), ('xgb', XGBoost), ('lgbm', LightGBM)], final_estimator=GaussianNB())
- no feature selection
- simple imputer
- standard scaler
--stacking2.csv

roc: 0.5675982866072379
accuracy: 0.77448

## Case 205 - stacking, lgbm, adaboost
- lgbm1 = lgb.LGBMClassifier( boosting_type='dart', n_estimators=1000, learning_rate=0.07, num_leaves=40, max_depth=7, min_child_samples=25, subsample=0.85, colsample_bytree=0.75, random_state=42, n_jobs=-1 )
- lgbm2 = lgb.LGBMClassifier( boosting_type='dart', n_estimators=1500, learning_rate=0.07, num_leaves=40, max_depth=7, min_child_samples=25, subsample=0.85, colsample_bytree=0.75, random_state=42, n_jobs=-1 )
- lgbm3 = lgb.LGBMClassifier(learning_rate=0.02, max_depth=2, n_estimators=3500, random_state=42)
- meta_learner = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1, algorithm='SAMME.R', random_state=42)
- model = StackingClassifier( estimators=[ ('lgbm1', lgbm1), ('lgbm2', lgbm2), ('lgbm3', lgbm3) ], final_estimator=meta_learner, cv=5, stack_method='predict_proba',  passthrough=False )
- simple imputer
- standard scaler
- xgb = xgb.XGBClassifier(max_depth=5, n_estimators=250, learning_rate=0.1, eval_metric='auc', random_state=42)
- xgb, X = featureImportance( xgb, 45 )
--stacking2.csv

roc: 0.5323537283118134
accuracy: 0.95299

## Case 206 - voting, DT, RF
- xgb = xgb.XGBClassifier( max_depth=5, n_estimators=250, learning_rate=0.1, eval_metric='auc', random_state=42 )
- xgb = featureImportance( xgb, 40 )
- model_1 = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60, min_samples_leaf=80)
- model_1 = BaggingClassifier(estimator=model_1, n_estimators=50, verbose=2)
- model_2 = RandomForestClassifier(max_depth=11, n_estimators=400, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80, verbose=1)
- model = VotingClassifier(estimators=[('decision_tree', model_1), ('random_forest', model_2)], voting='soft', verbose=True)
- simple imputer
- minmax scaler
--mixed1.csv
- 200min

model accuracy =  0.9972642442136056    
roc score =  0.5    
accuracy: 0.93243

## Case 207a - stacking, random forest
- model_1 = RandomForestClassifier(max_depth=11, n_estimators=400, criterion='entropy', min_samples_split=15, max_features=60, min_samples_leaf=80, verbose=2)
- model_2 = RandomForestClassifier(max_depth=13, n_estimators=300, criterion='entropy', min_samples_split=10, max_features=50, min_samples_leaf=60, verbose=2)
- model_3 = RandomForestClassifier(max_depth=9, n_estimators=500, criterion='entropy', min_samples_split=20, max_features=55, min_samples_leaf=70, verbose=2)
- estimators = [('rf1', model_1), ('rf2', model_2), ('rf3', model_3)]
- model = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier( max_depth=11, n_estimators=500, criterion='entropy', min_samples_split=10, max_features=40, min_samples_leaf=70, verbose=2))
- model_1 = featureImportance( model_1, 45 )
- simple imputer
- minmax scaler
- ERROR: failed after 540mins of running on kaggle, internet dsconnected and code stopped running

## Case 207b - lgb bagged
- xgb = xgb.XGBClassifier( max_depth=5, n_estimators=300, learning_rate=0.1, eval_metric='auc', random_state=42 )
- xgb = featureImportance(xgb, 41)
- model= BaggingClassifier( estimator=lgb.LGBMClassifier(learning_rate=0.02, max_depth=2, n_estimators=3600), max_features=0.8, max_samples=0.8, n_estimators=80,n_jobs=-1, random_state=42, verbose=2 )
- simple imputer
- minmax scaler

model accuracy =  0.9973319609409916    
roc score =  0.5271869812770643   
roc score = 0.9963278870156316
accuracy: 0.96407

## Case 208 - lgb bagged
- xgb = xgb.XGBClassifier( max_depth=5, n_estimators=300, learning_rate=0.1, eval_metric='auc', random_state=42 )
- xgb = featureImportance(xgb, 41)
- model= BaggingClassifier( estimator=lgb.LGBMClassifier(learning_rate=0.02, max_depth=2, n_estimators=3650), max_features=0.8, max_samples=0.8, n_estimators=80,n_jobs=-1, random_state=42, verbose=2 )
- simple imputer
- minmax scaler
- 102min

model accuracy =  0.9973319609409916    
roc score =  0.5271869812770643     
roc score = 0.996400764541722
accuracy: 0.96410

## Case 209 - stacking, xgb
xgb = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, subsample=1.0, device = 'cuda') #---40 features
LGBM = lgb.LGBMClassifier(learning_rate=0.02, max_depth=2, n_estimators=3500 , device = 'gpu')
Ada = AdaBoostClassifier(n_estimators = 170)
RF1 = RandomForestClassifier(criterion="entropy", max_depth=13, min_samples_leaf=60, min_samples_split=15, n_estimators=400)
DT1 = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=70, min_samples_split=18)
classifier = StackingClassifier(estimators=[('xgb', xgb),('LGBM',LGBM),('Ada', Ada) , ('RF1',RF1), ('DT1',DT1)],final_estimator = xgb)
model_4, X, trainX, trainY, testX, test_data_processed = featureImportance( xgb.XGBClassifier(), 45, X, trainX, trainY, testX, test_data_processed )

model accuracy =  0.9972642442136056    
roc score =  0.5    
accuracy: 0.95582

## Case 210 - stacking
xgb = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, subsample=1.0) #---40 features
RF1 = RandomForestClassifier(criterion="entropy", max_depth=13, min_samples_leaf=60, min_samples_split=15, n_estimators=400)
RF2 = RandomForestClassifier(criterion="entropy", max_depth=11, min_samples_leaf=70, min_samples_split=18, n_estimators=300)
s_clf = StackingClassifier(estimators=[('xgb', xgb), ('RF1', RF1),('RF2',RF2)],final_estimator = xgb)
model_4, X, trainX, trainY, testX, test_data_processed = featureImportance( xgb.XGBClassifier(), 45, X, trainX, trainY, testX, test_data_processed )

model accuracy =  0.9972642442136056    
roc score =  0.5    
acciracy: 0.94855