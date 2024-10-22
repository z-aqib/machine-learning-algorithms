# Results

## Case 1
- DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=20), 
- rows removal, 
- minmax scaling, 
- no grid

roc score = 0.5230564082198742   
Accuracy: 0.72913 

### ERROR SOLUTION
one error approached was that we had first splitted the data into features and target variables (X, Y) and then dropped all the NaN rows from X which made X and Y have differnet number of rows. after trying to "index" and match these rows, nothing was working. So a solution was found to restart the ipynb and first remove all the NaN rows from the train_data_processed and THEN split it into X, Y which resulted in same rows and this code line worked. 

## Case 2
- DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=20)
- rows removal,
- minmax scaling,
- no grid

roc score = 0.5229797007820421   
Accuracy: 0.83323  

## Case 3
- DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=15)
- rows removal
- minmax scaling
- no grid

roc score =  0.5203814609840954    
accuracy: 0.83327

## Case 4
- DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=15)
- rows removal
- minmax scaling
- no grid

roc score =  0.5306000850899    
accuracy: 0.78883

## Case 5
- DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=15)
- rows removal
- minmax scaling
- no grid

roc score =  0.5204372482116096    
accuracy: 0.85532

## Case 6
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15)
- rows removal
- minmax scaling
- no grid

roc score =  0.5230843018336313    
accuracy: 0.87815

## Case 7
- DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=15)
- rows removal
- minmax scaling
- no grid

roc score =  0.5025494259738718    
accuracy: 0.87296

## Case 8
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=10)
- rows removal
- minmax scaling
- no grid

roc score =  0.5154151037016982    
accuracy: 0.83207

## Case 9
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=50)
- rows removal
- minmax scaling
- no grid

roc score =  0.5128726511312657    
accuracy: 0.86817

## Case 10
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- rows removal
- minmax scaling
- no grid

roc score =  0.5205418492631988    
accuracy: 0.88788

## Case 11
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- SimpleImputer(strategy='mean')
- minmax scaling
- no grid

roc score =  0.5184642260176702   
accuracy: 0.89330

## Case 12
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- KNNImputer(n_neighbors=5)
- minmax scaling
- no grid

roc score =  0.5138345831202585      
accuracy: 0.87791

## Case 13
- DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=15, max_features=60)
- SimpleImputer(strategy='mean')
- StandardScaler()
- no grid

roc score =  0.5086297947750887    
accuracy: 0.77620   