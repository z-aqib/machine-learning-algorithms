# DAY 1: Monday 25th November 2024

## Case 01 - linear regression
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- train_test_split(X, Y, test_size=0.3, random_state=2)
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
- linearRegression1.csv

model score:  0.6611981837697488    
Mean squared error: 191699960168873.97    
Root Mean squared error: 13845575.47    
Mean absolute error: 6682687.71    
Coefficient of determination: 0.60    
score: 13443448.16606

### analyzing
lets shift to ridge

## Case 02 - ridge, minmax scaler
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=0.5))
])
- ridge1.csv

model score:  0.6610617032143522   
Mean squared error: 189920563566693.88    
Root Mean squared error: 13781166.99    
Mean absolute error: 6672565.38    
Coefficient of determination: 0.60     
score: 13414895.67411

### analyzing
lets put some grid search for ridge

## Case 03 - ridge with grid
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge())
])
- param_grid = {
    'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0] 
}
- best alpha: {'model__alpha': 100.0}
- ridge1.csv

Mean squared error: 178446807239989.81    
Root Mean squared error: 13358398.38    
Mean absolute error: 6620569.43    
Coefficient of determination: 0.63     
model score:  0.6429330718509376    
score: 13095760.55722

### Analyzing
okay, nice. lets increase alphas more to find the best one

## Case 04 - ridge with grid for alpha
- ran it with this grid: param_grid = {
    'model__alpha': [100.0, 150.0, 200.0, 300.0, 500.0]
}
- got {'model__alpha': 100.0}
- this is same parameters as before, so didnt want to waste an entry. stopped running and changed grid
- so ran it with this grid: param_grid = {
    'model__alpha': [98.0, 99.0, 100.0, 101.0, 102.0, 105.0]
}
- got {'model__alpha': 99.0}

- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge())
])
- ridge1.csv

Mean squared error: 178445219796271.12    
Root Mean squared error: 13358338.96    
Mean absolute error: 6620198.39    
Coefficient of determination: 0.63     
model score:  0.6429840577286432    
score: 13095804.97745

### analyzing
score decreased, looks like 100.0 was fine. lets try extreme alpha values

## Case 05a - ridge with grid for alpha
- ran it with this grid: param_grid = {
    "model__alpha": [100, 500, 1000, 5000, 10000]
}
- got {'model__alpha': 100}
- did not submit as same parameters as case 3. did not want to waste an entry.

## Case 05b - ridge with grid for model_solver
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=100))
])
- param_grid = {
    'model__solver': ['auto', 'svd', 'lsqr', 'sparse_cg']
}
- best solver: {'model__solver': 'lsqr'}
- ridge1.csv

Mean squared error: 178422961193358.22    
Root Mean squared error: 13357505.80    
Mean absolute error: 6620163.56    
Coefficient of determination: 0.63     
model score:  0.6428676656837684    
score: 13095300.21716

## Case 06a - ridge with grid for positive and fit intercept
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=100, solver='lsqr'))
])
- param_grid = {
    'model__fit_intercept': [True, False], 
    'model__positive': [True, False]
}
- best params: {'model__fit_intercept': True, 'model__positive': False}
- same as DEFAULT so no need to submit

## Case 06b - ridge with grid for max iterations
- param_grid = {
    'model__max_iter': [1000, 5000, 10000]
}
- best params: {'model__max_iter': 1000}
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=100, solver='lsqr'))
])

Mean squared error: 178422961193358.22    
Root Mean squared error: 13357505.80    
Mean absolute error: 6620163.56    
Coefficient of determination: 0.63     
model score:  0.6428676656837684    
score: 13095300.21716

### Analyzing
file had not changed. my entry wasted :(    
it chose 1000 max iterations which was the smallest value in grid. lets use smaller values than that and re-run the grid and see. this time i will check if the file has changed before submitting !

## Case 07a - ridge with grid for max iterations
- param_grid = {
    'model__max_iter': [100, 200, 300, 500, 900, 1000]
}
- best params: {'model__max_iter': 100}
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=100, solver='lsqr'))
])

Mean squared error: 178422961193358.22    
Root Mean squared error: 13357505.80    
Mean absolute error: 6620163.56    
Coefficient of determination: 0.63    
model score:  0.6428676656837684     
score: FILE DID NOT CHANGE. ENTRY IS NOT SUBMITTED.

### Analyzing
okay so it picked the least amount, again, 100. so now in next grid lets do 1 to 100 to find the best one.    
file did not change so this entry is not submitted.

## Case 07b - ridge with grid for max iterations
- param_grid = {
    'model__max_iter': [1, 10, 20, 50, 90, 100]
}
- best params: {'model__max_iter': 50}
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=100, solver='lsqr'))
])

Mean squared error: 178422961193358.22    
Root Mean squared error: 13357505.80    
Mean absolute error: 6620163.56    
Coefficient of determination: 0.63    
model score:  0.6428676656837684    
score: FILE DID NOT CHANGE. ENTRY WAS NOT SUBMITTED.

### Analyzing
the default max_iterations seems fine as file is not changing even if i do 1000, 100, or 50. lets do grid on something else.

## Case 07c - ridge with grid for tol
- param_grid = {
    'model__tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
}
- best params: {'model__tol': 0.001}
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=100, solver='lsqr'))
])

Mean squared error: 178399494100404.12    
Root Mean squared error: 13356627.35    
Mean absolute error: 6627800.65    
Coefficient of determination: 0.63     
model score:  0.6409230973228461     
score: 13091172.07257

### Analyzing
the default tol was 0.0001, our grid found 0.001. file changed successfully! improved, shukar. lets shift to lasso now.

## Case 08 - regressiontree
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(max_depth=10, random_state=0))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 174156211510543.12    
Root Mean squared error: 13196825.81    
Mean absolute error: 5696000.70    
Coefficient of determination: 0.64     
model score:  0.6796795918857109     
score: 12963023.56640

### analyzing
lasso was taking too long (20mins passed) so i shifted to poly, it failed, so shifted to knn, it was also taking too long so shifted to regressiontree. it was relatively fast (7min) and accuracy is good.      
lets depth grid search now

## Case 09 - lasso
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Lasso())
])
- lasso1.csv
- 15min + 25min

Mean squared error: 191668945999741.03    
Root Mean squared error: 13844455.42    
Mean absolute error: 6682058.36    
Coefficient of determination: 0.60  
model score:  0.6611991035918549     
score: 13442390.39444

### Analyzing
very low. lasso itself is very slow so grid would be a bit difficult. lets try though and start from alpha    
update: lasso and knn are running. RT is faster so we are running that

## Case 10 - regression tree grid for depth
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(random_state=0))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- param_grid = {
    'model__max_depth': [1, 2, 3, 4, 5, 10]
}
- best params: {'model__max_depth': 5}

Mean squared error: 167977657854855.94    
Root Mean squared error: 12960619.50    
Mean absolute error: 6016174.11    
Coefficient of determination: 0.65  
model score:  0.6562927030744047     
score: 12756207.92108

### Analyzing
nice! lets do more depth search, what if 6 was better? lets do it with 6 to 10

## Case 11 - xgbregressor
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb.XGBRegressor())
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 177766344966137.41    
Root Mean squared error: 13332904.60    
Mean absolute error: 5799980.84    
Coefficient of determination: 0.63     
model score:  0.7777357346516934    
score: 12982571.93294

## Case 12 - regression tree grid for depth
- param_grid = {
    'model__max_depth': [5, 6, 7, 8, 9, 10]
}
- best params: {'model__max_depth': 5}
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(random_state=0))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 167977657854855.94    
Root Mean squared error: 12960619.50    
Mean absolute error: 6016174.11    
Coefficient of determination: 0.65  
model score:  0.6562927030744047     
score: 12756207.92108

### Analyzing
file did not change...

## Case 13 - randomforest regressor
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(max_depth=6, max_features=4, min_samples_split=8, n_estimators=300))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 210712625322840.84    
Root Mean squared error: 14515943.83    
Mean absolute error: 8082056.71    
Coefficient of determination: 0.56     
model score:  0.5643253669923808     
score: 14311567.49859 

### Analyzing
done from kaggle. lets increase depth. 

## Case 14 - regression tree grid for maxfeatures
- param_grid = {
    'model__max_features': ['auto', 'sqrt', 'log2']
}
- best params: {'model__max_features': 'sqrt'}
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(random_state=0, max_depth=5))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 183853745621936.66    
Root Mean squared error: 13559267.89    
Mean absolute error: 7056495.43    
Coefficient of determination: 0.62     
model score:  0.6293951358082965     
score: 13234262.48727

### analyzing
this is very bad. accuracy fell alot. default of None was much better. 

## Case 15 - random forest, depth increased
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(max_depth=7, max_features=4, min_samples_split=8, n_estimators=300))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 201350035576811.72    
Root Mean squared error: 14189786.31    
Mean absolute error: 7820540.79    
Coefficient of determination: 0.58   
model score:  0.579537908188712  
score: 14062611.38234

### analyzing
improved, lets increase depth further

## Case 16 - random forest, depth increased
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(max_depth=10, max_features=4, min_samples_split=8, n_estimators=300))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 182064047829679.44    
Root Mean squared error: 13493111.12    
Mean absolute error: 7126153.18    
Coefficient of determination: 0.62     
model score:  0.6266156940965661    
score: 13283359.81617

## Case 17 - random forest, depth increased
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(max_depth=15, max_features=4, min_samples_split=8, n_estimators=300))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 173977492829199.50    
Root Mean squared error: 13190052.80    
Mean absolute error: 6625772.14    
Coefficient of determination: 0.64     
model score:  0.6527342751574265    
score: 13012520.35937

# DAY 2: Tuesday 26th November 2024

## Case 18a - random forest grid for maxdepth
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(max_features=4, min_samples_split=8, n_estimators=300))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- param_grid = {
    'model__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
- best params: {'model__max_depth': 10}
- did not submit as same as case 16

## Case 18b - random forest grid for maxdepth
- param_grid = {
    'model__max_depth': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}
- best params: {'model__max_depth': 20}
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(max_features=4, min_samples_split=8, n_estimators=300))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 170485082730009.75    
Root Mean squared error: 13056993.63    
Mean absolute error: 6315450.37    
Coefficient of determination: 0.64     
model score:  0.6802655875129153     
score: 12861372.95847

### Analyzing
lets decrease depth further

## Case 19 - lasso grid for alpha
- param_grid = {
    'model__alpha': [100, 1000, 10000]
}
- best params: {'model__alpha': 10000}
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Lasso())
])
- lasso1.csv
- 154min + 20min + 30min

Mean squared error: 178437965939460.09    
Root Mean squared error: 13358067.45    
Mean absolute error: 6738128.97    
Coefficient of determination: 0.63     
model score:  0.6336621905064037     
score: 13124664.68129

### Analyzing
lets increase alpha but this time with a sample so that it doesnt take too long

## Case 20 - xgb with grid for booster
- param_grid = {
    'model__booster': ['gbtree', 'gblinear', 'dart']
}
- best parameters: {'model__booster': 'gbtree'}
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb.XGBRegressor())
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- top 20 algorithm feature importances

Mean squared error: 178515304193197.03    
Root Mean squared error: 13360961.95    
Mean absolute error: 5808192.51    
Coefficient of determination: 0.63     
model score:  0.816158130981291    
score: 13112676.12178

### Analyzing
worsened. default was fine. 

## Case 21 - regression tree, grid for criterion
- param_grid = {
    'model__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
}
- best params: {'model__criterion': 'squared_error'}
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(random_state=0, max_depth=5, max_features='sqrt'))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 183853745621936.66    
Root Mean squared error: 13559267.89    
Mean absolute error: 7056495.43    
Coefficient of determination: 0.62     
model score:  0.6293951358082965     
score: 13234262.48727

### analyzing
looks like file didnt change??? what

## Case 22 - regression tree, grid for criterion
- param_grid = {
    'model__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
}
- best params: {'model__criterion': 'poisson'}
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(random_state=0, max_depth=5))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

i ran this file before but didnt save its values, csv saved, so submitting. poisson was used.    
score: 12765094.57261

## Case 23 - lasso grid for alpha
- param_grid = {
    'model__alpha': [10000, 20000, 50000, 100000]
}
- best params: {'model__alpha': 50000}
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Lasso())
])
- lasso1.csv    

Mean squared error: 181387236371446.19    
Root Mean squared error: 13468007.88    
Mean absolute error: 6947525.65    
Coefficient of determination: 0.62     
model score:  0.6260249178722197     
score: 13245107.59931

### Analyzing
decreased. 10000 was fine. 

## Case 24 - regression tree grid for min samples split
- param_grid = {
    'min_samples_split': [10, 30, 50, 100, 200, 1000]
}
- best params: {'min_samples_split': 10}
- model = DecisionTreeRegressor(random_state=0, max_depth=5, criterion='poisson')
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 167260228624135.78    
Root Mean squared error: 12932912.61    
Mean absolute error: 5984409.15    
Coefficient of determination: 0.65     
model score:  0.6550849217780055     
score: 12765094.57261

### Analyzing
i think the file didnt change from case 22...

## Case 25a - knnregressor
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", KNeighborsRegressor(n_neighbors=5))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- doesnt work, crashed after like 600min

## Case 25b - xgb with grid for maxdepth, increased feature imp
- param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
- best params: {'max_depth': 2}
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb.XGBRegressor())
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- top 40 algorithm feature importances

Mean squared error: 167743096759781.88    
Root Mean squared error: 12951567.35    
Mean absolute error: 5903178.89    
Coefficient of determination: 0.65     
model score:  0.6620939103856143     
score: 12737222.65783

## Case 26 - xgb grid for learning rate
- param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
}
- best params: {'learning_rate': 0.1}
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = xgb.XGBRegressor(max_depth=2)
- top 40 algorithm feature importances

Mean squared error: 166853913780964.91    
Root Mean squared error: 12917194.50    
Mean absolute error: 5887385.84    
Coefficient of determination: 0.65     
model score:  0.656760765654886     
score: 12709984.52150

## Case 27a - regression tree for min samples split
- param_grid = {
    'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
- best params: {'min_samples_split': 2}
- model = DecisionTreeRegressor(random_state=0, max_depth=5, criterion='poisson')
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- same as default, not submitted, file didnt change

## Case 27b - xgb grid for estimators
- param_grid = {
    'estimators': [10, 100, 200, 500, 1000, 2000, 3000]
}
- best params: {'estimators': 10}
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = xgb.XGBRegressor(max_depth=2, learning_rate=0.1)
- top 40 algorithm feature importances

Mean squared error: 166965016240907.06    
Root Mean squared error: 12921494.35    
Mean absolute error: 5895809.67    
Coefficient of determination: 0.65     
model score:  0.6567594926088847     
score: 12717549.88555

### Analyzing
decreased. default was much better

## Case 28a - regression tree grid for splitter
- param_grid = {
    'splitter': ['best', 'random']
}
- best params: {'splitter': 'best'}
- model = DecisionTreeRegressor(random_state=0, max_depth=5, criterion='poisson')
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- file didnt change, default, didnt submit

## Case 28b - xgb grid for estimators, learning rate
- param_grid = {
    'estimators': [10, 100, 200, 500, 1000, 2000, 3000], 
    'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
}
- best params: {'estimators': 10, 'learning_rate': 0.05}
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = xgb.XGBRegressor(max_depth=2, learning_rate=0.1)
- top 40 algorithm feature importances

Mean squared error: 167210261942516.41    
Root Mean squared error: 12930980.70    
Mean absolute error: 5951696.16    
Coefficient of determination: 0.65    
model score:  0.6540416179368541     
score: 12731027.64332

### Analyzing
default estimators and 0.1 learning rate was best

## Case 29 - random forest, grid for max depth
- param_grid = {
    'model__max_depth': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
}
- best params: {'model__max_depth': 30}
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(max_features=4, min_samples_split=8, n_estimators=300))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 164994167532686.22    
Root Mean squared error: 12845005.55    
Mean absolute error: 5787791.82    
Coefficient of determination: 0.66     
model score:  0.7501690798022409     
score: 12652092.92952

### analyzing
improved! kiss had tak jaon mein? :)

## Case 30 - xgb grid for estimators, learning rate, depth
- param_grid = {
    'estimators': [10, 100, 200, 500, 1000, 2000, 3000], 
    'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
- best params: {'estimators': 10, 'learning_rate': 0.05, 'max_depth': 4}
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = xgb.XGBRegressor(max_depth=2, learning_rate=0.1)
- top 40 algorithm feature importances

Mean squared error: 165326527443054.19    
Root Mean squared error: 12857936.36    
Mean absolute error: 5799900.84    
Coefficient of determination: 0.65     
model score:  0.6658882422122077     
score: 12662370.10532

### Analyzing
improved but not best. lets stop grid and tune parameters randomly

## Case 31 - xgb parameters
- model = xgb.XGBRegressor(max_depth=10, learning_rate=0.01, n_estimators=1000, subsample=0.8, colsample_bytree=0.8, reg_lambda=1, reg_alpha=0, random_state=42)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- top 40 algorithm feature importance

Mean squared error: 165149204783138.88    
Root Mean squared error: 12851039.05    
Mean absolute error: 5519579.84    
Coefficient of determination: 0.66     
model score:  0.8922213221590881     
score: 12648162.24473

## Case 32 - xgb parameters
- model = XGBRegressor(max_depth=10, learning_rate=0.01, n_estimators=1000, subsample=0.8, colsample_bytree=0.8, reg_lambda=1, reg_alpha=0, random_state=42)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)
- no algorithm feature importance

Mean squared error: 160024759236690.97    
Root Mean squared error: 12650089.30    
Mean absolute error: 5364775.15    
Coefficient of determination: 0.67     
model score:  0.8686829978840838     
score: 12578391.06997

### analyzing
improved and as much as i want to test it more, i cant as xgb ki kaafi entries hogayi hain. we will shift to some other algo for the moment and come back later. 

## Case 33 - lasso grid for max iterations
- param_grid = {
    'max_iter': [1000, 5000, 10000]
}
- best params: {'max_iter': 5000}
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = Lasso(alpha=10000)
- 120mins, full train data

Mean squared error: 178445511124957.69    
Root Mean squared error: 13358349.87    
Mean absolute error: 6738142.70    
Coefficient of determination: 0.63    
model score:  0.6336565804993961        
score: 13124809.78873

### analyzing
i think it improved, not sure though

## Case 34a - random forest, grid for max depth
- param_grid = {
    'model__max_depth': [30, 31, 32, 33, 34, 35]
}
- best params: {'model__max_depth': 35}
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(max_features=4, min_samples_split=8, n_estimators=300))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- random forest depth testing pe kaafi saari entries hogayi hain, after 2 hour grid search we found 35 is best. so lets stop it from training and do a further grid search. just trying to find breakeven point

## Case 34b - lasso grid for selection
- param_grid = {
    "selection": ["cyclic", "random"]
}
- best params: {'selection': 'random'}
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = Lasso(alpha=10000)

Mean squared error: 178442238660973.81    
Root Mean squared error: 13358227.38    
Mean absolute error: 6738059.58    
Coefficient of determination: 0.63     
model score:  0.6336676025426253     
score: 13124597.21980

### analyzing
improved for overall lasso, but not best of all.. far from it

## Case 35 - lasso grid for tol
- param_grid = {
    "tol": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 1]
}
- best params: {'tol': 0.001}
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- model = Lasso(alpha=10000, selection='random')
- 10mins max

Mean squared error: 178441627053489.91    
Root Mean squared error: 13358204.48    
Mean absolute error: 6738000.85    
Coefficient of determination: 0.63     
model score:  0.6336635394027026     
score: 13124614.48958

### analyzing
decreased a bit. that concludes lasso. the best params were Lasso(alpha=10000, selection='random')

## Case 36 - random forest, grid for max depth
- param_grid = {
    'model__max_depth': [35, 36, 37, 38, 39, 40]
}
- best params: {'model__max_depth': 39}
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(max_features=4, min_samples_split=8, n_estimators=300))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- 2h33min + 50min

Mean squared error: 163742544360962.56    
Root Mean squared error: 12796192.57    
Mean absolute error: 5584219.52    
Coefficient of determination: 0.66     
model score:  0.7963392035062864     
score: 12585886.63893

### Analyzing
breakeven found! we even got a high accuracy. amaxing. lets grid for n_estimators now

## Case 37 - random forest, grid for n_estimators
- param_grid = {
    'model__n_estimators': [50, 100, 200, 300, 400, 500]
}
- best params: {'model__n_estimators': 400}
- "model", RandomForestRegressor(max_depth=39, max_features=4, min_samples_split=8, verbose=1, n_jobs=-1)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- 1h 21min

Mean squared error: 163464054696822.44    
Root Mean squared error: 12785306.20    
Mean absolute error: 5571375.52    
Coefficient of determination: 0.66     
model score:  0.7958969398640308     
score: 12580644.37388

### Analyzing
okay nice, improved. we dont need to do mroe on estimators as we have reached breakeven. lets do some other grid as well

# DAY 3: Wednesday 27th November 2024

## Case 38 - xgb, kbest
- model = kbest(40)
- model = XGBRegressor(max_depth=10, learning_rate=0.01, n_estimators=1000, subsample=0.8, colsample_bytree=0.8, reg_lambda=1, reg_alpha=0, random_state=42)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 161226712153427.88    
Root Mean squared error: 12697508.11    
Mean absolute error: 5474978.07    
Coefficient of determination: 0.67     
model score:  0.9024452166332968     
score: 12631527.70549

### Analyzing
too less attributes, lets increase features in kbest

## Case 39 - knn, kbest
- model = kbest(model, 100)
- model = KNeighborsRegressor( n_neighbors=35, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', n_jobs=-1 )
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)=
- numerical scaler = MinMaxScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- get dummies encoding

Mean squared error: 171533481002074.34    
Root Mean squared error: 13097079.10    
Mean absolute error: 5987080.87    
Coefficient of determination: 0.64     
model score:  0.6652201842973957     
score: 12917545.45673

## Case 40 - knn, kbest increased
- model = kbest(model, 200)
- model = KNeighborsRegressor( n_neighbors=35, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', n_jobs=-1 )
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)=
- numerical scaler = MinMaxScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- get dummies encoding

Mean squared error: 172838676912414.31    
Root Mean squared error: 13146812.42    
Mean absolute error: 6037527.93    
Coefficient of determination: 0.64     
model score:  0.6647465044868816     
score: 12910355.24566

### analyzing
improved! okay. so higher kbest is better

## Case 41 - knn, kbest decreased
- model = kbest(model, 50)
- model = KNeighborsRegressor( n_neighbors=35, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', n_jobs=-1 )
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)=
- numerical scaler = MinMaxScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- get dummies encoding

Mean squared error: 175478062684853.31    
Root Mean squared error: 13246813.30    
Mean absolute error: 6048657.72    
Coefficient of determination: 0.63     
model score:  0.6598568005117571     
score: 12989247.62167

### Analyzing
worsened. higher kbest is better. in onwards cases we should experiment with higher k like 250?

## Case 42 - knn, knearest increased
- model = kbest(model, 100)
- model = KNeighborsRegressor( n_neighbors=50, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', n_jobs=-1 )
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)=
- numerical scaler = MinMaxScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- get dummies encoding

Mean squared error: 171632824649793.75    
Root Mean squared error: 13100871.14    
Mean absolute error: 6059310.66    
Coefficient of determination: 0.64     
model score:  0.659613520214208     
score: 12905957.49702

### analyzing
improved! lets grid for better n_neighbours and put kbest as 200

## Case 43 - random forest, grid for max_features
- RandomForestRegressor(max_depth=39, n_estimators=400, min_samples_split=8, verbose=1, n_jobs=-1)
- param_grid = {
    'model__max_features': [1, 'sqrt', 'log2']
}
- best params: {'model__max_features': 'sqrt'}
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 161225691081200.31    
Root Mean squared error: 12697467.90    
Mean absolute error: 5243660.99    
Coefficient of determination: 0.66     
model score:  0.9150250874518355     
score: 12501962.35127

### analyzing
wow! improved. this is highest. lets grid for min_samples_split

## Case 44a - knn, grid for n_neighbours
- param_grid = {
    'n_neighbors': [ 30, 35, 40, 45, 50 ]
}
- best params: {'n_neighbors': 50}
- model = kbest(model, 100)
- model = KNeighborsRegressor( weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', n_jobs=-1 )
- numerical scaler = MinMaxScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- get dummies encoding

not submitted as same as case 42

## Case 44b - knn, grid for n_neighbours
- param_grid = {
    'n_neighbors': [ 50, 55, 60, 65, 70, 75, 80 ]
}
- best params: {'n_neighbors': 70}
- model = kbest(model, 200)
- model = KNeighborsRegressor( weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', n_jobs=-1 )
- numerical scaler = MinMaxScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- get dummies encoding

Mean squared error: 171728848815368.84    
Root Mean squared error: 13104535.43    
Mean absolute error: 6152053.05    
Coefficient of determination: 0.64     
model score:  0.6565880782422875   
score: 12873928.68621

### Analyzing
improved! lets grid once more to find the exact value between 65 - 75

## Case 45 - knn, grid for n_neighbours
- param_grid = {
    'n_neighbors': [ 66, 67, 68, 69, 70, 71, 72, 73, 74 ]
}
- best params: {'n_neighbors': 67}
- model = kbest(model, 200)
- model = KNeighborsRegressor( weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', n_jobs=-1 )
- numerical scaler = MinMaxScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- get dummies encoding

Mean squared error: 171750939833772.78    
Root Mean squared error: 13105378.28    
Mean absolute error: 6145186.69    
Coefficient of determination: 0.64     
model score:  0.6569848211741741     
score: 12871457.06655

### Analyzing
improved! lets try grid for a different metric

## Case 46a - knn, grid for metric
- param_grid = {
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming']
}
- best params: 
- model = kbest(model, 200)
- model = KNeighborsRegressor( weights='uniform', algorithm='auto', leaf_size=30, p=2, n_neighbours=67, n_jobs=-1 )
- numerical scaler = MinMaxScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- get dummies encoding

file didnt change, not submitted

## Case 46b - adaboost
- model = AdaBoostRegressor(n_estimators=50,learning_rate=1.0)
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies
- 2199 columns

Mean squared error: 170579321926767.38    
Root Mean squared error: 13060601.90    
Mean absolute error: 6522100.20    
Coefficient of determination: 0.64     
model score:  0.6470543607545798     
score: 12859588.16179

## Case 47 - knn, grid on weights
- param_grid = {
    'weights': ['uniform', 'distance']
}
- best params: {'weights': 'distance'}
- model = kbest(model, 200)
- model = KNeighborsRegressor( n_neighbors=67, algorithm='auto', leaf_size=30, p=2, metric='euclidean', n_jobs=-1 )
- numerical scaler = MinMaxScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- get dummies encoding

Mean squared error: 168584198858619.34    
Root Mean squared error: 12983997.80    
Mean absolute error: 5829392.19    
Coefficient of determination: 0.65     
model score:  0.9999971891469528     
score: 12752650.09991

## Case 48 - adaboost, kbest applied
- model = AdaBoostRegressor(n_estimators=50,learning_rate=1.0)
- model = kbest(model, 200)
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies

Mean squared error: 170426388675063.00    
Root Mean squared error: 13054745.83    
Mean absolute error: 6502154.98    
Coefficient of determination: 0.64     
model score:  0.6467723520855394     
score: 12866587.85803

### Analyzing
worsened. lets try and increase estimators.

## Case 49 - adaboost, increased estimators
- model = AdaBoostRegressor(n_estimators=100,learning_rate=1.0)
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies
- 2199 columns

Mean squared error: 170844482328987.31    
Root Mean squared error: 13070749.11    
Mean absolute error: 6559888.10    
Coefficient of determination: 0.64     
model score:  0.6467437678882886     
score: 12866229.08047

### analyzing
improved however very slight difference. lets grid it. 

## Case 50 - gradboost
- model = GradientBoostingRegressor( n_estimators=100, learning_rate=0.1, max_depth=3, verbose=2 )
- numerical scaler = MinMaxScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- get dummies encoding

Mean squared error: 165936078475250.19    
Root Mean squared error: 12881617.85    
Mean absolute error: 5838357.88    
Coefficient of determination: 0.65     
model score:  0.6682961788999566    
score: 12678243.20278

### Analyzing
lets try kbest feature importance

## Case 51a - knn, grid on leaf size
- param_grid = {
    'leaf_size': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}
- best params: {'leaf_size': 10}
- model = kbest(model, 200)
- model = KNeighborsRegressor( n_neighbors=67, weights='distance', algorithm='auto', p=2, metric='euclidean', n_jobs=-1 )
- numerical scaler = MinMaxScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- get dummies encoding

Mean squared error: 168584198858619.34    
Root Mean squared error: 12983997.80    
Mean absolute error: 5829392.19    
Coefficient of determination: 0.65     
model score:  0.9999971891469528     
score: not submitted. file is same.

### analyzing
leaf_size does not matter and file remained same

## Case 51b - adaboost, grid for estimators
- param_grid = {
    'n_estimators': [ 100, 150, 200, 250 ]
}
- best params: {'n_estimators': 150}
- model = AdaBoostRegressor(learning_rate=1.0)
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies
- 2199 columns

Mean squared error: 170697294577910.81    
Root Mean squared error: 13065117.47    
Mean absolute error: 6523891.08    
Coefficient of determination: 0.64     
model score:  0.645097185496281     
score: 12911230.64029

### analyzing
eek! worsened. lets keep it at 100 and grid learning rate

## Case 52 - gradboost, kbest
- model = GradientBoostingRegressor( n_estimators=100, learning_rate=0.1, max_depth=3, verbose=2 )
- model = kbest(model, 200)
- numerical scaler = MinMaxScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- get dummies encoding

Mean squared error: 166346106634387.72    
Root Mean squared error: 12897523.28    
Mean absolute error: 5848139.33    
Coefficient of determination: 0.65     
model score:  0.6689946727395466     
score: 12694114.13982

### analyzing
deterioration. lets grid for depth now. 

## Case 53 - xgb, loop for best kbest features
- model = XGBRegressor(max_depth=10, learning_rate=0.01, n_estimators=1000, subsample=0.8, colsample_bytree=0.8, reg_lambda=1, reg_alpha=0, random_state=42)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- kbest loop: 50 to 250
- best: 88
- 10:01 am to 1:40am

Mean squared error: 159480942168010.12    
Root Mean squared error: 12628576.41    
Mean absolute error: 5378167.48    
Coefficient of determination: 0.67    
model score:  0.9200141191638683     
score: 12621819.10892

## Case 54 - gradboost, lower depth
- model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,     
    max_depth=1,           
    verbose=2
)
- cat_imputer = SimpleImputer(strategy="most_frequent")\
- num_imputer = SimpleImputer(strategy="mean")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)

Mean squared error: 170893869736288.38    
Root Mean squared error: 13072638.21    
Mean absolute error: 6162825.49    
Coefficient of determination: 0.64     
model score:  0.6472367878652435     
score: 12849989.26788

## Case 55 - xgb, increased estimators, subsample, removed col_by_sumsample
- model = XGBRegressor(max_depth=10, learning_rate=0.01, n_estimators=1500, subsample=0.85, reg_lambda=0.2, reg_alpha=0.8)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)

Mean squared error: 161018423401054.06    
Root Mean squared error: 12689303.50    
Mean absolute error: 5364447.09    
Coefficient of determination: 0.67     
model score:  0.8839744537224783     
score: 12572870.13849

## Case 56 - gradboost, estimators decreased
- model = GradientBoostingRegressor(
    n_estimators=10,    
    learning_rate=0.1,  
    max_depth=1, 
    verbose=2
)
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)

Mean squared error: 221794355757278.06    
Root Mean squared error: 14892761.86    
Mean absolute error: 8448052.63    
Coefficient of determination: 0.54     
model score:  0.5378510641445383     
score: 14720380.13598

## Case 57 - knn, grid for algorithm
- model = KNeighborsRegressor(
    n_neighbors=67,
    weights='distance',
    leaf_size=30,
    p=2, 
    metric='euclidean',  
    n_jobs=-1
)  
- param_grid = {
    'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto']
}
- best params: {'algorithm': 'ball_tree'}
- model, X, trainX, trainY, testX, test_data = kbest(model, 200, X, trainX, trainY, testX, test_data)
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)

Mean squared error: 168584204034058.12    
Root Mean squared error: 12983998.00    
Mean absolute error: 5829397.78    
Coefficient of determination: 0.65     
model score:  0.9999971968331348     
score: 12752648.99290

# DAY 4: Thursday 28th November 2024

## Case 58 - random forest, larger params
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(max_depth=31, n_estimators=1400, max_features='log2', min_samples_leaf=2, min_samples_split=3, bootstrap=True, verbose=2, n_jobs=-1))
])
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
- X = X.select_dtypes(include=["number"])
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)

Mean squared error: 160883805931493.09    
Root Mean squared error: 12683998.03    
Mean absolute error: 5189756.85    
Coefficient of determination: 0.66     
model score:  0.9186423892843449     
score: 12479819.76352

### analyzing
lets remove pipeline to see how it changes

## Case 59 - randomforest, pipeline removed + larger params
- no pipeline
- model = RandomForestRegressor(max_depth=31, n_estimators=1400, max_features='log2', min_samples_leaf=2, min_samples_split=3, bootstrap=True, verbose=2, n_jobs=-1)
- num_imputer = SimpleImputer(strategy="mean")
- scaler = MinMaxScaler()
- X = X.select_dtypes(include=["number"])
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)

Mean squared error: 160869442400043.56    
Root Mean squared error: 12683431.81    
Mean absolute error: 5192165.99    
Coefficient of determination: 0.66     
model score:  0.9192710031398919    
score: 12484520.86505

### analyzing
deterioration. lets go back to pipelibe and try finding the best min_samples_split

## Case 60 - randomforest, grid for min_samples_split
- param_grid = {
    'model__min_samples_split': [7, 8, 9, 10]
}
- best params: {'model__min_samples_split': 7}
- RandomForestRegressor(max_depth=39, n_estimators=400, max_features='sqrt', verbose=2, n_jobs=-1)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 160274940665930.53    
Root Mean squared error: 12659973.96    
Mean absolute error: 5047843.18    
Coefficient of determination: 0.67     
model score:  0.9334141991326709     
score: 12462904.72110

### analyzing
improved! lets further grid for min_samples_split to find breakeven

## Case 61 - adaboost, higher estimators
- model = AdaBoostRegressor(n_estimators=500, learning_rate=1.0)
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies
- 2199 columns

Mean squared error: 170937601415745.06    
Root Mean squared error: 13074310.74    
Mean absolute error: 6569128.59    
Coefficient of determination: 0.64     
model score:  0.6467055169974858     
score: 12879336.65792

### analyzing
i think it improved. lets make it 1000 estimators and 0.5 learning rate

## Case 62 - randomforest, grid for min_samples_split
- param_grid = {
    'model__min_samples_split': [4, 5, 6, 7]
}
- best params: {'model__min_samples_split': 6}
- RandomForestRegressor(max_depth=39, n_estimators=400, max_features='sqrt', verbose=2, n_jobs=-1)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 160523155422913.66    
Root Mean squared error: 12669773.30    
Mean absolute error: 5041015.38    
Coefficient of determination: 0.66     
model score:  0.938420659852937     
score: 12466009.74720

## Case 63 - adaboost, higher estimator, lower learning rate
- AdaBoostRegressor(learning_rate=0.5, n_estimators=1000)
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies
- 2199 columns

Mean squared error: 170435288226422.69    
Root Mean squared error: 13055086.68    
Mean absolute error: 6510890.39    
Coefficient of determination: 0.64    
model score:  0.6473047035253705    
score: 12862538.18829

### analyzing
improved, lets continue increasing estimators to find breakeven. grid i have tried but it takes too long and fails. 

## Case 64 - adaboost, higher estimators, standard scaler
- model = AdaBoostRegressor( n_estimators=1200, learning_rate=0.5 )
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = StandardScaler()
- get dummies
- 2199 columns

Mean squared error: 170738647766173.09    
Root Mean squared error: 13066699.96    
Mean absolute error: 6582025.10    
Coefficient of determination: 0.64     
model score:  0.6474534044681877     
score: 12855862.28037

## Case 65 - adaboost, higher estimators
- model = AdaBoostRegressor( n_estimators=1500, learning_rate=0.5 )
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = StandardScaler()
- get dummies
- 2199 columns

Mean squared error: 170961650403776.41    
Root Mean squared error: 13075230.41    
Mean absolute error: 6600065.80    
Coefficient of determination: 0.64     
model score:  0.647399973422075     
score: 12858613.50877

## Case 66 - stacking rf+rf+rf
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols)
        # ("cat", cat_transformer, categorical_cols)
    ]
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- rf1 = RandomForestRegressor(max_depth=39, n_estimators=400, max_features='sqrt', verbose=2, n_jobs=-1, min_samples_split=7)
- rf2 = RandomForestRegressor(max_depth=31, n_estimators=1400, max_features='log2', min_samples_leaf=2, min_samples_split=3, bootstrap=True, verbose=2, n_jobs=-1)
- meta_regressor = RandomForestRegressor(max_depth=32, n_estimators=1500, max_features='log2', min_samples_leaf=2, min_samples_split=3, bootstrap=True, verbose=2, n_jobs=-1)
- stacking = StackingRegressor(estimators=[('rf1', rf1), ('rf2', rf2)],
    final_estimator=meta_regressor,
    passthrough=False
)
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", stacking)
])

Mean squared error: 167351990230424.88    
Root Mean squared error: 12936459.73    
Mean absolute error: 5138216.02    
Coefficient of determination: 0.65     
no model score   
score: 12637627.79394

## Case 67 - ridgecv
- model = RidgeCV(
    alphas=[0.1, 1.0, 10.0, 100.0],
    fit_intercept=True,
    cv=5
)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- no feature importance

Mean squared error: 178446807239989.81    
Root Mean squared error: 13358398.38    
Mean absolute error: 6620569.43    
Coefficient of determination: 0.63     
model score:  0.6429316560418115     
score: 13095757.51710

## Case 68 - adaboost, dt estimator with larger params, kbest 70
- model = AdaBoostRegressor(n_estimators=100, estimator = DecisionTreeRegressor(max_depth=10, splitter='best', criterion='poisson', min_samples_leaf=5, min_samples_split=3))
- model, X, trainX, trainY, testX, test_data = kbest(model, 70, X, trainX, trainY, testX, test_data)
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies method
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)

RMSE: 12774943.590350837     
score: 12584851.34447

## Case 69 - xgb, higher parameters, no feature selection
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ]
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)
- model = XGBRegressor(
    max_depth=18, 
    learning_rate=0.009, 
    n_estimators=2000, 
    subsample=0.85,
    colsample_bytree=0.8, 
    reg_lambda=0.2,  
    reg_alpha=0.8,
    device = 'cuda',
    verbose = 2,
    tree_method = 'gpu_hist',
    predictor = 'gpu_predictor'
)

RMSE: 12682809.26539975     
score: 12466344.80314

## Case 70 - adaboost, parameters changed
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies method
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
    learning_rate=0.5
)

Mean squared error: 170489722078728.28    
Root Mean squared error: 13057171.29    
Mean absolute error: 6522477.00    
Coefficient of determination: 0.64     
model test score:  0.6465356142089589     
score: 12874165.01203

## Case 71 - stacking rf+dt+rf
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols)
        # ("cat", cat_transformer, categorical_cols)
    ]
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- rf1 = RandomForestRegressor(
    max_depth=39,
    n_estimators=400,
    max_features='sqrt',
    verbose=2,
    n_jobs=-1,
    min_samples_split=7
)
- dt2 = DecisionTreeRegressor(
    max_depth=31,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=3
)
- meta_regressor = RandomForestRegressor(
    max_depth=32,
    n_estimators=1500,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=3,
    bootstrap=True,
    verbose=2,
    n_jobs=-1
)
- stacking = StackingRegressor(
    estimators=[('rf1', rf1), ('dt2', dt2)],
    final_estimator=meta_regressor,
    passthrough=False
)
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", stacking)
])

Mean squared error: 165907217060102.69    
Root Mean squared error: 12880497.55    
Mean absolute error: 5108580.96    
Coefficient of determination: 0.65     
model test score:  0.7467883722827993    
score: 12743497.47981

## Case 72 - adaboost1, random parameters
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies method
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model = AdaBoostRegressor(
    n_estimators=50,
    learning_rate=1.0
)

Mean squared error: 170625153398977.44    
Root Mean squared error: 13062356.35    
Mean absolute error: 6564950.21    
Coefficient of determination: 0.64     
model test score:  0.6470294875622229     
score: 12865378.75132

## Case 73 - gboost1, multi file running
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies method
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0, 
    verbose=3
)

Mean squared error: 166009546737258.38    
Root Mean squared error: 12884469.21    
Mean absolute error: 5839338.77    
Coefficient of determination: 0.65     
model test score:  0.6682961788999566     
score: 12677408.56511

## Case 74 - gradboost, higher parameters
- model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, min_samples_leaf=4, min_samples_split=2, random_state=2,verbose=2, max_features='log2', max_depth= 35)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- get dummies
- scaler = MinMaxScaler()
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")

RMSE: 12929883.886812469    
score: 12758903.07745

## Case 75 - gradboost, estimators increased
- model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1,min_samples_leaf=4, min_samples_split=2, random_state=2,verbose=2,max_features='log2' )
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- get dummies
- scaler = MinMaxScaler()
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")

RMSE: 12977232.74952204    
score: 12753991.47323

## Case 76 - gradboost, estimators inc, learning rate dec, min_samples_leaf inc, split inc, max_depth introduced
- model = GradientBoostingRegressor(n_estimators=600, learning_rate=0.01,min_samples_leaf=5, min_samples_split=3, random_state=2,verbose=2,max_features='log2',max_depth= 12)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- get dummies
- scaler = MinMaxScaler()
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")

RMSE: 12785491.593735853     
score: 12584218.94004   

## Case 77 - adaboost3, dt
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=2),
    n_estimators=200,
    learning_rate=0.1
)

Mean squared error: 175228180857318.75    
Root Mean squared error: 13237378.17    
Mean absolute error: 6941579.37    
Coefficient of determination: 0.63     
model test score:  0.636744979476696     
score: 13043233.80775

# DAY 5: Friday 29th November 2024

## Case 78a - lassocv
- LassoCV(
    alphas=[0.1, 0.5, 1.0, 5.0, 10.0],
    fit_intercept=True,
    max_iter=1000,
    cv=5
)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- no feature importance
- too long, 6h on traindata and no update

## Case 78b - stacking rf+dt+rf
-rf1 = RandomForestRegressor(
    max_depth=39,
    n_estimators=400,
    max_features='sqrt',
    verbose=2,
    n_jobs=-1,
    min_samples_split=7
)
- dt2 = DecisionTreeRegressor(
    max_depth=31,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=3
)
- meta_regressor = RandomForestRegressor(
    max_depth=32,
    n_estimators=1500,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=3,
    bootstrap=True,
    verbose=2,
    n_jobs=-1
)
- stacking = StackingRegressor(
    estimators=[('rf1', rf1), ('dt2', dt2)],
    final_estimator=meta_regressor,
    passthrough=False
)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols)
        # ("cat", cat_transformer, categorical_cols)
    ]
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)

Mean squared error: 167418138128401.97    
Root Mean squared error: 12939016.12    
Mean absolute error: 5114919.80    
Coefficient of determination: 0.65     
model test score:  0.6841359514075278     
score: 12704290.23240

## Case 79 - neural networks
- nn_model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[lr_scheduler, early_stopping]
)
- lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
- early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
-   model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
- no feature selection
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- numerical_scaler = MinMaxScaler()
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 179050228228766.62    
Root Mean squared error: 13380965.15    
Mean absolute error: 6277095.75    
Coefficient of determination: 0.63     
no model score
score: 12910899.71065

## Case 80 - stacking rf+dt+xgb+rf
- rf1 = RandomForestRegressor(
    max_depth=39,
    n_estimators=400,
    max_features='sqrt',
    verbose=2,
    n_jobs=-1,
    min_samples_split=7
)
- dt2 = DecisionTreeRegressor(
    max_depth=31,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=3
)
- xgb = XGBRegressor(
    max_depth=10, 
    learning_rate=0.01, 
    n_estimators=1000, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    reg_lambda=1, 
    reg_alpha=0, 
    random_state=42, 
    verbose=True
)
- meta_regressor = RandomForestRegressor(
    max_depth=32,
    n_estimators=1500,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=3,
    bootstrap=True,
    verbose=2,
    n_jobs=-1
)
- stacking = StackingRegressor(
    estimators=[('rf1', rf1), ('dt2', dt2), ('xgb1', xgb)],
    final_estimator=meta_regressor,
    passthrough=False
)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols)
        # ("cat", cat_transformer, categorical_cols)
    ]
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)

Mean squared error: 163785884848689.34    
Root Mean squared error: 12797885.95    
Mean absolute error: 5030737.99    
Coefficient of determination: 0.66    
model test score:  0.8204545348337849     
score: 12612182.01775

### Analyzing
improved, the xgb one improved the score. lets remove the decisiontree one and add xgb one

## Case 81 - neural networks, more layer, higher dropout, tanh activation function
- nn_model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[lr_scheduler, early_stopping]
)
- lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
- early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
-   model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
- no feature selection
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- numerical_scaler = MinMaxScaler()
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

Mean squared error: 697070495518187.50    
Root Mean squared error: 26402092.64    
Mean absolute error: 14772166.09    
Coefficient of determination: -0.46     
no model score
score: 26264836.83072

### analyzing
ok so andazan i think the problem is the tanh function. lets remove that and perhaps do some other tuning

## Case 82 - neural network, tanh removed, standard scaler, simple mean imputer
- nn_model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[lr_scheduler, early_stopping]
)
- lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
- early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
-   model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ]
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)

Mean squared error: 175078256634941.47    
Root Mean squared error: 13231714.05    
Mean absolute error: 6347543.50    
Coefficient of determination: 0.63     
no model score
score: 13057066.02581

### analyzing
just ok. we have scored better in NN than this. too many layers? wrong scaler? i dunno

## Case 83 - neural network, l2 regularization, epoch inc, batch_size halved
- nn_model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=150,
    batch_size=16,
    verbose=2,
    callbacks=[lr_scheduler, early_stopping]
)
- lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
- early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
-   model = Sequential()
    model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- preprocessor = ColumnTransformer( 
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ]
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- epoch early stopping at 17

Mean squared error: 171660426605292.75    
Root Mean squared error: 13101924.54    
Mean absolute error: 6327963.11    
Coefficient of determination: 0.64     
no model score
score: 12845636.43455

### analyzing
ok good, slight improvement. l2 regularization and higher epoch and more batches performed good

## Case 84 - randomforest, larger params
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(max_depth=36, n_estimators=1800, min_samples_split=2, min_samples_leaf=1, max_features=0.45, bootstrap=True, verbose=2, n_jobs=-1))
])
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols)
        # ("cat", cat_transformer, categorical_cols)
    ]
)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 159463537422322.97    
Root Mean squared error: 12627887.29    
Mean absolute error: 4930593.56    
Coefficient of determination: 0.67     
model score:  0.952527263193879     
score: 12437836.73841

## case 85 - linear regression, kbest=200
- model, X, trainX, trainY, testX, test_data = kbest(model, 200, X, trainX, trainY, testX, test_data)
- model = LinearRegression()
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ]
)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 445472403693967.00    
Root Mean squared error: 21106217.18    
Mean absolute error: 12611617.42    
Coefficient of determination: 0.07     
model score:  0.07577281726634111     
score: 20986342.15397

### analyzing
very bad, kbest ruined it as alone it was in 134

## Case 86 - adaboost, higher DT depth, lower estimators
- model = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=10),
    n_estimators=50,
    learning_rate=1.0
)
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = StandardScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- 2199 columns

Mean squared error: 164174620580665.84    
Root Mean squared error: 12813064.45    
Mean absolute error: 5610232.60    
Coefficient of determination: 0.66     
model test score:  0.686733332002083     
score: 12597515.55623

## case 87 - adaboost, higher estimators, lower DT depth
- model = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=500,
    learning_rate=0.5
)
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = StandardScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- 2199 columns

Mean squared error: 170259012465408.72    
Root Mean squared error: 13048333.70    
Mean absolute error: 6493107.63    
Coefficient of determination: 0.64     
model test score:  0.647649089203941    
score: 12860041.39208

### analyzing
deterioration, looks like higher dt depth and lower estimators of ada were better

## Case 88 - adaboost, RF, lower learning rate
- model = AdaBoostRegressor(
    estimator=RandomForestRegressor(n_estimators=10, max_depth=3),
    n_estimators=100,
    learning_rate=0.8
)
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = StandardScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- 2199 columns

Mean squared error: 169506153648570.00    
Root Mean squared error: 13019452.89    
Mean absolute error: 6382041.83    
Coefficient of determination: 0.65    
model test score:  0.6474311343354947     
score: 12862143.13762

### analyzing
this further strengthened our understanding that the base_estimator shoud have high depth for a better result

## Case 89 - linear regression, kbest dec
- model, X, trainX, trainY, testX, test_data = kbest(model, 100, X, trainX, trainY, testX, test_data)
- model = LinearRegression()
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ]
)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 464005048513003.69    
Root Mean squared error: 21540776.41    
Mean absolute error: 13045122.22    
Coefficient of determination: 0.03     
model score:  0.03604177888147497     
score: 21413824.17276

### analyzing
still bad. worsened. lets try PCA. 

## Case 90 - gboost, lower depth, lower estimators, higher learning rate
- model2 = GradientBoostingRegressor(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.2,
    subsample=0.8, 
    verbose=3
)
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- 2199 columns

Mean squared error: 167300856852361.72    
Root Mean squared error: 12934483.25    
Mean absolute error: 5922902.55    
Coefficient of determination: 0.65     
model test score:  0.6594729919504378     
score: 12727187.19395

## Case 91 - gboost, higher estimators, higher depth, lowest learning rate
- model3 = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9, 
    verbose=3
)
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- 2199 columns

Mean squared error: 166411087802967.31    
Root Mean squared error: 12900042.16    
Mean absolute error: 5816306.79    
Coefficient of determination: 0.65     
model test score:  0.6899587895131241     
score: 12673824.44438

### analyzing
nice however even lower has been achieved

## Case 92 - linear regression, no pipeline, no feature selection
- model = LinearRegression()
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ]
)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 191656155099850.69    
Root Mean squared error: 13843993.47    
Mean absolute error: 6681690.71    
Coefficient of determination: 0.60     
model score:  0.6612014260584386     
score: 13442363.53969

### analyzing
so we see that without any feature selection we get a better result which is faster, but itself is really bad
code file has error and has ridge defined when it was linear regression

## Case 93 - ridge, kbest
- model = Ridge(alpha=100, solver='lsqr', tol=0.001)
- model, X, trainX, trainY, testX, test_data = kbest(model, 200, X, trainX, trainY, testX, test_data)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ]
)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 470769261792904.81    
Root Mean squared error: 21697217.84    
Mean absolute error: 13284759.95    
Coefficient of determination: 0.02     
model score:  0.024202988176320828     
score: 21511049.80223

### analyzing
worsened by alot. ridge itself was good but kbest ruined it. 

## Case 94 - stacking ridge+lasso+linear+rf
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols), 
        ("cat", cat_transformer, categorical_cols)
    ]
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- rf1 = Ridge(alpha=100, solver='lsqr', tol=0.001, max_iter=1000)
- dt2 = Lasso(alpha=10000, max_iter=1000)
- xgb = LinearRegression(n_jobs=-1)
- meta_regressor = RandomForestRegressor(
    max_depth=32,
    n_estimators=1500,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=3,
    bootstrap=True,
    verbose=2,
    n_jobs=-1
)
- stacking = StackingRegressor(
    estimators=[('rf1', rf1), ('dt2', dt2), ('xgb1', xgb)],
    final_estimator=meta_regressor,
    passthrough=False, n_jobs=-1, verbose=2
)
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", stacking)
])

Mean squared error: 176640562174842.00    
Root Mean squared error: 13290619.33    
Mean absolute error: 6466013.71    
Coefficient of determination: 0.63     
model test score:  0.6314408654849808     
score: 13131537.11232

## Case 95 - lasso, kbest
- model = Lasso(alpha=10000, selection='random')
- model, X, trainX, trainY, testX, test_data = kbest(model, 200, X, trainX, trainY, testX, test_data)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ]
)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 478331947015584.38    
Root Mean squared error: 21870801.24    
Mean absolute error: 13415352.03    
Coefficient of determination: 0.00     
model score:  0.000803374426354142     
score: 21743808.64479

### analyzing
proved that kbest is bad on ridge lasso and linear

## Case 96 - regressiontree, kbest
- model, X, trainX, trainY, testX, test_data = kbest(model, 200, X, trainX, trainY, testX, test_data )
- model = DecisionTreeRegressor(random_state=0, max_depth=5, criterion='poisson')
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 477725537825579.81    
Root Mean squared error: 21856933.40    
Mean absolute error: 13379169.34    
Coefficient of determination: 0.00     
model score:  0.005283516977046321     
score: 21696485.17191

### analyzing
lets do one more to check if lowered kbest would work

## Case 97 - regressiontree, kbest lowered
- model, X, trainX, trainY, testX, test_data = kbest(model, 100, X, trainX, trainY, testX, test_data )
- model = DecisionTreeRegressor(random_state=0, max_depth=5, criterion='poisson')
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 477658555118472.06    
Root Mean squared error: 21855401.05    
Mean absolute error: 13385455.52    
Coefficient of determination: 0.00     
model score:  0.0049625472744322385     
score: 21707456.19722

### anayzing
ruined further. kbest is not good atm. 

# DAY 6: Saturday 30th November 2024

## Case 98 - linear regression, forward selection
- model = LinearRegression()
- model = fbselection( "forward", model, 10 )
- forward selection sample: sample_train = train_data.sample(frac=0.1)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ]
)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 192403144980823.88    
Root Mean squared error: 13870946.07    
Mean absolute error: 7041594.36    
Coefficient of determination: 0.60     
model score:  0.6019278394572414     
score: 13689504.35699

## Case 99 - regression tree, PCA
- this was done 2-3 times, as PCA after one-hot was all 1.0 (this graph has been committed) then to cater this, i did a variance filter of 1%
- model = DecisionTreeRegressor(random_state=0, max_depth=5, criterion='poisson')
- selector = VarianceThreshold(threshold=0.01) 
- 273 features by variance
- pca = PCA(n_components=0.95)
- 205 features by PCA
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")
- scaler = StandardScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)

Mean squared error: 177197379480826.09    
Root Mean squared error: 13311550.60    
Mean absolute error: 6710773.96    
Coefficient of determination: 0.63     
model score:  0.6363686000238731     
score: 13089744.37642

## Case 100 - linear regression, forward decreased
- model = LinearRegression()
- model = fbselection( "forward", model, 5 )
- forward selection sample: sample_train = train_data.sample(frac=0.1)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ]
)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

Mean squared error: 204458095850321.47    
Root Mean squared error: 14298884.43    
Mean absolute error: 7365368.83    
Coefficient of determination: 0.57     
model score:  0.5781924233746147     
score: 14101310.75498

## Case 101 - adaboost7
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- reg = VerboseAdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=4),
    n_estimators=1000,
    learning_rate=0.05
)
- class VerboseAdaBoostRegressor(AdaBoostRegressor):

    def _boost(self, iboost, X, y, sample_weight, random_state):

        print(f"Training estimator {iboost + 1}/{self.n_estimators}")

        return super()._boost(iboost, X, y, sample_weight, random_state)

Mean squared error: 167240887330895.66    
Root Mean squared error: 12932164.84    
Mean absolute error: 6138320.20    
Coefficient of determination: 0.65     
model test score:  0.6541378354788046     
score: 12746850.37282

## Case 102 - adaboost9
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model = make_pipeline(
    SelectFromModel(RandomForestRegressor(n_estimators=10)),
    VerboseAdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=3), n_estimators=100, learning_rate=0.5)
)

Mean squared error: 170347705223805.91    
Root Mean squared error: 13051731.89    
Mean absolute error: 6504873.65    
Coefficient of determination: 0.64     
model test score:  0.6476724660303605     
score: 12852251.22953

## Case 103 - neural networks
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ]
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model
- lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=2)
- early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=2)
- nn_model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=150,
    batch_size=8,
    verbose=2,
    callbacks=[lr_scheduler, early_stopping]
)

Mean squared error: 174259745175685.91    
Root Mean squared error: 13200747.90    
Mean absolute error: 6573614.96    
Coefficient of determination: 0.64     
no score     
score: 12950464.58027

## Case 104 - neural networks, kbest
- num_imputer = SimpleImputer(strategy="median")
- scaler = StandardScaler()
- get dummies
- nn_model, X, trainX, trainY, testX, test_data = kbest(100, X, trainX, trainY, testX, test_data)
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model
- lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=2)
- early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=2)
- nn_model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=150,
    batch_size=16,
    verbose=2,
    callbacks=[lr_scheduler, early_stopping]
)

Mean squared error: 466469850753343.81    
Root Mean squared error: 21597913.11    
Mean absolute error: 12689852.71    
Coefficient of determination: 0.03     
no score
score: 21491457.48558

## Case 105 - stacking gb+ada+xgb+rf
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols), 
        ("cat", cat_transformer, categorical_cols)
    ]
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- rf1 = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.2, subsample=0.8, verbose=3)
- dt2 = VerboseAdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=10), n_estimators=50, learning_rate=1.0)
- xgb = XGBRegressor(verbosity=1, n_jobs=-1)
- meta_regressor = RandomForestRegressor(
    max_depth=32,
    n_estimators=1500,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=3,
    bootstrap=True,
    verbose=2,
    n_jobs=-1
)
- stacking = StackingRegressor(
    estimators=[('rf1', rf1), ('dt2', dt2), ('xgb1', xgb)],
    final_estimator=meta_regressor,
    passthrough=False, n_jobs=-1, verbose=2
)
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", stacking)
])

Mean squared error: 167815793317897.97    
Root Mean squared error: 12954373.52    
Mean absolute error: 5709140.71    
Coefficient of determination: 0.65     
model test score:  0.653873812224862     
score: 12753400.83857

## Case 106 - neural networks
- num_imputer = SimpleImputer(strategy="median")
- scaler = StandardScaler()
- get dummies
- selector = VarianceThreshold(threshold=0.01) 
- pca = PCA(n_components=0.95)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model
- lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=2)
- early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=2)
- nn_model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=150,
    batch_size=16,
    verbose=2,
    callbacks=[lr_scheduler, early_stopping]
)

Mean squared error: 173255813799230.69    
Root Mean squared error: 13162667.43    
Mean absolute error: 6386024.64    
Coefficient of determination: 0.64     
no model score    
score: 12957902.03457

## Case 107 - stacking grad+ada+xgb+rf, xgb feature importance
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="median")
- scaler = StandardScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- class VerboseAdaBoostRegressor(AdaBoostRegressor):
    def _boost(self, iboost, X, y, sample_weight, random_state):
        print(f"Training estimator {iboost + 1}/{self.n_estimators}")
        return super()._boost(iboost, X, y, sample_weight, random_state)
- rf1 = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.2, subsample=0.8, verbose=3)
- dt2 = VerboseAdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=10), n_estimators=50, learning_rate=1.0)
- xgb = XGBRegressor(verbosity=1, n_jobs=-1)
- meta_regressor = RandomForestRegressor(
    max_depth=32,
    n_estimators=1500,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=3,
    bootstrap=True,
    verbose=2,
    n_jobs=-1
)
- model = StackingRegressor(
    estimators=[('rf1', rf1), ('dt2', dt2), ('xgb1', xgb)],
    final_estimator=meta_regressor,
    passthrough=False, n_jobs=-1, verbose=2
)
- xgb, X, trainX, trainY, testX, test_data = featureImportance(
    xgb, 100, X, trainX, trainY, testX, test_data
)

Mean squared error: 167237398338203.06    
Root Mean squared error: 12932029.94    
Mean absolute error: 5665898.98    
Coefficient of determination: 0.65     
model test score:  0.6629559532356173     
score: 12748650.72259

## Case 108 - polyReg + linearReg
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")
- scaler = StandardScaler()
- get dummies
- poly = PolynomialFeatures(degree=2, include_bias=False)
- sample_train = train_data.sample(frac=0.01)
- selector = VarianceThreshold(threshold=0.9)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model = LinearRegression()

Mean squared error: 4977202288234251394482176.00    
Root Mean squared error: 2230964430069.26    
Mean absolute error: 783351980733.68    
Coefficient of determination: -9273329890.07     
model score:  0.9999837163784626     
score: 549926540162064.00000

## Case 109 - gradboost4
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model4 = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.85, 
    verbose=3
)

Mean squared error: 167918807493816.97    
Root Mean squared error: 12958348.95    
Mean absolute error: 5779233.04    
Coefficient of determination: 0.65     
model test score:  0.7234083877221744     
score: 12719710.06437

## Case 110 - gradboost5
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model5 = GradientBoostingRegressor(
    n_estimators=150,
    max_depth=2,
    learning_rate=0.15,
    subsample=0.7, 
    verbose=3
)

Mean squared error: 168175441514300.56    
Root Mean squared error: 12968247.43    
Mean absolute error: 5928197.06    
Coefficient of determination: 0.65    
model test score:  0.6665440365909008     
score: 12729943.36599

## Case 111 - gradboost6
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model6 = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    max_features=0.3, 
    verbose=3
)

Mean squared error: 166345885246589.03    
Root Mean squared error: 12897514.69    
Mean absolute error: 5828300.91    
Coefficient of determination: 0.65     
model test score:  0.6797302754028947     
score: 12699980.40753

## Case 112 - gradboost7
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model7 = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.9,
    min_samples_split=10, 
    verbose=3
)

Mean squared error: 166088771575068.12    
Root Mean squared error: 12887543.27    
Mean absolute error: 5848968.50    
Coefficient of determination: 0.65     
model test score:  0.6683665057074892     
score: 12696306.98443
 
## Case 113 - gradboost8
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model8 = GradientBoostingRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.3,
    subsample=0.75, 
    verbose=3
)

Mean squared error: 169678225768784.62    
Root Mean squared error: 13026059.49    
Mean absolute error: 5929284.63    
Coefficient of determination: 0.65     
model test score:  0.6727837630822178     
score: 12748711.96422

## Case 114 - gradboost9
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model9 = GradientBoostingRegressor(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.85,
    max_features='sqrt', 
    verbose=3
)

Mean squared error: 168921601061000.22    
Root Mean squared error: 12996984.31    
Mean absolute error: 6003153.51    
Coefficient of determination: 0.65     
model test score:  0.6636951317688753     
score: 12810788.62827

## Case 115 - gradboost10
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model0 = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.6, 
    verbose=3
)

score: 12738919.98609

## Case 116 - knn + variance fiter + PCA
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")
- scaler = MinMaxScaler()
- get dummies
- variance filter: selector = VarianceThreshold(threshold=0.01) 
- 273 columns
- pca = PCA(n_components=0.95)
- 267 columns
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model = KNeighborsRegressor(
    n_neighbors=67, 
    weights='distance',  
    leaf_size=30,     
    p=2, 
    metric='euclidean', =
    n_jobs=-1  
)

Mean squared error: 167815929949379.91    
Root Mean squared error: 12954378.79    
Mean absolute error: 5813474.85    
Coefficient of determination: 0.65     
model score:  0.9999973414753984     
score: 12759323.06052

## Case 117 - gboost9 + pCA
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies
- selector = VarianceThreshold(threshold=0.001) 
- pca = PCA(n_components=0.95)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model9 = GradientBoostingRegressor(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.85,
    max_features='sqrt', 
    verbose=3
)

Mean squared error: 177446742877784.25    
Root Mean squared error: 13320913.74    
Mean absolute error: 6629774.05    
Coefficient of determination: 0.63     
model test score:  0.6616145252063277     
score: 13021221.99635

# DAY 7: Sunday 1st December 2024

## Case 118 - regTree, forward selection
- took over 10hours
- model = fbselection( "forward", model, 10 )
- selected_features = [0, 37, 49, 111, 171, 207, 216, 231, 250, 1113]
- sample_train = train_data.sample(frac=0.1)
- model = DecisionTreeRegressor(random_state=0, max_depth=5, criterion='poisson')
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- get dummies
- scaler = StandardScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")

Mean squared error: 169740577777018.22    
Root Mean squared error: 13028452.62    
Mean absolute error: 6097042.84    
Coefficient of determination: 0.65     
model score:  0.6500690352657126     
score: 12814602.59947

## Case 119 - regTree, variance+correlation, maxabs scaler
- variance_filter = VarianceThreshold(threshold=0.001)
- to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
- model = DecisionTreeRegressor(random_state=0, max_depth=5, criterion='poisson')
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- get dummies
- scaler = MaxAbsScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- 419 columns

Mean squared error: 167281092333096.22    
Root Mean squared error: 12933719.20    
Mean absolute error: 5984727.18    
Coefficient of determination: 0.65     
model score:  0.6550849217780055     
score: 12765094.57261

## Case 120 - regTree, algorithm feature importance
- model = featureImportance( model, 200 )
- model = DecisionTreeRegressor(random_state=0, max_depth=5, criterion='poisson')
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- get dummies
- scaler = MaxAbsScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- 200 columns

Mean squared error: 167260228624135.78    
Root Mean squared error: 12932912.61    
Mean absolute error: 5984409.15    
Coefficient of determination: 0.65     
model score:  0.6555538165539976    
score: 12730638.81527

### analyzing
ok improved from best DT. lets decrease features and try again. 

## Case 121 - regTree, algo feature imp decreased, robust scaler
- model = featureImportance( model, 100 )
- model = DecisionTreeRegressor(random_state=0, max_depth=5, criterion='poisson')
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- get dummies
- scaler = RobustScaler()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- 100 columns

Mean squared error: 167257164304757.75    
Root Mean squared error: 12932794.14    
Mean absolute error: 5984224.86    
Coefficient of determination: 0.65   
model score:  0.6555538165539977    
score: 12730638.81527

### analyzing
so no difference in 200 or 100 algorithm feature importance features

## Case 122 - regTree, normalizer scaler
- model = featureImportance( model, 200 )
- model = DecisionTreeRegressor(random_state=0, max_depth=5, criterion='poisson')
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- get dummies
- scaler = Normalizer()
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- 200 columns

Mean squared error: 179202537001242.47    
Root Mean squared error: 13386655.18    
Mean absolute error: 6801823.75    
Coefficient of determination: 0.63     
model score:  0.6289503354006472     
score: 13178538.38246

## Case 123 - stacking gb+dt+knn+rf
- decision tree algorithm feature importance 200 features
- rf1 = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.2, subsample=0.8, verbose=3)
- dt2 = DecisionTreeRegressor(random_state=0, max_depth=5, criterion='poisson')
- xgb = KNeighborsRegressor( n_neighbors=67, algorithm='auto', leaf_size=30, p=2, metric='euclidean', n_jobs=-1 )
- meta_regressor = RandomForestRegressor(
    max_depth=32,
    n_estimators=1500,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=3,
    bootstrap=True,
    verbose=2,
    n_jobs=-1
)
- model = StackingRegressor(
    estimators=[('rf1', rf1), ('dt2', dt2), ('xgb1', xgb)],
    final_estimator=meta_regressor,
    passthrough=False, n_jobs=-1, verbose=2
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- get dummies
- scaler = RobustScaler()
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")

Mean squared error: 173474665659447.97    
Root Mean squared error: 13170978.16    
Mean absolute error: 5765039.99    
Coefficient of determination: 0.64     
model test score:  0.6464478155060429    
score: 12980226.72961

## Case 124 - stacking knn+knn+knn+knn
- kbest feature selection, 200 features
- model1 = KNeighborsRegressor( n_neighbors=30, algorithm='auto', leaf_size=50, p=3, metric='euclidean', n_jobs=-1, weights="distance" )
- model2 = KNeighborsRegressor( n_neighbors=50, algorithm='auto', leaf_size=70, p=2, metric='minkowski', n_jobs=-1, weights="uniform" )
- model3 = KNeighborsRegressor( n_neighbors=80, algorithm='auto', leaf_size=30, p=2, metric='euclidean', n_jobs=-1, weights="distance" )
- meta_regressor = KNeighborsRegressor( n_neighbors=67, algorithm='auto', leaf_size=10, p=2, metric='euclidean', n_jobs=-1, weights="distance" )
- model = StackingRegressor( 
    estimators=[('model1', model1), ('model2', model2), ('model3', model3)], 
    final_estimator=meta_regressor, 
    passthrough=False, n_jobs=-1, verbose=2
)
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = RobustScaler()
- get dummies, drop_first=False

Mean squared error: 164870186705817.69    
Root Mean squared error: 12840178.61    
Mean absolute error: 5372458.49    
Coefficient of determination: 0.66     
model test score:  0.8268361791338406     
score: 12689253.23345

## case 125 - randomforest, best + algo feature imp
- i am a bit iffy on this as file on kaggle stopped running upon model_predict_on_test however there was a file in output so i downloaded and submitted
- num_imputer = SimpleImputer(strategy="median")
- scaler = StandardScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model = RandomForestRegressor(max_depth=36, n_estimators=1800, min_samples_split=2, min_samples_leaf=1, max_features=0.45, bootstrap=True, verbose=2, n_jobs=-1)
- model, X, trainX, trainY, testX, test_data = featureImportance(model, 100, X, trainX, trainY, testX, test_data)

Mean squared error: 159625169771914.56    
Root Mean squared error: 12634285.49    
Mean absolute error: 4947876.31    
Coefficient of determination: 0.67    
no model score
score: 12452109.12910

## Case 126 - polyReg + randomforest
- model = RandomForestRegressor(max_depth=39, n_estimators=400, max_features='sqrt', verbose=2, n_jobs=-1)
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = StandardScaler()
- get dummies
- poly = PolynomialFeatures(degree=2, include_bias=False)
- X Y is 3% sample of train
- selector = VarianceThreshold(threshold=0.9)
- selector = VarianceThreshold(threshold=0.9)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)

Mean squared error: 189663005652428.50    
Root Mean squared error: 13771819.26    
Mean absolute error: 6575550.63    
Coefficient of determination: 0.67     
model score:  0.9502352626846423     
score: 13065940.61641

## Case 127 - polyReg + random forest
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = StandardScaler()
- get dummies
- poly = PolynomialFeatures(degree=2, include_bias=False)
- sample_train = train_data.sample(frac=0.01)
- selector = VarianceThreshold(threshold=0.9)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model = RandomForestRegressor(max_depth=35, n_estimators=500, max_features='sqrt', verbose=2, n_jobs=-1, min_samples_split=7)

Mean squared error: 151440466941947.53    
Root Mean squared error: 12306115.02    
Mean absolute error: 5501600.22    
Coefficient of determination: 0.62     
model score:  0.9302153974842944     
score: 12918342.46040

## Case 128 - xgb + PCA
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = StandardScaler()
- get dummies
- selector = VarianceThreshold(threshold=0.001) 
- pca = PCA(n_components=0.95)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)
- model = XGBRegressor(
    max_depth=10,  
    learning_rate=0.009, 
    n_estimators=1400,  
    subsample=0.8,  
    colsample_bytree=0.7,  
    reg_lambda=0.2, 
    reg_alpha=0.8,  
    verbose = 2,
    tree_method='hist',   
    predictor='auto'   
)

Mean squared error: 165138032534075.53    
Root Mean squared error: 12850604.36    
Mean absolute error: 6067012.38    
Coefficient of determination: 0.66     
model test score:  0.9717697146871356     
score: 12749993.51698

## Case 129 - stacking knn+knn+dt+knn
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = StandardScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model1 = KNeighborsRegressor( n_neighbors=30, algorithm='auto', leaf_size=50, p=3, metric='euclidean', n_jobs=-1, weights="distance" )
- model2 = KNeighborsRegressor( n_neighbors=50, algorithm='auto', leaf_size=70, p=2, metric='minkowski', n_jobs=-1, weights="uniform" )
- model4 = DecisionTreeRegressor(max_depth=5)
- meta_regressor = KNeighborsRegressor( n_neighbors=67, algorithm='auto', leaf_size=10, p=2, metric='euclidean', n_jobs=-1, weights="distance" )
- model = StackingRegressor( 
    estimators=[('model4', model4)], 
    final_estimator=meta_regressor, 
    passthrough=False, n_jobs=-1, verbose=2
)
- m, X, trainX, trainY, testX, test_data = kbest(
    None, 200, X, trainX, trainY, testX, test_data
)

Mean squared error: 169763989219714.88    
Root Mean squared error: 13029351.07    
Mean absolute error: 6047039.85    
Coefficient of determination: 0.65     
model test score:  0.646672768265262     
score: 12858957.89820

## Case 130 - polyReg + DT + stacking
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = StandardScaler()
- get dummies
- poly = PolynomialFeatures(degree=2, include_bias=False)
- sample_train = train_data.sample(frac=0.005)
- selector = VarianceThreshold(threshold=0.999999)
- 123 columns
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model = StackingRegressor( 
    estimators=[('model4', DecisionTreeRegressor(max_depth=1))], 
    final_estimator=DecisionTreeRegressor(max_depth=2), 
    passthrough=False, n_jobs=-1, verbose=2
)

Mean squared error: 399497786442319.06    
Root Mean squared error: 19987440.72    
Mean absolute error: 10181137.66    
Coefficient of determination: 0.23     
model score:  0.5160564308290309     
score: 16573081.40223

## Case 131 - gb + kbest
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = MinMaxScaler()
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model = GradientBoostingRegressor(n_estimators=600, learning_rate=0.01,min_samples_leaf=5, min_samples_split=3, random_state=2,verbose=2,max_features='log2',max_depth= 12)
- model, X, trainX, trainY, testX, test_data = kbest(model, 200, X, trainX, trainY, testX, test_data)

score: 12595325.58113

## Case 132 - polyReg + KNN
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = StandardScaler()
- get dummies
- poly = PolynomialFeatures(degree=2, include_bias=False)
- sample_train = train_data.sample(frac=0.1)
- selector = VarianceThreshold(threshold=0.999)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model = KNeighborsRegressor( n_neighbors=50, algorithm='auto', leaf_size=70, p=2, metric='minkowski', n_jobs=-1, weights="uniform" )

Mean squared error: 545826372861946.94    
Root Mean squared error: 23362927.32    
Mean absolute error: 9904841.71    
Coefficient of determination: -0.12    
model score:  -0.06714449369575992     
score: 22873886.79445

## Case 133 - xgb
- num_imputer = SimpleImputer(strategy="mean")
- cat_imputer = SimpleImputer(strategy="most_frequent")
- scaler = StandardScaler()
- get dummies
- selector = VarianceThreshold(threshold=0.001) 
- 405 columns
- pca = PCA(n_components=0.95)
- 205 columns
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)
- model = XGBRegressor(
    max_depth=10,  
    learning_rate=0.009, 
    n_estimators=1400,  
    subsample=0.8,  
    colsample_bytree=0.7,  
    reg_lambda=0.2, 
    reg_alpha=0.8,  
    verbose = 2,
    tree_method='hist',   
    predictor='auto'   
)

Mean squared error: 160932596212293.22    
Root Mean squared error: 12685921.18    
Mean absolute error: 5603029.43    
Coefficient of determination: 0.67     
model test score:  0.9982051964188322     
score: 12570455.68262

## Case 134 - neural network, 9 layers, higher xgb params
- lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1)
- early_stopping = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
- nn_model = nn_model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=250,
    batch_size=64,
    verbose=2,
    callbacks=[lr_scheduler, early_stopping]
)
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(2048, activation="relu", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.0015), loss="mse")
    return model
- xgb_model =  XGBRegressor( max_depth=19, learning_rate=0.006, n_estimators=2400, subsample=0.88, reg_lambda=0.15, reg_alpha=0.85, device = 'cuda', verbose = 2, tree_method = 'gpu_hist', predictor = 'gpu_predictor' )
- xgb_model, X, trainX, trainY, testX, test_data = featureImportance(xgb_model, 50, X, trainX, trainY, testX, test_data)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=2)
- get dummies
- scaler = StandardScaler()
- fillna(.median)
- no pipeline

RMSE: 12679833.120798344      
score: 12623184.80764

## Case 135 - neural network, 7 layers
- lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1)
- early_stopping = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
- nn_model = nn_model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=250,
    batch_size=64,
    verbose=2,
    callbacks=[lr_scheduler, early_stopping]
)
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(512, activation="relu", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model
- xgb_model = XGBRegressor(n_estimators=400, random_state=42)
- xgb_model, X, trainX, trainY, testX, test_data = featureImportance(xgb_model, 50, X, trainX, trainY, testX, test_data)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=2)
- get dummies
- scaler = StandardScaler()
- fillna(.median)
- no pipeline

RMSE: 12698692.002773136    
score: 12646712.27371

## Case 136 - neural network, xgb reduced
- lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1)
- early_stopping = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
- nn_model = nn_model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=250,
    batch_size=64,
    verbose=2,
    callbacks=[lr_scheduler, early_stopping]
)
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(1024, activation="relu", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.0015), loss="mse")
    return model
- xgb_model = XGBRegressor(n_estimators=400, random_state=42)
- xgb_model, X, trainX, trainY, testX, test_data = featureImportance(xgb_model, 50, X, trainX, trainY, testX, test_data)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=2)
- get dummies
- scaler = StandardScaler()
- fillna(.median)
- no pipeline

RMSE: 12695095.274658825    
score: 12633762.47409

## Case 137 - neural network, xgb feature imp
- lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1)
- early_stopping = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
- nn_model = nn_model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=250,
    batch_size=64,
    verbose=2,
    callbacks=[lr_scheduler, early_stopping]
)
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(1024, activation="relu", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.0015), loss="mse")
    return model
- xgb_model = XGBRegressor(n_estimators=100, random_state=42)
- xgb_model, X, trainX, trainY, testX, test_data = featureImportance(xgb_model, 50, X, trainX, trainY, testX, test_data)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=2)
- get dummies
- scaler = StandardScaler()
- fillna(.median)
- no pipeline

RMSE: 12663292.577469092     
score: 12612230.41653

# DAY 8: Monday 2nd December 2024

## Case 138 - randomforest + algorithm feature importance
- num_imputer = SimpleImputer(strategy="median")
- scaler = StandardScaler()
- no imputation on categorical
- get dummies
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- model = RandomForestRegressor(max_depth=36, n_estimators=1850, min_samples_split=2, min_samples_leaf=1, max_features=0.45, bootstrap=True, verbose=2, n_jobs=-1)
- model, X, trainX, trainY, testX, test_data = featureImportance(model, 100, X, trainX, trainY, testX, test_data)

Mean squared error: 159635519508012.88    
Root Mean squared error: 12634695.07    
Mean absolute error: 4946789.83    
Coefficient of determination: 0.67     
model score:  0.9523381486393312     
score: 12442789.30317

## Case 139 - randomforest + best par higher max_features
- model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf)
])
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols)
        # ("cat", cat_transformer, categorical_cols)
    ]
)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- rf = RandomForestRegressor(max_depth=36, n_estimators=1800, min_samples_split=2, min_samples_leaf=1, max_features=0.5, bootstrap=True, verbose=2, n_jobs=-1)

not trained on trainX, trainY, no prediction on testY     
model score:  0.9528415052518412    
score: 12438069.97995

## Case 140 - stacking rf+gb+ada
- meta_regressor = AdaBoostRegressor(n_estimators=10)
- model = StackingRegressor( 
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=100, max_depth = 5, random_state=42, verbose=2)),
        ('gb', GradientBoostingRegressor(n_estimators=10, random_state=42, verbose=1))
    ], 
    final_estimator=meta_regressor, 
    passthrough=False, n_jobs=-1, verbose=2
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- get dummies
- scaler = RobustScaler()
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")

file was run but parameters werent recorded...    
score: 12738231.25142

## Case 141 - stacking rf+gb+ada
- meta_regressor = AdaBoostRegressor(n_estimators=10)
- model = StackingRegressor( 
    estimators=[
        ('rf', RandomForestRegressor(max_depth=5, n_jobs=-1, random_state=42, verbose=2)),
        ('gb', GradientBoostingRegressor(n_estimators=10,random_state=42, verbose=1))
    ], 
    final_estimator=meta_regressor, 
    passthrough=False, n_jobs=-1, verbose=2
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- get dummies
- scaler = RobustScaler()
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")

file was run but parameters werent recorded...    
score: 12737584.77444

## Case 142 - stacking rf+xgb+ada
- meta_regressor = AdaBoostRegressor(n_estimators=10)
- model = StackingRegressor( 
    estimators= [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth = 5, random_state=42, verbose=2,n_jobs=-1)),
        ('xgb', XGBRegressor(n_estimators=200, max_depth=5,learning_rate=0.05, random_state=42))
    ], 
    final_estimator=meta_regressor, 
    passthrough=False, n_jobs=-1, verbose=2
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- get dummies
- scaler = RobustScaler()
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")

file was run but parameters werent recorded...    
score: 12661050.51915

## Case 143 - stacking rf+ridge
- meta_regressor = LinearRegression()
- model = StackingRegressor( 
    estimators= [
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', bootstrap=True, random_state=42, n_jobs=-1)),
        ('ridge', Ridge(alpha=10.0, solver='saga', tol=1e-4, max_iter=10000, random_state=42))
    ], 
    final_estimator=meta_regressor, 
    passthrough=False, n_jobs=-1, verbose=2
)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=2)
- get dummies
- scaler = RobustScaler()
- cat_imputer = SimpleImputer(strategy="most_frequent")
- num_imputer = SimpleImputer(strategy="mean")

file was run but parameters werent recorded...    
score: 12490377.08595

## Case 144 - neural networks, 3 layers, no early stopping, categorical dropped
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(64, activation=""relu"", input_dim=input_dim))
    model.add(Dense(32, activation=""relu""))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss=""mse"")
    return model
- nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
- StandardScaler	
- SimpleImputer(strategy="median")	
- dropped categorical columns
- 257 columns
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=2)

score: 13089512.03435

## Case 145 - neural networks, xgb feature imp, 4 layers
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.1))  # Leaky ReLU
    model.add(Dropout(0.4))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=""mse"")
    return model
- lr_scheduler = ReduceLROnPlateau(monitor=""val_loss"", factor=0.5, patience=5, verbose=1)
- early_stopping = EarlyStopping(monitor=""val_loss"", patience=10, restore_best_weights=True, verbose=1)
- nn_model = nn_model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[lr_scheduler, early_stopping]
)
- train_data[numerical_cols] = train_data[numerical_cols].fillna(train_data[numerical_cols].median())
- test_data[numerical_cols] = test_data[numerical_cols].fillna(test_data[numerical_cols].median())
- scaler = StandardScaler()
- drop columns categorical
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=2)
- xgb_model = XGBRegressor(n_estimators=100, random_state=42)
- xgb_model, X, trainX, trainY, testX, test_data = featureImportance(xgb_model, 50, X, trainX, trainY, testX, test_data)

score: 12823441.95400

## Case 146 - neural networks, larger network
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(256, activation=""relu"", kernel_regularizer=l2(0.01), input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation=""relu"", kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation=""relu"", kernel_regularizer=l2(0.01)))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss=""mse"")
    return model
- lr_scheduler = ReduceLROnPlateau(monitor=""val_loss"", factor=0.5, patience=5, verbose=1)
- early_stopping = EarlyStopping(monitor=""val_loss"", patience=10, restore_best_weights=True, verbose=1)
- nn_model = nn_model.fit(
    X_train_preprocessed, y_train_scaled,
    validation_data=(X_val_preprocessed, y_val_scaled),
    epochs=200,
    batch_size=64,
    verbose=1,
    callbacks=[lr_scheduler, early_stopping]
)
- num_imputer = SimpleImputer(strategy="median")
- scaler = StandardScaler()
- get dummies
- simple imputer(most frequent)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=2)

score: 12763701.04221

### analyzing
Key Changes
- Missing Values:
Added SimpleImputer to fill missing values for both numerical and categorical features.
- Larger Network:
Increased neurons in the hidden layers for better representation of high-dimensional data.
Added L2 regularization to dense layers to reduce overfitting.
- Dropout Rate:
Increased dropout to 0.3 for better regularization.
Increased Epochs and Batch Size:
Increased epochs to 200 and batch size to 64 for better convergence.
- Learning Rate Scheduler:
Added a ReduceLROnPlateau callback for dynamic learning rate adjustment.

## Case 147 - neural networks, no imputation
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(128, activation=""relu"", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation=""relu""))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation=""relu""))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss=""mse"")
    return model
- lr_scheduler = ReduceLROnPlateau(monitor=""val_loss"", factor=0.5, patience=5, verbose=1)
- early_stopping = EarlyStopping(monitor=""val_loss"", patience=10, restore_best_weights=True, verbose=1)
- nn_model.fit(
    X_train_preprocessed, y_train_scaled,
    validation_data=(X_val_preprocessed, y_val_scaled),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[lr_scheduler, early_stopping]
)
- preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)
- no imputation
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=2)

score: 12750468.62401   

## Case 148 - neural networks, higher xgb params
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(128, activation=""relu"", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation=""relu""))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation=""relu""))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss=""mse"")
    return model
- lr_scheduler = ReduceLROnPlateau(monitor=""val_loss"", factor=0.5, patience=5, verbose=1)
- early_stopping = EarlyStopping(monitor=""val_loss"", patience=10, restore_best_weights=True, verbose=1)
- nn_model = nn_model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[lr_scheduler, early_stopping]
)
- xgb_model = XGBRegressor(
    n_estimators=500, 
    learning_rate=0.01,
    random_state=42
)
- xgb_model, X, trainX, trainY, testX, test_data = featureImportance(xgb_model, 50, X, trainX, trainY, testX, test_data)
- standard scaler
- fillna.median
- drop categorical cols

score: 12703628.73551

## Case 149 - neural networks, minmax sclaing, mlp regressor, kbest feature selection
- nn_model = MLPRegressor(hidden_layer_sizes=(50, 200, 100), activation='tanh', alpha=0.0001, learning_rate='adaptive', learning_rate_init=0.003, max_iter=400, tol=1e-4, random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
- num_imputer = SimpleImputer(strategy="most-frequent")
- num_imputer = SimpleImputer(strategy="median")
- one hot encoding
- scaler = MinMaxScaler()
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=2)
- m, X, trainX, trainY, testX, test_data = kbest(210, X, trainX, trainY, testX, test_data)

score: 26266883.67793

## Case 150 - polyReg + ridge
- num_transformer = Pipeline(steps=[
    (""imputer"", SimpleImputer(strategy=""median"")),
    (""scaler"", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    (""imputer"", SimpleImputer(strategy=""most_frequent"")),
    (""onehot"", OneHotEncoder(handle_unknown=""ignore""))
])
- SimpleImputer(strategy="median") 
- SimpleImputer(strategy="most_frequent")
- standardscaler
- Ridge(alpha=1000)
- trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=2)

score: 13175268.83308

## Case 151 - neural networks, mlp regressor, more params
- regressor = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25),   
    activation='relu',                
    alpha=0.0001,                     
    learning_rate='adaptive',         
    learning_rate_init=0.001,          
    max_iter=200,                      
    tol=1e-4,                          
    random_state=42,                   
    early_stopping=True,    
    validation_fraction=0.1,         
    n_iter_no_change      )
- KBEST	210	
- min max	
- simple imputer mean for numerical 
- simple imputer most frequent for categorical
- one hot encoding

score: 12667829.54214

## Case 152 - neural networks
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(128, activation=""relu"", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation=""relu""))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation=""relu""))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss=""mse"")
    return model
- lr_scheduler = ReduceLROnPlateau(monitor=""val_loss"", factor=0.5, patience=5, verbose=1)
- early_stopping = EarlyStopping(monitor=""val_loss"", patience=10, restore_best_weights=True, verbose=1)
- nn_model = nn_model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[lr_scheduler, early_stopping]
)
- xgb_model = XGBRegressor(
    max_depth=10, 
    learning_rate=0.01,
    n_estimators=1000, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    reg_lambda=1,
    reg_alpha=0, 
    random_state=42
)	
- algo feature importance 50	
- StandardScaler	
- .fillna(train[numerical_cols].median())

score: 12697007.37604

## Case 153 - neural network
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(256, activation=""relu"", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))  # Increase dropout to prevent overfitting
    model.add(Dense(128, activation=""relu""))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation=""relu""))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation=""relu""))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=""mse"")
    return model
- XGBRegressor(n_estimators=100, random_state=42)
- xgb_model.feature_importances_ = 50	
- StandardScaler	
- .fillna(train[numerical_cols].median())

score: 12682240.55308

## Case 154 - neural networks
- def build_nn(input_dim):
    model = Sequential([
        Dense(128, activation=""""relu"""", input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation=""""relu""""),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation=""""relu""""),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=Nadam(learning_rate=0.001), loss=""""mse"""")
    return model
- lr_scheduler = ReduceLROnPlateau(monitor=""""val_loss"""", factor=0.5, patience=5, verbose=1)
- early_stopping = EarlyStopping(monitor=""""val_loss"""", patience=10, restore_best_weights=True, verbose=1)
- history = nn_model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[lr_scheduler, early_stopping]
)
- xgb feature importance = 38	
- StandardScaler	
- .fillna(X.median())	
- Using Nadam optimizer: model.compile(optimizer=Nadam(learning_rate=0.001), loss=""mse"")

- top 38 numerical features (from XGBoost importance)
- top_features = [
    ""mosque_count_500"", ""culture_objects_top_25_raion"", ""cafe_count_3000_price_2500"",
    ""cafe_count_5000_price_high"", ""cafe_count_3000"", ""cafe_count_1500"",
    ""cafe_count_5000_price_2500"", ""build_count_monolith"", ""sport_count_3000"",
    ""full_sq"", ""church_count_1500"", ""cafe_count_3000_price_4000"", 
    ""cafe_avg_price_5000"", ""build_count_before_1920"", ""cafe_count_2000_price_4000"",
    ""cafe_count_1000_na_price"", ""cafe_count_3000_price_1000"", ""leisure_count_1000"",
    ""school_km"", ""metro_km_avto"", ""big_church_count_5000"", 
    ""cafe_count_500_price_500"", ""raion_build_count_with_material_info"",
    ""cafe_count_3000_price_high"", ""cafe_count_2000_price_2500"", 
    ""cafe_count_500_price_2500"", ""office_count_3000"", ""cafe_count_5000_price_1000"",
    ""fitness_km"", ""green_part_2000"", ""railroad_station_avto_min"", 
    ""sport_count_2000"", ""bulvar_ring_km"", ""market_count_3000"",
    ""exhibition_km"", ""build_count_1921-1945"" #, ""cat__nuclear_reactor_raion_no"",
    #""cat__railroad_1line_no"",
     ,""metro_min_avto"", ""mosque_km""
]

score: 12678377.93830

## Case 155 - neural networks
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(128, activation=""relu"", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation=""relu""))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation=""relu""))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss=""mse"")
    return model
- lr_scheduler = ReduceLROnPlateau(monitor=""val_loss"", factor=0.5, patience=5, verbose=1)
- early_stopping = EarlyStopping(monitor=""val_loss"", patience=10, restore_best_weights=True, verbose=1)
- history = nn_model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[lr_scheduler, early_stopping]
)
- XGBRegressor(n_estimators=100, random_state=42
- xgb_model.feature_importances_ = 50	
- StandardScaler	
- fillna(train[numerical_cols].median()	

score: 12664792.06688

### analyzing
- Explanation of Key Changes
- Dropped Categorical Columns:
Focused only on numerical columns (float64 and int64) to simplify the preprocessing.
- Used XGBoost for Feature Selection:
Trained an XGBoost Regressor and extracted feature importances.
Selected the top 50 features based on importance, which balances accuracy and computational efficiency.
- Scaling of Features and Target:
Scaled both features and the target variable using StandardScaler to improve neural network performance.
- Neural Network Training:
Adjusted architecture with dropout layers to reduce overfitting.
Tuned callbacks (e.g., ReduceLROnPlateau, EarlyStopping) for dynamic learning rate and training termination.

## Case 156 - neural networks
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(128, activation=""relu"", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation=""relu""))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation=""relu""))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss=""mse"")
    return model
- lr_scheduler = ReduceLROnPlateau(monitor=""val_loss"", factor=0.5, patience=5, verbose=1)
- early_stopping = EarlyStopping(monitor=""val_loss"", patience=10, restore_best_weights=True, verbose=1)
- history = nn_model.fit(
    X_train, y_train_scaled,
    validation_data=(X_val, y_val_scaled),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[lr_scheduler, early_stopping]
)		"
- Feature selection using RandomForestRegressor" 
rf = RandomForestRegressor(n_estimators=100, random_state=42) = 50
- standard scaler
- numerical_imputer = SimpleImputer(strategy=""median"")
- categorical_imputer = SimpleImputer(strategy=""most_frequent"")"	
- Selected Features: ['cafe_count_1000_price_2500', 'green_part_1500', 'area_m', 'green_zone_km', 'office_sqm_5000', 'power_transmission_line_km', 'cafe_count_1000', 'water_treatment_km', 'railroad_1line', 'big_road2_km', 'zd_vokzaly_avto_km', 'cafe_count_3000_price_1000', 'museum_km', 'sport_count_1000', 'build_count_1971-1995', 'ID_bus_terminal', 'trc_sqm_1500', 'hospice_morgue_km', 'build_count_1946-1970', 'ID_big_road1', 'incineration_km', 'build_count_1921-1945', 'stadium_km', 'public_transport_station_min_walk', 'detention_facility_km', 'cafe_count_3000_price_500', 'build_count_before_1920', 'church_count_500', 'ID_metro', 'raion_build_count_with_builddate_info', 'cafe_count_1000_price_1000', 'big_market_km', 'build_count_foam', 'cafe_count_1500_price_2500', 'railroad_km', 'green_part_2000', 'raion_popul', 'radiation_km', 'life_sq', 'floor', 'cafe_count_2000', 'prom_part_3000', 'market_count_3000', 'cafe_count_2000_price_2500', 'cafe_count_3000', 'market_count_2000', 'prom_part_5000', 'sport_objects_raion', 'full_sq', 'prom_part_500'](ReduceLROnPlateau) and early stopping to ensure efficient training."

score: 12657135.79789

### analyzing
Key Points
- Feature Selection:
Used RandomForestRegressor to rank feature importance.
Kept only the top 50 most important features for training.
- Reduced Input Dimensions:
By focusing only on the most relevant columns, we reduce model complexity, making it faster to train and less prone to overfitting.
- Preprocessing for Important Features:
Scaled only the selected features to ensure optimal performance for neural networks.
- Callbacks:
Included learning rate adjustment 

## Case 157 - neural networks
- def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.1))  # Leaky ReLU
    model.add(Dropout(0.4))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=""mse"")
    return model
- lr_scheduler = ReduceLROnPlateau(monitor=""val_loss"", factor=0.5, patience=5, verbose=1)
- early_stopping = EarlyStopping(monitor=""val_loss"", patience=10, restore_best_weights=True, verbose=1)
- history = nn_model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[lr_scheduler, early_stopping]
)
- xgb feature importance = 50	
- StandardScaler	
- fillna(train[numerical_cols].median()
- xgb_model = XGBRegressor(
    max_depth=10,     
    learning_rate=0.01, 
    n_estimators=1000, 
    subsample=0.8,   
    colsample_bytree=0.8,   
    reg_lambda=1,   
    reg_alpha=0,        
    random_state=42
)
- Selected Features: ['mosque_count_500' 'mosque_count_1000' 'culture_objects_top_25_raion'
 'mosque_count_2000' 'cafe_count_5000_price_high' 'full_sq'
 'cafe_count_500_price_high' 'cafe_count_3000_price_2500'
 'cafe_count_3000_price_high' 'cafe_count_5000_price_2500'
 'mosque_count_1500' 'cafe_count_3000' 'cafe_count_2000_price_high'
 'cafe_count_1000_price_high' 'mosque_count_3000'
 'cafe_count_2000_price_4000' 'cafe_count_2000'
 'cafe_count_1000_price_4000' 'mosque_count_5000'
 'cafe_count_3000_price_4000' 'market_count_2000' 'market_count_500'
 'raion_build_count_with_builddate_info' 'leisure_count_5000'
 'leisure_count_500' 'cafe_count_5000' 'cafe_avg_price_5000'
 'leisure_count_3000' 'kremlin_km' 'church_count_1500' 'sport_count_3000'
 'sadovoe_km' 'leisure_count_1000' 'cafe_count_3000_na_price'
 'big_church_count_500' 'build_count_foam' 'cafe_count_1500'
 'leisure_count_2000' 'cafe_count_5000_price_500'
 'cafe_count_1500_price_high' 'sport_count_2000' 'build_count_1921-1945'
 'build_count_before_1920' 'cafe_count_5000_price_1500'
 'additional_education_km' 'cafe_avg_price_3000' 'ID_railroad_terminal'
 'bulvar_ring_km' 'cafe_count_5000_price_1000' 'market_count_1500']
- drop categorical cols

score: 12619655.52095