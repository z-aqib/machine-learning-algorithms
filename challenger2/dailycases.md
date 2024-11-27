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
okay nice, decreased. we dont need to do mroe on estimators as we have reached breakeven. lets do some other grid as well

// need to fix knn-regressor code

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

## Case X - xgb, loop for best kbest features
- model = XGBRegressor(max_depth=10, learning_rate=0.01, n_estimators=1000, subsample=0.8, colsample_bytree=0.8, reg_lambda=1, reg_alpha=0, random_state=42)
- num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
- cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
- kbest loop: 50 to 271
- best: 