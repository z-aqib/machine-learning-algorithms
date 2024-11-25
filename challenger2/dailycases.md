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