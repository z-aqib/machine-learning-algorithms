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