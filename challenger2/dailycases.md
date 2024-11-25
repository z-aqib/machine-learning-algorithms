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
model score:  0.6611981837697488

Mean squared error: 191699960168873.97    
Root Mean squared error: 13845575.47    
Mean absolute error: 6682687.71    
Coefficient of determination: 0.60  
score: 13443448.16606