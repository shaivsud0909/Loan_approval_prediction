from sklearn.model_selection import RandomizedSearchCV

def tune_random_forest(pipeline, X_train, y_train):

    param_dist = {

        'classifier__n_estimators': [100, 200, 300,400,500,600,700,800],
        'classifier__max_depth': [None, 10, 20, 30],

        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1,2,4,5,6,7,8],

        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='accuracy', 
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    return random_search
