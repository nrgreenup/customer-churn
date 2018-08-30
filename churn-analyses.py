#### Fit models for Telco Customer Churn Data ####

### IMPORTANT: 'churn-clean_preprocess.py' must be run prior to this file

"""
Package versions:
python        : 3.6.5
scitkit-learn : 0.19.1
pandas        : 0.23.0
numpy         : 1.14.3
matplotlib    : 2.2.2
seaborn       : 0.8.1
"""

### Split into training for df_enc data
X = df_enc.drop('Churn', axis = 1)
y= df_enc['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3,
                                                    stratify = y,
                                                    random_state = 30)

### Fit KNN Model
## Start time
knn_start = time.time()

## Instantiate classifier

knn = KNeighborsClassifier()

## Set up hyperparameter grid for tuning
knn_param_grid = {'n_neighbors' : np.arange(5,26),
                  'weights' : ['uniform', 'distance']}

## Tune hyperparameters
knn_cv = GridSearchCV(knn, param_grid = knn_param_grid, cv = 5)

## Fit knn to training data
knn_cv.fit(X_train, y_train)

## Get info about best hyperparameters
print("Tuned KNN Parameters: {}".format(knn_cv.best_params_))
print("Best KNN Training Score: {}".format(knn_cv.best_score_)) 

## Predict knn on test data
print("KNN Test Performance: {}".format(knn_cv.score(X_test, y_test)))

## Obtain model performance metrics
knn_pred_prob = knn_cv.predict_proba(X_test)[:,1]
knn_auroc = roc_auc_score(y_test, knn_pred_prob)
print("KNN AUROC: {}".format(knn_auroc))
knn_y_pred = knn_cv.predict(X_test)
print(classification_report(y_test, knn_y_pred))

## End time
knn_end = time.time()
print(knn_end - knn_start)

### Fit logistic regression model
## Start time
lr_start = time.time()

## Instantiate classifier
lr = LogisticRegression(random_state = 30)

## Set up hyperparameter grid for tuning
lr_param_grid = {'C' : [0.0001, 0.001, 0.01, 0.05, 0.1] }

## Tune hyperparamters
lr_cv = GridSearchCV(lr, param_grid = lr_param_grid, cv = 5)

## Fit lr to training data
lr_cv.fit(X_train, y_train)

## Get info about best hyperparameters
print("Tuned LR Parameters: {}".format(lr_cv.best_params_))
print("Best LR Training Score:{}".format(lr_cv.best_score_)) 

## Predict lr on test data
print("LR Test Performance: {}".format(lr_cv.score(X_test, y_test)))

## Obtain model performance metrics
lr_pred_prob = lr_cv.predict_proba(X_test)[:,1]
lr_auroc = roc_auc_score(y_test, lr_pred_prob)
print("LR AUROC: {}".format(lr_auroc))
lr_y_pred = lr_cv.predict(X_test)
print(classification_report(y_test, lr_y_pred))

## End time
lr_end = time.time()
print(lr_end - lr_start)

### Fit Random Forest
## Start time
rf_start = time.time()

## Instatiate classifier
rf = RandomForestClassifier(random_state = 30)

## Set up hyperparameter grid for tuning
rf_param_grid = {'n_estimators': [200, 250, 300, 350, 400, 450, 500],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [3, 4, 5, 6, 7],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4]}

## Tune hyperparameters
rf_cv = RandomizedSearchCV(rf, param_distributions = rf_param_grid, cv = 5, 
                           random_state = 30, n_iter = 20)

## Fit RF to training data
rf_cv.fit(X_train, y_train)

## Get info about best hyperparameters
print("Tuned RF Parameters: {}".format(rf_cv.best_params_))
print("Best RF Training Score:{}".format(rf_cv.best_score_)) 

## Predict RF on test data
print("RF Test Performance: {}".format(rf_cv.score(X_test, y_test)))

## Obtain model performance metrics
rf_pred_prob = rf_cv.predict_proba(X_test)[:,1]
rf_auroc = roc_auc_score(y_test, rf_pred_prob)
print("RF AUROC: {}".format(rf_auroc))
rf_y_pred = rf_cv.predict(X_test)
print(classification_report(y_test, rf_y_pred))

## Inspect feature importances
rf_optimal = rf_cv.best_estimator_
rf_feat_importances = pd.Series(rf_optimal.feature_importances_, 
                             index=X_train.columns)
rf_feat_importances.nlargest(5).plot(kind='barh', color = 'r')
plt.title('Feature Importances from Random Forest Classifier')
plt.xlabel('Relative Importance')
plt.ylabel('Feature Name')
plt.savefig('model-rf_feature_importances.png', dpi = 200, 
            bbox_inches = 'tight')
plt.show()

## End time
rf_end = time.time()
print(rf_end - rf_start)

### Fit GradientBoostingClassifier model
## Start time
sgb_start = time.time()

## Instantiate classifier
sgb = GradientBoostingClassifier(random_state = 30)

## Set up hyperparameter grid for tuning
sgb_param_grid = {'n_estimators' : [200, 300, 400, 500],
                  'learning_rate' : [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
                  'max_depth' : [3, 4, 5, 6, 7],
                  'min_samples_split': [2, 5, 10, 20],
                  'min_weight_fraction_leaf': [0.001, 0.01, 0.05],
                  'subsample' : [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                  'max_features': ['sqrt', 'log2']}

## Tune hyperparamters
sgb_cv = RandomizedSearchCV(sgb, param_distributions = sgb_param_grid, cv = 5, 
                            random_state = 30, n_iter = 20)

## Fit SGB to training data
sgb_cv.fit(X_train, y_train)

## Get info about best hyperparameters
print("Tuned SGB Parameters: {}".format(sgb_cv.best_params_))
print("Best SGB Training Score:{}".format(sgb_cv.best_score_)) 

## Predict SGB on test data
print("SGB Test Performance: {}".format(sgb_cv.score(X_test, y_test)))

## Obtain model performance metrics
sgb_pred_prob = sgb_cv.predict_proba(X_test)[:,1]
sgb_auroc = roc_auc_score(y_test, sgb_pred_prob)
print("SGB AUROC: {}".format(sgb_auroc))
sgb_y_pred = sgb_cv.predict(X_test)
print(classification_report(y_test, sgb_y_pred))

## Inspect feature importances
sgb_optimal = sgb_cv.best_estimator_
sgb_feat_importances = pd.Series(sgb_optimal.feature_importances_, 
                                 index=X_train.columns)
sgb_feat_importances.nlargest(5).plot(kind='barh', color = 'r')
plt.title('Feature Importances from Stochastic Gradient Boosting Classifier')
plt.xlabel('Relative Importance')
plt.ylabel('Feature Name')
plt.savefig('model-sgb_feature_importances.png', dpi = 200, 
            bbox_inches = 'tight')
plt.show()

## End time
sgb_end = time.time()
print(sgb_end - sgb_start)

### Plot ROC for all models
knn_fpr, knn_tpr, knn_thresh = roc_curve(y_test, knn_pred_prob)
plt.plot(knn_fpr,knn_tpr,label="KNN: auc="+str(round(knn_auroc, 3)),
         color = 'blue')

lr_fpr, lr_tpr, lr_thresh = roc_curve(y_test, lr_pred_prob)
plt.plot(lr_fpr,lr_tpr,label="LR: auc="+str(round(lr_auroc, 3)),
         color = 'red')

rf_fpr, rf_tpr, rf_thresh = roc_curve(y_test, rf_pred_prob)
plt.plot(rf_fpr,rf_tpr,label="RF: auc="+str(round(rf_auroc, 3)),
         color = 'green')

sgb_fpr, sgb_tpr, sgb_thresh = roc_curve(y_test, sgb_pred_prob)
plt.plot(sgb_fpr,sgb_tpr,label="SGB: auc="+str(round(sgb_auroc, 3)),
         color = 'yellow')

plt.plot([0, 1], [0, 1], color='gray', lw = 1, linestyle='--', 
         label = 'Random Guess')

plt.legend(loc = 'best', frameon = True, facecolor = 'lightgray')
plt.title('ROC Curve for Classification Models')
plt.xlabel('False Positive Rate (1 - specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.text(0.85,0.75, 'threshold = 0', fontsize = 8)
plt.arrow(0.85,0.8, 0.14,0.18, head_width = 0.01)
plt.text(0.05,0, 'threshold = 1', fontsize = 8)
plt.arrow(0.05,0, -0.03,0, head_width = 0.01)
plt.savefig('plot-ROC_4models.png', dpi = 500)
plt.show()



"""
### Fit SVC model
## Instantiate classifier
svc = SVC(random_state = 30, probability = True)

## Set up hyperparameter grid for tuning
svc_param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1]}

## Tune hyperparamters
svc_cv = RandomizedSearchCV(svc, param_distributions = svc_param_grid, cv = 5, 
                            random_state = 30, n_iter = 20)

## Fit SVC to training data
svc_cv.fit(X_train, y_train)

## Get info about best hyperparameters
print("Tuned SVC Parameters: {}".format(svc_cv.best_params_))
print("Best SVC Training Score:{}".format(svc_cv.best_score_)) 

## Predict SVC on test data
print("SVC Test Performance: {}".format(svc_cv.score(X_test, y_test)))

## Obtain model performance metrics
svc_pred_prob = svc_cv.predict_proba(X_test)[:,1]
svc_auroc = roc_auc_score(y_test, svc_pred_prob)
print("SVC AUROC: {}".format(svc_auroc))
svc_y_pred = svc_cv.predict(X_test)
print(classification_report(y_test, svc_y_pred))
"""