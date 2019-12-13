import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import mean_squared_error as mse, r2_score, roc_curve, confusion_matrix, classification_report, \
    roc_auc_score

plt.close('all')

# ===========================================
desired_width = 320
pd.set_option('display.width', desired_width)  # Show columns horizontally in console
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)  # Show as many columns as I want in console
pd.set_option('display.max_rows', 1000)  # Show as many rows as I want in console
# ===========================================

# +++++++++++++++ On partial dependence plots ++++++++++++++ #
#                                                            #
# https://www.kaggle.com/dansbecker/partial-dependence-plots #
#                                                            #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

# Load data
data = pd.read_csv('salary.csv')
print(data.head(5))
print(data.info())
print(data.describe())

# =================================================================================== EDA
# Quick EDA
plt.figure()
sns.barplot(x='Education', y='Salary', data=data, ci=None)
plt.figure()
sns.barplot(x='Education', y='Salary', hue='Management', data=data, ci=None)
# plt.figure()
sns.lmplot(x='Experience', y='Salary', hue='Management', data=data, ci=None)

data = pd.get_dummies(data, drop_first=True)  # Transform categorical data ('Education')

# Correlation matrix
corr = pd.DataFrame(data).corr()
print(corr)

f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240, 10, as_cmap=True),
            square=True, ax=ax)

# ===================================================================================  Simple Regressor

seed = 123

X = data.drop(['Salary'], axis=1)  # Features
y = data['Salary']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

lr = LinearRegression()

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

lr_error = mse(y_test, y_pred)**(1/2)
lr_accuracy = r2_score(y_test, y_pred)

print('Test set RMSE of linear regressor: {:.2f}'.format(lr_error))
print('Test set score of linear regressor: {:.4f}'.format(lr_accuracy))

# plt.figure()
plot_partial_dependence(lr, features=[0, 1, 2, 3], X=X_train, feature_names=X.columns, grid_resolution=20)

'''
# ============================================================================================ GridSearch RandomForest

rf_g = RandomForestRegressor(random_state=seed)

params_rf = {'n_estimators': [10, 15, 20], 'max_depth': [5, 7, 10], 'min_samples_leaf': np.arange(0.01, 0.1, 0.02)}

grid_rf = GridSearchCV(estimator=rf_g, param_grid=params_rf, cv=3, scoring='neg_mean_squared_error',
                       verbose=0, n_jobs=-1)

grid_rf.fit(X_train, y_train)

best_hyperparams = grid_rf.best_params_
print('Best hyperparameters:\n', best_hyperparams)

best_model = grid_rf.best_estimator_
y_pred = best_model.predict(X_test)

grid_rf_error = mse(y_test, y_pred)**(1/2)
grid_rf_accuracy = r2_score(y_test, y_pred)

print('Test set RMSE of complex random forest: {:.2f}'.format(grid_rf_error))
print('Test set score of complex random forest: {:.4f}'.format(grid_rf_accuracy))

importances_rf = pd.Series(best_model.feature_importances_, index=X.columns)
sorted_importances_rf = importances_rf.sort_values()
plt.figure()
sorted_importances_rf.plot(kind='barh', color='lightgreen')

plot_partial_dependence(lr, features=[0, 1, 2, 3], X=X_train, feature_names=X.columns, grid_resolution=20)

'''
# ===================================================================================  Simple Logistic Regression

# Transform Boolean to categorical
data['Management'] = data['Management'].astype('category')
data['Management'] = data['Management'].cat.codes

X = data.drop(['Management'], axis=1)  # Features
y = data['Management']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

logreg_error = mse(y_test, y_pred)**(1/2)
logreg_accuracy = r2_score(y_test, y_pred)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot ROC curve
plt.figure()
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')

print(logreg.coef_)
print(roc_auc_score(y_test, y_pred_prob))

plot_partial_dependence(logreg, features=[0, 1, 2, 3], X=X_train, feature_names=X.columns, grid_resolution=20)

# ==============
plt.show()
