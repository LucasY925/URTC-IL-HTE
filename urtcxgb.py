# XGBoost
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import shap
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('/Users/lucasyanney/Downloads/more_2122_anon.csv')
encoder = LabelEncoder()
categorical_columns = ['item_id', 'passage_type', 'item_lab']
for col in categorical_columns:
    data[f'{col}_encoded'] = encoder.fit_transform(data[col])
X = data.drop(['s_correct'] + categorical_columns, axis=1)
y = data['s_correct']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_params = {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.01, 0.001], 'max_depth': [3, 5, 7]}
xgb_model = XGBClassifier(random_state=42)
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='f1')
xgb_grid.fit(X_train, y_train)
xgb_best_model = xgb_grid.best_estimator_
xgb_y_pred = xgb_best_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
xgb_f1 = f1_score(y_test, xgb_y_pred)
print("XGBoost Classifier:")
print(f"Accuracy: {xgb_accuracy:.2f}")
print(f"F1-score: {xgb_f1:.2f}")

explainer = shap.TreeExplainer(xgb_best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('XGBoost Feature Importance (SHAP)')
plt.tight_layout()
plt.savefig('xgb_shap_importance.png')
plt.close()
