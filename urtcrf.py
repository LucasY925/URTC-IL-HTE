# Random Forest
from sklearn.ensemble import RandomForestClassifier
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

rf_params = {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}
rf_model = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='f1')
rf_grid.fit(X_train, y_train)
rf_best_model = rf_grid.best_estimator_
rf_y_pred = rf_best_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_f1 = f1_score(y_test, rf_y_pred)
print("Random Forest Classifier:")
print(f"Accuracy: {rf_accuracy:.2f}")
print(f"F1-score: {rf_f1:.2f}")

explainer = shap.TreeExplainer(rf_best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('Random Forest Feature Importance (SHAP)')
plt.tight_layout()
plt.savefig('rf_shap_importance.png')
plt.close()
