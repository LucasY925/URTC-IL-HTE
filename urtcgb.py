# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
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

gb_params = {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.01, 0.001], 'max_depth': [3, 5, 7]}
gb_model = GradientBoostingClassifier(random_state=42)
gb_grid = GridSearchCV(gb_model, gb_params, cv=5, scoring='f1')
gb_grid.fit(X_train, y_train)
gb_best_model = gb_grid.best_estimator_
gb_y_pred = gb_best_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_y_pred)
gb_f1 = f1_score(y_test, gb_y_pred)
print("Gradient Boosting Classifier:")
print(f"Accuracy: {gb_accuracy:.2f}")
print(f"F1-score: {gb_f1:.2f}")

explainer = shap.TreeExplainer(gb_best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('Gradient Boosting Feature Importance (SHAP)')
plt.tight_layout()
plt.savefig('gb_shap_importance.png')
plt.close()
