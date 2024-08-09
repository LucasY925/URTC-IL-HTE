# KNN
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import shap
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/lucasyanney/Downloads/more_2122_anon.csv')

encoder = LabelEncoder()
categorical_columns = ['item_id', 'passage_type', 'item_lab']
for col in categorical_columns:
    data[f'{col}_encoded'] = encoder.fit_transform(data[col])

X = data.drop(['s_correct'] + categorical_columns, axis=1)
y = data['s_correct']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
knn_model = KNeighborsClassifier()
knn_grid = GridSearchCV(knn_model, knn_params, cv=5, scoring='f1')
knn_grid.fit(X_train, y_train)
knn_best_model = knn_grid.best_estimator_

knn_y_pred = knn_best_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
knn_f1 = f1_score(y_test, knn_y_pred)

print("KNN Classifier:")
print(f"Accuracy: {knn_accuracy:.2f}")
print(f"F1-score: {knn_f1:.2f}")

background = shap.sample(X_test, 100)
explainer = shap.KernelExplainer(knn_best_model.predict, background)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('KNN Feature Importance (SHAP)')
plt.tight_layout()
plt.savefig('knn_shap_importance.png')
plt.close()
