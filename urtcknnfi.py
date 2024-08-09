# KNN Feature Importance
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/lucasyanney/Downloads/more_2122_anon.csv')

encoder = LabelEncoder()
categorical_columns = ['item_id', 'passage_type', 'item_lab']
for col in categorical_columns:
    data[f'{col}_encoded'] = encoder.fit_transform(data[col])

X = data.drop(['s_correct'] + categorical_columns, axis=1)
y = data['s_correct']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

perm_importance = permutation_importance(knn_model, X_test, y_test, n_repeats=30, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.title('KNN Feature Importance')
plt.xlabel('Mean Decrease in Accuracy')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('knn_feature_importance.png')
plt.close()
