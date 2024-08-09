# Heatmap
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/Users/lucasyanney/Downloads/more_2122_anon.csv')

encoder = LabelEncoder()
categorical_columns = ['item_id', 'passage_type', 'item_lab']
for col in categorical_columns:
    data[f'{col}_encoded'] = encoder.fit_transform(data[col])

X = data.drop(['s_correct'] + categorical_columns, axis=1)
y = data['s_correct']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

correlation_matrix = X.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

lasso_model = LassoCV(cv=5, random_state=42)
lasso_model.fit(X_train, y_train)

lasso_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs(lasso_model.coef_)
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=lasso_importance)
plt.title('LASSO Feature Importance')
plt.tight_layout()
plt.savefig('lasso_feature_importance.png')
plt.close()
