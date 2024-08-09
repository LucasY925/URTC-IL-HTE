# EIRM Feature Importance
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('/Users/lucasyanney/Downloads/more_2122_anon.csv')

encoder = LabelEncoder()
categorical_columns = ['item_id', 'passage_type', 'item_lab']
for col in categorical_columns:
    data[f'{col}_encoded'] = encoder.fit_transform(data[col])

X_train, X_test, y_train, y_test = train_test_split(data, data['s_correct'], test_size=0.2, random_state=42)

formula = 's_correct ~ s_mapritread2122f + s_itt_2122 + passage_type_encoded + item_lab_encoded + (1|item_id_encoded)'
model = smf.mixedlm(formula, data=X_train, groups=X_train['item_id_encoded'])
result = model.fit()

coef = result.fe_params
coef = coef.drop('Intercept')

plt.figure(figsize=(10, 6))
coef.abs().sort_values().plot(kind='barh')
plt.title('EIRM Feature Importance')
plt.xlabel('Coefficient Value (Absolute)')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('eirm_feature_importance.png')
plt.close()
