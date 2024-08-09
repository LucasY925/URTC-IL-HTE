# EIRM
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import shap

data = pd.read_csv('/Users/lucasyanney/Downloads/more_2122_anon.csv')
encoder = LabelEncoder()
data['item_id_encoded'] = encoder.fit_transform(data['item_id'])
data['passage_type_encoded'] = encoder.fit_transform(data['passage_type'])
data['item_lab_encoded'] = encoder.fit_transform(data['item_lab'])
X_train, X_test, y_train, y_test = train_test_split(data, data['s_correct'], test_size=0.2, random_state=42)

formula = 's_correct ~ s_mapritread2122f + s_itt_2122 + passage_type_encoded + item_lab_encoded + (1|item_id_encoded)'
model = smf.mixedlm(formula, data=X_train, groups=X_train['item_id_encoded'])
result = model.fit()
y_pred = result.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
print("EIRM Model Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1-score: {f1:.2f}")

explainer = shap.LinearExplainer(result)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('EIRM Feature Importance (SHAP)')
plt.tight_layout()
plt.savefig('eirm_shap_importance.png')
plt.close()
