import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# 1. Загрузка данных
df = pd.read_csv('data/diabetes.csv')

# 2. Проверка пропусков только в нужных столбцах
cols_with_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_missing:
    df[col] = df[col].replace(0, pd.NA)
    median = df[col].median()
    df[col] = df[col].fillna(median)

# 3. Разделение данных
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 4. Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Оценка модели
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nКлассификационный отчет:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {roc_auc:.3f}")

# 8. Сохранение модели и scaler
joblib.dump(model, 'diabetes_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Модель и scaler сохранены.")