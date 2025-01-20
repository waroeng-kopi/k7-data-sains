# Import library yang diperlukan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 1. Memuat dataset
# Gantilah 'path_to_your_dataset.csv' dengan path ke dataset Anda
data = pd.read_csv('Dataset Update pasien Gagal Jantung.csv')

# 2. Pra-pemrosesan Data
data.fillna(data.mean(), inplace=True)
data['sex'] = data['sex'].map({'M': 1, 'F': 0})
data['anaemia'] = data['anaemia'].map({'Yes': 1, 'No': 0})
data['high_blood_pressure'] = data['high_blood_pressure'].map({'Yes': 1, 'No': 0})
data['diabetes'] = data['diabetes'].map({'Yes': 1, 'No': 0})
data['smoking'] = data['smoking'].map({'Yes': 1, 'No': 0})

# Memisahkan fitur dan target
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# 3. Membagi Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Membangun Model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Evaluasi Model
y_pred = model.predict(X_test)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy:.2f}')

# Menampilkan confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Menampilkan classification report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# Visualisasi pohon keputusan
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Survive', 'Death'])
plt.title('Decision Tree for Heart Failure Prediction')
plt.show()
