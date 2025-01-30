import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Membaca dataset
df = pd.read_csv('diabetes.csv')

# Pra-pemrosesan
# Pastikan tidak ada missing values
print(df.isnull().sum())

# Memisahkan fitur dan target
X = df.drop('Outcome', axis=1)  # Fitur
y = df['Outcome']  # Target

# Membagi dataset menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iterasi 1: Bangun model hanya dengan satu tingkat pohon
model_iter1 = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=42)
model_iter1.fit(X_train, y_train)

# Evaluasi model pada iterasi pertama
y_pred_iter1 = model_iter1.predict(X_test)
print(f"Iterasi 1 - Akurasi: {accuracy_score(y_test, y_pred_iter1)}")
print("Iterasi 1 - Laporan Klasifikasi:")
print(classification_report(y_test, y_pred_iter1))

# Visualisasi pohon keputusan untuk iterasi 1
plt.figure(figsize=(10, 10))
plot_tree(model_iter1, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Pohon Keputusan - Iterasi 1", fontsize = 40)
plt.savefig("decision_tree_plot_iterasi_1.svg", format='svg', bbox_inches='tight', dpi=1200)
plt.savefig("decision_tree_plot_iterasi_1.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.savefig("decision_tree_plot_iterasi_1.png", format='png', bbox_inches='tight', dpi=1200)
plt.show()



# Iterasi 2: Bangun model dengan dua tingkat pohon
model_iter2 = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=42)
model_iter2.fit(X_train, y_train)

# Evaluasi model pada iterasi kedua
y_pred_iter2 = model_iter2.predict(X_test)
print(f"Iterasi 2 - Akurasi: {accuracy_score(y_test, y_pred_iter2)}")
print("Iterasi 2 - Laporan Klasifikasi:")
print(classification_report(y_test, y_pred_iter2))

# Visualisasi pohon keputusan untuk iterasi 2
plt.figure(figsize=(10, 10))
plot_tree(model_iter2, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Pohon Keputusan - Iterasi 2", fontsize = 40)
plt.savefig("decision_tree_plot_iterasi_2.svg", format='svg', bbox_inches='tight', dpi=1200)
plt.savefig("decision_tree_plot_iterasi_2.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.savefig("decision_tree_plot_iterasi_2.png", format='png', bbox_inches='tight', dpi=1200)
plt.show()



# Iterasi 3: Bangun model pohon keputusan penuh
model_iter3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
model_iter3.fit(X_train, y_train)

# Evaluasi model pada iterasi ketiga
y_pred_iter3 = model_iter3.predict(X_test)
print(f"Iterasi 3 - Akurasi: {accuracy_score(y_test, y_pred_iter3)}")
print("Iterasi 3 - Laporan Klasifikasi:")
print(classification_report(y_test, y_pred_iter3))

# Visualisasi pohon keputusan untuk iterasi 3
plt.figure(figsize=(40, 10))
plot_tree(model_iter3, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Pohon Keputusan - Iterasi 3", fontsize = 40)
plt.savefig("decision_tree_plot_iterasi_3.svg", format='svg', bbox_inches='tight', dpi=1200)
plt.savefig("decision_tree_plot_iterasi_3.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.savefig("decision_tree_plot_iterasi_3.png", format='png', bbox_inches='tight', dpi=1200)
plt.show()
