import pandas as pd

# Load the files
df_mahasiswa_baru = pd.read_csv("./data/Daftar-mahasiswa-baru.csv")
df_yudisium = pd.read_csv("./data/Yudisium-2014-2023.csv")

# Remove the dots from the NPM column in both datasets
df_mahasiswa_baru['NPM'] = df_mahasiswa_baru['NPM'].str.replace('.', '', regex=False)
df_yudisium['NPM'] = df_yudisium['NPM'].str.replace('.', '', regex=False)

# Merge the dataframes based on the 'NPM' column
merged_data = pd.merge(df_mahasiswa_baru, df_yudisium, on='NPM', how='left', indicator=True)

# Create a new column 'Status' to determine if the student is graduated or DO
merged_data['Status'] = merged_data['_merge'].apply(lambda x: 'Lulus' if x == 'both' else 'DO')

# Tabulate the data to count how many students are 'Lulus' and how many are 'DO'
status_counts = merged_data['Status'].value_counts()

# Display the status counts and the first few rows of the merged data
print(status_counts)
print(merged_data[['NPM', 'Nama', 'Program Studi', 'Status']].head())

# Filter only DO students
do_students = merged_data[merged_data['Status'] == 'DO']

# Extract year of entry from the 'NPM' column
do_students['Tahun Masuk'] = do_students['NPM'].str[2:6]

# Filter students from the 2017 cohort
do_students_2017 = do_students[do_students['Tahun Masuk'] == '2017']

# Tabulate the data by program study
tabulated_data = do_students_2017['Program Studi'].value_counts()

# Display the tabulated data
print("Tabulasi DO Angkatan 2017 per Program Studi:")
print(tabulated_data)

# Optional: Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
tabulated_data.plot(kind='bar', color='orange', edgecolor='black')
plt.title("Jumlah Mahasiswa DO Angkatan 2017 per Program Studi")
plt.xlabel("Program Studi")
plt.ylabel("Jumlah DO")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
