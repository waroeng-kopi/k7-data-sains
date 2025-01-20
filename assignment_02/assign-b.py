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
