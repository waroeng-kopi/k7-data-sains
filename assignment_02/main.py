import pandas as pd

# Load the Daftar-mahasiswa-baru.csv file
df_mahasiswa_baru = pd.read_csv("./data/Daftar-mahasiswa-baru.csv")

# Extract unique program studies from the 'Program Studi' column
unique_program_studies = df_mahasiswa_baru['Program Studi'].nunique()

# List of unique program studies
unique_program_studies_list = df_mahasiswa_baru['Program Studi'].unique()

print("Jumlah Program Studi Unik:", unique_program_studies)
print("Daftar Program Studi Unik:", unique_program_studies_list)
