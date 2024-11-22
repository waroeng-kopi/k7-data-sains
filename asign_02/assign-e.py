import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math  # Import modul math

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

# Select only graduated students for further analysis
graduated_students = merged_data[merged_data['Status'] == 'Lulus'].copy()

# Process graduation years and program
graduated_students['kode_lulus'] = graduated_students['kode_lulus'].astype(str)
graduated_students['Tahun Lulus'] = graduated_students['kode_lulus'].str[:4]
graduated_students['Semester'] = graduated_students['kode_lulus'].str[-1]

filtered_graduates = graduated_students[graduated_students['Tahun Lulus'].isin(['2021', '2022', '2023'])]
grouped_data = filtered_graduates.groupby(['Program Studi', 'Tahun Lulus']).size().unstack(fill_value=0)

# Custom colors for each department
jurusan_colormap = {
    "S1 Teknik Kendaraan Ringan": "#FFA07A",  # Light Salmon
    "S1 Teknik Astronomi": "#ADD8E6",  # Light Blue
    "S1 Teknik Otomotif": "#FF6347",  # Tomato
    "S1 Mekatronika": "#D3D3D3",  # Light Grey
    "S1 Teknik Perhotelan": "#FFC0CB",  # Pink
    "S1 Teknik Tata Boga": "#D8BFD8",  # Thistle
    "S1 Teknik Telekomunikasi": "#FFFFE0",  # Light Yellow
    "S1 Teknik Industri": "#F4A300",  # Amber
    "S1 Teknik Satelit": "#9B30FF",  # Purple
    "S1 Teknik Perkapalan": "#4169E1",  # Royal Blue
    "S1 Teknik Kayu": "#98FB98",  # Pale Green
    "S1 Teknik K3": "#FFA500",  # Orange
    "S1 Teknik Nuklir": "#B22222",  # Firebrick
    "S1 Teknik Kimia": "#FFD700",  # Gold
    "S2 Manajemen": "#F5F5DC",  # Beige
    "S2 Hukum": "#9370DB",  # Medium Purple
}

# Create gradient colors for years
jurusan_color_map = {}
for jurusan, base_color in jurusan_colormap.items():
    base_rgb = mcolors.to_rgb(base_color)
    gradient = [
        mcolors.to_hex((base_rgb[0] * (1 - i / 8),
                        base_rgb[1] * (1 - i / 8),
                        base_rgb[2] * (1 - i / 8)))
        for i in range(3)
    ]
    jurusan_color_map[jurusan] = gradient

# Data for Nested Pie Chart
outer_labels = grouped_data.index
outer_sizes = grouped_data.sum(axis=1)
inner_labels = grouped_data.columns.tolist() * len(outer_labels)
inner_sizes = grouped_data.values.flatten()

# Outer and inner colors
outer_colors = [jurusan_colormap[jurusan] for jurusan in outer_labels]
inner_colors = []
for jurusan in outer_labels:
    inner_colors.extend(jurusan_color_map[jurusan])

# Plot Nested Pie Chart
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(aspect="equal"))

# Inner pie
inner_pie = ax.pie(inner_sizes, radius=1, colors=inner_colors,
                   wedgeprops=dict(width=0.3))

# Outer pie
outer_pie, texts = ax.pie(outer_sizes, labels=outer_labels, radius=1.3, colors=outer_colors,
                          wedgeprops=dict(width=0.3), labeldistance=1.4)

# Add guide lines for all outer pie labels
for i, (pie_wedge, label) in enumerate(zip(outer_pie, texts)):
    # Ensure we process even small segments
    angle = (pie_wedge.theta1 + pie_wedge.theta2) / 2  # Calculate the angle of the wedge
    x = 1.3 * math.cos(math.radians(angle))  # X-coordinate for the wedge center
    y = 1.3 * math.sin(math.radians(angle))  # Y-coordinate for the wedge center

    # Get label position or force it to a visible position
    label_pos = label.get_position() if label is not None else (4 * math.cos(math.radians(angle)),
                                                                4 * math.sin(math.radians(angle)))

    # Force annotate to draw lines for even the smallest wedge
    ax.annotate(
        outer_labels[i],  # Label text
        xy=(x, y),  # Point on the pie chart
        xytext=label_pos,  # Label position (calculated or forced)
        arrowprops=dict(arrowstyle="-", color="black", lw=1),
        ha="center", va="center"
    )

# Add legend with two columns
handles = []
for jurusan, colors in jurusan_color_map.items():
    for i, year in enumerate(['2021', '2022', '2023']):
        handles.append(plt.Line2D([0], [0], color=colors[i], lw=6, label=f"{jurusan} - {year}"))

legend = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.4, 1.2), title="Legend",
                   ncol=2, fontsize=10, frameon=True, framealpha=0.8, borderpad=1, edgecolor='black')

# Adjust layout for better spacing
plt.subplots_adjust(left=0.05, right=0.55, top=0.8, bottom=0.2)  # Move chart to the left
plt.title("Nested Pie Chart of Graduates (2021-2023)", pad=40)  # Increased padding for title
plt.show()
