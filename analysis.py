
"""
Auspify Technologies - Data Analysis Using Python Internship
Netflix Dataset Analysis
Tasks 1-4:
1. Data Cleaning and Preparation
2. Content Type Analysis
3. Country-Wise Analysis
4. Release-Year Trend Analysis
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = Path(
    r"C:\SIMON\auspify internship\project\netflix_titles_cleaned.csv"
)

PROJECT_DIR = INPUT_FILE.parent
OUTPUT_DIR = PROJECT_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
DATA_DIR = OUTPUT_DIR / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)



print("=" * 60)
print("NETFLIX DATA ANALYSIS USING PYTHON")
print("=" * 60)

if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}\n"
        "Check the file path and filename."
    )

df = pd.read_csv(INPUT_FILE, encoding="cp1252")

print("\nOriginal dataset shape:", df.shape)
print("\nAvailable columns:")
print(df.columns.tolist())
print("\nFirst five records:")
print(df.head())


# ==================================================
# TASK 1: DATA CLEANING AND PREPARATION
# ==================================================

print("\nTASK 1: DATA CLEANING")

# Standardize column names
df.columns = [
    str(column).strip().lower().replace(" ", "_")
    for column in df.columns
]

required_columns = [
    "type",
    "title",
    "country",
    "release_year",
    "rating",
    "listed_in",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Show missing values before cleaning
print("\nMissing values before cleaning:")
print(df.isna().sum().sort_values(ascending=False))

# Remove unwanted whitespace and normalize empty strings
for column in df.select_dtypes(
    include=["object", "string"]
).columns:
    df[column] = df[column].astype("string").str.strip()

    df[column] = df[column].replace(
        ["", "nan", "None", "NULL"], pd.NA
    )

# Standardize selected columns
df["type"] = df["type"].str.title()
df["rating"] = df["rating"].str.upper()

df["country"] = df["country"].str.replace(
    r"\s*,\s*", ", ", regex=True
)

# Convert release year to numeric integer values
df["release_year"] = pd.to_numeric(
    df["release_year"], errors="coerce"
).astype("Int64")

# Remove exact duplicate records
duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

print("\nExact duplicates removed:", duplicates_removed)
print("Cleaned dataset shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isna().sum().sort_values(ascending=False))

# Save cleaned dataset
cleaned_output = DATA_DIR / "netflix_titles_cleaned.csv"
df.to_csv(cleaned_output, index=False)

print("\nCleaned dataset saved to:", cleaned_output)


# ==================================================
# TASK 2: CONTENT TYPE ANALYSIS
# ==================================================

print("\nTASK 2: CONTENT TYPE ANALYSIS")

type_counts = df["type"].value_counts()
type_percentages = (
    df["type"].value_counts(normalize=True) * 100
).round(2)

type_summary = pd.DataFrame({
    "content_type": type_counts.index,
    "count": type_counts.values,
    "percentage": [
        type_percentages[content_type]
        for content_type in type_counts.index
    ],
})

print("\nContent type summary:")
print(type_summary.to_string(index=False))

type_summary.to_csv(
    OUTPUT_DIR / "type_summary.csv", index=False
)

# Bar chart
plt.figure(figsize=(8, 5))
type_counts.plot(kind="bar")

plt.title("Netflix Titles by Content Type")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    CHARTS_DIR / "content_type_counts.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()
plt.close()

# Pie chart
plt.figure(figsize=(7, 7))
type_counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90,
)

plt.title("Netflix Content Type Distribution")
plt.ylabel("")
plt.tight_layout()

plt.savefig(
    CHARTS_DIR / "content_type_distribution.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()
plt.close()


# ==================================================
# TASK 3: COUNTRY-WISE CONTENT ANALYSIS
# ==================================================

print("\nTASK 3: COUNTRY-WISE ANALYSIS")

# Split countries because one title may list multiple countries
country_data = (
    df["country"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

country_data = country_data[country_data.ne("")]

country_counts = country_data.value_counts()
top_countries = country_counts.head(10)

print("\nTop 10 countries:")
print(top_countries.to_string())

country_summary = (
    top_countries
    .rename_axis("country")
    .reset_index(name="title_count")
)

country_summary.to_csv(
    OUTPUT_DIR / "top_countries.csv",
    index=False,
)

# Horizontal bar chart
plt.figure(figsize=(10, 6))

top_countries.sort_values().plot(kind="barh")

plt.title("Top 10 Countries by Netflix Title Count")
plt.xlabel("Number of Title-Country Associations")
plt.ylabel("Country")
plt.tight_layout()

plt.savefig(
    CHARTS_DIR / "top_countries.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()
plt.close()


# ==================================================
# TASK 4: RELEASE-YEAR TREND ANALYSIS
# ==================================================

print("\nTASK 4: RELEASE-YEAR TREND ANALYSIS")

year_data = df["release_year"].dropna()
year_counts = year_data.value_counts().sort_index()

print("\nTitle counts by release year:")
print(year_counts.to_string())

year_summary = (
    year_counts
    .rename_axis("release_year")
    .reset_index(name="title_count")
)

year_summary.to_csv(
    OUTPUT_DIR / "yearly_release_counts.csv",
    index=False,
)

if not year_counts.empty:
    peak_year = int(year_counts.idxmax())
    peak_count = int(year_counts.max())

    print("\nMost represented release year:", peak_year)
    print("Number of titles:", peak_count)
    print(
        "Release-year range:",
        int(year_counts.index.min()),
        "to",
        int(year_counts.index.max()),
    )

    # Line chart
    plt.figure(figsize=(12, 6))

    plt.plot(
        year_counts.index.astype(int),
        year_counts.values,
        marker="o",
        markersize=3,
    )

    plt.title("Netflix Titles by Release Year")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Titles")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        CHARTS_DIR / "release_year_trend.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()
else:
    print("No valid release-year values were found.")


# ==================================================
# ADDITIONAL ANALYSIS: CONTENT RATINGS
# ==================================================

print("\nADDITIONAL ANALYSIS: CONTENT RATINGS")

rating_counts = df["rating"].value_counts().head(15)

print("\nMost common ratings:")
print(rating_counts.to_string())

rating_summary = (
    rating_counts
    .rename_axis("rating")
    .reset_index(name="count")
)

rating_summary.to_csv(
    OUTPUT_DIR / "rating_summary.csv",
    index=False,
)

plt.figure(figsize=(9, 6))
rating_counts.sort_values().plot(kind="barh")

plt.title("Most Common Netflix Content Ratings")
plt.xlabel("Number of Titles")
plt.ylabel("Rating")
plt.tight_layout()

plt.savefig(
    CHARTS_DIR / "rating_distribution.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()
plt.close()


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nTotal cleaned records:", len(df))
print("Exact duplicates removed:", duplicates_removed)

print("\nContent type summary:")
print(type_summary.to_string(index=False))

print("\nTop 10 countries:")
print(top_countries.to_string())

print("\nAll charts saved in:")
print(CHARTS_DIR)

print("\nAll summary CSV files saved in:")
print(OUTPUT_DIR)

print("\nCleaned dataset saved in:")
print(cleaned_output)

print("\nAll selected tasks have been processed.")
