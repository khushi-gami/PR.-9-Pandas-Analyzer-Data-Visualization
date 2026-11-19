# 📊 Retail Sales Data Analyzer

A Python-based **Retail Sales Data Analyzer** built to perform data exploration, cleaning, manipulation, statistical analysis, aggregation, and visualization on retail sales datasets.

This project uses **Object-Oriented Programming (OOP)** along with NumPy, Pandas, Matplotlib, and Seaborn to analyze retail sales data through an interactive, menu-driven application.

The application allows users to load CSV files, perform different data analysis operations, generate visualizations, and export processed results.

---

## ✨ Key Features

### 📂 1. Load Dataset
- Accept a CSV file path from the user.
- Check whether the file exists.
- Validate the CSV file extension.
- Handle empty files and invalid CSV formats.
- Load the dataset into a Pandas DataFrame.
- Display dataset dimensions and the first five records.

### 🔍 2. Explore Data
Explore the loaded dataset using different operations:

- Display the first `n` rows using `head()`.
- Display the last `n` rows using `tail()`.
- Display random records using `sample()`.
- Check dataset dimensions using `shape`.
- View column names and data types.
- Display dataset information using `info()`.
- Identify duplicate records using `duplicated()`.

### 🧮 3. DataFrame & NumPy Operations
Perform mathematical and numerical operations:

- Convert Sales and Profit columns into numeric data types.
- Create a new Expense column using `Sales - Profit`.
- Calculate total Sales and Profit.
- Perform addition, subtraction, multiplication, and division on selected numeric columns.
- Convert numeric DataFrame columns into NumPy arrays.
- Display NumPy array dimensions using `shape`.

### 🧹 4. Data Cleaning & Missing Value Handling
Clean the dataset and improve data quality:

- Identify missing values using `isna()`.
- Remove rows containing missing values using `dropna()`.
- Fill missing numeric values using Mean and Median.
- Fill missing text values using Mode.
- Replace `"unknown"` values with `NaN`.
- Remove duplicate records using `drop_duplicates()`.
- Convert Date columns into datetime format.
- Convert Sales and Profit columns into numeric format using `to_numeric()`.

### 🔗 5. Combine Datasets
Combine multiple CSV datasets using Pandas:

**Concatenation**
- Add rows from another CSV file using `pd.concat()`.

**Merging**
- Merge datasets using a common column with `pd.merge()`.
- Support four types of joins:

| Join Type | Description |
|---|---|
| Inner Join | Returns only matching records from both datasets. |
| Left Join | Returns all records from the left dataset and matching records from the right. |
| Right Join | Returns all records from the right dataset and matching records from the left. |
| Outer Join | Returns all records from both datasets. |

### ✂️ 6. Split Data
- Split dataset rows into two parts using a user-defined percentage.
- Select specific columns from the dataset.
- Store selected columns in a separate result DataFrame.

### 🔎 7. Search, Sort & Filter
Perform data retrieval and filtering operations:

- Search for values inside selected columns.
- Perform case-insensitive text searches using `str.contains()`.
- Sort records by column values using `sort_values()`.
- Sort records by index using `sort_index()`.
- Filter numeric values using greater than, less than, and equal to conditions.
- Apply multiple conditions using Boolean indexing.
- Filter records using `isin()`.

### 📈 8. Aggregation Functions
Perform numerical aggregation operations:

- Sum
- Mean
- Minimum
- Maximum
- Count

**GroupBy Analysis**

Group records by:
- Product
- Region
- Year

Apply multiple aggregation functions using `groupby()` and `agg()`.

### 📊 9. Statistical Analysis
Perform statistical calculations on numeric columns:

- Descriptive statistics using `describe()`.
- Median using `median()`.
- Variance using `var()`.
- Standard deviation using `std()`.
- Correlation analysis using `corr()`.

### 📋 10. Pivot Table Analysis
Generate summarized reports using Pandas Pivot Tables:

- Product-wise sales by region.
- Region-wise sales by product.
- Combined Product and Region summaries.
- Calculate Sum, Mean, and Count.

### 📉 11. Data Visualization
Generate visual representations of retail sales data using Matplotlib and Seaborn.

| Visualization | Purpose |
|---|---|
| Bar Chart | Analyze total Sales by Product. |
| Bar Chart | Analyze total Profit by Region. |
| Histogram | Understand Sales distribution. |
| Scatter Plot | Analyze the relationship between Sales and Profit. |
| Line Chart | Visualize Sales trends over time. |
| Heatmap | Visualize correlations between numeric columns. |

### 💾 12. Save Dataset & Results
- Export the updated dataset into a CSV file.
- Export the last analysis result into a CSV file.
- Automatically handle Pandas Series results before exporting.
- Display the absolute path of the saved file.

### 🖼️ 13. Save Visualizations
- Save the last generated chart as a PNG image.
- Use Matplotlib's `savefig()` method.
- Save images with a user-provided filename.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming language |
| 🔢 NumPy | Numerical computations and array operations |
| 🐼 Pandas | Data manipulation, cleaning, and analysis |
| 📊 Matplotlib | Creating charts and visualizations |
| 🎨 Seaborn | Statistical visualization and heatmaps |
| 📁 OS Module | File existence checks and file path operations |

---

## 📁 Project Structure

```text
PR. 9 Pandas Analyzer & Data/
│
├── 📄 .gitignore
│
├── 🐍 main_ui.py
│
├── 🐍 main.py
│
├── 📘 README.md
│
└── 📊 sales_data.csv
```

### 📌 File Description

| File | Description |
|---|---|
| `.gitignore` | Specifies files and folders Git should ignore. |
| `main_ui.py` | UI-related Python file in the project structure. |
| `main.py` | Contains the `SalesDataAnalyzer` class, data analysis methods, and main menu in the shared code. |
| `README.md` | Project documentation and usage instructions. |
| `sales_data.csv` | Sample retail sales dataset used for analysis. |

**Note:** In the currently shared code, the main menu and analyzer logic are both inside `main.py`. The separate `main_ui.py` file is present in the folder structure, but its implementation has not been included here.

---

## 📊 Sample Dataset Format

The project uses CSV files containing retail sales information.

Example:

```csv
SalesID,Date,Product,Region,Sales,Profit,Year
1001,2026-01-03,Product A,North,500,120,2026
1002,2026-01-05,Product B,South,700,180,2026
1003,2026-01-08,Product C,East,900,250,2026
1004,2026-01-10,Product A,West,600,150,2026
1005,2026-01-12,Product B,North,800,200,2026
```

### Dataset Columns

| Column | Description |
|---|---|
| `SalesID` | Unique sales transaction ID. |
| `Date` | Date of the sales transaction. |
| `Product` | Name of the product sold. |
| `Region` | Geographical sales region. |
| `Sales` | Total sales amount. |
| `Profit` | Profit earned from the transaction. |
| `Year` | Year of the sales transaction. |

**Note:** Individual operations require specific columns. For example, pivot table analysis requires `Product`, `Region`, and `Sales`.

---

## 🧠 Python Concepts Implemented

This project demonstrates practical implementation of the following concepts:

### Object-Oriented Programming
- Classes and Objects.
- Constructor (`__init__()`).
- Instance Attributes.
- Encapsulation.
- Methods and Object Interaction.

### Python Fundamentals
- Functions and User-Defined Methods.
- Conditional Statements.
- While Loops.
- Exception Handling.
- Input Validation.
- File Handling.
- OS Module.

### NumPy & Pandas
- NumPy Arrays.
- DataFrame and Series Operations.
- Data Cleaning and Transformation.
- Boolean Indexing.
- Sorting and Filtering.
- GroupBy and Aggregation.
- Merging and Concatenation.
- Pivot Tables.
- Statistical Analysis.

### Data Visualization
- Matplotlib Charts.
- Seaborn Heatmaps.
- Saving Visualizations as Images.

---

## 🎯 Project Objective

The main objective of this project is to develop practical skills in Python programming, Object-Oriented Programming, NumPy, Pandas, and data visualization.

It demonstrates how raw retail sales data can be transformed into meaningful information through data cleaning, analysis, statistical calculations, and visual representation.

---
