import contextlib
import io
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


# ======================================================================
# Small helpers
# ======================================================================


def clean_path(path):
    """Remove spaces and quotes around a pasted path."""
    return str(path).strip().strip('"').strip("'")


def float_value(text):
    try:
        return float(str(text).strip())
    except ValueError:
        raise ValueError("Please enter a valid number.")


def int_value(text, minimum, maximum):
    try:
        number = int(str(text).strip())
    except ValueError:
        raise ValueError(
            f"Please enter a whole number between {minimum} and {maximum}."
        )
    if not minimum <= number <= maximum:
        raise ValueError(
            f"Please enter a whole number between {minimum} and {maximum}."
        )
    return number


def capture(function):
    """Run function() and return everything it printed as one string."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        function()
    return buffer.getvalue()


# ======================================================================
# Chart catalogue used by the Visualization page
# ======================================================================

CHARTS = {
    "Bar chart - Sales by Product": "bar_product",
    "Bar chart - Profit by Region": "bar_region",
    "Line chart - Sales trend by Date": "line",
    "Scatter plot - Sales vs Profit": "scatter",
    "Pie chart - Sales share": "pie",
    "Histogram - Sales distribution": "hist",
    "Stack plot - Sales over time": "stack",
    "Seaborn - Correlation heatmap": "heatmap",
    "Seaborn - Box plot of Sales": "box",
    "Seaborn - Bar plot (average Sales)": "sbar",
    "Subplots - 2x2 dashboard": "dashboard",
}
SEABORN_CHARTS = {"heatmap", "box", "sbar"}
SNS_STYLES = ["darkgrid", "whitegrid", "dark", "white", "ticks"]
SNS_PALETTES = ["deep", "muted", "pastel", "bright", "dark", "colorblind"]


# ======================================================================
# Data layer: SalesDataAnalyzer
# (methods print their output; the GUI shows that text in the output box)
# ======================================================================


class SalesDataAnalyzer:
    """Encapsulates all sales data loading, cleaning, analysis and plotting."""

    NUMERIC_COLUMNS = ["Sales", "Profit"]

    # ---- constructor / destructor / special methods -----------------

    def __init__(self, file_path=None):
        self.data = None
        self.result = None
        self.file_path = None
        self.last_figure = None
        self.parts = {}
        self.sns_style = "darkgrid"
        self.sns_palette = "deep"
        if file_path:
            self.load_data(file_path)

    def __del__(self):
        """Destructor: release the data and the figure held by this object."""
        try:
            self.data = None
            self.result = None
            self.last_figure = None
            self.parts = {}
        except Exception:
            pass

    def __len__(self):
        return 0 if self.data is None else len(self.data)

    def __repr__(self):
        if self.data is None:
            return "SalesDataAnalyzer(no data loaded)"
        return f"SalesDataAnalyzer(rows={self.data.shape[0]}, columns={self.data.shape[1]})"

    def __add__(self, other):
        """Operator overloading: analyzer1 + analyzer2 -> new analyzer with stacked rows."""
        if not isinstance(other, SalesDataAnalyzer):
            return NotImplemented
        if self.data is None or other.data is None:
            raise ValueError("Both analyzers must have data loaded.")
        combined = type(self)()
        combined.data = pd.concat([self.data, other.data], ignore_index=True)
        return combined

    # ---- helpers -----------------------------------------------------

    @property
    def loaded(self):
        return self.data is not None

    @staticmethod
    def _has_columns(data, *columns):
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise ValueError(f"Required column(s) not found: {', '.join(missing)}")

    def _numeric(self, column):
        return pd.to_numeric(self.data[column], errors="coerce")

    def _prepare_plot_data(self):
        """Copy of the data with Sales/Profit numeric and Date as datetime."""
        data = self.data.copy()
        for column in self.NUMERIC_COLUMNS:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        return data

    @staticmethod
    def _read_csv(path):
        path = clean_path(path)
        if not path:
            raise ValueError("Please choose the second CSV file first.")
        try:
            other = pd.read_csv(path)
        except FileNotFoundError:
            raise ValueError("Second CSV file not found.")
        except pd.errors.EmptyDataError:
            raise ValueError("The second CSV file is empty.")
        if other.empty:
            raise ValueError("The second CSV file is empty.")
        return other

    # ---- 1. load -----------------------------------------------------

    def load_data(self, file_path):
        file_path = clean_path(file_path)

        if not os.path.exists(file_path):
            raise ValueError("File not found!")
        if not file_path.lower().endswith(".csv"):
            raise ValueError("Please select a CSV file!")

        try:
            data = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            raise ValueError("File is empty!")
        except pd.errors.ParserError:
            raise ValueError("Invalid CSV file format!")
        except PermissionError:
            raise ValueError("Permission denied!")
        except UnicodeDecodeError:
            raise ValueError(
                "Could not read the file. Please use a UTF-8 encoded CSV file."
            )
        except OSError as error:
            raise ValueError(f"Operating system error: {error}")

        if data.empty:
            raise ValueError("File is empty!")

        self.file_path = file_path
        self.data = data
        self.result = None
        self.parts = {}

    # ---- 2. explore --------------------------------------------------

    def explore(self, kind, n=5):
        data = self.data
        n = max(1, min(int(n), len(data)))

        if kind == "head":
            print(data.head(n))
        elif kind == "tail":
            print(data.tail(n))
        elif kind == "sample":
            print(data.sample(n))
        elif kind == "shape":
            print(f"Rows   : {data.shape[0]}")
            print(f"Columns: {data.shape[1]}")
        elif kind == "columns":
            print("Column names:")
            print(list(data.columns))
        elif kind == "dtypes":
            print("Data types:")
            print(data.dtypes)
        elif kind == "info":
            data.info()
        elif kind == "duplicates":
            duplicate_count = int(data.duplicated().sum())
            print(f"Duplicate rows: {duplicate_count}")
            if duplicate_count > 0:
                print()
                print(data[data.duplicated(keep=False)])

    # ---- 3. DataFrame / NumPy operations ------------------------------

    def convert_numeric(self):
        found = []
        for column in self.NUMERIC_COLUMNS:
            if column in self.data.columns:
                self.data[column] = pd.to_numeric(self.data[column], errors="coerce")
                found.append(column)
        if not found:
            raise ValueError("Sales and Profit columns were not found.")
        print("Sales and Profit converted to numeric.\n")
        print(self.data[found].dtypes)

    def add_expense(self):
        if not {"Sales", "Profit"}.issubset(self.data.columns):
            raise ValueError("Sales and Profit columns are required.")
        self.data["Expense"] = self._numeric("Sales") - self._numeric("Profit")
        print("Expense column added (Sales - Profit).\n")
        print(self.data[["Sales", "Profit", "Expense"]].head(10))

    def totals(self):
        shown = False
        for column in self.NUMERIC_COLUMNS:
            if column in self.data.columns:
                print(f"Total {column}: {self._numeric(column).sum():,.2f}")
                shown = True
        if not shown:
            raise ValueError("Sales and Profit columns were not found.")

    def apply_operation(self, column, operation, number):
        values = self._numeric(column)
        if values.notna().sum() == 0:
            raise ValueError("This column does not contain numeric values.")
        if operation == "divide" and number == 0:
            raise ValueError("Cannot divide by zero.")

        if operation == "add":
            self.result = values + number
        elif operation == "subtract":
            self.result = values - number
        elif operation == "multiply":
            self.result = values * number
        else:
            self.result = values / number

        self.result.name = f"{column}_{operation}_{number:g}"
        print(f"Result of {operation} on '{column}' with {number:g} (first 10 rows):\n")
        print(self.result.head(10))

    def elementwise(self, first, second, operation):
        a = self._numeric(first).to_numpy(dtype=float)
        b = self._numeric(second).to_numpy(dtype=float)
        if np.isnan(a).all() or np.isnan(b).all():
            raise ValueError("Both columns must contain numeric values.")

        with np.errstate(divide="ignore", invalid="ignore"):
            if operation == "add":
                values = np.add(a, b)
            elif operation == "subtract":
                values = np.subtract(a, b)
            elif operation == "multiply":
                values = np.multiply(a, b)
            else:
                values = np.divide(a, b)

        self.result = pd.Series(
            values, index=self.data.index, name=f"{first}_{operation}_{second}"
        )
        print(
            f"NumPy element-wise {operation}: '{first}' and '{second}' (first 10 rows):\n"
        )
        print(self.result.head(10))

    def show_numpy_array(self):
        numeric_data = self.data.select_dtypes(include="number")
        if numeric_data.empty:
            raise ValueError("No numeric columns found.")
        array = numeric_data.to_numpy()
        print(f"Columns: {list(numeric_data.columns)}\n")
        print("NumPy array:")
        print(array)
        print("\nArray shape:", array.shape)

    def numpy_indexing_demo(self):
        numeric_data = self.data.select_dtypes(include="number")
        if numeric_data.empty:
            raise ValueError("No numeric columns found. Convert Sales/Profit first.")

        columns = list(numeric_data.columns)
        array = numeric_data.to_numpy(dtype=float)
        if array.shape[0] == 0:
            raise ValueError("The dataset has no rows.")

        print("Columns in the array:", columns)
        print("Array shape:", array.shape)

        print("\n--- Indexing ---")
        print("array[0]       (first row)  :", array[0])
        print("array[-1]      (last row)   :", array[-1])
        print("array[0, 0]    (first value):", array[0, 0])

        print("\n--- Slicing ---")
        print(f"array[:5, 0]   (first 5 values of '{columns[0]}'):")
        print(array[:5, 0])
        print("array[2:5, :2] (rows 2-4, first two columns):")
        print(array[2:5, :2])
        print("array[::2][:3] (every 2nd row, first 3):")
        print(array[::2][:3])

        print("\n--- Boolean indexing ---")
        first_column = array[:, 0]
        mean_value = np.nanmean(first_column)
        above_mean = first_column[first_column > mean_value]
        print(
            f"Values in '{columns[0]}' above its mean ({mean_value:.2f}): {above_mean.size}"
        )

        print("\n--- Element-wise operations ---")
        print("array[:5, 0] * 2         :", array[:5, 0] * 2)
        print("np.round(array[:5,0] / 3):", np.round(array[:5, 0] / 3, 2))

        print("\n--- Column-wise statistics ---")
        print("Mean    :", np.round(np.nanmean(array, axis=0), 2))
        print("Std dev :", np.round(np.nanstd(array, axis=0), 2))
        print("Percentiles (25, 50, 75):")
        print(np.round(np.nanpercentile(array, [25, 50, 75], axis=0), 2))

    # ---- 4. clean data -----------------------------------------------

    def check_missing(self):
        print("Missing values in each column:")
        print(self.data.isna().sum())
        print("\nTotal missing values:", int(self.data.isna().sum().sum()))
        rows_with_missing = self.data[self.data.isna().any(axis=1)]
        if not rows_with_missing.empty:
            print(f"\nRows containing missing values ({len(rows_with_missing)}):")
            print(rows_with_missing)

    def drop_missing(self):
        before = len(self.data)
        self.data = self.data.dropna().copy()
        print(f"Removed {before - len(self.data)} row(s) containing missing values.")
        print(f"Rows remaining: {len(self.data)}")

    def fill_numeric(self, method):
        numeric_columns = self.data.select_dtypes(include="number").columns
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found. Convert Sales/Profit first.")

        before = int(self.data[numeric_columns].isna().sum().sum())
        for column in numeric_columns:
            fill_value = (
                self.data[column].mean()
                if method == "mean"
                else self.data[column].median()
            )
            self.data[column] = self.data[column].fillna(fill_value)
        print(f"Missing numeric values filled with the column {method}.")
        print(f"Values filled: {before}")

    def fill_text_mode(self):
        text_columns = self.data.select_dtypes(exclude=["number", "datetime"]).columns
        before = int(self.data[text_columns].isna().sum().sum())
        for column in text_columns:
            mode_values = self.data[column].mode()
            if not mode_values.empty:
                self.data[column] = self.data[column].fillna(mode_values[0])
        after = int(self.data[text_columns].isna().sum().sum())
        print("Missing text values filled with each column's mode.")
        print(f"Values filled: {before - after}")

    def replace_unknown(self):
        text_columns = self.data.select_dtypes(exclude=["number", "datetime"]).columns
        before = int(self.data[text_columns].isna().sum().sum())
        self.data[text_columns] = self.data[text_columns].replace(
            r"(?i)^\s*unknown\s*$", np.nan, regex=True
        )
        after = int(self.data[text_columns].isna().sum().sum())
        print(f"'unknown' values replaced with NaN: {after - before}")

    def drop_duplicates(self):
        before = len(self.data)
        self.data = self.data.drop_duplicates().copy()
        print(f"Removed {before - len(self.data)} duplicate row(s).")
        print(f"Rows remaining: {len(self.data)}")

    def convert_date(self):
        if "Date" not in self.data.columns:
            raise ValueError("Date column not found.")
        self.data["Date"] = pd.to_datetime(self.data["Date"], errors="coerce")
        print("Date column converted to datetime.")
        print(f"Invalid dates (now NaT): {int(self.data['Date'].isna().sum())}")

    # ---- 5. combine data ---------------------------------------------

    def combine_concat(self, path):
        other = self._read_csv(path)
        self.data = pd.concat([self.data, other], ignore_index=True)
        print("Files concatenated successfully (rows added).")
        print("Updated dataset shape:", self.data.shape)

    def combine_merge(self, path, common_column, how):
        other = self._read_csv(path)
        print("Columns in current dataset :", list(self.data.columns))
        print("Columns in second dataset  :", list(other.columns))
        common_column = common_column.strip()
        if common_column not in self.data.columns:
            raise ValueError("Column not found in current dataset.")
        if common_column not in other.columns:
            raise ValueError("Column not found in second dataset.")
        self.data = pd.merge(
            self.data, other, on=common_column, how=how, suffixes=("_left", "_right")
        )
        print(f"\nFiles merged successfully ({how} join).")
        print("Updated dataset shape:", self.data.shape)

    def combine_plus(self, path):
        other = type(self)()
        other.data = self._read_csv(path)
        combined = self + other  # uses __add__ (operator overloading)
        self.data = combined.data
        print("Analyzers combined using the + operator (operator overloading).")
        print("Updated dataset shape:", self.data.shape)

    # ---- 6. split data -----------------------------------------------

    def split_rows(self, percent):
        if len(self.data) < 2:
            raise ValueError("At least 2 rows are required.")
        first_size = int(len(self.data) * percent / 100)
        first_part = self.data.head(first_size)
        second_part = self.data.tail(len(self.data) - first_size)

        self.result = first_part.copy()
        print(f"First part ({len(first_part)} rows):")
        print(first_part)
        print(f"\nSecond part ({len(second_part)} rows):")
        print(second_part)
        print("\n(The first part is kept as the current result for saving.)")

    def select_columns(self, columns):
        if not columns:
            raise ValueError("Select at least one column.")
        self.result = self.data[columns].copy()
        print(f"Selected columns: {columns}\n")
        print(self.result.head(15))

    def split_by(self, column):
        self.parts = {key: group.copy() for key, group in self.data.groupby(column)}
        if not self.parts:
            raise ValueError("No groups found in this column.")
        print(f"DataFrame split into {len(self.parts)} parts by '{column}':\n")
        for key, part in self.parts.items():
            print(f"  {key}: {len(part)} rows")
        print("\nChoose a value in the box above and click 'Show part'.")

    def show_part(self, key):
        for name, part in self.parts.items():
            if str(name) == str(key):
                self.result = part.copy()
                print(f"Rows for the selected value '{name}' ({len(part)} rows):\n")
                print(part)
                return
        raise ValueError("Please split the data first, then choose a value.")

    # ---- 7. search / sort / filter ------------------------------------

    def search(self, column, value):
        if not value.strip():
            raise ValueError("Enter a value to search.")
        matches = self.data[
            self.data[column]
            .astype(str)
            .str.contains(value.strip(), case=False, na=False, regex=False)
        ]
        self.result = matches
        print(f"Rows matching '{value.strip()}' in '{column}': {len(matches)}\n")
        print(matches)

    def sort_by_column(self, column, ascending):
        self.result = self.data.sort_values(by=column, ascending=ascending)
        print(f"Sorted by '{column}' ({'ascending' if ascending else 'descending'}):\n")
        print(self.result)

    def sort_by_index(self, ascending):
        self.result = self.data.sort_index(ascending=ascending)
        print(f"Sorted by index ({'ascending' if ascending else 'descending'}):\n")
        print(self.result)

    def filter_numeric(self, column, operator, limit):
        values = self._numeric(column)
        if values.notna().sum() == 0:
            raise ValueError("Selected column has no numeric values.")
        if operator == ">":
            self.result = self.data[values > limit]
        elif operator == "<":
            self.result = self.data[values < limit]
        else:
            self.result = self.data[values == limit]
        print(f"Rows where {column} {operator} {limit:g}: {len(self.result)}\n")
        print(self.result)

    def filter_multi(self, min_sales, min_profit):
        self._has_columns(self.data, "Sales", "Profit")
        sales = self._numeric("Sales")
        profit = self._numeric("Profit")
        self.result = self.data[(sales >= min_sales) & (profit >= min_profit)]
        print(
            f"Rows with Sales >= {min_sales:g} AND Profit >= {min_profit:g}: "
            f"{len(self.result)}\n"
        )
        print(self.result)

    def filter_isin(self, column, values_text):
        selected = [item.strip() for item in values_text.split(",") if item.strip()]
        if not selected:
            raise ValueError("Enter at least one value (separate with commas).")
        self.result = self.data[self.data[column].astype(str).isin(selected)]
        print(f"Rows where {column} is in {selected}: {len(self.result)}\n")
        print(self.result)

    def filter_dates(self, start_text, end_text):
        self._has_columns(self.data, "Date")
        try:
            start = pd.to_datetime(start_text.strip())
            end = pd.to_datetime(end_text.strip())
        except (ValueError, TypeError):
            raise ValueError("Invalid date. Please use the format YYYY-MM-DD.")
        if start > end:
            raise ValueError("Start date must not be after the end date.")
        dates = pd.to_datetime(self.data["Date"], errors="coerce")
        self.result = self.data[(dates >= start) & (dates <= end)]
        print(f"Rows between {start.date()} and {end.date()}: {len(self.result)}\n")
        print(self.result)

    # ---- 8. aggregation ------------------------------------------------

    def aggregate_basic(self, column, function):
        values = self._numeric(column)
        if values.notna().sum() == 0:
            raise ValueError("Selected column does not contain numeric data.")
        labels = {
            "sum": ("Sum", values.sum()),
            "mean": ("Mean", values.mean()),
            "min": ("Minimum", values.min()),
            "max": ("Maximum", values.max()),
            "count": ("Count of non-missing values", values.count()),
        }
        label, value = labels[function]
        print(f"{label} of '{column}': {value}")

    def group_by(self, group_column):
        if group_column in self.data.columns:
            group_key = self.data[group_column]
        elif group_column == "Year" and "Date" in self.data.columns:
            group_key = pd.to_datetime(
                self.data["Date"], errors="coerce"
            ).dt.year.rename("Year")
        else:
            raise ValueError(f"{group_column} column not found.")

        for column in self.NUMERIC_COLUMNS:
            if column in self.data.columns:
                self.data[column] = pd.to_numeric(self.data[column], errors="coerce")

        # ID and Year columns make no sense to sum or average, so they are skipped.
        skipped = {group_column, "SalesID", "Year"}
        numeric_columns = [
            column
            for column in self.data.select_dtypes(include="number").columns
            if column not in skipped
        ]
        if not numeric_columns:
            raise ValueError("No numeric columns to aggregate.")

        grouped = (
            self.data[numeric_columns].groupby(group_key).agg(["sum", "mean", "count"])
        )
        self.result = grouped
        print(f"Aggregation grouped by {group_column}:\n")
        print(grouped)

    # ---- 9. statistics ---------------------------------------------------

    def statistics(self, percentiles_text="25, 50, 75, 90"):
        try:
            percentiles = [float(p) for p in percentiles_text.replace(",", " ").split()]
        except ValueError:
            raise ValueError("Percentiles must be numbers, e.g. 25, 50, 75, 90.")
        if not percentiles or any(p < 0 or p > 100 for p in percentiles):
            raise ValueError("Percentiles must be between 0 and 100.")

        numeric_data = self.data.select_dtypes(include="number")
        if numeric_data.empty:
            raise ValueError("No numeric columns found. Convert Sales/Profit first.")

        description = numeric_data.describe()
        self.result = description

        print("Descriptive statistics (describe):")
        print(description)
        print("\nMedian:")
        print(numeric_data.median())
        print("\nVariance (var):")
        print(numeric_data.var())
        print("\nStandard deviation (std):")
        print(numeric_data.std())
        print(f"\nPercentiles (quantile {', '.join(f'{p:g}%' for p in percentiles)}):")
        print(numeric_data.quantile([p / 100 for p in percentiles]))
        print("\nCorrelation:")
        if numeric_data.shape[1] >= 2:
            print(numeric_data.corr())
        else:
            print("At least two numeric columns are needed for correlation.")

    # ---- 10. pivot table -------------------------------------------------

    def pivot(self, kind):
        self._has_columns(self.data, "Product", "Region", "Sales")
        data_copy = self.data.copy()
        data_copy["Sales"] = pd.to_numeric(data_copy["Sales"], errors="coerce")

        if kind == 1:
            pivot = pd.pivot_table(
                data_copy,
                values="Sales",
                index="Product",
                columns="Region",
                aggfunc="sum",
                fill_value=0,
            )
        elif kind == 2:
            pivot = pd.pivot_table(
                data_copy,
                values="Sales",
                index="Region",
                columns="Product",
                aggfunc="sum",
                fill_value=0,
            )
        else:
            pivot = pd.pivot_table(
                data_copy,
                values="Sales",
                index=["Product", "Region"],
                aggfunc=["sum", "mean", "count"],
                fill_value=0,
            )

        self.result = pivot
        print("Pivot table:\n")
        print(pivot)

    # ---- advanced operations (groupby + transform, re-index, labels) -----

    def groupby_transform(self, group_column, value_column):
        values = self._numeric(value_column)
        if values.notna().sum() == 0:
            raise ValueError("The selected value column is not numeric.")

        grouped = values.groupby(self.data[group_column])
        total_name = f"{value_column}_{group_column}_Total"
        share_name = f"{value_column}_Share_%"
        diff_name = f"{value_column}_Diff_From_{group_column}_Mean"

        group_total = grouped.transform("sum")
        self.data[total_name] = group_total
        self.data[share_name] = (values / group_total * 100).round(2)
        self.data[diff_name] = (values - grouped.transform("mean")).round(2)

        print("New columns added using groupby() + transform():\n")
        print(
            self.data[
                [group_column, value_column, total_name, share_name, diff_name]
            ].head(15)
        )

    def reset_index(self, start):
        self.data = self.data.reset_index(drop=True)
        if start == 1:
            self.data.index = range(1, len(self.data) + 1)
        print(f"Index reset (starting at {start}).\n")
        print(self.data.head(10))

    def sort_and_reindex(self, column, ascending):
        self.data = self.data.sort_values(by=column, ascending=ascending).reset_index(
            drop=True
        )
        print(f"Data sorted by '{column}' and re-indexed.\n")
        print(self.data.head(10))

    def rename_column(self, old_name, new_name):
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("Enter the new column name.")
        if new_name in self.data.columns:
            raise ValueError("A column with that name already exists.")
        self.data = self.data.rename(columns={old_name: new_name})
        print(f"Column '{old_name}' renamed to '{new_name}'.")

    # ---- summary report ----------------------------------------------------

    def summary_lines(self):
        data = self._prepare_plot_data()
        lines = [
            "SALES ANALYSIS SUMMARY",
            "=" * 44,
            f"Records analysed   : {len(data)}",
        ]

        if "Date" in data.columns and data["Date"].notna().any():
            lines.append(
                f"Date range         : {data['Date'].min().date()} to "
                f"{data['Date'].max().date()}"
            )
        if "Sales" in data.columns:
            lines.append(f"Total Sales        : {data['Sales'].sum():,.2f}")
            lines.append(f"Average Sales      : {data['Sales'].mean():,.2f}")
            lines.append(f"Highest Sale       : {data['Sales'].max():,.2f}")
        if "Profit" in data.columns:
            lines.append(f"Total Profit       : {data['Profit'].sum():,.2f}")
        if {"Sales", "Profit"}.issubset(data.columns) and data["Sales"].sum() != 0:
            margin = data["Profit"].sum() / data["Sales"].sum() * 100
            lines.append(f"Profit margin      : {margin:.2f}%")
        if {"Product", "Sales"}.issubset(data.columns):
            by_product = data.groupby("Product")["Sales"].sum()
            if not by_product.empty:
                lines.append(
                    f"Top product        : {by_product.idxmax()} ({by_product.max():,.2f})"
                )
                lines.append(
                    f"Weakest product    : {by_product.idxmin()} ({by_product.min():,.2f})"
                )
        if {"Region", "Profit"}.issubset(data.columns):
            by_region = data.groupby("Region")["Profit"].sum()
            if not by_region.empty:
                lines.append(
                    f"Top region (profit): {by_region.idxmax()} ({by_region.max():,.2f})"
                )

        lines.append(f"Missing values     : {int(self.data.isna().sum().sum())}")
        lines.append(f"Duplicate rows     : {int(self.data.duplicated().sum())}")
        return lines

    # ---- 11. visualisation ---------------------------------------------------

    def _default_category(self, data):
        for column in ("Product", "Region"):
            if column in data.columns:
                return column
        raise ValueError("A Product or Region column is required for this chart.")

    def _plot_bar(self, ax, data, category, value, color):
        self._has_columns(data, category, value)
        totals = data.groupby(category)[value].sum()
        if totals.empty:
            raise ValueError("No data available for this chart.")
        ax.bar(
            totals.index.astype(str), totals.values, color=color, label=f"Total {value}"
        )
        ax.set_title(f"Total {value} by {category}")
        ax.set_xlabel(category)
        ax.set_ylabel(value)
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

    def _plot_line(self, ax, data):
        self._has_columns(data, "Date", "Sales")
        trend = (
            data.dropna(subset=["Date", "Sales"])
            .groupby("Date")["Sales"]
            .sum()
            .sort_index()
        )
        if trend.empty:
            raise ValueError("No valid Date/Sales data available.")

        label = "Daily Sales"
        if len(trend) > 60:
            monthly = trend.groupby(trend.index.to_period("M")).sum()
            monthly.index = monthly.index.to_timestamp()
            trend, label = monthly, "Monthly Sales"

        ax.plot(
            trend.index,
            trend.values,
            marker="o",
            markersize=4,
            color="tab:green",
            label=label,
        )
        ax.set_title("Sales Trend Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

    def _plot_scatter(self, ax, data):
        self._has_columns(data, "Sales", "Profit")
        points = data.dropna(subset=["Sales", "Profit"])
        if points.empty:
            raise ValueError("No valid Sales/Profit data available.")

        if "Region" in points.columns:
            for region, group in points.groupby(points["Region"].fillna("Unknown")):
                ax.scatter(
                    group["Sales"], group["Profit"], label=str(region), alpha=0.75
                )
            ax.legend(title="Region", fontsize="small")
        else:
            ax.scatter(points["Sales"], points["Profit"], label="Orders", alpha=0.75)
            ax.legend()
        ax.set_title("Sales vs Profit")
        ax.set_xlabel("Sales")
        ax.set_ylabel("Profit")

    def _plot_pie(self, ax, data, category):
        self._has_columns(data, category, "Sales")
        totals = data.groupby(category)["Sales"].sum()
        totals = totals[totals > 0]
        if totals.empty:
            raise ValueError("No positive Sales values available for a pie chart.")
        ax.pie(totals.values, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"Sales Share by {category}")
        ax.legend(
            totals.index.astype(str),
            title=category,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )

    def _plot_hist(self, ax, data):
        self._has_columns(data, "Sales")
        values = data["Sales"].dropna()
        if values.empty:
            raise ValueError("No valid Sales values available.")
        ax.hist(values, bins=10, color="tab:orange", edgecolor="black", label="Sales")
        ax.set_title("Sales Distribution")
        ax.set_xlabel("Sales")
        ax.set_ylabel("Frequency")
        ax.legend()

    def _plot_stack(self, ax, data, category):
        self._has_columns(data, "Sales", category)
        time_column = next((c for c in ("Date", "Year") if c in data.columns), None)
        if time_column is None:
            raise ValueError("A stack plot needs a Date or Year column.")

        work = data.dropna(subset=[time_column, "Sales"]).copy()
        if time_column == "Date":
            work["Period"] = work["Date"].dt.to_period("M").dt.to_timestamp()
            x_column = "Period"
        else:
            x_column = time_column

        pivot = work.pivot_table(
            index=x_column,
            columns=category,
            values="Sales",
            aggfunc="sum",
            fill_value=0,
        ).sort_index()
        if len(pivot) < 2:
            raise ValueError("At least two time periods are needed for a stack plot.")

        ax.stackplot(
            pivot.index, pivot.T.values, labels=[str(c) for c in pivot.columns]
        )
        ax.set_title(f"Sales Over Time Stacked by {category}")
        ax.set_xlabel("Month" if time_column == "Date" else time_column)
        ax.set_ylabel("Sales")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title=category, loc="upper left", fontsize="small")

    def _plot_heatmap(self, ax, data):
        numeric_data = data.select_dtypes(include="number").drop(
            columns=["SalesID"], errors="ignore"
        )
        numeric_data = numeric_data.loc[:, numeric_data.nunique() > 1]
        if numeric_data.shape[1] < 2:
            raise ValueError(
                "At least two numeric columns with varying values are needed."
            )
        sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")

    def _plot_box(self, ax, data, category):
        self._has_columns(data, category, "Sales")
        work = data.dropna(subset=[category, "Sales"])
        if work.empty:
            raise ValueError("No valid data available for a box plot.")
        try:
            sns.boxplot(
                data=work,
                x=category,
                y="Sales",
                hue=category,
                palette=self.sns_palette,
                legend=False,
                ax=ax,
            )
        except (TypeError, AttributeError):
            sns.boxplot(
                data=work, x=category, y="Sales", palette=self.sns_palette, ax=ax
            )
        ax.set_title(f"Sales Spread by {category} (Box Plot)")
        ax.set_xlabel(category)
        ax.set_ylabel("Sales")
        ax.tick_params(axis="x", rotation=45)

    def _plot_seaborn_bar(self, ax, data, category):
        self._has_columns(data, category, "Sales")
        means = data.groupby(category)["Sales"].mean().reset_index()
        if means.empty:
            raise ValueError("No data available for this chart.")
        try:
            sns.barplot(
                data=means,
                x=category,
                y="Sales",
                hue=category,
                palette=self.sns_palette,
                legend=False,
                ax=ax,
            )
        except (TypeError, AttributeError):
            sns.barplot(
                data=means, x=category, y="Sales", palette=self.sns_palette, ax=ax
            )
        ax.set_title(f"Average Sales by {category} (Seaborn)")
        ax.set_xlabel(category)
        ax.set_ylabel("Average Sales")
        ax.tick_params(axis="x", rotation=45)

    def build_figure(self, chart, category=None):
        """Create a matplotlib Figure (not tied to pyplot) for the chosen chart."""
        data = self._prepare_plot_data()
        style = (
            sns.axes_style(self.sns_style)
            if chart in SEABORN_CHARTS
            else plt.style.context("default")
        )

        with style:
            if chart == "dashboard":
                figure = Figure(figsize=(12, 8))
                axes = figure.subplots(2, 2).flatten()
                jobs = [
                    lambda ax: self._plot_bar(
                        ax, data, "Product", "Sales", "steelblue"
                    ),
                    lambda ax: self._plot_bar(ax, data, "Region", "Profit", "seagreen"),
                    lambda ax: self._plot_hist(ax, data),
                    lambda ax: self._plot_scatter(ax, data),
                ]
                drawn = 0
                for ax, job in zip(axes, jobs):
                    try:
                        job(ax)
                        drawn += 1
                    except ValueError:
                        ax.axis("off")
                        ax.text(
                            0.5, 0.5, "Data not available", ha="center", va="center"
                        )
                if drawn == 0:
                    raise ValueError("None of the dashboard charts could be drawn.")
                figure.suptitle("Sales Dashboard", fontsize=16)
            else:
                figure = Figure(figsize=(9, 5.5))
                ax = figure.add_subplot(111)

                if chart == "bar_product":
                    self._plot_bar(ax, data, "Product", "Sales", "steelblue")
                elif chart == "bar_region":
                    self._plot_bar(ax, data, "Region", "Profit", "seagreen")
                elif chart == "line":
                    self._plot_line(ax, data)
                elif chart == "scatter":
                    self._plot_scatter(ax, data)
                elif chart == "hist":
                    self._plot_hist(ax, data)
                elif chart == "heatmap":
                    self._plot_heatmap(ax, data)
                else:
                    category = category or self._default_category(data)
                    if chart == "pie":
                        self._plot_pie(ax, data, category)
                    elif chart == "stack":
                        self._plot_stack(ax, data, category)
                    elif chart == "box":
                        self._plot_box(ax, data, category)
                    elif chart == "sbar":
                        self._plot_seaborn_bar(ax, data, category)

        try:
            figure.set_layout_engine("tight")
        except (AttributeError, ValueError):
            figure.tight_layout()

        self.last_figure = figure
        return figure


# ======================================================================
# Graphical user interface
# ======================================================================

PAGE_NAMES = [
    "Dataset Preview",
    "Explore Data",
    "DataFrame & NumPy",
    "Clean Data",
    "Combine Data",
    "Split Data",
    "Search / Sort / Filter",
    "Aggregation",
    "Statistics",
    "Pivot Table",
    "Advanced Operations",
    "Visualization",
    "Save & Report",
]


class SalesAnalyzerApp(tk.Tk):
    BG = "#f4f6f9"
    SIDEBAR = "#1e293b"
    SIDEBAR_HOVER = "#334155"
    ACCENT = "#2563eb"
    ACCENT_DARK = "#1d4ed8"
    TEXT = "#1f2933"

    def __init__(self):
        super().__init__()
        self.az = SalesDataAnalyzer()
        self.column_boxes = []  # (combobox, preferred default)
        self.column_lists = []  # multi-select listboxes
        self.category_boxes = []  # comboboxes with Product / Region
        self.part_box = None
        self.pages = {}
        self.nav_buttons = {}
        self.last_summary = ""

        self.title("Sales Data Analyzer - Pandas, NumPy, Matplotlib & Seaborn")
        self.geometry("1320x800")
        self.minsize(1080, 660)
        self.configure(bg=self.BG)

        self._configure_styles()
        self._build_layout()
        self.show_page("Dataset Preview")
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

    # ------------------------------------------------------------------
    # Look and feel
    # ------------------------------------------------------------------

    def _configure_styles(self):
        base = tkfont.nametofont("TkDefaultFont")
        base.configure(size=10)
        self.mono = tkfont.nametofont("TkFixedFont").copy()
        self.mono.configure(size=10)
        self.bold = base.copy()
        self.bold.configure(weight="bold")
        self.title_font = base.copy()
        self.title_font.configure(size=15, weight="bold")

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=self.BG, foreground=self.TEXT, font=base)
        style.configure("TFrame", background=self.BG)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT)
        style.configure("TLabelframe", background=self.BG, bordercolor="#cbd5e1")
        style.configure(
            "TLabelframe.Label",
            background=self.BG,
            foreground=self.SIDEBAR,
            font=self.bold,
        )
        style.configure("TRadiobutton", background=self.BG)
        style.configure("TButton", padding=(10, 5))
        style.configure(
            "Accent.TButton",
            background=self.ACCENT,
            foreground="white",
            padding=(12, 6),
            font=self.bold,
        )
        style.map(
            "Accent.TButton",
            background=[("active", self.ACCENT_DARK), ("pressed", self.ACCENT_DARK)],
        )

        style.configure("Top.TFrame", background="white")
        style.configure("Top.TLabel", background="white", foreground=self.SIDEBAR)
        style.configure(
            "Title.TLabel",
            background="white",
            foreground=self.ACCENT,
            font=self.title_font,
        )
        style.configure("Status.TFrame", background="#e2e8f0")
        style.configure("Status.TLabel", background="#e2e8f0", foreground=self.TEXT)
        style.configure(
            "Page.TLabel",
            background=self.BG,
            foreground=self.SIDEBAR,
            font=self.title_font,
        )

        style.configure(
            "Nav.TButton",
            background=self.SIDEBAR,
            foreground="#e2e8f0",
            anchor="w",
            padding=(18, 9),
            borderwidth=0,
            relief="flat",
        )
        style.map("Nav.TButton", background=[("active", self.SIDEBAR_HOVER)])
        style.configure(
            "NavSel.TButton",
            background=self.ACCENT,
            foreground="white",
            anchor="w",
            padding=(18, 9),
            borderwidth=0,
            relief="flat",
            font=self.bold,
        )
        style.map("NavSel.TButton", background=[("active", self.ACCENT)])

        style.configure(
            "Treeview", rowheight=24, background="white", fieldbackground="white"
        )
        style.configure("Treeview.Heading", font=self.bold, background="#e2e8f0")

    # ------------------------------------------------------------------
    # Layout: top bar, sidebar, pages, status bar
    # ------------------------------------------------------------------

    def _build_layout(self):
        # Top bar: title + CSV chooser
        top = ttk.Frame(self, style="Top.TFrame", padding=(14, 10))
        top.pack(side="top", fill="x")
        ttk.Label(top, text="Sales Data Analyzer", style="Title.TLabel").pack(
            side="left"
        )

        ttk.Button(
            top, text="Load", style="Accent.TButton", command=self.load_file
        ).pack(side="right", padx=(6, 0))
        ttk.Button(top, text="Browse...", command=self.browse_file).pack(
            side="right", padx=(6, 0)
        )
        self.file_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.file_var, width=48).pack(
            side="right", padx=(6, 0)
        )
        ttk.Label(top, text="CSV file:", style="Top.TLabel").pack(side="right")

        # Status bar
        status = ttk.Frame(self, style="Status.TFrame", padding=(10, 4))
        status.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="Welcome! Load a CSV file to begin.")
        self.dataset_var = tk.StringVar(value="No dataset loaded")
        ttk.Label(status, textvariable=self.status_var, style="Status.TLabel").pack(
            side="left"
        )
        ttk.Label(status, textvariable=self.dataset_var, style="Status.TLabel").pack(
            side="right"
        )

        # Body: sidebar + content
        body = ttk.Frame(self)
        body.pack(side="top", fill="both", expand=True)

        sidebar = tk.Frame(body, bg=self.SIDEBAR, width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="MAIN MENU",
            bg=self.SIDEBAR,
            fg="#94a3b8",
            font=self.bold,
            anchor="w",
        ).pack(fill="x", padx=18, pady=(16, 8))

        self.content = ttk.Frame(body)
        self.content.pack(side="left", fill="both", expand=True)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        builders = {
            "Dataset Preview": self.build_preview,
            "Explore Data": self.build_explore,
            "DataFrame & NumPy": self.build_numpy,
            "Clean Data": self.build_clean,
            "Combine Data": self.build_combine,
            "Split Data": self.build_split,
            "Search / Sort / Filter": self.build_search,
            "Aggregation": self.build_aggregation,
            "Statistics": self.build_statistics,
            "Pivot Table": self.build_pivot,
            "Advanced Operations": self.build_advanced,
            "Visualization": self.build_visualization,
            "Save & Report": self.build_save,
        }

        for name in PAGE_NAMES:
            button = ttk.Button(
                sidebar,
                text=name,
                style="Nav.TButton",
                command=lambda n=name: self.show_page(n),
            )
            button.pack(fill="x")
            self.nav_buttons[name] = button

            page = ttk.Frame(self.content)
            page.grid(row=0, column=0, sticky="nsew")
            ttk.Label(page, text=name, style="Page.TLabel").pack(
                anchor="w", padx=12, pady=(10, 0)
            )
            builders[name](page)
            self.pages[name] = page

        ttk.Button(
            sidebar, text="Exit", style="Nav.TButton", command=self.exit_app
        ).pack(side="bottom", fill="x", pady=(0, 12))

    def show_page(self, name):
        self.pages[name].tkraise()
        for page_name, button in self.nav_buttons.items():
            button.configure(
                style="NavSel.TButton" if page_name == name else "Nav.TButton"
            )

    def exit_app(self):
        try:
            plt.close("all")
        finally:
            self.destroy()

    # ------------------------------------------------------------------
    # Reusable widget helpers
    # ------------------------------------------------------------------

    def set_status(self, text):
        self.status_var.set(text)

    def section(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(fill="x", padx=12, pady=(8, 0))
        return frame

    def button(self, parent, text, command, accent=False):
        return ttk.Button(
            parent,
            text=text,
            command=command,
            style="Accent.TButton" if accent else "TButton",
        )

    def column_combo(self, parent, preferred=None, width=16, editable=False):
        box = ttk.Combobox(
            parent, state="normal" if editable else "readonly", width=width
        )
        self.column_boxes.append((box, preferred))
        self._fill_column_box(box, preferred)
        return box

    def category_combo(self, parent, width=14):
        box = ttk.Combobox(parent, state="readonly", width=width)
        self.category_boxes.append(box)
        self._fill_category_box(box)
        return box

    def output(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, padx=12, pady=12)
        text = tk.Text(
            frame,
            wrap="none",
            font=self.mono,
            bg="white",
            fg=self.TEXT,
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6,
            state="disabled",
        )
        ys = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        text.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        return text

    def write(self, widget, content):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content.rstrip() + "\n")
        widget.configure(state="disabled")
        widget.see("1.0")

    def need_data(self):
        if not self.az.loaded:
            messagebox.showwarning("No dataset", "Please load a CSV dataset first.")
            return False
        return True

    def run(self, output, action, refresh=False, success=None):
        """Run an analyzer action, show its printed text, and handle errors."""
        if not self.need_data():
            return
        try:
            text = capture(action)
        except Exception as error:
            messagebox.showerror("Error", str(error))
            self.set_status(f"Error: {error}")
            return
        self.write(output, text)
        if refresh:
            self.refresh_all()
        self.set_status(success or "Done.")

    # ------------------------------------------------------------------
    # Loading and refreshing
    # ------------------------------------------------------------------

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.file_var.set(path)
            self.load_file()

    def load_file(self, path=None):
        path = path or self.file_var.get()
        if not clean_path(path):
            messagebox.showwarning(
                "No file", "Please enter or browse for a CSV file first."
            )
            return
        try:
            self.az.load_data(path)
        except Exception as error:
            messagebox.showerror("Could not load file", str(error))
            self.set_status(f"Load failed: {error}")
            return

        self.file_var.set(self.az.file_path)
        self.refresh_all()
        self.show_page("Dataset Preview")
        self.set_status(
            f"Data loaded successfully: {os.path.basename(self.az.file_path)}"
        )

    def _fill_column_box(self, box, preferred):
        if not self.az.loaded:
            box["values"] = []
            box.set("")
            return
        columns = [str(c) for c in self.az.data.columns]
        box["values"] = columns
        if box.get() not in columns:
            box.set(preferred if preferred in columns else columns[0])

    def _fill_category_box(self, box):
        options = (
            [c for c in ("Product", "Region") if c in self.az.data.columns]
            if self.az.loaded
            else []
        )
        box["values"] = options
        if box.get() not in options:
            box.set(options[0] if options else "")

    def refresh_all(self):
        """Update every widget that depends on the dataset."""
        for box, preferred in self.column_boxes:
            self._fill_column_box(box, preferred)
        for box in self.category_boxes:
            self._fill_category_box(box)
        for listbox in self.column_lists:
            listbox.delete(0, "end")
            if self.az.loaded:
                for column in self.az.data.columns:
                    listbox.insert("end", str(column))
        if self.part_box is not None:
            self.part_box["values"] = []
            self.part_box.set("")
        self.populate_table()
        self.update_dataset_info()

    def update_dataset_info(self):
        if not self.az.loaded:
            self.dataset_var.set("No dataset loaded")
            self.info_var.set("")
            return
        data = self.az.data
        missing = int(data.isna().sum().sum())
        duplicates = int(data.duplicated().sum())
        text = (
            f"Rows: {data.shape[0]}   Columns: {data.shape[1]}   "
            f"Missing values: {missing}   Duplicate rows: {duplicates}"
        )
        self.dataset_var.set(
            f"{os.path.basename(self.az.file_path or '')}  |  "
            f"{data.shape[0]} rows x {data.shape[1]} columns"
        )
        self.info_var.set(text)

    @staticmethod
    def _format_cell(value):
        if pd.isna(value):
            return "NaN"
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def populate_table(self, limit=500):
        tree = self.tree
        tree.delete(*tree.get_children())
        if not self.az.loaded:
            tree["columns"] = ()
            return

        frame = self.az.data.head(limit)
        columns = [str(c) for c in frame.columns]
        tree["columns"] = columns
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=max(90, len(column) * 11 + 24), anchor="center")

        has_missing = frame.isna().any(axis=1).tolist()
        for flag, row in zip(has_missing, frame.itertuples(index=False)):
            tree.insert(
                "",
                "end",
                values=[self._format_cell(v) for v in row],
                tags=("missing",) if flag else (),
            )

    # ------------------------------------------------------------------
    # Page 1: Dataset preview
    # ------------------------------------------------------------------

    def build_preview(self, page):
        self.info_var = tk.StringVar(value="")
        ttk.Label(page, textvariable=self.info_var).pack(
            anchor="w", padx=12, pady=(4, 0)
        )
        ttk.Label(
            page,
            text="Rows with missing values are highlighted in red "
            "(first 500 rows shown).",
        ).pack(anchor="w", padx=12)

        frame = ttk.Frame(page)
        frame.pack(fill="both", expand=True, padx=12, pady=10)
        self.tree = ttk.Treeview(frame, show="headings")
        self.tree.tag_configure("missing", background="#fde2e2")
        ys = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

    # ------------------------------------------------------------------
    # Page 2: Explore data
    # ------------------------------------------------------------------

    def build_explore(self, page):
        box = self.section(page, "Explore the dataset")
        ttk.Label(box, text="Rows to display:").grid(
            row=0, column=0, sticky="w", pady=4
        )
        rows = tk.StringVar(value="5")
        ttk.Spinbox(box, from_=1, to=1000, width=6, textvariable=rows).grid(
            row=0, column=1, sticky="w", padx=6
        )

        actions = [
            ("First rows", "head"),
            ("Last rows", "tail"),
            ("Random rows", "sample"),
            ("Dataset shape", "shape"),
            ("Column names", "columns"),
            ("Data types", "dtypes"),
            ("Dataset information", "info"),
            ("Check duplicate rows", "duplicates"),
        ]
        for index, (label, kind) in enumerate(actions):
            self.button(
                box,
                label,
                lambda k=kind: self.run(
                    out, lambda: self.az.explore(k, int_value(rows.get(), 1, 100000))
                ),
            ).grid(row=1 + index // 4, column=index % 4, padx=4, pady=4, sticky="ew")
        out = self.output(page)

    # ------------------------------------------------------------------
    # Page 3: DataFrame & NumPy operations
    # ------------------------------------------------------------------

    def build_numpy(self, page):
        quick = self.section(page, "Quick operations")
        quick_actions = [
            ("Convert Sales & Profit to numeric", self.az_convert_numeric, True),
            ("Add column Expense (Sales - Profit)", self.az_add_expense, True),
            ("Calculate total Sales & Profit", self.az_totals, False),
            ("View NumPy array", self.az_show_array, False),
            ("NumPy indexing & slicing demo", self.az_numpy_demo, False),
        ]
        for index, (label, handler, _) in enumerate(quick_actions):
            self.button(quick, label, lambda h=handler: h(out)).grid(
                row=index // 3, column=index % 3, padx=4, pady=4, sticky="ew"
            )

        op_names = {
            "Add (+)": "add",
            "Subtract (-)": "subtract",
            "Multiply (*)": "multiply",
            "Divide (/)": "divide",
        }

        single = self.section(page, "Operation between a column and a number")
        column = self.column_combo(single, "Sales")
        operation = ttk.Combobox(
            single, values=list(op_names), state="readonly", width=14
        )
        operation.set("Add (+)")
        number = tk.StringVar(value="1000")
        ttk.Label(single, text="Column:").grid(row=0, column=0, padx=4)
        column.grid(row=0, column=1, padx=4)
        ttk.Label(single, text="Operation:").grid(row=0, column=2, padx=4)
        operation.grid(row=0, column=3, padx=4)
        ttk.Label(single, text="Number:").grid(row=0, column=4, padx=4)
        ttk.Entry(single, textvariable=number, width=12).grid(row=0, column=5, padx=4)
        self.button(
            single,
            "Apply",
            lambda: self.run(
                out,
                lambda: self.az.apply_operation(
                    column.get(), op_names[operation.get()], float_value(number.get())
                ),
            ),
            accent=True,
        ).grid(row=0, column=6, padx=8)

        double = self.section(page, "NumPy element-wise operation between two columns")
        first = self.column_combo(double, "Sales")
        second = self.column_combo(double, "Profit")
        operation2 = ttk.Combobox(
            double, values=list(op_names), state="readonly", width=14
        )
        operation2.set("Subtract (-)")
        ttk.Label(double, text="First:").grid(row=0, column=0, padx=4)
        first.grid(row=0, column=1, padx=4)
        ttk.Label(double, text="Second:").grid(row=0, column=2, padx=4)
        second.grid(row=0, column=3, padx=4)
        ttk.Label(double, text="Operation:").grid(row=0, column=4, padx=4)
        operation2.grid(row=0, column=5, padx=4)
        self.button(
            double,
            "Apply",
            lambda: self.run(
                out,
                lambda: self.az.elementwise(
                    first.get(), second.get(), op_names[operation2.get()]
                ),
            ),
            accent=True,
        ).grid(row=0, column=6, padx=8)

        out = self.output(page)

    def az_convert_numeric(self, out):
        self.run(out, self.az.convert_numeric, refresh=True)

    def az_add_expense(self, out):
        self.run(out, self.az.add_expense, refresh=True)

    def az_totals(self, out):
        self.run(out, self.az.totals)

    def az_show_array(self, out):
        self.run(out, self.az.show_numpy_array)

    def az_numpy_demo(self, out):
        self.run(out, self.az.numpy_indexing_demo)

    # ------------------------------------------------------------------
    # Page 4: Clean data
    # ------------------------------------------------------------------

    def build_clean(self, page):
        box = self.section(page, "Handle missing data and clean the dataset")
        actions = [
            ("Check missing values", self.az.check_missing, False),
            ("Remove rows with missing values", self.az.drop_missing, True),
            ("Fill numeric with mean", lambda: self.az.fill_numeric("mean"), True),
            ("Fill numeric with median", lambda: self.az.fill_numeric("median"), True),
            ("Fill text with mode", self.az.fill_text_mode, True),
            ("Replace 'unknown' with NaN", self.az.replace_unknown, True),
            ("Remove duplicate rows", self.az.drop_duplicates, True),
            ("Convert Date to datetime", self.az.convert_date, True),
            ("Convert Sales & Profit to numeric", self.az.convert_numeric, True),
        ]
        for index, (label, action, refresh) in enumerate(actions):
            self.button(
                box, label, lambda a=action, r=refresh: self.run(out, a, refresh=r)
            ).grid(row=index // 3, column=index % 3, padx=4, pady=4, sticky="ew")
        for column in range(3):
            box.grid_columnconfigure(column, weight=1)
        out = self.output(page)

    # ------------------------------------------------------------------
    # Page 5: Combine data
    # ------------------------------------------------------------------

    def build_combine(self, page):
        box = self.section(page, "Second CSV file")
        self.second_path = tk.StringVar()
        ttk.Entry(box, textvariable=self.second_path, width=70).grid(
            row=0, column=0, padx=4
        )
        self.button(box, "Browse...", self.browse_second).grid(row=0, column=1, padx=4)

        concat = self.section(page, "Concatenate (add rows)")
        self.button(
            concat,
            "Concatenate second file",
            lambda: self.run(
                out,
                lambda: self.az.combine_concat(self.second_path.get()),
                refresh=True,
            ),
            accent=True,
        ).grid(row=0, column=0, padx=4)
        self.button(
            concat,
            "Combine using + operator",
            lambda: self.run(
                out, lambda: self.az.combine_plus(self.second_path.get()), refresh=True
            ),
        ).grid(row=0, column=1, padx=4)

        merge = self.section(page, "Merge (join using a common column)")
        ttk.Label(merge, text="Common column:").grid(row=0, column=0, padx=4)
        common = self.column_combo(merge, "SalesID", editable=True)
        common.grid(row=0, column=1, padx=4)
        ttk.Label(merge, text="Join type:").grid(row=0, column=2, padx=4)
        join = ttk.Combobox(
            merge,
            values=["inner", "left", "right", "outer"],
            state="readonly",
            width=10,
        )
        join.set("inner")
        join.grid(row=0, column=3, padx=4)
        self.button(
            merge,
            "Merge second file",
            lambda: self.run(
                out,
                lambda: self.az.combine_merge(
                    self.second_path.get(), common.get(), join.get()
                ),
                refresh=True,
            ),
            accent=True,
        ).grid(row=0, column=4, padx=8)

        out = self.output(page)

    def browse_second(self):
        path = filedialog.askopenfilename(
            title="Select the second CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.second_path.set(path)

    # ------------------------------------------------------------------
    # Page 6: Split data
    # ------------------------------------------------------------------

    def build_split(self, page):
        rows_box = self.section(page, "Split rows into two parts")
        ttk.Label(rows_box, text="First part percentage (1-99):").grid(
            row=0, column=0, padx=4
        )
        percent = tk.StringVar(value="70")
        ttk.Spinbox(rows_box, from_=1, to=99, width=6, textvariable=percent).grid(
            row=0, column=1, padx=4
        )
        self.button(
            rows_box,
            "Split rows",
            lambda: self.run(
                out, lambda: self.az.split_rows(int_value(percent.get(), 1, 99))
            ),
            accent=True,
        ).grid(row=0, column=2, padx=8)

        cols_box = self.section(page, "Select specific columns")
        listbox = tk.Listbox(
            cols_box, selectmode="multiple", height=4, exportselection=False
        )
        self.column_lists.append(listbox)
        listbox.grid(row=0, column=0, padx=4, rowspan=2)
        ttk.Label(cols_box, text="Hold Ctrl / Shift to select several columns.").grid(
            row=0, column=1, padx=8, sticky="sw"
        )
        self.button(
            cols_box,
            "Show selected columns",
            lambda: self.run(
                out,
                lambda: self.az.select_columns(
                    [listbox.get(i) for i in listbox.curselection()]
                ),
            ),
            accent=True,
        ).grid(row=1, column=1, padx=8, sticky="nw")

        value_box = self.section(
            page, "Split by the values of a column (Region / Product / ...)"
        )
        ttk.Label(value_box, text="Column:").grid(row=0, column=0, padx=4)
        column = self.column_combo(value_box, "Region")
        column.grid(row=0, column=1, padx=4)
        self.button(
            value_box,
            "Split",
            lambda: self.split_by_clicked(column.get(), out),
            accent=True,
        ).grid(row=0, column=2, padx=8)
        ttk.Label(value_box, text="Value:").grid(row=0, column=3, padx=4)
        self.part_box = ttk.Combobox(value_box, state="readonly", width=16)
        self.part_box.grid(row=0, column=4, padx=4)
        self.button(
            value_box,
            "Show part",
            lambda: self.run(out, lambda: self.az.show_part(self.part_box.get())),
        ).grid(row=0, column=5, padx=8)

        out = self.output(page)

    def split_by_clicked(self, column, out):
        self.run(out, lambda: self.az.split_by(column))
        self.part_box["values"] = [str(key) for key in self.az.parts]
        if self.az.parts:
            self.part_box.set(str(next(iter(self.az.parts))))

    # ------------------------------------------------------------------
    # Page 7: Search / sort / filter
    # ------------------------------------------------------------------

    def build_search(self, page):
        search = self.section(page, "Search for a value")
        column = self.column_combo(search, "Product")
        value = tk.StringVar()
        ttk.Label(search, text="Column:").grid(row=0, column=0, padx=4)
        column.grid(row=0, column=1, padx=4)
        ttk.Label(search, text="Value:").grid(row=0, column=2, padx=4)
        ttk.Entry(search, textvariable=value, width=20).grid(row=0, column=3, padx=4)
        self.button(
            search,
            "Search",
            lambda: self.run(out, lambda: self.az.search(column.get(), value.get())),
            accent=True,
        ).grid(row=0, column=4, padx=8)

        sort = self.section(page, "Sort")
        sort_column = self.column_combo(sort, "Sales")
        order = tk.StringVar(value="asc")
        ttk.Label(sort, text="Column:").grid(row=0, column=0, padx=4)
        sort_column.grid(row=0, column=1, padx=4)
        ttk.Radiobutton(sort, text="Ascending", variable=order, value="asc").grid(
            row=0, column=2
        )
        ttk.Radiobutton(sort, text="Descending", variable=order, value="desc").grid(
            row=0, column=3
        )
        self.button(
            sort,
            "Sort by column",
            lambda: self.run(
                out,
                lambda: self.az.sort_by_column(sort_column.get(), order.get() == "asc"),
            ),
            accent=True,
        ).grid(row=0, column=4, padx=8)
        self.button(
            sort,
            "Sort by index",
            lambda: self.run(out, lambda: self.az.sort_by_index(order.get() == "asc")),
        ).grid(row=0, column=5)

        numeric = self.section(page, "Filter numeric values")
        num_column = self.column_combo(numeric, "Sales")
        operator = ttk.Combobox(
            numeric, values=[">", "<", "=="], state="readonly", width=4
        )
        operator.set(">")
        limit = tk.StringVar(value="10000")
        ttk.Label(numeric, text="Column:").grid(row=0, column=0, padx=4)
        num_column.grid(row=0, column=1, padx=4)
        operator.grid(row=0, column=2, padx=4)
        ttk.Entry(numeric, textvariable=limit, width=12).grid(row=0, column=3, padx=4)
        self.button(
            numeric,
            "Filter",
            lambda: self.run(
                out,
                lambda: self.az.filter_numeric(
                    num_column.get(), operator.get(), float_value(limit.get())
                ),
            ),
            accent=True,
        ).grid(row=0, column=4, padx=8)

        multi = self.section(
            page, "Filter using multiple conditions (Sales AND Profit)"
        )
        min_sales = tk.StringVar(value="10000")
        min_profit = tk.StringVar(value="2000")
        ttk.Label(multi, text="Minimum Sales:").grid(row=0, column=0, padx=4)
        ttk.Entry(multi, textvariable=min_sales, width=12).grid(row=0, column=1, padx=4)
        ttk.Label(multi, text="Minimum Profit:").grid(row=0, column=2, padx=4)
        ttk.Entry(multi, textvariable=min_profit, width=12).grid(
            row=0, column=3, padx=4
        )
        self.button(
            multi,
            "Filter",
            lambda: self.run(
                out,
                lambda: self.az.filter_multi(
                    float_value(min_sales.get()), float_value(min_profit.get())
                ),
            ),
            accent=True,
        ).grid(row=0, column=4, padx=8)

        isin = self.section(page, "Filter using isin()  (values separated by commas)")
        isin_column = self.column_combo(isin, "Region")
        isin_values = tk.StringVar(value="North, South")
        ttk.Label(isin, text="Column:").grid(row=0, column=0, padx=4)
        isin_column.grid(row=0, column=1, padx=4)
        ttk.Entry(isin, textvariable=isin_values, width=30).grid(
            row=0, column=2, padx=4
        )
        self.button(
            isin,
            "Filter",
            lambda: self.run(
                out, lambda: self.az.filter_isin(isin_column.get(), isin_values.get())
            ),
            accent=True,
        ).grid(row=0, column=3, padx=8)

        dates = self.section(page, "Filter by date range  (YYYY-MM-DD)")
        start = tk.StringVar(value="2026-01-01")
        end = tk.StringVar(value="2026-03-31")
        ttk.Label(dates, text="Start:").grid(row=0, column=0, padx=4)
        ttk.Entry(dates, textvariable=start, width=14).grid(row=0, column=1, padx=4)
        ttk.Label(dates, text="End:").grid(row=0, column=2, padx=4)
        ttk.Entry(dates, textvariable=end, width=14).grid(row=0, column=3, padx=4)
        self.button(
            dates,
            "Filter",
            lambda: self.run(out, lambda: self.az.filter_dates(start.get(), end.get())),
            accent=True,
        ).grid(row=0, column=4, padx=8)

        out = self.output(page)

    # ------------------------------------------------------------------
    # Page 8: Aggregation
    # ------------------------------------------------------------------

    def build_aggregation(self, page):
        basic = self.section(page, "Aggregate a single column")
        column = self.column_combo(basic, "Sales")
        ttk.Label(basic, text="Column:").grid(row=0, column=0, padx=4)
        column.grid(row=0, column=1, padx=4)
        for index, (label, function) in enumerate(
            [
                ("Sum", "sum"),
                ("Mean", "mean"),
                ("Minimum", "min"),
                ("Maximum", "max"),
                ("Count", "count"),
            ]
        ):
            self.button(
                basic,
                label,
                lambda f=function: self.run(
                    out, lambda: self.az.aggregate_basic(column.get(), f)
                ),
            ).grid(row=0, column=2 + index, padx=4)

        group = self.section(
            page, "Group by (sum, mean and count of the numeric columns)"
        )
        for index, name in enumerate(["Product", "Region", "Year"]):
            self.button(
                group,
                f"Group by {name}",
                lambda n=name: self.run(out, lambda: self.az.group_by(n), refresh=True),
                accent=True,
            ).grid(row=0, column=index, padx=4)

        out = self.output(page)

    # ------------------------------------------------------------------
    # Page 9: Statistics
    # ------------------------------------------------------------------

    def build_statistics(self, page):
        box = self.section(page, "Statistical analysis of the numeric columns")
        percentiles = tk.StringVar(value="25, 50, 75, 90")
        ttk.Label(box, text="Percentiles (0-100):").grid(row=0, column=0, padx=4)
        ttk.Entry(box, textvariable=percentiles, width=22).grid(row=0, column=1, padx=4)
        self.button(
            box,
            "Run analysis",
            lambda: self.run(out, lambda: self.az.statistics(percentiles.get())),
            accent=True,
        ).grid(row=0, column=2, padx=8)
        ttk.Label(
            box,
            text="Shows describe(), median, var(), std(), quantile() and correlation.",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=4, pady=(6, 0))
        out = self.output(page)

    # ------------------------------------------------------------------
    # Page 10: Pivot table
    # ------------------------------------------------------------------

    def build_pivot(self, page):
        box = self.section(page, "Create a pivot table (Sales)")
        kind = tk.IntVar(value=1)
        for index, label in enumerate(
            [
                "Product-wise sales by region",
                "Region-wise sales by product",
                "Product and region summary",
            ],
            start=1,
        ):
            ttk.Radiobutton(box, text=label, variable=kind, value=index).grid(
                row=0, column=index - 1, padx=8
            )
        self.button(
            box,
            "Create pivot table",
            lambda: self.run(out, lambda: self.az.pivot(kind.get())),
            accent=True,
        ).grid(row=0, column=3, padx=8)
        out = self.output(page)

    # ------------------------------------------------------------------
    # Page 11: Advanced operations
    # ------------------------------------------------------------------

    def build_advanced(self, page):
        transform = self.section(page, "groupby() + transform()")
        group = self.column_combo(transform, "Region")
        value = self.column_combo(transform, "Sales")
        ttk.Label(transform, text="Group by:").grid(row=0, column=0, padx=4)
        group.grid(row=0, column=1, padx=4)
        ttk.Label(transform, text="Value column:").grid(row=0, column=2, padx=4)
        value.grid(row=0, column=3, padx=4)
        self.button(
            transform,
            "Add total / share % / difference columns",
            lambda: self.run(
                out,
                lambda: self.az.groupby_transform(group.get(), value.get()),
                refresh=True,
            ),
            accent=True,
        ).grid(row=0, column=4, padx=8)

        index_box = self.section(page, "Re-index")
        start = tk.StringVar(value="0")
        ttk.Label(index_box, text="Start index at:").grid(row=0, column=0, padx=4)
        ttk.Spinbox(index_box, from_=0, to=1, width=4, textvariable=start).grid(
            row=0, column=1, padx=4
        )
        self.button(
            index_box,
            "Reset index",
            lambda: self.run(
                out,
                lambda: self.az.reset_index(int_value(start.get(), 0, 1)),
                refresh=True,
            ),
        ).grid(row=0, column=2, padx=8)

        sort_column = self.column_combo(index_box, "Sales")
        order = tk.StringVar(value="asc")
        ttk.Label(index_box, text="Sort column:").grid(row=0, column=3, padx=(24, 4))
        sort_column.grid(row=0, column=4, padx=4)
        ttk.Radiobutton(index_box, text="Asc", variable=order, value="asc").grid(
            row=0, column=5
        )
        ttk.Radiobutton(index_box, text="Desc", variable=order, value="desc").grid(
            row=0, column=6
        )
        self.button(
            index_box,
            "Sort + re-index",
            lambda: self.run(
                out,
                lambda: self.az.sort_and_reindex(
                    sort_column.get(), order.get() == "asc"
                ),
                refresh=True,
            ),
        ).grid(row=0, column=7, padx=8)

        rename = self.section(page, "Alter a column label")
        old = self.column_combo(rename, "Sales")
        new = tk.StringVar()
        ttk.Label(rename, text="Column:").grid(row=0, column=0, padx=4)
        old.grid(row=0, column=1, padx=4)
        ttk.Label(rename, text="New name:").grid(row=0, column=2, padx=4)
        ttk.Entry(rename, textvariable=new, width=20).grid(row=0, column=3, padx=4)
        self.button(
            rename,
            "Rename",
            lambda: self.run(
                out, lambda: self.az.rename_column(old.get(), new.get()), refresh=True
            ),
            accent=True,
        ).grid(row=0, column=4, padx=8)

        out = self.output(page)

    # ------------------------------------------------------------------
    # Page 12: Visualization
    # ------------------------------------------------------------------

    def build_visualization(self, page):
        wrapper = ttk.Frame(page)
        wrapper.pack(fill="both", expand=True)

        options = ttk.LabelFrame(wrapper, text="Chart options", padding=10)
        options.pack(side="left", fill="y", padx=(12, 6), pady=10)

        ttk.Label(options, text="Chart:").pack(anchor="w")
        self.chart_box = ttk.Combobox(
            options, values=list(CHARTS), state="readonly", width=34
        )
        self.chart_box.set(list(CHARTS)[0])
        self.chart_box.pack(anchor="w", pady=(0, 8))

        ttk.Label(options, text="Category (pie, stack, box, bar):").pack(anchor="w")
        self.category_box = self.category_combo(options, width=20)
        self.category_box.pack(anchor="w", pady=(0, 8))

        ttk.Label(options, text="Seaborn style:").pack(anchor="w")
        self.style_box = ttk.Combobox(
            options, values=SNS_STYLES, state="readonly", width=20
        )
        self.style_box.set("darkgrid")
        self.style_box.pack(anchor="w", pady=(0, 8))

        ttk.Label(options, text="Seaborn palette:").pack(anchor="w")
        self.palette_box = ttk.Combobox(
            options, values=SNS_PALETTES, state="readonly", width=20
        )
        self.palette_box.set("deep")
        self.palette_box.pack(anchor="w", pady=(0, 14))

        self.button(options, "Draw chart", self.draw_chart, accent=True).pack(
            fill="x", pady=3
        )
        self.button(options, "Save chart as image...", self.save_visualization).pack(
            fill="x", pady=3
        )

        ttk.Label(
            options,
            wraplength=240,
            justify="left",
            text="Tip: Seaborn style and palette apply to the three Seaborn charts. "
            "Save as .png or .jpg.",
        ).pack(anchor="w", pady=(14, 0))

        self.chart_area = ttk.Frame(wrapper)
        self.chart_area.pack(
            side="left", fill="both", expand=True, padx=(6, 12), pady=10
        )
        ttk.Label(self.chart_area, text="Choose a chart and click 'Draw chart'.").pack(
            expand=True
        )

    def draw_chart(self):
        if not self.need_data():
            return
        self.az.sns_style = self.style_box.get()
        self.az.sns_palette = self.palette_box.get()
        try:
            figure = self.az.build_figure(
                CHARTS[self.chart_box.get()], self.category_box.get() or None
            )
        except Exception as error:
            messagebox.showerror("Chart error", str(error))
            self.set_status(f"Chart error: {error}")
            return
        self.show_figure(figure)
        self.set_status(f"Chart drawn: {self.chart_box.get()}")

    def show_figure(self, figure):
        for widget in self.chart_area.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(figure, master=self.chart_area)
        toolbar = NavigationToolbar2Tk(canvas, self.chart_area, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        canvas.draw()

    def save_visualization(self):
        if self.az.last_figure is None:
            messagebox.showwarning("No chart", "Please draw a chart first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save chart as image",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg *.jpeg")],
        )
        if not path:
            return
        try:
            self.az.last_figure.savefig(path, bbox_inches="tight", dpi=150)
        except Exception as error:
            messagebox.showerror("Could not save chart", str(error))
            return
        self.set_status(f"Visualization saved: {path}")
        messagebox.showinfo("Saved", f"Visualization saved successfully:\n{path}")

    # ------------------------------------------------------------------
    # Page 13: Save & report
    # ------------------------------------------------------------------

    def build_save(self, page):
        box = self.section(page, "Export")
        self.button(
            box, "Save current dataset as CSV...", self.save_dataset, accent=True
        ).grid(row=0, column=0, padx=4, pady=4)
        self.button(box, "Save last result as CSV...", self.save_result).grid(
            row=0, column=1, padx=4, pady=4
        )
        self.button(box, "Save chart as image...", self.save_visualization).grid(
            row=0, column=2, padx=4, pady=4
        )

        report = self.section(page, "Summary report (conclusion)")
        self.button(
            report, "Generate summary", self.generate_summary, accent=True
        ).grid(row=0, column=0, padx=4, pady=4)
        self.button(report, "Save summary as text file...", self.save_summary).grid(
            row=0, column=1, padx=4, pady=4
        )

        self.summary_out = self.output(page)

    def save_dataset(self):
        if not self.need_data():
            return
        path = filedialog.asksaveasfilename(
            title="Save dataset",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        try:
            self.az.data.to_csv(path, index=False)
        except Exception as error:
            messagebox.showerror("Could not save file", str(error))
            return
        self.set_status(f"Dataset saved: {path}")
        messagebox.showinfo("Saved", f"File saved successfully:\n{path}")

    def save_result(self):
        if not self.need_data():
            return
        if self.az.result is None:
            messagebox.showwarning(
                "No result", "No result available. Perform an operation first."
            )
            return
        path = filedialog.asksaveasfilename(
            title="Save last result",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        try:
            result = self.az.result
            if isinstance(result, pd.Series):
                result = result.to_frame()
            result.to_csv(path, index=True)
        except Exception as error:
            messagebox.showerror("Could not save file", str(error))
            return
        self.set_status(f"Result saved: {path}")
        messagebox.showinfo("Saved", f"File saved successfully:\n{path}")

    def generate_summary(self):
        if not self.need_data():
            return
        try:
            self.last_summary = "\n".join(self.az.summary_lines())
        except Exception as error:
            messagebox.showerror("Error", str(error))
            return
        self.write(self.summary_out, self.last_summary)
        self.set_status("Summary generated.")

    def save_summary(self):
        if not self.last_summary:
            if not self.need_data():
                return
            self.generate_summary()
        path = filedialog.asksaveasfilename(
            title="Save summary",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as file:
                file.write(self.last_summary + "\n")
        except OSError as error:
            messagebox.showerror("Could not save file", str(error))
            return
        self.set_status(f"Summary saved: {path}")
        messagebox.showinfo("Saved", f"Summary saved:\n{path}")


# ======================================================================
# Entry point
# ======================================================================


def find_default_csv():
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.getcwd()):
        candidate = os.path.join(folder, "sales_data.csv")
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    try:  # sharper text on high-DPI Windows screens
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = SalesAnalyzerApp()
    default_csv = find_default_csv()
    if default_csv:
        app.load_file(default_csv)
    app.mainloop()


if __name__ == "__main__":
    main()
