import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# safe user input


def int_input(message, minimum, maximum):
    while True:
        try:
            number = int(input(message))
            if minimum <= number <= maximum:
                return number

            print(f"Please enter a number between {minimum} and {maximum}.\n")

        except ValueError:
            print("Invalid input! Please enter a whole number.\n")


def text_input(message):
    while True:
        value = input(message).strip()
        if value:
            return value
        print("Input cannot be empty. Please try again.\n")


class SalesDataAnalyzer:

    def __init__(self):
        self.data = None
        self.result = None
        self.last_figure = None
        self.file_path = None

    def check_data(self):
        if self.data is None:
            print("\nPlease load a CSV dataset first (Main Menu option 1).\n")
            return False
        return True

    def show_columns(self):
        print("\nAvailable columns:")
        for column in self.data.columns:
            print(f"{column}")

    def choose_column(self, allow_all=False):
        self.show_columns()

        if allow_all:
            print("all")

        while True:
            column = input("\nEnter column name: ").strip()

            if allow_all and column.lower() == "all":
                return "all"

            if column in self.data.columns:
                return column

            print("Column not found.\n")

    # 1. Load data ------------------------------------------------------------------------------

    def load_data(self, file_path):
        try:
            if not os.path.exists(file_path):
                print("File not found!")
                return

            if not file_path.lower().endswith(".csv"):
                print("Please enter a CSV file!")
                return

            data = pd.read_csv(file_path)

            if data.empty:
                print("File is empty!")
                return

            self.file_path = file_path
            self.data = data
            self.result = None

            print("\nData loaded successfully!")
            print("\nTotal rows:", self.data.shape[0])
            print("Total columns:", self.data.shape[1])
            print("\nFirst 5 rows:\n", self.data.head(5))

        except pd.errors.EmptyDataError:
            print("File is empty!")

        except pd.errors.ParserError:
            print("Invalid CSV file format!")

        except FileNotFoundError:
            print("File not found!")

        except PermissionError:
            print("Permission denied!")

        except UnicodeDecodeError:
            print("Could not read the file. Please use a UTF-8 encoded CSV file.")

        except OSError as error:
            print(f"Operating system error: {error}")

    # 2. Explore data ------------------------------------------------------------------------

    def explore_data(self):
        if not self.check_data():
            return

        if self.data.empty:
            print("The dataset has no rows to explore.")
            return

        while True:
            print("\n---------------- EXPLORE DATA ----------------")
            print("1. Display first rows")
            print("2. Display last rows")
            print("3. Display random rows")
            print("4. Dataset shape")
            print("5. Column names")
            print("6. Data types")
            print("7. Dataset information")
            print("8. Check duplicate rows")
            print("9. Back to main menu")

            choice = int_input("Enter your choice: ", 1, 9)

            if choice == 1:
                n = int_input("Enter rows : ", 1, len(self.data))
                print(self.data.head(n))

            elif choice == 2:
                n = int_input("Enter rows : ", 1, len(self.data))
                print(self.data.tail(n))

            elif choice == 3:
                n = int_input("Enter rows : ", 1, len(self.data))
                print(self.data.sample(n))

            elif choice == 4:
                print(f"\nRows: {self.data.shape[0]}")
                print(f"Columns: {self.data.shape[1]}")

            elif choice == 5:
                print("\nColumn names:")
                print(list(self.data.columns))

            elif choice == 6:
                print("\nData types:")
                print(self.data.dtypes)

            elif choice == 7:
                print("\nDataset information:")
                self.data.info()

            elif choice == 8:
                duplicate_count = self.data.duplicated().sum()
                print(f"\nDuplicate rows: {duplicate_count}")
                if duplicate_count > 0:
                    print(self.data[self.data.duplicated(keep=False)])

            elif choice == 9:
                break

    # 3. Dataframe and numpy operation ---------------------------------------------------------

    def mathematical_operations(self):
        if not self.check_data():
            return

        while True:
            print("\n------------- DATAFRAME / NUMPY OPERATIONS -------------")
            print("1. Convert Sales and Profit to numeric")
            print("2. Add a new column (Sales - Profit)")
            print("3. Calculate total Sales and Profit")
            print("4. Apply a mathematical operation to a numeric column")
            print("5. View a NumPy array from the dataset")
            print("6. Back to main menu")

            choice = int_input("Enter your choice: ", 1, 6)

            if choice == 1:
                available_columns = []
                for column in ["Sales", "Profit"]:
                    if column in self.data.columns:
                        self.data[column] = pd.to_numeric(
                            self.data[column], errors="coerce"
                        )
                        available_columns.append(column)

                if not available_columns:
                    print("Sales and Profit columns were not found.")
                    continue

                print("\nSales and Profit converted to numeric.")
                print(self.data[available_columns].dtypes)

            elif choice == 2:
                if not {"Sales", "Profit"}.issubset(self.data.columns):
                    print("Sales and Profit columns are required.")
                    continue

                sales = pd.to_numeric(self.data["Sales"], errors="coerce")
                profit = pd.to_numeric(self.data["Profit"], errors="coerce")
                self.data["Expense"] = sales - profit
                print("\nExpense column added (Sales - Profit).")
                print(self.data[["Sales", "Profit", "Expense"]].head())

            elif choice == 3:
                for column in ["Sales", "Profit"]:
                    if column in self.data.columns:
                        values = pd.to_numeric(self.data[column], errors="coerce")
                        print(f"Total {column}: {values.sum()}")

            elif choice == 4:
                column = self.choose_column()
                values = pd.to_numeric(self.data[column], errors="coerce")

                if values.notna().sum() == 0:
                    print("This column does not contain numeric values.")
                    continue

                print("1. Add a number")
                print("2. Subtract a number")
                print("3. Multiply by a number")
                print("4. Divide by a number")
                operation = int_input("Choose operation: ", 1, 4)

                while True:
                    try:
                        number = float(input("Enter number: "))
                        if operation == 4 and number == 0:
                            print("Cannot divide by zero.")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid number.")

                if operation == 1:
                    self.result = values + number
                elif operation == 2:
                    self.result = values - number
                elif operation == 3:
                    self.result = values * number
                else:
                    self.result = values / number

                print("\nResult:")
                print(self.result.head(10))

            elif choice == 5:
                numeric_data = self.data.select_dtypes(include="number")
                if numeric_data.empty:
                    print("No numeric columns found.")
                else:
                    array = numeric_data.to_numpy()
                    print("\nNumPy array:")
                    print(array)
                    print("Array shape:", array.shape)

            elif choice == 6:
                break

    # 4. Clean data / Handle missing values -------------------------------------------------

    def clean_data(self):
        if not self.check_data():
            return

        while True:
            print("\n---------------- CLEAN DATA ----------------")
            print("1. Check missing values")
            print("2. Remove rows containing missing values")
            print("3. Fill missing numeric values with mean")
            print("4. Fill missing numeric values with median")
            print("5. Fill missing text values with mode")
            print("6. Replace 'unknown' with missing value")
            print("7. Remove duplicate rows")
            print("8. Convert Date column to datetime")
            print("9. Convert Sales and Profit to numeric")
            print("10. Back to main menu")

            choice = int_input("Enter your choice: ", 1, 10)

            if choice == 1:
                print("\nMissing values in each column:")
                print(self.data.isna().sum())
                print("\nTotal missing values:", self.data.isna().sum().sum())

            elif choice == 2:
                before = len(self.data)
                self.data = self.data.dropna().copy()
                print(f"Removed {before - len(self.data)} rows.")

            elif choice in [3, 4]:
                numeric_columns = self.data.select_dtypes(include="number").columns

                if len(numeric_columns) == 0:
                    print("No numeric columns found. Convert Sales/Profit first.")
                    continue

                for column in numeric_columns:
                    if choice == 3:
                        self.data[column] = self.data[column].fillna(
                            self.data[column].mean()
                        )
                    else:
                        self.data[column] = self.data[column].fillna(
                            self.data[column].median()
                        )

                print("Missing numeric values filled successfully.")

            elif choice == 5:
                text_columns = self.data.select_dtypes(include="object").columns
                for column in text_columns:
                    mode_values = self.data[column].mode()
                    if not mode_values.empty:
                        self.data[column] = self.data[column].fillna(mode_values[0])
                print("Missing text values filled with each column's mode.")

            elif choice == 6:
                self.data.replace("unknown", np.nan, inplace=True)
                print("'unknown' values replaced with NaN.")

            elif choice == 7:
                before = len(self.data)
                self.data = self.data.drop_duplicates().copy()
                print(f"Removed {before - len(self.data)} duplicate rows.")

            elif choice == 8:
                if "Date" not in self.data.columns:
                    print("Date column not found.")
                else:
                    self.data["Date"] = pd.to_datetime(
                        self.data["Date"], errors="coerce"
                    )
                    print("Date column converted to datetime.")

            elif choice == 9:
                for column in ["Sales", "Profit"]:
                    if column in self.data.columns:
                        self.data[column] = pd.to_numeric(
                            self.data[column], errors="coerce"
                        )
                print("Sales and Profit converted to numeric.")

            elif choice == 10:
                break

    # 5. Combine data

    def combine_data(self):
        if not self.check_data():
            return

        print("\n---------------- COMBINE DATA ----------------")
        print("1. Concatenate another CSV file (add rows)")
        print("2. Merge another CSV file (join using a common column)")
        print("3. Back")

        choice = int_input("Enter your choice: ", 1, 3)

        if choice == 3:
            return

        path = text_input("Enter the path of the second CSV file: ")

        try:
            other_data = pd.read_csv(path)

            if choice == 1:
                self.data = pd.concat([self.data, other_data], ignore_index=True)
                print("Files concatenated successfully.")

            elif choice == 2:
                print("\nColumns in current dataset:")
                print(list(self.data.columns))
                print("Columns in second dataset:")
                print(list(other_data.columns))

                common_column = text_input("Enter the common column name for merging: ")

                if common_column not in self.data.columns:
                    print("Column not found in current dataset.")
                    return
                if common_column not in other_data.columns:
                    print("Column not found in second dataset.")
                    return

                print("1. Inner join")
                print("2. Left join")
                print("3. Right join")
                print("4. Outer join")
                join_choice = int_input("Choose join type: ", 1, 4)

                join_types = {1: "inner", 2: "left", 3: "right", 4: "outer"}

                self.data = pd.merge(
                    self.data,
                    other_data,
                    on=common_column,
                    how=join_types[join_choice],
                    suffixes=("_left", "_right"),
                )
                print("Files merged successfully.")

            print("Updated dataset shape:", self.data.shape)

        except FileNotFoundError:
            print("Second CSV file not found.")
        except pd.errors.EmptyDataError:
            print("The second CSV file is empty.")
        except Exception as error:
            print(f"Unable to combine files: {error}")

    # 6. Split data -----------------------------------------------------------------------

    def split_data(self):
        if not self.check_data():
            return

        print("\n---------------- SPLIT DATA ----------------")
        print("1. Split rows into two parts")
        print("2. Select specific columns")
        print("3. Back")

        choice = int_input("Enter your choice: ", 1, 3)

        if choice == 1:
            if len(self.data) < 2:
                print("at least 2 rows are required.")
                return

            split_percent = int_input("Enter first part percentage (1-99): ", 1, 99)
            first_part_size = int(len(self.data) * split_percent / 100)

            first_part = self.data.head(first_part_size)
            remaining_rows = len(self.data) - first_part_size
            second_part = self.data.tail(remaining_rows)

            print("\nFirst part:")
            print(first_part)
            print("\nSecond part:")
            print(second_part)
            print(f"\nFirst part rows: {len(first_part)}")
            print(f"Second part rows: {len(second_part)}")

        elif choice == 2:
            self.show_columns()
            while True:
                selected = input("Enter column names separated by commas: ").strip()
                columns = [item.strip() for item in selected.split(",") if item.strip()]

                valid_columns = True
                for column in columns:
                    if column not in self.data.columns:
                        valid_columns = False
                        break

                if columns and valid_columns:
                    self.result = self.data[columns].copy()
                    print("\nSelected columns:")
                    print(self.result.head())
                    break

                print("column names are invalid. try again.")

    # 7. Search , sort , filter ----------------------------------------------------------------

    def search_sort_filter(self):
        if not self.check_data():
            return

        while True:
            print("\n------------- SEARCH / SORT / FILTER -------------")
            print("1. Search for a value")
            print("2. Sort by column values")
            print("3. Sort by index")
            print("4. Filter numeric values")
            print("5. Filter using multiple conditions")
            print("6. Filter using isin()")
            print("7. Back to main menu")

            choice = int_input("Enter your choice: ", 1, 7)

            if choice == 1:
                column = self.choose_column()
                value = text_input("Enter value to search: ")
                matches = self.data[
                    self.data[column]
                    .astype(str)
                    .str.contains(value, case=False, na=False)
                ]
                print(matches)
                self.result = matches

            elif choice == 2:
                column = self.choose_column()
                order = int_input("1. Ascending  2. Descending: ", 1, 2)
                self.result = self.data.sort_values(by=column, ascending=(order == 1))
                print(self.result)

            elif choice == 3:
                order = int_input("1. Ascending  2. Descending: ", 1, 2)
                self.result = self.data.sort_index(ascending=(order == 1))
                print(self.result)

            elif choice == 4:
                column = self.choose_column()
                values = pd.to_numeric(self.data[column], errors="coerce")

                if values.notna().sum() == 0:
                    print("Selected column has no numeric values.")
                    continue

                while True:
                    try:
                        limit = float(input("Enter comparison value: "))
                        break
                    except ValueError:
                        print("Enter a valid number.")

                print("1. Greater than")
                print("2. Less than")
                print("3. Equal to")
                operator = int_input("Choose condition: ", 1, 3)

                if operator == 1:
                    self.result = self.data[values > limit]
                elif operator == 2:
                    self.result = self.data[values < limit]
                else:
                    self.result = self.data[values == limit]

                print(self.result)

            elif choice == 5:
                if not {"Sales", "Profit"}.issubset(self.data.columns):
                    print("Sales and Profit columns are required for this example.")
                    continue

                sales = pd.to_numeric(self.data["Sales"], errors="coerce")
                profit = pd.to_numeric(self.data["Profit"], errors="coerce")
                while True:
                    try:
                        min_sales = float(input("Minimum Sales: "))
                        min_profit = float(input("Minimum Profit: "))
                        break
                    except ValueError:
                        print("Enter valid numbers.")

                self.result = self.data[(sales >= min_sales) & (profit >= min_profit)]
                print(self.result)

            elif choice == 6:
                column = self.choose_column()
                values = input("Enter values separated by commas: ").strip()

                if not values:
                    print("Please enter at least one value.")
                    continue

                selected_values = [item.strip() for item in values.split(",")]
                self.result = self.data[
                    self.data[column].astype(str).isin(selected_values)
                ]
                print(self.result)

            elif choice == 7:
                break

    # 8. Aggregation functions ---------------------------------------------------------------

    def aggregate_functions(self):
        if not self.check_data():
            return

        while True:
            print("\n---------------- AGGREGATION ----------------")
            print("1. Sum")
            print("2. Mean")
            print("3. Minimum")
            print("4. Maximum")
            print("5. Count")
            print("6. Group by Product")
            print("7. Group by Region")
            print("8. Group by Year")
            print("9. Back to main menu")

            choice = int_input("Enter your choice: ", 1, 9)

            if choice in [1, 2, 3, 4, 5]:
                column = self.choose_column()
                values = pd.to_numeric(self.data[column], errors="coerce")

                if values.notna().sum() == 0:
                    print("Selected column does not contain numeric data.")
                    continue

                if choice == 1:
                    print("Sum:", values.sum())
                elif choice == 2:
                    print("Mean:", values.mean())
                elif choice == 3:
                    print("Minimum:", values.min())
                elif choice == 4:
                    print("Maximum:", values.max())
                else:
                    print("Count of non-missing values:", values.count())

            elif choice in [6, 7, 8]:
                group_column = {6: "Product", 7: "Region", 8: "Year"}[choice]

                if group_column not in self.data.columns:
                    print(f"{group_column} column not found.")
                    continue

                for column in ["Sales", "Profit"]:
                    if column in self.data.columns:
                        self.data[column] = pd.to_numeric(
                            self.data[column], errors="coerce"
                        )

                numeric_columns = self.data.select_dtypes(include="number").columns
                if len(numeric_columns) == 0:
                    print("No numeric columns to aggregate.")
                    continue

                grouped = self.data.groupby(group_column)[list(numeric_columns)].agg(
                    ["sum", "mean", "count"]
                )

                print(grouped)

            elif choice == 9:
                break

    # 9. Statistical analysis ----------------------------------------------------------------

    def statistical_analysis(self):
        if not self.check_data():
            return

        print("\n---------------- STATISTICAL ANALYSIS ----------------")

        numeric_data = self.data.select_dtypes(include="number")

        if numeric_data.empty:
            print("No numeric columns found.")
            return

        print("\nDescriptive statistics:")
        print(numeric_data.describe())

        print("\nMedian:")
        print(numeric_data.median())

        print("\nVariance:")
        print(numeric_data.var())

        print("\nStandard deviation:")
        print(numeric_data.std())

        print("\nCorrelation:")
        if numeric_data.shape[1] >= 2:
            print(numeric_data.corr())
        else:
            print("At least two numeric columns are needed for correlation.")

    # 10. Pivot table -----------------------------------------------------------------------

    def create_pivot_table(self):
        if not self.check_data():
            return

        required = ["Product", "Region", "Sales"]
        has_required_columns = True
        for column in required:
            if column not in self.data.columns:
                has_required_columns = False
                break

        if not has_required_columns:
            print("Pivot table requires Product, Region and Sales columns.")
            return

        data_copy = self.data.copy()
        data_copy["Sales"] = pd.to_numeric(data_copy["Sales"], errors="coerce")

        print("\n---------------- PIVOT TABLE ----------------")
        print("1. Product-wise sales by region")
        print("2. Region-wise sales by product")
        print("3. Product and region summary")

        choice = int_input("Enter your choice: ", 1, 3)

        if choice == 1:
            pivot = pd.pivot_table(
                data_copy,
                values="Sales",
                index="Product",
                columns="Region",
                aggfunc="sum",
                fill_value=0,
            )
        elif choice == 2:
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
        print("\nPivot table:")
        print(pivot)

    # 11. Visualize data ------------------------------------------------------------------

    def visualize_data(self):
        if not self.check_data():
            return

        while True:
            print("\n---------------- DATA VISUALIZATION ----------------")
            print("1. Sales by Product (bar chart)")
            print("2. Profit by Region (bar chart)")
            print("3. Sales distribution (histogram)")
            print("4. Sales vs Profit (scatter plot)")
            print("5. Sales trend by Date (line chart)")
            print("6. Correlation heatmap")
            print("7. Back to main menu")

            choice = int_input("Enter your choice: ", 1, 7)

            try:
                data_copy = self.data.copy()

                for column in ["Sales", "Profit"]:
                    if column in data_copy.columns:
                        data_copy[column] = pd.to_numeric(
                            data_copy[column], errors="coerce"
                        )

                figure = plt.figure(figsize=(9, 5))

                if choice == 1:
                    if (
                        "Product" not in data_copy.columns
                        or "Sales" not in data_copy.columns
                    ):
                        print("Product and Sales columns are required for this chart.")
                        plt.close(figure)
                        continue
                    chart_data = data_copy.groupby("Product")["Sales"].sum()
                    chart_data.plot(kind="bar")
                    plt.title("Total Sales by Product")
                    plt.xlabel("Product")
                    plt.ylabel("Sales")

                elif choice == 2:
                    if (
                        "Region" not in data_copy.columns
                        or "Profit" not in data_copy.columns
                    ):
                        print("Region and Profit columns are required for this chart.")
                        plt.close(figure)
                        continue
                    chart_data = data_copy.groupby("Region")["Profit"].sum()
                    chart_data.plot(kind="bar")
                    plt.title("Total Profit by Region")
                    plt.xlabel("Region")
                    plt.ylabel("Profit")

                elif choice == 3:
                    if "Sales" not in data_copy.columns:
                        print("Sales column not found.")
                        plt.close(figure)
                        continue
                    data_copy["Sales"].dropna().plot(kind="hist", bins=10)
                    plt.title("Sales Distribution")
                    plt.xlabel("Sales")
                    plt.ylabel("Frequency")

                elif choice == 4:
                    if (
                        "Sales" not in data_copy.columns
                        or "Profit" not in data_copy.columns
                    ):
                        print("Sales and Profit columns are required for this chart.")
                        plt.close(figure)
                        continue
                    chart_data = data_copy.dropna(subset=["Sales", "Profit"])
                    plt.scatter(chart_data["Sales"], chart_data["Profit"])
                    plt.title("Sales vs Profit")
                    plt.xlabel("Sales")
                    plt.ylabel("Profit")

                elif choice == 5:
                    if (
                        "Date" not in data_copy.columns
                        or "Sales" not in data_copy.columns
                    ):
                        print("Date and Sales columns are required for this chart.")
                        plt.close(figure)
                        continue
                    data_copy["Date"] = pd.to_datetime(
                        data_copy["Date"], errors="coerce"
                    )
                    chart_data = data_copy.dropna(subset=["Date", "Sales"])
                    chart_data = chart_data.groupby("Date")["Sales"].sum().sort_index()
                    chart_data.plot(kind="line", marker="o")
                    plt.title("Sales Trend Over Time")
                    plt.xlabel("Date")
                    plt.ylabel("Sales")
                    plt.xticks(rotation=45)

                elif choice == 6:
                    numeric_data = data_copy.select_dtypes(include="number")
                    if numeric_data.shape[1] < 2:
                        print("At least two numeric columns are needed.")
                        plt.close(figure)
                        continue
                    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
                    plt.title("Correlation Heatmap")

                elif choice == 7:
                    plt.close(figure)
                    break

                plt.tight_layout()
                self.last_figure = figure
                plt.show()

            except Exception as error:
                plt.close()
                print(f"Unable to create chart: {error}")

    # 12. Save result / visualization ------------------------------------------------------

    def save_result(self):
        if not self.check_data():
            return

        print("\n---------------- SAVE DATA ----------------")
        print("1. Save current dataset as CSV")
        print("2. Save last result as CSV")
        print("3. Back")

        choice = int_input("Enter your choice: ", 1, 3)

        if choice == 3:
            return

        filename = text_input("Enter output file name (without extension): ")
        filename = filename + ".csv"

        try:
            if choice == 1:
                self.data.to_csv(filename, index=False)
            else:
                if self.result is None:
                    print("No result available. Perform an operation first.")
                    return

                if isinstance(self.result, pd.Series):
                    self.result = self.result.to_frame()

                self.result.to_csv(filename, index=True)

            print(f"File saved successfully: {os.path.abspath(filename)}")

        except Exception as error:
            print(f"Unable to save file: {error}")

    def save_visualization(self):
        if not self.check_data():
            return

        if self.last_figure is None:
            print("\nPlease create a chart from Data Visualization first.")
            return

        filename = text_input("Enter image file name (without extension): ")
        filename = filename + ".png"

        try:
            self.last_figure.savefig(filename, bbox_inches="tight")
            print(f"Visualization saved successfully: {os.path.abspath(filename)}")
        except Exception as error:
            print(f"Unable to save visualization: {error}")


analysis = SalesDataAnalyzer()

while True:
    print("\n================ MAIN MENU ================")
    print("1. Load Dataset")
    print("2. Explore Data")
    print("3. DataFrame and NumPy Operations")
    print("4. Handle Missing Data / Clean Data")
    print("5. Combine Data (Concat / Merge)")
    print("6. Split Data")
    print("7. Search, Sort and Filter")
    print("8. Aggregation Functions")
    print("9. Statistical Analysis")
    print("10. Create Pivot Table")
    print("11. Data Visualization")
    print("12. Save Dataset / Result")
    print("13. Save Visualization")
    print("14. Exit")

    choice = int_input("Enter your choice: ", 1, 14)

    if choice == 1:
        file_path = text_input("Enter CSV file name: ")
        analysis.load_data(file_path)
    elif choice == 2:
        analysis.explore_data()
    elif choice == 3:
        analysis.mathematical_operations()
    elif choice == 4:
        analysis.clean_data()
    elif choice == 5:
        analysis.combine_data()
    elif choice == 6:
        analysis.split_data()
    elif choice == 7:
        analysis.search_sort_filter()
    elif choice == 8:
        analysis.aggregate_functions()
    elif choice == 9:
        analysis.statistical_analysis()
    elif choice == 10:
        analysis.create_pivot_table()
    elif choice == 11:
        analysis.visualize_data()
    elif choice == 12:
        analysis.save_result()
    elif choice == 13:
        analysis.save_visualization()
    elif choice == 14:
        print("\nExiting the program. Goodbye!")
        break
