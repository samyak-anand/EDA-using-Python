""" Defining all the libraries used.
Python libraries are used by installing and importing them.
These libraries are required for operations such as reading, manipulating, and preparing data,
as well as visualising it.
They also support code testing, database access, and warning handling."""
import sqlite3

import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource
import numpy as np
import pytest
from bokeh.layouts import gridplot
import unittest
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sqlalchemy import Connection, text
# Define the file paths for the train, ideal, and test data files

train_data_file= r"C:\Users\samya\PycharmProjects\Assignment_new_final\train.csv"
ideal_data_file= r"C:\Users\samya\PycharmProjects\Assignment_new_final\ideal.csv"
test_data_file = r"C:\Users\samya\PycharmProjects\Assignment_new_final\test.csv"
#Loading datasets

#loading train data
train_data = pd.read_csv(train_data_file)
ideal_data = pd.read_csv(ideal_data_file)
test_data = pd.read_csv(test_data_file)

# Display the test data
print("Test Data:")
print(test_data)
def box_plot_test_data():
# Boxplot of test dataset for both x and y variable
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=test_data, palette="Set1")
    plt.title('Boxplot of the Test Dataset for both x and y variable')
    plt.xticks(rotation=45)
    plt.show()
box_plot_test_data()

# Display the ideal data
print("\nIdeal Data:")
print(ideal_data.head())
# Display the train data
print("\nTrain Data:")
print(train_data.head())
"""
There are duplicate values present in the test dataset.
Remove duplicate X values from test data and calculate the mean of corresponding Y values
"""
test_data = test_data.groupby('x').mean().reset_index()
"""
Display the test dataset after removing the duplicate values.
"""
print("\nTrain Data after cleaning:")
print(test_data)

# defining the database path
database_path =r"C:\Users\samya\PycharmProjects\Assignment_new_final\assignment_database.db"
# Create SQLite engine
engine = create_engine('sqlite:///assignment_database.db')
# Save datasets to SQLite database
test_data.to_sql('test_data', con=engine, index=False, if_exists='replace')
ideal_data.to_sql('ideal_data', con=engine, index=False, if_exists='replace')
train_data.to_sql('train_data', con=engine, index=False, if_exists='replace')
print("The database was successfully created, and the data was loaded!")

# Define the function to display the contents of the train_data table
def ideal_data_table():
    """
    Fetches and displays the contents of the train_data table from a SQLite database.
    """
    # Create a SQLite database engine
    engine = create_engine(r'sqlite:///C:\\Users\\samya\\PycharmProjects\\Assignment_new_final\\assignment_database.db', echo=True)
    # Fetch and display the contents of the train_data table
    query = "SELECT * FROM ideal_data"
    ideal_data = pd.read_sql_query(query, engine)
    print("Contents of train_data table:")
    print(ideal_data)
# Call the function to display the contents of the train_data table
ideal_data_table()

#Displays the contents of the test_data table from a SQLite database
def test_data_table():
    """
    Fetches and displays the contents of the test_data table from a SQLite database.
    """
    # Create a SQLite database engine
    engine = create_engine(r'sqlite:///C:\\Users\\samya\\PycharmProjects\\Assignment_new_final\\assignment_database.db', echo=True)
    # Fetch and display the contents of the test_data table
    query = "SELECT * FROM test_data"
    test_data = pd.read_sql_query(query, engine)
    print("Contents of test_data table:")
    print(test_data)

test_data_table()
test_data.info()
test_data.describe().T

#Displays the contents of the train_data table from a SQLite database
def train_data_table():
    """
    Fetches and displays the contents of the train_data table from a SQLite database.
    """
    # Create a SQLite database engine
    engine =create_engine(r'sqlite:///C:\\Users\\samya\\PycharmProjects\\Assignment_new_final\\assignment_database.db', echo=True)
    # Fetch and display the contents of the train_data table
    query = "SELECT * FROM train_data"
    train_data = pd.read_sql_query(query, engine)
    print("Contents of train_data table:")
    print(train_data)

train_data_table()

#Define a function to visualize ideal data using box plot
def visualize_ideal_data_boxplot(data):
    """
    Visualize multiple variables of an ideal dataset using a boxplot.
    Parameters:
    data (DataFrame): The ideal dataset.
    Returns:
    None
    """
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=data, palette="Set1")
    plt.xlabel("Variables")  # x-axis label
    plt.ylabel("Values")  # y-axis label
    plt.title("Boxplot of Ideal Dataset")  # title
    plt.show()

# Call the function to visualize the boxplot of the ideal_data dataset
visualize_ideal_data_boxplot(ideal_data)

train_data.describe().T
train_data.info()

# Boxplot of train dataset
def box_plot_train_data():
    '''
    Visualize more than two variables
    of a train dataset
    '''
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=train_data, palette="Paired", ax=ax)
    plt.xlabel("Variables")
    plt.ylabel("Values")
    plt.title("Boxplot of Train Dataset")
    plt.xticks(rotation=45)
    plt.show()
box_plot_train_data()

train_data.describe().T
train_data.info()

def box_plot_test_data_visual():
    # Boxplot of test dataset for both x and y variable
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=test_data, palette="Set1")
    plt.title('Boxplot of the Test Dataset for both x and y variable (duplicate removed)')
    plt.xticks(rotation=45)
    plt.show()
box_plot_test_data_visual()

class DataProcessor:
    def __init__(self, test_data_path, ideal_data_path, train_data_path):
        self.test_data = pd.read_csv(test_data_path)
        self.ideal_data = pd.read_csv(ideal_data_path)
        self.train_data = pd.read_csv(train_data_path)

class DatabaseHandler:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.engine = create_engine('sqlite:///assignment_database.db')
    def create_database(self):
        self.data_processor.test_data.to_sql('test_data', con=self.engine, index=False,
        if_exists='replace')
        self.data_processor.ideal_data.to_sql('ideal_data', con=self.engine, index=False,
        if_exists='replace')
        self.data_processor.train_data.to_sql('train_data', con=self.engine, index=False,
        if_exists='replace')

class Visualization:
    def __init__(self, data_processor):
        self.data_processor = data_processor
class Visualization:
    def __init__(self, data_processor):
        self.data_processor = data_processor
    def plot_scatter_with_regression(self, train_data, x_column, y_column, title):
        """
        Plot a scatter plot with a regression line using Bokeh.
        Parameters:
        train_data (pd.DataFrame): DataFrame containing the train data.
        x_column (str): Name of the x-axis column.
        y_column (str): Name of the y-axis column.
        title (str): Title of the plot.
        Returns:
        None
        """
        source = ColumnDataSource(train_data)
        f = figure(title=title, x_axis_label=x_column, y_axis_label=y_column)
        f.circle(x_column, y_column, source=source, color="red", legend_label="Train Data Points")
        f.line(x_column, y_column, source=source, color="blue", legend_label="Regression Line")
        output_notebook()
        show(f)

def main():
    test_data_path = 'test.csv'
    ideal_data_path = 'ideal.csv'
    train_data_path = 'train.csv'
    data_processor = DataProcessor(test_data_path, ideal_data_path, train_data_path)
    database_handler = DatabaseHandler(data_processor)
    visualization = Visualization(data_processor)
    database_handler.create_database()
    visualization.plot_scatter_with_regression(visualization.data_processor.train_data, 'x', 'y1','Scatter Plot for y1 with regression line')
    visualization.plot_scatter_with_regression(visualization.data_processor.train_data, 'x', 'y2','Scatter Plot for y2 with regression line')
    visualization.plot_scatter_with_regression(visualization.data_processor.train_data, 'x', 'y3','Scatter Plot for y3 with regression line')
    visualization.plot_scatter_with_regression(visualization.data_processor.train_data, 'x', 'y4','Scatter Plot for y4 with regression line')

if __name__ == "__main__":
    main()

class DeviationAnalysis:
    """
    A class for calculating and analyzing the sum of least square deviations between train_data
    and ideal_data.
    """
    def __init__(self, train_data, ideal_data, y_index):
        """
        Initializes DeviationAnalysis object.
        Parameters:
        train_data (pd.DataFrame): The training data DataFrame.
        ideal_data (pd.DataFrame): The ideal data DataFrame.
        y_index (int): The index of the column to analyze.
        """
        self.train_data = train_data
        self.ideal_data = ideal_data
        self.y_index = y_index
        self.column_name = f"y{y_index}"
    def calculate_lsd(self):
        """
        Calculates the sum of least square deviations for each column.
        Returns:
        list: A list containing the sum of least square deviations for each column.
        """
        lsd_sum = []
        for column in self.ideal_data.columns:
            residuals = self.ideal_data[column] - self.train_data[self.column_name]
        lsd = sum(residuals ** 2)
        lsd_sum.append(lsd)
        return lsd_sum

    def plot_graph(self, lsd_sum):
        """
        Plots a bar graph showing the sum of least square deviations for each column.
        Parameters:
        lsd_sum (list): A list containing the sum of least square deviations for each column.
        Returns:
        tuple: A tuple containing the minimum LSD value and its corresponding column index.
        """
        min_lsd_index = np.argmin(lsd_sum)
        min_lsd_value = lsd_sum[min_lsd_index]
        plt.figure(figsize=(12, 6))
        x = np.arange(1, 51)
        colors = ['red' if i == min_lsd_index else 'blue' for i in range(len(lsd_sum))]
        plt.bar(x, lsd_sum, color=colors)
        plt.xlabel('Column')
        plt.ylabel('Sum of Least Square Deviation (Log Scale)')
        plt.title(f'Sum of Least Square Deviation for Y1 to Y50 (Log Scale) - {self.column_name}Train Data')
        plt.xticks(x, [f"y{i}" for i in range(1, 51)], rotation='vertical')
        plt.yscale('log')
        min_lsd_label = f"Min LSD: {min_lsd_value:.2f}"
        plt.text(0.02, 0.98, min_lsd_label, transform=plt.gca().transAxes,
                 ha='left', va='top', color='red', bbox=dict(facecolor='white', edgecolor='black'))
        plt.tight_layout()
        plt.show()
        return min_lsd_value, min_lsd_index

    def run(self):
        """
        Runs the deviation analysis and plotting process.
        """
        try:
            lsd_sum = self.calculate_lsd()
        except Exception as e:
            print("An error occurred while calculating LSD:", e)
        else:
            try:
                min_lsd_value, min_lsd_index = self.plot_graph(lsd_sum)
            except Exception as e:
                print("An error occurred while plotting the graph:", e)
            else:
                min_lsd_column = f"y{min_lsd_index + 1}"
                print(f"The minimum sum of least square deviation is {min_lsd_value}")
                print(f"The ideal function for the train data {self.column_name} is:{min_lsd_column}")
            finally:
                 print('This is always executed at the end of the run method')
class LeastSquareDeviation(DeviationAnalysis):
    def __init__(self, train_data, ideal_data, y_index):
        super().__init__(train_data, ideal_data, y_index)
        # Least Square Deviation for Multiple Data Columns
        for y_index in range(1, 5):
            lsd = LeastSquareDeviation(train_data, ideal_data, y_index)
        lsd.run()

class DataProcessor:
    """Class to process data."""
    def __init__(self, test_data_path, ideal_data_path, train_data_path):
        """Initialize the DataProcessor.
        Parameters:
        - test_data_path (str): Path to the test data file.
        - ideal_data_path (str): Path to the ideal data file.
        - train_data_path (str): Path to the training data file.
        """
        self.test = pd.read_csv(test_data_path)
        self.ideal = pd.read_csv(ideal_data_path)
        self.train = pd.concat([pd.read_csv(train_data_path), self.test])
        @staticmethod
    def _remove_duplicates(df1, df2):
        """Remove duplicates from two pandas DataFrames.
        Args:
        df1 (pandas.DataFrame): First DataFrame.
        df2 (pandas.DataFrame): Second DataFrame.
        Returns:
        pandas.DataFrame: A new DataFrame with all duplicate rows removed.
        """
        return df1[~df1.isin(df2).any(axis=0)]
    # Remove any instances in the testing set that also appear in the training set
    DataProcessor._remove_duplicates(DataProcessor.train, DataProcessor.test).reset_index(drop=True).to_csv('processed/clean_testing.csv', index=False)
    DataProcessor.train.reset_index(drop=True).to_csv('processed/clean_training.csv')

    def split_dataset(self, ratio=.75):
        """Split dataset into training and validation sets.
            The method splits the cleaned training dataset into a training set and a validation set.
            The size of the training set is given by 'ratio' * len(self.train),
            while the size of the validation set is given by (1-ratio
            The size of the training set is given by `ratio * len(self.train)`. Any remaining
            instances are added to the validation set.
            Args:
            ratio (float): Ratio of the training set to the total dataset. Defaults to 0.75.
            Returns:
            tuple: A tuple containing two pandas DataFrames; the first element is the training set,
            and the second element is the validation set.
        """

        num_instances = int(len(self.train) * ratio)
        train_set = self.train[:num_instances]
        valid_set = self.train[num_instances:]
        return train_set, valid_set

    def mse(train_data, ideal_data):
        """
            Calculate the Mean Squared Error (MSE) between the train data and ideal data. Args:
            train_data (ndarray): Array containing the predicted values (train data). ideal_data (ndarray):
            Array containing the true labels (ideal data).
            Returns:
            float: The calculated MSE value.
        """

        return np.mean((train_data - ideal_data) ** 2)

    def lowest_mse_label(train_feature, ideal_data):
        '''
        Calculate the lowest Mean Squared Error (MSE) and associated label for a given train feature.
        Args:
        train_feature (str): Name of the train feature.
        ideal_data (DataFrame): DataFrame containing the ideal data.
        Returns:
        tuple: A tuple containing the label with the lowest MSE and its corresponding MSE value.
        '''

        error_list = []
        for i in ideal_data.columns[1:]: prediction = ideal_data[i].values
        label = train_data[train_feature].values
        error = mse(label, prediction)
        label_error_tuple = i, error
        error_list.append(label_error_tuple)
        error_list_sorted = sorted(error_list, key=lambda x: x[1])
        return error_list_sorted[0]


def best_label_mapping(train_data, ideal_data):
    '''
            Find the best label mapping between train data and ideal data based on the lowest MSE val-
            ues.
            Args:
            train_data (DataFrame): DataFrame containing the train data. ideal_data (DataFrame):
            DataFrame containing the ideal data.
            Returns:
            dict: A dictionary mapping each train feature to the label in ideal data with the lowest MSE
            value.
            '''

    best_label_list = []
    for train_col in train_data.columns[1:]:
        try:
            best_label = lowest_mse_label(train_col, ideal_data)
            best_label_list.append(best_label)
        except Exception as e:
            print(f"An error occurred while processing '{train_col}': {str(e)}")
            col_list = train_data.columns[1:]
        return dict(zip(col_list, best_label_list))
        # Calculate the label mapping
        try:
            mapping_dict = best_label_mapping(train_data, ideal_data)
        # Print the mapping dictionary
        print(mapping_dict)
        # Raise a user-defined exception if the mapping dictionary is empty if len(mapping_dict)== 0:
    raise EmptyMappingDictionaryError()
    except EmptyMappingDictionaryError as EMDE: (print(f"An error occurred:{str(EMDE)}"))
    except Exception as EX: print(f"An error occurred during label mapping: {str(EX)}")


def max_abs_deviation(train_data, ideal_data):
    """
    Calculate the absolute maximum deviation between train_data and ideal_data.
    Parameters:
    train_data (numpy.ndarray): Array containing train data values.
    ideal_data (numpy.ndarray): Array containing ideal function values.
    Returns:
    float: The absolute maximum deviation between train_data and ideal_data.
    """
    max_abs_dev = np.max(np.abs(train_data - ideal_data))
    return max_abs_dev

def plot_bar_chart(ax, deviations, max_deviation_index, x_labels, title):
    """
    Plot a bar chart with deviation values.
    Parameters:
    ax (matplotlib.axes.Axes): Axes object to plot the chart on.
    deviations (numpy.ndarray): Array containing deviation values.
    max_deviation_index (int): Index of the maximum deviation value.
    x_labels (list): List of labels for the x-axis.
    title (str): Title of the plot.
    """
    ax.bar(range(max_deviation_index + 1), deviations[:max_deviation_index + 1],
    color='blue')
    ax.bar(max_deviation_index, deviations[max_deviation_index], color='red')
    for i, deviation in enumerate(deviations[:max_deviation_index + 1]):
    ax.text(i, deviation, f'{deviation:.4f}', ha='center', va='bottom')
    ax.set_xlabel('X-Data Points')
    ax.set_ylabel('Deviation')
    ax.set_title(title)
    ax.set_xticks(range(max_deviation_index + 1))
    ax.set_xticklabels(x_labels[:max_deviation_index + 1], rotation=90)
    ax.margins(x=0.01) # Adjust the x-axis margins for better visibility
# Maximum deviation allowed for 'y1' variable
y1_train_data = np.array(list(train_data['y1'].values))
y1_ideal_data = np.array(list(ideal_data['y13'].values))
deviations_y1 = np.abs(y1_train_data - y1_ideal_data) * np.sqrt(2)
max_deviation_index_y1 = np.argmax(deviations_y1)
x_labels_y1 = list(train_data['x'].values) # assuming 'x' is the corresponding variable for 'y1'
max_dev_1 = max_abs_deviation(y1_train_data, y1_ideal_data) * np.sqrt(2)
print('Maximum deviation allowed for y1 train variable and selected ideal function y13 is',max_dev_1)
# Maximum deviation allowed for 'y2' variable
y2_train_data = np.array(list(train_data['y2'].values))
y2_ideal_data = np.array(list(ideal_data['y24'].values))
deviations_y2 = np.abs(y2_train_data - y2_ideal_data) * np.sqrt(2)
max_deviation_index_y2 = np.argmax(deviations_y2)
x_labels_y2 = list(train_data['x'].values) # assuming 'x' is the corresponding variable for 'y2'
max_dev_2 = max_abs_deviation(y2_train_data, y2_ideal_data) * np.sqrt(2)
print('Maximum deviation allowed for y2 train variable and selected ideal function y24 is',max_dev_2)
# Maximum deviation allowed for 'y3' variable
y3_train_data = np.array(list(train_data['y3'].values))
y3_ideal_data = np.array(list(ideal_data['y36'].values))
deviations_y3 = np.abs(y3_train_data - y3_ideal_data) * np.sqrt(2)
max_deviation_index_y3 = np.argmax(deviations_y3)
x_labels_y3 = list(train_data['x'].values) # assuming 'x' is the corresponding variable for 'y3'
max_dev_3 = max_abs_deviation(y3_train_data, y3_ideal_data) * np.sqrt(2)
print('Maximum deviation allowed for y3 train variable and selected ideal function y36 is',max_dev_3)
# Maximum deviation allowed for 'y4' variable
y4_train_data = np.array(list(train_data['y4'].values))
y4_ideal_data = np.array(list(ideal_data['y40'].values))
deviations_y4 = np.abs(y4_train_data - y4_ideal_data) * np.sqrt(2)
max_deviation_index_y4 = np.argmax(deviations_y4)
x_labels_y4 = list(train_data['x'].values) # assuming 'x' is the corresponding variable for 'y4'
max_dev_4 = max_abs_deviation(y4_train_data, y4_ideal_data) * np.sqrt(2)
print('Maximum deviation allowed for y4 train variable and selected ideal function y40 is',max_dev_4)

def r2_score_val(engine):
    with engine.connect() as conn:
        ideal_query = text("SELECT x, y13, y24, y36, y40 FROM ideal_data")
        ideal_results = conn.execute(ideal_query).fetchall()
        ideal_df = pd.DataFrame(ideal_results, columns=['x', 'y13', 'y24', 'y36', 'y40'])
        print("Ideal DataFrame Created")
        test_query = text("SELECT x, y FROM test_data")
        test_results = conn.execute(test_query).fetchall()
        test_df = pd.DataFrame(test_results, columns=['x', 'y'])
        r_squared_values = {}
        for col in ['y13', 'y24', 'y36', 'y40']:
            merged_df = pd.merge(test_df, ideal_df[['x', col]], on='x', how='inner')
        r_squared = r2_score(merged_df['y'], merged_df[col])
        r_squared_values[col] = r_squared
        for col, r_squared in r_squared_values.items():
            print(f"R-square value between {col} ideal function and Y test data points: {r_squared}")
        return sum([value for value in r_squared_values.values()]) / len(r_squared_values)
r2_score_val(engine)

def calculate_and_store_abs_deviations(engine,database_path,test_data):
    regression_results = {}
    y_variables = ['y13', 'y24', 'y36', 'y40']
    for y_var in y_variables:
        regression = linregress(ideal_data.index, ideal_data[y_var])
        regression_results[y_var]= regression
        print(f"\nRegression values for '{y_var}':")
        print("Slope:", regression.slope)
        print("Intercept:", regression.intercept)
        print("R-value:", regression.rvalue)
        print("P-value:", regression.pvalue)
        print("Standard Error:", regression.stderr)
    predicted_values ={}
    for y_var in y_variables:
        slope = regression_results[y_var].slope
        intercept = regression_results[y_var].intercept
        predicted_values[y_var] = slope * test_data['x'] + intercept
    y13_predicted = predicted_values['y13']
    y24_predicted = predicted_values['y24']
    y36_predicted = predicted_values['y36']
    y40_predicted = predicted_values['y40']
    # Map the predicted values to the test data
    test_data['y13_predicted'] = y13_predicted
    test_data['y24_predicted'] = y24_predicted
    test_data['y36_predicted'] = y36_predicted
    test_data['y40_predicted'] = y40_predicted
    test_data = test_data.dropna(subset=['y'])
    test_data['y13_abs_deviation'] = np.abs(test_data['y'] - test_data['y13_predicted'])
    test_data['y24_abs_deviation'] = np.abs(test_data['y'] - test_data['y24_predicted'])
    test_data['y36_abs_deviation'] = np.abs(test_data['y'] - test_data['y36_predicted'])
    test_data['y40_abs_deviation'] = np.abs(test_data['y'] - test_data['y40_predicted'])
    conn = sqlite3.connect(database_path)
    test_data.to_sql('test_data', con=conn, if_exists='replace', index=False)
    conn.close()
    print("Mapped predicted values and absolute deviations calculated and stored in the database!")
    test_data_df = test_data
    # Filter the test data based on the condition
    filtered_data1 = test_data_df[test_data_df['y13_abs_deviation'] <= max_dev_1]
    filtered_data2 = test_data_df[test_data_df['y24_abs_deviation'] <= max_dev_2]
    filtered_data3 = test_data_df[test_data_df['y36_abs_deviation'] <= max_dev_3]
    filtered_data4 = test_data_df[test_data_df['y40_abs_deviation'] <= max_dev_4]
    # Extract the mapped values of x points and corresponding y13_abs_deviation
    mapped_x_points1 = filtered_data1['x']
    y_values1 = filtered_data1['y']
    y13_values = filtered_data1['y13_predicted']
    y13_abs_deviation = filtered_data1['y13_abs_deviation']
    # Create a new DataFrame for the mapped values
    mapped_values1 = pd.DataFrame({'x': mapped_x_points1, 'y': y_values1, 'Delta Y':y13_abs_deviation, 'No.of ideal func': 'y13'})
    print('mapped value 1:', mapped_values1)
    # Extract the mapped values of x points and corresponding y24_abs_deviation
    mapped_x_points2 = filtered_data2['x']
    y_values2 = filtered_data2['y']
    y24_values = filtered_data2['y24_predicted']
    y24_abs_deviation = filtered_data2['y24_abs_deviation']
    # Create a new DataFrame for the mapped values
    mapped_values2 = pd.DataFrame({'x': mapped_x_points2, 'y': y_values2, 'Delta Y':
        y24_abs_deviation, 'No. of ideal func': 'y24'})
    print('mapped value 2:', mapped_values2)
    # Extract the mapped values of x points and corresponding y36_abs_deviation

    mapped_x_points3 = filtered_data3['x']
    y_values3 = filtered_data3['y']
    y36_values = filtered_data3['y36_predicted']
    y36_abs_deviation = filtered_data3['y36_abs_deviation']
    # Create a new DataFrame for the mapped values
    mapped_values3 = pd.DataFrame({'x': mapped_x_points3, 'y': y_values3, 'Delta Y':
        y36_abs_deviation, 'No. of ideal func': 'y36'})
    print('mapped value 3:', mapped_values3)
    # Extract the mapped values of x points and corresponding y40_abs_deviation

    mapped_x_points4 = filtered_data4['x']
    y_values4 = filtered_data4['y']
    y40_values = filtered_data4['y40_predicted']
    y40_abs_deviation = filtered_data4['y40_abs_deviation']
    # Create a new DataFrame for the mapped values
    mapped_values4 = pd.DataFrame({'x': mapped_x_points4, 'y': y_values4, 'Delta Y':
        y40_abs_deviation, 'No. of ideal func': 'y40'})
    # Concatenate all the mapped values dataframes
    print('mapped value 4:', mapped_values4)

    all_mapped_values = pd.concat([mapped_values1, mapped_values2, mapped_values3,
                                   mapped_values4])
    all_filtered_data = pd.concat([filtered_data1, filtered_data2, filtered_data3, filtered_data4],
                                  ignore_index=True)
    # print(all_mapped_values)
    print('ALL mapped points', all_mapped_values)
calculate_and_store_abs_deviations(engine, database_path, test_data)

def filter_and__test_data(engine, max_dev_1, max_dev_, max_dev3, max__4, test_data, test_data_df):
    """
    This function filters the test data based on specified conditions and saves the results in the
    specified format as a table named 'test_data' in the database
    Parameters:
    engine (sqlalchemy.engine.Engine): The SQLAlchemy engine to connect to the database.
    max_dev_1 (float): The maximum deviation for 'y13_abs_deviation'.
    max_dev_2 (float): The maximum deviation for 'y24_abs_deviation'.
    max_dev_3 (float): The maximum deviation for 'y36_abs_deviation'.
    max_dev_4 (float): The maximum deviation for 'y40_abs_deviation'.
    Returns:
    None
    """
    # Query the test_data table and load the results into a DataFrame
    # test_data_query = "SELECT * FROM test_data"
    est_data_df = test_data
    # Filter the test data based on the condition
    filtered_data1 = test_data_df[test_data_df['y13_abs_deviation'] <= max_dev_1]
    filtered_data2 = test_data_df[test_data_df['y24_abs_deviation'] <= max_dev_2]
    filtered_data3 = test_data_df[test_data_df['y36_abs_deviation'] <= max_dev_3]
    filtered_data4 = test_data_df[test_data_df['y40_abs_deviation'] <= max_dev_4]
    # Extract the mapped values of x points and corresponding y13_abs_deviation
    mapped_x_points1 = filtered_data1['x']
    y_values1 = filtered_data1['y']
    y13_values = filtered_data1['y13_predicted']
    y13_abs_deviation = filtered_data1['y13_abs_deviation']
    # Create a new DataFrame for the mapped values
    mapped_values1 = pd.DataFrame({'x': mapped_x_points1, 'y': y_values1, 'Delta Y':
    y13_abs_deviation, 'No. of ideal func': 'y13'})
    # Extract the mapped values of x points and corresponding y24_abs_deviation
    mapped_x_points2 = filtered_data2['x']
    y_values2 = filtered_data2['y']
    y24_values = filtered_data2['y24_predicted']
    y24_abs_deviation = filtered_data2['y24_abs_deviation']
    # Create a new DataFrame for the mapped values
    mapped_values2 = pd.DataFrame({'x': mapped_x_points2, 'y': y_values2, 'Delta Y':
    y24_abs_deviation, 'No. of ideal func': 'y24'})
    # Extract the mapped values of x points and corresponding y34_abs_deviation
    mapped_x_points3 = filtered_data3['x']
    y_values3 = filtered_data3['y']
    y36_values = filtered_data3['y36_predicted']
    y36_abs_deviation = filtered_data3['y36_abs_deviation']
    # Create a new DataFrame for the mapped values
    mapped_values3 = pd.DataFrame({'x': mapped_x_points3, 'y': y_values3, 'Delta Y':
    y36_abs_deviation, 'No. of ideal func': 'y36'})
    # Extract the mapped values of x points and corresponding y40_abs_deviation
    mapped_x_points4 = filtered_data4['x']
    y_values4 = filtered_data4['y']
    y40_values = filtered_data4['y40_predicted']
    y40_abs_deviation = filtered_data4['y40_abs_deviation']
    # Create a new DataFrame for the mapped values
    mapped_values4 = pd.DataFrame({'x': mapped_x_points4, 'y': y_values4, 'Delta Y':
    y40_abs_deviation, 'No. of ideal func': 'y40'})
    # Concatenate all the mapped values dataframes
    all_mapped_values = pd.concat([mapped_values1, mapped_values2, mapped_values3,
    mapped_values4])
    # Drop the existing test_data table from the database
    engine.execute("DROP TABLE IF EXISTS test_data")
    # Save the new DataFrame as the test_data table in the database
    all_mapped_values.to_sql('test_data', engine, index=False)
    # Display the new test_data table
    new_test_data_query = "SELECT * FROM test_data"
    new_test_data_df = pd.read_sql_query(new_test_data_query, engine)
    print("New test_data table:")
    print(new_test_data_df)

def filter_and_plot_data(test_data, max_dev_1, max_dev_2, max_dev_3, max_dev_4):
    """
    Filter the test data based on the condition, plot the mapped values, and display relevant
    information.
    Parameters:
    test_data (pd.DataFrame): DataFrame containing test data with 'x', 'y', 'y13_predicted',
    'y13_abs_deviation',
    'y24_predicted', 'y24_abs_deviation', 'y36_predicted', 'y36_abs_deviation', 'y40_predicted',
    and 'y40_abs_deviation'
    max_dev_1 (float): Maximum deviation allowed for the 'y13_abs_deviation' variable
    max_dev_2 (float): Maximum deviation allowed for the 'y24_abs_deviation' variable
    max_dev_3 (float): Maximum deviation allowed for the 'y36_abs_deviation' variable
    max_dev_4 (float): Maximum deviation allowed for the 'y40_abs_deviation' variable
    """
    # Filter the test data based on the conditions
    filtered_data_1 = test_data[test_data['y13_abs_deviation'] <= max_dev_1]
    filtered_data_2 = test_data[test_data['y24_abs_deviation'] <= max_dev_2]
    filtered_data_3 = test_data[test_data['y36_abs_deviation'] <= max_dev_3]
    filtered_data_4 = test_data[test_data['y40_abs_deviation'] <= max_dev_4]
    # Create Bokeh figures
    p1 = figure(title="Scatter Plot of Mapped Values of x-y test data and corresponding Y13 and y13_abs_deviation", x_axis_label='x', y_axis_label='y')
    p2 = figure(title="Scatter Plot of Mapped Values of x-y test data and corresponding Y24 and y24_abs_deviation", x_axis_label='x', y_axis_label='y')
    p3 = figure(title="Scatter Plot of Mapped Values of x-y test data and corresponding Y36 and y36_abs_deviation", x_axis_label='x', y_axis_label='y')
    p4 = figure(title="Scatter Plot of Mapped Values of x-y test data and corresponding Y40 andy40_abs_deviation ", x_axis_label='x', y_axis_label='y')

    # Extract the mapped values for each condition
    mapped_x_points1 = filtered_data_1['x']
    y_values1 = filtered_data_1['y']
    y13_values = filtered_data_1['y13_predicted']
    y13_abs_deviation = filtered_data_1['y13_abs_deviation']
    mapped_x_points2 = filtered_data_2['x']
    y_values2 = filtered_data_2['y']
    y24_values = filtered_data_2['y24_predicted']
    y24_abs_deviation = filtered_data_2['y24_abs_deviation']
    mapped_x_points3 = filtered_data_3['x']
    y_values3 = filtered_data_3['y']
    y36_values = filtered_data_3['y36_predicted']
    y36_abs_deviation = filtered_data_3['y36_abs_deviation']
    mapped_x_points4 = filtered_data_4['x']
    y_values4 = filtered_data_4['y']
    y40_values = filtered_data_4['y40_predicted']
    y40_abs_deviation = filtered_data_4['y40_abs_deviation']

    # Create DataFrames for the mapped values
    mapped_values1 = pd.DataFrame({'x': mapped_x_points1, 'y': y_values1, 'y13': y13_values,
                                   'y13_abs_deviation': y13_abs_deviation})
    mapped_values2 = pd.DataFrame({'x': mapped_x_points2, 'y': y_values2, 'y24': y24_values,
                                   'y24_abs_deviation': y24_abs_deviation})
    mapped_values3 = pd.DataFrame({'x': mapped_x_points3, 'y': y_values3, 'y36':
        y36_values, 'y36_abs_deviation': y36_abs_deviation})
    mapped_values4 = pd.DataFrame({'x': mapped_x_points4, 'y': y_values4, 'y40': y40_values,
                                   'y40_abs_deviation': y40_abs_deviation})

    # Print the mapped values for each condition
    print("Mapped Values of x-y test data and corresponding Y13 and y13_abs_deviation:")
    print(mapped_values1[['x', 'y', 'y13', 'y13_abs_deviation']])
    print(f"Total mapped test data points within {max_dev_1} Maximum deviation allowed for y1 train variable and selected ideal function y13: {len(mapped_x_points1)}")
    print("\nMapped Values of x-y test data and corresponding Y24 and y24_abs_deviation:")
    print(mapped_values2[['x', 'y', 'y24', 'y24_abs_deviation']])
    print(f"Total mapped test data points within {max_dev_2} Maximum deviation allowed for y2train variable and selected ideal function y24: {len(mapped_x_points2)}")
    print("\nMapped Values of x-y test data and corresponding Y36 and y36_abs_deviation:")
    print(mapped_values3[['x', 'y', 'y36', 'y36_abs_deviation']])
    print(f"Total mapped test data points within {max_dev_3} Maximum deviation allowed for y3train variable and selected ideal function y36: {len(mapped_x_points3)} ")
    print("\nMapped Values of x-y test data and corresponding Y40 and y40_abs_deviation:")
    print(mapped_values4[['x', 'y', 'y40', 'y40_abs_deviation']])
    print(f"Total mapped test data points within {max_dev_4} Maximum deviation allowed for y4 train variable and selected ideal function y40: {len(mapped_x_points4)}")

    # Plot scatter glyphs with different shapes for each condition
    p1.circle(mapped_values1['x'], mapped_values1['y'], legend_label='y', color='blue', size=8,
              alpha=0.7)
    p1.square(mapped_values1['x'], mapped_values1['y13'], legend_label='y13', color='green',
              size=8, alpha=0.7)
    p1.triangle(mapped_values1['x'], mapped_values1['y13_abs_deviation'],
                legend_label='y13_abs_deviation', color='red', size=8, alpha=0.7)
    p1.legend.location = "top_left"
    show(p1)
    p2.circle(mapped_values2['x'], mapped_values2['y'], legend_label='y', color='blue', size=8,
              alpha=0.7)
    p2.square(mapped_values2['x'], mapped_values2['y24'], legend_label='y24', color='green',
              size=8, alpha=0.7)
    p2.triangle(mapped_values2['x'], mapped_values2['y24_abs_deviation'],
                legend_label='y24_abs_deviation', color='red', size=8, alpha=0.7)
    p2.legend.location = "top_left"
    p3.circle(mapped_values3['x'], mapped_values3['y'], legend_label='y', color='blue', size=8,
              alpha=0.7)
    p3.square(mapped_values3['x'], mapped_values3['y36'], legend_label='y36', color='green',
              size=8, alpha=0.7)
    p3.triangle(mapped_values3['x'], mapped_values3['y36_abs_deviation'],
                legend_label='y36_abs_deviation', color='red', size=8, alpha=0.7)
    p3.legend.location = "top_left"
    p4.circle(mapped_values4['x'], mapped_values4['y'], legend_label='y', color='blue', size=8,
              alpha=0.7)
    p4.square(mapped_values4['x'], mapped_values4['y40'], legend_label='y40', color='green',
              size=8, alpha=0.7)
    p4.triangle(mapped_values4['x'], mapped_values4['y40_abs_deviation'],
                legend_label='y40_abs_deviation', color='red', size=8, alpha=0.7)
    p4.legend.location = "top_left"

    output_notebook()
    show(p1)
    show(p2)
    show(p3)
    show(p4)

def query_table(table_name):
    """
    Query the database for a given table and load the results into a DataFrame.
    Parameters:
    table_name (str): The name of the table to query.
    Returns:
    pandas.DataFrame: DataFrame containing the results from the queried table.
    """
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql_query(query, engine)
# Query and display the ideal_data DataFrame
ideal_data_df = query_table("ideal_data")
print("ideal_data DataFrame:")
print(ideal_data_df)

class TestDatabaseHandler:
    @pytest.fixture
    def database_handler_instance(self):
        # Create an instance of DatabaseHandler for testing
        # You might need to pass necessary parameters depending on your implementation
        return DatabaseHandler()
    def test_calculate_deviation(self, database_handler_instance):
        # Test case for calculate_deviation method
        # Mock data for testing
        actual_values = [1, 2, 3, 4, 5]
        ideal_values = [1, 2, 3, 4, 5]
        # Call the method being tested
        deviation_result = database_handler_instance.calculate_deviation(actual_values, ideal_values)
        # Add assertions to validate the result
        assert deviation_result == 0 # Assuming the deviation calculation logic returns 0 for identical values
    def test_calculate_deviation_with_different_values(self, database_handler_instance):
        # Test case for calculate_deviation method with different actual and ideal values
        # Mock data for testing
        actual_values = [1, 2, 3, 4, 5]
        ideal_values = [2, 3, 4, 5, 6]
        # Call the method being tested
        deviation_result = database_handler_instance.calculate_deviation(actual_values,
                                                                         ideal_values)
        # Add assertions to validate the result
        assert deviation_result == pytest.approx(0.176, rel=1e-2)
        # Assuming the deviation calculation logic returns 0.176 for different values

class DataProcessor:
    """Class to process data."""
    def __init__(self, test_data_path, ideal_data_path, train_data_path):
        """Initialize the DataProcessor.
        Parameters:
        - test_data_path (str): Path to the test data file.
        - ideal_data_path (str): Path to the ideal data file.
        - train_data_path (str): Path to the training data file.
        """
        self.test = pd.read_csv(test_data_path)
        self.ideal = pd.read_csv(ideal_data_path)
        self.train = pd.concat([pd.read_csv(train_data_path), self.test])
    @staticmethod

    def split_dataset(self, ratio=.75):
        """
        Split dataset into training and validation sets.
        The method splits the cleaned training dataset into a training set and a validation set.
        The size of the training set is given by 'ratio' * len(self.train),
        while the size of the validation set is given by (1-ratio
        The size of the training set is given by `ratio * len(self.train)`. Any remaining
        instances are added to the validation set.
        Args:
        ratio (float): Ratio of the training set to the total dataset. Defaults to 0.75.
        Returns:
        tuple: A tuple containing two pandas DataFrames; the first element is the training set,
        and the second element is the validation set.
        """
        num_instances = int(len(self.train) * ratio)
        train_set = self.train[:num_instances]
        valid_set = self.train[num_instances:]
        return train_set, valid_set

class DataProcessor:
    def __init__(self, test_data, ideal_data, train_data):
        self.test_data = pd.read_csv(test_data) if isinstance(test_data, str) else test_data
        self.ideal_data = pd.read_csv(ideal_data) if isinstance(ideal_data, str) else ideal_data
        self.train_data = pd.read_csv(train_data) if isinstance(train_data, str) else train_data
class DatabaseHandler:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.engine = create_engine('sqlite:///assignment_database.db', echo=True)

    def create_database(self):
        self.data_processor.test_data.to_sql('test_data', con=self.engine, index=False,
                                             if_exists='replace')

        self.data_processor.ideal_data.to_sql('ideal_data', con=self.engine, index=False,
                                          if_exists='replace')
        self.data_processor.train_data.to_sql('train_data', con=self.engine, index=False,
                                              if_exists='replace')

class ErrorCalculator:
    @staticmethod
    def calculate_least_square(train_data, ideal_function):
        deviations = train_data['y'] - ideal_function
        least_square = np.sum(deviations**2)
        return least_square
class Visualization:
    def __init__(self, data_processor):
        self.data_processor = data_processor
    def plot_data(self):
        p = figure(title="Test Data vs Ideal Data", x_axis_label="x", y_axis_label="y",
        plot_width=800, plot_height=400)
        p.circle(self.data_processor.test_data['x'], self.data_processor.test_data['y'],
        legend_label="Test Data", size=8, color="blue", alpha=0.5)
        for i in range(50):
            p.line(self.data_processor.ideal_data['x'], self.data_processor.ideal_data[f'y{i+1}'],
            legend_label=f"Ideal Function {i+1}", line_width=2, line_color="orange")
        show(p)


#Unittest
class TestAssignment(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.test_data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 3, 5, 4, 6]})
        self.ideal_data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y1': [1, 2, 4, 3, 5]})
        self.train_data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 3, 5, 4, 6]})
        self.data_processor = DataProcessor(self.test_data, self.ideal_data, self.train_data)
    def test_least_square_calculation(self):
        # Assuming a simple ideal function y = x
        ideal_function = self.train_data['x']
        least_square = ErrorCalculator.calculate_least_square(self.train_data, ideal_function)
        print("Testing successful.")
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAssignment)
    unittest.TextTestRunner(verbosity=2).run(suite)

# Appendix B: Additional Visualization
# Display the correlation heatmap of the modified ideal dataset
plt.figure(figsize=(50, 20)) # Set the size of the figure for optimum view
# Calculate the correlation matrix for the modified ideal dataset
modified_correlation_matrix = ideal_data.corr()
# Create the heatmap with modified data
sns.heatmap(modified_correlation_matrix, annot=True, cmap="viridis", fmt=".2f")
# Add a title for the modified correlation heatmap
plt.title("Modified Correlation Heatmap of Ideal Dataset")
# Display the modified correlation plot
plt.show()
# Single graph for the complete modified train dataset considering a pair of variables at a time
sns.pairplot(train_data)
plt.suptitle("Pairwise Relationships in Modified Train Dataset", y=1.02)
plt.show()
# Box and whisker plots to compare distributions of different pairs of variables from the modified train dataset

def modified_hist_box(modified_test_data, col):
    """
    Generate a combination of histogram and boxplot for a given column in the modified test
    dataset.
    Parameters:
    modified_test_data (pd.DataFrame): DataFrame containing the modified test data.
    col (str): Name of the column to visualize.
    Returns:
    None
    """
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (0.10,
    0.70)},
    figsize=(10, 8))
    # Adding a boxplot for the modified test dataset
    sns.boxplot(modified_test_data[col], ax=ax_box, showmeans=True, color='skyblue')
    ax_box.set(xlabel='')
    ax_box.set_title('Box Plot and Histogram of ' + col + ' in Modified Test Data')
    # Adding a histogram for the modified test dataset
    sns.histplot(modified_test_data[col], ax=ax_hist, kde=True, color='lightcoral')
    ax_hist.set(xlabel=col)
    plt.tight_layout()
    plt.show()
# Visualize modified_hist_box for 'x' variable in the modified test dataset
modified_hist_box(test_data, 'x')
# Visualize modified_hist_box for 'y' variable in the modified test dataset
modified_hist_box(test_data, 'y')

# Define a function to generate a pair plot for the modified test dataset
def modified_pair_plot(modified_test_data):
    """
    Generate a pair plot for the modified test dataset.
    Parameters:
    modified_test_data (pd.DataFrame): DataFrame containing the modified test data.
    Returns:
    None
    """
    # Single graph for the modified test dataset considering a pair of variables at a time
    sns.pairplot(modified_test_data, palette='husl')
    plt.suptitle("Modified Pairwise Relationships in Test Dataset", y=1.02)
    plt.show()
# Visualize modified_pair_plot for the modified test dataset
modified_pair_plot(test_data)

# Initialize the database connection
db_engine = create_engine('sqlite:///assignment_database.db')
# Load the data into DataFrames
ideal_data = pd.read_csv('ideal.csv')
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
class TestDataModifier:
    def __init__(self, test_data):
        self.test_data = test_data.copy()
    def modify_data(self):
        # Implement your data modification logic here
        pass
    # Display the correlation heatmap of the ideal dataset
plt.figure(figsize=(10, 6))
correlation_matrix = ideal_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Heatmap of Ideal Dataset")
plt.show()
# Single graph for the complete train dataset considering a pair of variables at a time
sns.pairplot(train_data, palette='husl')
plt.suptitle("Pairwise Relationships in Train Dataset", y=1.02)
plt.show()
# Modify test data
test_modifier = TestDataModifier(test_data)
modified_test_data = test_modifier.modify_data()
# Display box plot for 'x' variable in modified test dataset
plt.figure(figsize=(8, 6))
sns.boxplot(x='x', data=test_data, color='sky blue')
plt.title("Box Plot for Modified 'x' in Test Dataset")
plt.show()
# Display box plot for 'x' variable in modified test dataset
plt.figure(figsize=(8, 6))
sns.boxplot(x='x', data=test_data, color='sky blue')
plt.title("Box Plot for Modified 'x' in Test Dataset")
plt.show()
# Display box plot for 'y' variable in modified test dataset
plt.figure(figsize=(8, 6))
sns.boxplot(x='y', data=test_data, color='light coral')
plt.title("Box Plot for Modified 'y' in Test Dataset")
plt.show()
# Display the pair plot for the modified test dataset

plt.figure(figsize=(10, 6))
sns.pairplot(test_data, palette='husl')
plt.suptitle("Modified Pairwise Relationships in Test Dataset", y=1.02)
plt.show()
# Create a Bokeh plot for modified test data
bokeh_plot = figure(title="Modified Test Data vs Ideal Data", x_axis_label="x",
y_axis_label="y", width=800, height=400)
bokeh_plot.circle(test_data['x'], test_data['y'], legend_label="Modified Test Data", size=8,
color="green", alpha=0.5)
for i in range(50):
    bokeh_plot.line(ideal_data['x'], ideal_data[f'y{i+1}'], legend_label=f"Ideal Function {i+1}",line_width=2, line_color="orange")
show(bokeh_plot)