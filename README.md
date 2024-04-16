# EDA-using-Python

Real-world Scenario: Imagine a company that builds electronics. As a data scientist, your job is to predict when parts might fail by analyzing how well they perform. You have data from experiments on these parts (training data) and information on how they should ideally function (ideal data).

Data Breakdown:

    Training data (sets A): Information on how parts performed under different conditions (like temperature or manufacturing process).
    Test data (set B): Separate data used to test how well predictions work on unseen situations.
    Ideal data (set C): Information on how parts should perfectly function.

Our Goal: We want to create a Python program that:

    Analyzes training data to find the 4 best matches from 50 ideal functions.
    Uses these chosen functions to analyze the test data.
    Checks if each test data point (x, y) can be described by one of the chosen functions within an acceptable deviation range.
    If a match is found, map the data point to the chosen function and record the deviation.

Overall Objective: Build a reliable Python program that selects the best ideal functions based on training data and then analyzes the test data using those functions, considering deviations from the ideal.


Contact:
Author: www.linkedin.com/in/samyak-anand-496a1143