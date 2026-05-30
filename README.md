# AI & ML Internship - Task 1

## Data Cleaning and Preprocessing on Titanic Dataset

### Objective

The objective of this task is to clean and preprocess raw data to make it suitable for machine learning applications.

### Dataset

Titanic Dataset

### Tasks Performed

#### 1. Dataset Exploration

* Loaded the Titanic dataset using Pandas.
* Analyzed dataset structure, shape, and data types.
* Identified missing values in different columns.

#### 2. Missing Value Handling

* Filled missing values in the Age column using the median value.
* Filled missing values in the Embarked column using the mode value.
* Removed the Cabin column due to a large number of missing values.

#### 3. Categorical Feature Encoding

* Converted categorical features such as Sex and Embarked into numerical values using Label Encoding.

#### 4. Feature Scaling

* Standardized numerical features such as Age and Fare using StandardScaler.

#### 5. Outlier Detection and Removal

* Visualized outliers using Boxplots.
* Removed outliers using the Interquartile Range (IQR) method.

### Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn

### Project Files

* task1.py
* Titanic-Dataset.csv
* Cleaned_titanic_Dataset.csv
* requirements.txt
* missing_values.png
* boxplot_before.png
* boxplot_after.png

### Results

* Successfully cleaned and preprocessed the dataset.
* Generated a cleaned dataset ready for machine learning applications.
* Visualized missing values and outliers using plots.

### Outcome

This project demonstrates essential data preprocessing techniques including missing value treatment, feature encoding, feature scaling, and outlier handling, which are important steps in the machine learning workflow.
