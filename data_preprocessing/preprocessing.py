import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    This class is used to clean and transform the data before training.
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self, data, columns):
        """
        Removes the given columns from a pandas dataframe.
        Output: A pandas DataFrame after removing the specified columns.
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.data = data
        self.columns = columns
        try:
            # Drop the labels specified in the columns
            self.useful_data = self.data.drop(labels=self.columns, axis=1)
            self.logger_object.log(self.file_object,
                                   'Column removal Successful. Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in remove_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
        Separates the features and a label column.
        Output: Returns two separate DataFrames, one containing features and the other containing labels.
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            # Drop the label column to get features
            self.X = data.drop(labels=label_column_name, axis=1)
            # Get the label column
            self.Y = data[label_column_name]
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X, self.Y
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def dropUnnecessaryColumns(self, data, columnNameList):
        """
        Drops the unwanted columns as discussed in EDA section.
        """
        data = data.drop(columnNameList, axis=1)
        return data

    def replaceInvalidValuesWithNull(self, data):
        """
        Replaces invalid values i.e. '?' with null, as discussed in EDA.
        """
        for column in data.columns:
            count = data[column][data[column] == '?'].count()
            if count != 0:
                data[column] = data[column].replace('?', np.nan)
        return data

    def is_null_present(self, data):
        """
        Checks whether there are null values present in the pandas DataFrame or not.
        Output: Returns True if null values are present in the DataFrame, False if not, and
        returns the list of columns for which null values are present.
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        self.cols_with_missing_values = []
        self.cols = data.columns
        try:
            # Check for the count of null values per column
            self.null_counts = data.isna().sum()
            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    self.null_present = True
                    self.cols_with_missing_values.append(self.cols[i])
            if self.null_present:
                # Write the logs to see which columns have null values
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = data.columns
                self.dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                self.dataframe_with_null.to_csv('preprocessing_data/null_values.csv')  # Store null column info to file
            self.logger_object.log(self.file_object, 'Finding missing values is a success. Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def encodeCategoricalValues(self, data):
        """
        Encodes all the categorical values in the training set.
        Output: A DataFrame which has all the categorical values encoded.
        On Failure: Raise Exception
        """
        # Map class column to numeric values
        data["class"] = data["class"].map({'p': 1, 'e': 2})

        # One-hot encode all other columns except 'class'
        for column in data.drop(['class'], axis=1).columns:
            data = pd.get_dummies(data, columns=[column])

        return data

    def encodeCategoricalValuesPrediction(self, data):
        """
        Encodes all the categorical values in the prediction set.
        Output: A DataFrame which has all the categorical values encoded.
        On Failure: Raise Exception
        """
        # One-hot encode all columns
        for column in data.columns:
            data = pd.get_dummies(data, columns=[column])

        return data

    # def handleImbalanceDataset(self, X, Y):
    #     """
    #     Handles the imbalance in the dataset by oversampling.
    #     Output: A DataFrame which is balanced now.
    #     On Failure: Raise Exception
    #     """
    #     rdsmple = RandomOverSampler()
    #     x_sampled, y_sampled = rdsmple.fit_sample(X, Y)
    #     return x_sampled, y_sampled

    def standardScalingData(self, X):
        """
        Applies standard scaling to the data.
        Output: Scaled feature array.
        """
        scalar = StandardScaler()
        X_scaled = scalar.fit_transform(X)
        return X_scaled

    def logTransformation(self, X):
        """
        Applies log transformation to all columns in the DataFrame.
        Output: Transformed DataFrame.
        """
        for column in X.columns:
            X[column] += 1
            X[column] = np.log(X[column])
        return X

    def impute_missing_values(self, data):
        """
        Replaces all the missing values in the DataFrame using KNN Imputer.
        Output: A DataFrame which has all the missing values imputed.
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data = data
        try:
            # Impute the missing values using KNNImputer
            imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
            self.new_array = imputer.fit_transform(self.data)
            # Convert the ndarray returned to a DataFrame
            self.new_data = pd.DataFrame(data=(self.new_array), columns=self.data.columns)
            self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    def get_columns_with_zero_std_deviation(self, data):
        """
        Finds out the columns which have a standard deviation of zero.
        Output: List of the columns with standard deviation of zero.
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        self.columns = data.columns
        self.data_n = data.describe()
        self.col_to_drop = []
        try:
            # Check if standard deviation is zero for each column
            for x in self.columns:
                if (self.data_n[x]['std'] == 0):
                    self.col_to_drop.append(x)
            self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            return self.col_to_drop
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()
