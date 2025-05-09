import pandas as pd

class Data_Getter:
    """
    This class is used for obtaining the data from the source for training.
    """

    def __init__(self, file_object, logger_object):
        # Path to the training data file
        self.training_file = 'Training_FileFromDB/InputFile.csv'
        # File object for logging
        self.file_object = file_object
        # Logger object for logging events
        self.logger_object = logger_object

    def get_data(self):
        """
        Reads the data from the source CSV file.

        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.

        Raises:
            Exception: If data loading fails.
        """
        self.logger_object.log(self.file_object, 'Entered the get_data method of the Data_Getter class')
        try:
            # Reading the data file into a pandas DataFrame
            self.data = pd.read_csv(self.training_file)
            self.logger_object.log(self.file_object, 'Data Load Successful. Exited the get_data method of the Data_Getter class')
            return self.data
        except Exception as e:
            # Logging the exception if data loading fails
            self.logger_object.log(self.file_object, 'Exception occurred in get_data method of the Data_Getter class. Exception message: ' + str(e))
            self.logger_object.log(self.file_object, 'Data Load Unsuccessful. Exited the get_data method of the Data_Getter class')
            raise Exception()
