from datetime import datetime
from Training_Raw_data_validation.rawValidation import Raw_Data_validation
from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
from DataTransform_Training.DataTransformation import dataTransform
from application_logging import logger

class train_validation:
    """
    Train_validation class is responsible for orchestrating the end-to-end validation and insertion process for training data files. 
    Attributes:
        raw_data (Raw_Data_validation): Handles validation of raw data files.
        dataTransform (dataTransform): Performs data transformation operations.
        dBOperation (dBOperation): Manages database operations such as table creation and data insertion.
        file_object (file): File object for logging.
        log_writer (logger.App_Logger): Logger for writing logs.
    Methods:
        __init__(path):
            Initializes the train_validation class with the provided data path and sets up required components.
        train_validation():
            Executes the complete validation workflow for training data files, including:
                - Logging the start of validation.
                - Extracting schema details (date/time stamp lengths, column names, number of columns).
                - Validating file names using regex and schema information.
                - Validating the number of columns in each file.
                - Checking for columns with all missing values.
                - Creating the training database and tables as per schema.
                - Inserting validated data into the database.
                - Deleting the Good_Data folder after successful insertion.
                - Archiving bad files and deleting the Bad_Data folder.
                - Exporting the data from the database table to a CSV file.
                - Logging each step of the process.
            Raises:
                Exception: Propagates any exceptions encountered during the process.
    """
    

    def __init__(self, path):
        # Initialize raw data validator, data transformer, DB operation, logger, and log file
        self.raw_data = Raw_Data_validation(path)
        self.dataTransform = dataTransform()
        self.dBOperation = dBOperation()
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()

    def train_validation(self):
        """
        This method performs validation on training data files, 
        creates database and tables, inserts data, and exports data to CSV.
        """
        try:
            # Log the start of validation
            self.log_writer.log(self.file_object, 'Start of Validation on files for prediction!!')

            # Extract schema values: date/time stamp lengths, column names, and number of columns
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.raw_data.valuesFromSchema()

            # Create regex for filename validation
            regex = self.raw_data.manualRegexCreation()

            # Validate filenames using regex and schema info
            self.raw_data.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)

            # Validate number of columns in files
            self.raw_data.validateColumnLength(noofcolumns)

            # Check for columns with all missing values
            self.raw_data.validateMissingValuesInWholeColumn()
            self.log_writer.log(self.file_object, "Raw Data Validation Complete!!")

            # Log database and table creation
            self.log_writer.log(self.file_object, "Creating Training_Database and tables on the basis of given schema!!!")

            # Create database and table as per schema
            self.dBOperation.createTableDb('Training', column_names)
            self.log_writer.log(self.file_object, "Table creation Completed!!")

            # Log start of data insertion
            self.log_writer.log(self.file_object, "Insertion of Data into Table started!!!!")

            # Insert validated data into the table
            self.dBOperation.insertIntoTableGoodData('Training')
            self.log_writer.log(self.file_object, "Insertion in Table completed!!!")

            # Delete Good Data folder after insertion
            self.log_writer.log(self.file_object, "Deleting Good Data Folder!!!")
            self.raw_data.deleteExistingGoodDataTrainingFolder()
            self.log_writer.log(self.file_object, "Good_Data folder deleted!!!")

            # Move bad files to archive and delete Bad_Data folder
            self.log_writer.log(self.file_object, "Moving bad files to Archive and deleting Bad_Data folder!!!")
            self.raw_data.moveBadFilesToArchiveBad()
            self.log_writer.log(self.file_object, "Bad files moved to archive!! Bad folder Deleted!!")

            # Log completion of validation operation
            self.log_writer.log(self.file_object, "Validation Operation completed!!")

            # Export data from table to CSV file
            self.log_writer.log(self.file_object, "Extracting csv file from table")
            self.dBOperation.selectingDatafromtableintocsv('Training')

            # Close the log file
            self.file_object.close()

        except Exception as e:
            # Raise any exceptions encountered
            raise e
