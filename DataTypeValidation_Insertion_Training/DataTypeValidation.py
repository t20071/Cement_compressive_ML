import shutil
import sqlite3
from datetime import datetime
from os import listdir
import os
import csv
from application_logging.logger import App_Logger

class dBOperation:
    """
    This class shall be used for handling all the SQL operations.
    """

    def __init__(self):
        # Path to store the training database
        self.path = 'Training_Database/'
        # Path for bad raw files
        self.badFilePath = "Training_Raw_files_validated/Bad_Raw"
        # Path for good raw files
        self.goodFilePath = "Training_Raw_files_validated/Good_Raw"
        # Logger instance for logging
        self.logger = App_Logger()

    def dataBaseConnection(self, DatabaseName):
        """
        Method Name: dataBaseConnection
        Description: This method creates the database with the given name and if Database already exists then opens the connection to the DB.
        Output: Connection to the DB
        On Failure: Raise ConnectionError
        """
        try:
            # Create or connect to the database
            conn = sqlite3.connect(self.path + DatabaseName + '.db')
            file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Opened %s database successfully" % DatabaseName)
            file.close()
        except ConnectionError:
            # Log connection error
            file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Error while connecting to database: %s" % ConnectionError)
            file.close()
            raise ConnectionError
        return conn

    def createTableDb(self, DatabaseName, column_names):
        """
        Method Name: createTableDb
        Description: This method creates a table in the given database which will be used to insert the Good data after raw data validation.
        Output: None
        On Failure: Raise Exception
        """
        try:
            # Establish database connection
            conn = self.dataBaseConnection(DatabaseName)
            c = conn.cursor()

            # Check if the table 'Good_Raw_Data' already exists
            c.execute("SELECT count(name) FROM sqlite_master WHERE type = 'table' AND name = 'Good_Raw_Data'")
            if c.fetchone()[0] == 1:
                # Table exists, log and close connection
                conn.close()
                file = open("Training_Logs/DbTableCreateLog.txt", 'a+')
                self.logger.log(file, "Tables created successfully!!")
                file.close()

                file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
                self.logger.log(file, "Closed %s database successfully" % DatabaseName)
                file.close()
            else:
                # Table does not exist, create table or add columns
                for key in column_names.keys():
                    type = column_names[key]
                    try:
                        # Try to add column if table exists
                        conn.execute('ALTER TABLE Good_Raw_Data ADD COLUMN "{column_name}" {dataType}'.format(column_name=key, dataType=type))
                    except:
                        # If table does not exist, create it with the first column
                        conn.execute('CREATE TABLE Good_Raw_Data ({column_name} {dataType})'.format(column_name=key, dataType=type))

                # Close connection after table creation
                conn.close()

                # Log table creation
                file = open("Training_Logs/DbTableCreateLog.txt", 'a+')
                self.logger.log(file, "Tables created successfully!!")
                file.close()

                file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
                self.logger.log(file, "Closed %s database successfully" % DatabaseName)
                file.close()

        except Exception as e:
            # Log error and close connection
            file = open("Training_Logs/DbTableCreateLog.txt", 'a+')
            self.logger.log(file, "Error while creating table: %s " % e)
            file.close()
            conn.close()
            file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Closed %s database successfully" % DatabaseName)
            file.close()
            raise e

    def insertIntoTableGoodData(self, Database):
        """
        Method Name: insertIntoTableGoodData
        Description: This method inserts the Good data files from the Good_Raw folder into the
                     above created table.
        Output: None
        On Failure: Raise Exception
        """
        # Establish database connection
        conn = self.dataBaseConnection(Database)
        goodFilePath = self.goodFilePath
        badFilePath = self.badFilePath
        onlyfiles = [f for f in listdir(goodFilePath)]
        log_file = open("Training_Logs/DbInsertLog.txt", 'a+')

        for file in onlyfiles:
            try:
                # Open each good data file
                with open(goodFilePath + '/' + file, "r") as f:
                    next(f)  # Skip header
                    reader = csv.reader(f, delimiter="\n")
                    for line in enumerate(reader):
                        for list_ in (line[1]):
                            try:
                                # Insert each row into the table
                                conn.execute('INSERT INTO Good_Raw_Data values ({values})'.format(values=(list_)))
                                self.logger.log(log_file, " %s: File loaded successfully!!" % file)
                                conn.commit()
                            except Exception as e:
                                raise e

            except Exception as e:
                # If error occurs, rollback and move file to bad folder
                conn.rollback()
                self.logger.log(log_file, "Error while creating table: %s " % e)
                shutil.move(goodFilePath + '/' + file, badFilePath)
                self.logger.log(log_file, "File Moved Successfully %s" % file)
                log_file.close()
                conn.close()

        conn.close()
        log_file.close()

    def selectingDatafromtableintocsv(self, Database):
        """
        Method Name: selectingDatafromtableintocsv
        Description: This method exports the data in GoodData table as a CSV file in a given location.
        Output: None
        On Failure: Raise Exception
        """
        # Set output directory and file name
        self.fileFromDb = 'Training_FileFromDB/'
        self.fileName = 'InputFile.csv'
        log_file = open("Training_Logs/ExportToCsv.txt", 'a+')
        try:
            # Connect to database and fetch data
            conn = self.dataBaseConnection(Database)
            sqlSelect = "SELECT *  FROM Good_Raw_Data"
            cursor = conn.cursor()

            cursor.execute(sqlSelect)

            results = cursor.fetchall()
            # Get the headers of the csv file
            headers = [i[0] for i in cursor.description]

            # Make the CSV output directory if it doesn't exist
            if not os.path.isdir(self.fileFromDb):
                os.makedirs(self.fileFromDb)

            # Open CSV file for writing
            csvFile = csv.writer(open(self.fileFromDb + self.fileName, 'w', newline=''), delimiter=',', lineterminator='\r\n', quoting=csv.QUOTE_ALL, escapechar='\\')

            # Add the headers and data to the CSV file
            csvFile.writerow(headers)
            csvFile.writerows(results)

            self.logger.log(log_file, "File exported successfully!!!")
            log_file.close()

        except Exception as e:
            # Log export failure
            self.logger.log(log_file, "File exporting failed. Error : %s" % e)
            log_file.close()
