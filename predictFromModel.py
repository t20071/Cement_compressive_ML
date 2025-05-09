# import necessary libraries
import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation

class prediction:
    """
    The `prediction` class handles the end-to-end process of making predictions using a pre-trained machine learning model for cement strength regression.
    """

    def __init__(self, path):
        # Open log file for appending prediction logs
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        # Initialize logger
        self.log_writer = logger.App_Logger()
        # Initialize prediction data validation
        self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):
        """
        Executes the prediction pipeline:
            - Deletes previous prediction output
            - Loads and validates input data
            - Handles missing values and applies log transformation
            - Scales the data
            - Loads KMeans clustering model and assigns clusters
            - For each cluster, loads the corresponding regression model and makes predictions
            - Aggregates predictions and saves them to a CSV file
            - Logs the process
        Returns:
            str: Path to the CSV file containing prediction results.
        """
        try:
            # Delete existing prediction file from last run
            self.pred_data_val.deletePredictionFile()
            self.log_writer.log(self.file_object, 'Start of Prediction')

            # Load prediction data
            data_getter = data_loader_prediction.Data_Getter_Pred(self.file_object, self.log_writer)
            data = data_getter.get_data()

            # Initialize preprocessor
            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)

            # Check and impute missing values if present
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)
            if is_null_present:
                data = preprocessor.impute_missing_values(data)

            # Apply log transformation to data
            data = preprocessor.logTransformation(data)

            # Scale the prediction data
            data_scaled = pandas.DataFrame(preprocessor.standardScalingData(data), columns=data.columns)

            # Initialize file loader for models
            file_loader = file_methods.File_Operation(self.file_object, self.log_writer)
            # Load KMeans clustering model
            kmeans = file_loader.load_model('KMeans')

            # Predict clusters for the data
            clusters = kmeans.predict(data_scaled)
            data_scaled['clusters'] = clusters
            clusters = data_scaled['clusters'].unique()

            result = []  # List to store predictions

            # For each cluster, load the corresponding model and make predictions
            for i in clusters:
                # Select data for the current cluster
                cluster_data = data_scaled[data_scaled['clusters'] == i]
                cluster_data = cluster_data.drop(['clusters'], axis=1)
                # Find and load the correct model for this cluster
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                # Make predictions and append to result list
                for val in (model.predict(cluster_data.values)):
                    result.append(val)

            # Convert results to DataFrame
            result = pandas.DataFrame(result, columns=['Predictions'])
            path = "Prediction_Output_File/Predictions.csv"
            # Save predictions to CSV file
            result.to_csv(path, header=True)
            self.log_writer.log(self.file_object, 'End of Prediction')
        except Exception as ex:
            # Log any errors that occur during prediction
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path
