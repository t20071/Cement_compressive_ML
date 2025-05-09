"""
This is the Entry point for Training the Machine Learning Model.
"""

# Import necessary libraries and modules
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger

# Class for training the machine learning model
class trainModel:
    """
    A class to handle the end-to-end training pipeline for predicting concrete compressive strength.
    This class performs the following steps:
        - Initializes logging for the training process.
        - Loads and preprocesses the dataset, including handling missing values and log transformation.
        - Applies clustering to group similar data points.
        - For each cluster, splits the data, scales features, and finds the best machine learning model.
        - Saves the trained model for each cluster.
        - Logs the progress and outcome of the training process.
    Attributes:
        log_writer: Logger object for writing logs.
        file_object: File object for logging.
    Methods:
        __init__():
            Initializes the logger and log file.
        trainingModel():
            Executes the complete training pipeline, including data ingestion, preprocessing,
            clustering, model selection, training, and saving the best models for each cluster.
    """

    def __init__(self):
        # Initialize logger and open log file
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # ===================== Data Ingestion =====================
            # Getting the data from the source
            data_getter = data_loader.Data_Getter(self.file_object, self.log_writer)
            data = data_getter.get_data()

            # ===================== Data Preprocessing =====================
            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)

            # Check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)

            # If missing values are there, replace them appropriately
            if is_null_present:
                data = preprocessor.impute_missing_values(data)  # Missing value imputation

            # Get encoded values for categorical data (if needed)
            # data = preprocessor.encodeCategoricalValues(data)

            # Create separate features and labels
            X, Y = preprocessor.separate_label_feature(data, label_column_name='Concrete_compressive _strength')

            # Apply log transformation to features
            X = preprocessor.logTransformation(X)

            # ===================== Clustering =====================
            # Initialize clustering object
            kmeans = clustering.KMeansClustering(self.file_object, self.log_writer)
            # Find the number of optimum clusters using elbow plot
            number_of_clusters = kmeans.elbow_plot(X)

            # Divide the data into clusters
            X = kmeans.create_clusters(X, number_of_clusters)

            # Create a new column in the dataset consisting of the corresponding cluster assignments
            X['Labels'] = Y

            # Get the unique clusters from our dataset
            list_of_clusters = X['Cluster'].unique()

            # ===================== Model Training for Each Cluster =====================
            # Parse all the clusters and look for the best ML algorithm to fit on individual cluster
            for i in list_of_clusters:
                # Filter the data for one cluster
                cluster_data = X[X['Cluster'] == i]

                # Prepare the feature and Label columns
                cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
                cluster_label = cluster_data['Labels']

                # Split the data into training and test set for each cluster
                x_train, x_test, y_train, y_test = train_test_split(
                    cluster_features, cluster_label, test_size=1 / 3, random_state=36
                )

                # Scale the training and test data
                x_train_scaled = preprocessor.standardScalingData(x_train)
                x_test_scaled = preprocessor.standardScalingData(x_test)

                # Initialize model finder object
                model_finder = tuner.Model_Finder(self.file_object, self.log_writer)

                # Get the best model for each of the clusters
                best_model_name, best_model = model_finder.get_best_model(
                    x_train_scaled, y_train, x_test_scaled, y_test
                )

                # Save the best model to the directory
                file_op = file_methods.File_Operation(self.file_object, self.log_writer)
                save_model = file_op.save_model(best_model, best_model_name + str(i))

            # ===================== End of Training =====================
            # Logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception:
            # Logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception