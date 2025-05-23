import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from file_operations import file_methods

class KMeansClustering:
    """
    This class is used to divide the data into clusters before training.
    """

    def __init__(self, file_object, logger_object):
        # Initialize with file and logger objects for logging
        self.file_object = file_object
        self.logger_object = logger_object

    def elbow_plot(self, data):
        """
        Method Name: elbow_plot
        Description: This method saves the plot to decide the optimum number of clusters to the file.
        Output: A picture saved to the directory
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the elbow_plot method of the KMeansClustering class')
        wcss = []  # List to store within-cluster sum of squares for each cluster count
        try:
            # Calculate WCSS for cluster counts from 1 to 10
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            # Plot WCSS vs number of clusters
            plt.plot(range(1, 11), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.savefig('preprocessing_data/K-Means_Elbow.PNG')  # Save the elbow plot locally

            # Find the optimal number of clusters using KneeLocator
            self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            self.logger_object.log(
                self.file_object,
                'The optimum number of clusters is: ' + str(self.kn.knee) + ' . Exited the elbow_plot method of the KMeansClustering class'
            )
            return self.kn.knee

        except Exception as e:
            self.logger_object.log(
                self.file_object,
                'Exception occured in elbow_plot method of the KMeansClustering class. Exception message:  ' + str(e)
            )
            self.logger_object.log(
                self.file_object,
                'Finding the number of clusters failed. Exited the elbow_plot method of the KMeansClustering class'
            )
            raise Exception()

    def create_clusters(self, data, number_of_clusters):
        """
        Method Name: create_clusters
        Description: Create a new dataframe consisting of the cluster information.
        Output: A dataframe with cluster column
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the create_clusters method of the KMeansClustering class')
        self.data = data
        try:
            # Initialize KMeans with the optimal number of clusters
            self.kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            # Fit the data and predict cluster for each sample
            self.y_kmeans = self.kmeans.fit_predict(data)

            # Save the KMeans model using file_operations
            self.file_op = file_methods.File_Operation(self.file_object, self.logger_object)
            self.save_model = self.file_op.save_model(self.kmeans, 'KMeans')

            # Add cluster assignments as a new column in the dataframe
            self.data['Cluster'] = self.y_kmeans
            self.logger_object.log(
                self.file_object,
                'Successfully created ' + str(self.kn.knee) + ' clusters. Exited the create_clusters method of the KMeansClustering class'
            )
            return self.data
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                'Exception occured in create_clusters method of the KMeansClustering class. Exception message:  ' + str(e)
            )
            self.logger_object.log(
                self.file_object,
                'Fitting the data to clusters failed. Exited the create_clusters method of the KMeansClustering class'
            )
            raise Exception()