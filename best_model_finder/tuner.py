from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

class Model_Finder:
    
    """    
    Model_Finder is a class for automated selection and hyperparameter tuning of regression models.
    This class provides methods to:
    - Tune hyperparameters for RandomForestRegressor and LinearRegression using GridSearchCV.
    - Train models with the best-found parameters.
    - Compare the performance of the tuned models using R2 score and select the best one.
    Attributes:
        file_object: File object for logging.
        logger_object: Logger object for logging messages.
        linearReg: Instance of LinearRegression.
        RandomForestReg: Instance of RandomForestRegressor.
    Methods:
        get_best_params_for_Random_Forest_Regressor(train_x, train_y):
            Tunes RandomForestRegressor hyperparameters using GridSearchCV and returns the trained model.
        get_best_params_for_linearReg(train_x, train_y):
            Tunes LinearRegression hyperparameters using GridSearchCV and returns the trained model.
        get_best_model(train_x, train_y, test_x, test_y):
            Compares the tuned LinearRegression and RandomForestRegressor models using R2 score on test data,
            and returns the name and instance of the best-performing model.
    """

    def __init__(self, file_object, logger_object):
        # Initialize logger and file objects, and model instances
        self.file_object = file_object
        self.logger_object = logger_object
        self.linearReg = LinearRegression()
        self.RandomForestReg = RandomForestRegressor()

    def get_best_params_for_Random_Forest_Regressor(self, train_x, train_y):
        """
        Get the best hyperparameters for RandomForestRegressor using GridSearchCV.
        Returns the trained model with the best parameters.
        """
        self.logger_object.log(self.file_object,
                               'Entered the RandomForestReg method of the Model_Finder class')
        try:
            # Define parameter grid for RandomForestRegressor
            self.param_grid_Random_forest_Tree = {
                "n_estimators": [10, 20, 30],
                "max_features": ["auto", "sqrt", "log2"],
                "min_samples_split": [2, 4, 8],
                "bootstrap": [True, False]
            }

            # Perform grid search with cross-validation
            self.grid = GridSearchCV(self.RandomForestReg, self.param_grid_Random_forest_Tree, verbose=3, cv=5)
            self.grid.fit(train_x, train_y)

            # Extract best parameters
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.max_features = self.grid.best_params_['max_features']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.bootstrap = self.grid.best_params_['bootstrap']

            # Train model with best parameters
            self.decisionTreeReg = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                bootstrap=self.bootstrap
            )
            self.decisionTreeReg.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'RandomForestReg best params: ' + str(
                                       self.grid.best_params_) + '. Exited the RandomForestReg method of the Model_Finder class')
            return self.decisionTreeReg
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in RandomForestReg method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'RandomForestReg Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_linearReg(self, train_x, train_y):
        """
        Get the best hyperparameters for LinearRegression using GridSearchCV.
        Returns the trained model with the best parameters.
        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_linearReg method of the Model_Finder class')
        try:
            # Define parameter grid for LinearRegression
            self.param_grid_linearReg = {
                'fit_intercept': [True, False],
                'normalize': [True, False],
                'copy_X': [True, False]
            }
            # Perform grid search with cross-validation
            self.grid = GridSearchCV(self.linearReg, self.param_grid_linearReg, verbose=3, cv=5)
            self.grid.fit(train_x, train_y)

            # Extract best parameters
            self.fit_intercept = self.grid.best_params_['fit_intercept']
            self.normalize = self.grid.best_params_['normalize']
            self.copy_X = self.grid.best_params_['copy_X']

            # Train model with best parameters
            self.linReg = LinearRegression(
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                copy_X=self.copy_X
            )
            self.linReg.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'LinearRegression best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_linearReg method of the Model_Finder class')
            return self.linReg
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_linearReg method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'LinearReg Parameter tuning  failed. Exited the get_best_params_for_linearReg method of the Model_Finder class')
            raise Exception()

    def get_best_model(self, train_x, train_y, test_x, test_y):
        """
        Compares LinearRegression and RandomForestRegressor models using R2 score.
        Returns the name and instance of the best model.
        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        try:
            # Train and evaluate Linear Regression
            self.LinearReg = self.get_best_params_for_linearReg(train_x, train_y)
            self.prediction_LinearReg = self.LinearReg.predict(test_x)
            self.LinearReg_error = r2_score(test_y, self.prediction_LinearReg)

            # Train and evaluate Random Forest Regressor
            self.randomForestReg = self.get_best_params_for_Random_Forest_Regressor(train_x, train_y)
            self.prediction_randomForestReg = self.randomForestReg.predict(test_x)
            self.prediction_randomForestReg_error = r2_score(test_y, self.prediction_randomForestReg)

            # Compare models and return the best one
            if self.LinearReg_error < self.prediction_randomForestReg_error:
                return 'RandomForestRegressor', self.randomForestReg
            else:
                return 'LinearRegression', self.LinearReg

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()
