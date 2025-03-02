from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Importing the parent: DataPreprocessing class from data_preprocess.py
from src.data_preprocess import DataPreprocessing 


class ModelBuilder(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)

    def dt(self, X_train, X_test, y_train, y_test):
        #Create DT model
        DT_classifier = DecisionTreeClassifier()

        #Train the model
        DT_classifier.fit(X_train, y_train)

        #Test the model
        DT_predicted = DT_classifier.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(DT_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        #get performance
        self.accuracy = accuracy_score(y_test, DT_predicted)

        return DT_classifier
    
class annModel(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(annModel, self).__init__(*args, **kwargs)

    def dt(self, X_train, X_test, y_train, y_test):
        # Initialize the MLPRegressor
        mlp = MLPRegressor(random_state=42)

        # Define hyperparameter grid
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 100), (100, 50), (100, 50, 25, 10)],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [200, 500, 1000]
        }
        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)

        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)

        # Get the best estimator
        best_mlp = grid_search.best_estimator_

        # Fit the best estimator on the training data
        best_mlp.fit(X_train, y_train)

        # Predict on the test data
        y_pred_best_mlp = best_mlp.predict(X_test)

        # Calculate the MSE and R² score
        mse_best_mlp = mean_squared_error(y_test, y_pred_best_mlp)
        r2_best_mlp = r2_score(y_test, y_pred_best_mlp)

        print(f"Optimized ANN Regression - MSE: {mse_best_mlp:.4f}, R² Score: {r2_best_mlp:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}")

        #get performance
        self.mse_best_mlp = mean_squared_error(y_test, y_pred_best_mlp)
        self.r2_best_mlp = r2_score(y_test, y_pred_best_mlp)

        return mlp