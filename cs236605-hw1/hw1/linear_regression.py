import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y
import torch

class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, 'weights_')

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = np.dot(X,self.weights_)
        #y_pred(N,1)
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)
        # X(N,2), y(N,1)
        # TODO: Calculate the optimal weights using the closed-form solution
        # Use only numpy functions.

        w_opt = None
        # ====== YOUR CODE: ======
        # Take regularization into consideration
        #w_opt = np.dot(np.linalg.inv(np.dot(np.transpose(X), X) / X.shape[0]
             #                + self.reg_lambda * np.eye(X.shape[1])),
             #                 np.dot(np.transpose(X),y) / X.shape[0])/2
        #pseudo_temp1 = np.dot(X.T, X) / float(X.shape[0])
        pseudo_temp1 = np.dot(X.T, X)
        pseudo_temp2 = np.add(pseudo_temp1, self.reg_lambda * np.eye(X.shape[1]))
        pseudo_temp3 = np.linalg.inv(pseudo_temp2)

        #w_opt = (np.dot(pseudo_temp3, np.dot(X.T, y))/ float(X.shape[0]))
        w_opt = np.dot(pseudo_temp3, np.dot(X.T, y))
        #w_opt = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        :param X: A tensor of shape (N,D) where N is the batch size or of shape
            (D,) (which assumes N=1).
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X)

        # TODO: Add bias term to X as the first feature.

        xb = None
        # ====== YOUR CODE: ======
        b = np.ones((X.shape[0],1))
        xb = np.hstack((b,X))
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """
    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        self.unique_indices = None
        #self.poly = PolynomialFeatures(degree=degree,include_bias=False)
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)
        # check_is_fitted(self, ['n_features_', 'n_output_features_'])

        # TODO: Transform the features of X into new features in X_transformed
        # Note: You can count on the order of features in the Boston dataset
        # (this class is "Boston-specific"). For example X[:,1] is the second
        # feature ('ZN').
        X_transformed = None
        # ====== YOUR CODE: ======
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        #X_transformed = poly.fit_transform(X)
        X_transformed_temp = poly.fit_transform(X)
        #delete features with same values
        if self.unique_indices is not None:
            X_transformed = X_transformed_temp[:, self.unique_indices]
        else:
            X_transformed, unique_indices = np.unique(X_transformed_temp, axis=1, return_index=True)
            self.unique_indices = unique_indices

        #(156,119) 119=14*13/2+14+14
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    num_columns = len(df.columns)
    #14
    #calculate correlation between all column vectors except target_feature
    corrs = []
    for i in range(num_columns - 1):
        corrs_temp=df[target_feature].corr(df.iloc[:,i])
        corrs.append(corrs_temp)
    corrs_arr = np.abs(np.array(corrs))
    top_n_corr = corrs_arr[corrs_arr.argsort()[-n:][::-1]]
    top_n_features = df.columns.values[corrs_arr.argsort()[-n:][::-1]]
    # ========================

    return top_n_features, top_n_corr


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                        degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #
    # Notes:
    # - You can implement it yourself or use the built in sklearn utilities
    #   (recommended). See the docs for the sklearn.model_selection package
    #   http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    # - If your model has more hyperparameters (not just lambda and degree)
    #   you should add them to the search.
    # - Use get_params() on your model to see what hyperparameters is has
    #   and their names. The parameters dict you return should use the same
    #   names as keys.
    # - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======

    #from sklearn.model_selection import GridSearchCV
    parameters = {'bostonfeaturestransformer__degree':degree_range, 'linearregressor__reg_lambda':lambda_range}
    scorer = sklearn.metrics.make_scorer(sklearn.metrics.r2_score)
    clf = sklearn.model_selection.GridSearchCV(model, parameters, cv=k_folds, scoring=scorer)
    clf.fit(X,y)
    best_params = clf.best_params_
    #print
    #score, rsq = evaluate_accuracy(y,y_pred)
    # ========================

    return best_params
