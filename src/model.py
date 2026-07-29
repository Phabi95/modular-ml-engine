from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def train_linear_svms(X_scaled, y):
    svm_clf1 = SVC(kernel="linear", C=0.1)
    svm_clf1.fit(X_scaled, y)

    svm_clf2 = SVC(kernel="linear", C=1000)
    svm_clf2.fit(X_scaled, y)

    y_pred1 = svm_clf1.predict(X_scaled)
    y_pred2 = svm_clf2.predict(X_scaled)

    logging.info(f"F1 score for Classifier 1 (C=0.1): {f1_score(y, y_pred1)}")
    logging.info(f"F1 score for Classifier 2 (C=1000): {f1_score(y, y_pred2)}")
    logging.info(
        f"Number of support vectors for svm_clf1: {len(svm_clf1.support_vectors_)}"
    )
    logging.info(
        f"Number of support vectors for svm_clf2: {len(svm_clf2.support_vectors_)}"
    )

    return svm_clf1, svm_clf2


def train_rbf_svm(X_scaled, y):
    svm_rbf = SVC(kernel="rbf")
    param_grid = {"C": [0.1, 1, 10, 100], "gamma": [0.1, 1, 10, 100]}

    grid_search = GridSearchCV(
        estimator=svm_rbf, param_grid=param_grid, scoring="f1", cv=5
    )
    grid_search.fit(X_scaled, y)

    model_best_est = grid_search.best_estimator_
    n_support_vectors = model_best_est.support_

    logging.info(f"Best hyperparameters: {grid_search.best_params_}")
    logging.info(f"Best cross-validation accuracy: {grid_search.best_score_}")
    logging.info(f"Number of Support Vectors: {len(n_support_vectors)}")

    return model_best_est
