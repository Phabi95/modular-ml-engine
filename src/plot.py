import os
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "screenshots"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_initial_data(X_df, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_df[y == 0]["mean concave points"],
        X_df[y == 0]["worst area"],
        label="Malignant",
        color="red",
    )
    plt.scatter(
        X_df[y == 1]["mean concave points"],
        X_df[y == 1]["worst area"],
        label="Benign",
        color="blue",
    )
    plt.title("Breast Cancer Data Visualization", fontsize=16)
    plt.xlabel("Mean Concave Points", fontsize=14)
    plt.ylabel("Worst Area", fontsize=14)
    plt.legend(title="Class", fontsize=12)
    plt.savefig(f"{OUT_DIR}/breast_cancer_plot.png")


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]
    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    svs = svm_clf.support_vectors_

    plt.plot(x0, decision_boundary, "k-", linewidth=2, zorder=-2)
    plt.plot(x0, gutter_up, "k--", linewidth=2, zorder=-2)
    plt.plot(x0, gutter_down, "k--", linewidth=2, zorder=-2)
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors="#AAA", zorder=-1)


def plot_linear_boundaries(X_scaled, y, svm_clf1, svm_clf2):
    fig, axes = plt.subplots(ncols=2, figsize=(10, 2.7), sharey=True)

    plt.sca(axes[0])
    plt.plot(X_scaled[:, 0][y == 1], X_scaled[:, 1][y == 1], "g^", label="Benign")
    plt.plot(X_scaled[:, 0][y == 0], X_scaled[:, 1][y == 0], "bs", label="Malignant")
    plot_svc_decision_boundary(svm_clf1, X_scaled[:, 0].min(), X_scaled[:, 0].max())
    plt.xlabel("mean concave points")
    plt.ylabel("worst area")
    plt.legend(loc="upper left")
    plt.title(f"$C = {svm_clf1.C}$")
    plt.axis([-1, 2.5, 1, 2.5])
    plt.grid()

    plt.sca(axes[1])
    plt.plot(X_scaled[:, 0][y == 1], X_scaled[:, 1][y == 1], "g^", label="Benign")
    plt.plot(X_scaled[:, 0][y == 0], X_scaled[:, 1][y == 0], "bs", label="Malignant")
    plot_svc_decision_boundary(svm_clf2, X_scaled[:, 0].min(), X_scaled[:, 0].max())
    plt.xlabel("mean concave points")
    plt.title(f"$C = {svm_clf2.C}$")
    plt.axis([-1, 2.5, 1, 2.5])
    plt.grid()
    plt.savefig(f"{OUT_DIR}/my_plot1.png")


def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)
    plt.grid(True)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$", rotation=0)


def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


def plot_rbf_boundary(X_scaled, y, model):
    plot_predictions(model, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X_scaled, y, [-1.5, 2.45, -1, 1.5])
    plt.title(f"gamma={model.gamma}, C={model.C}")
    plt.savefig(f"{OUT_DIR}/my_plot2.png")
