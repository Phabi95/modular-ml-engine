from src.data import load_and_scale_data
from src.model import train_linear_svms, train_rbf_svm
from src.plot import plot_initial_data, plot_linear_boundaries, plot_rbf_boundary


def main():
    X_df, X_scaled, y = load_and_scale_data()

    plot_initial_data(X_df, y)

    svm_clf1, svm_clf2 = train_linear_svms(X_scaled, y)
    plot_linear_boundaries(X_scaled, y, svm_clf1, svm_clf2)

    best_rbf = train_rbf_svm(X_scaled, y)
    plot_rbf_boundary(X_scaled, y, best_rbf)


if __name__ == "__main__":
    main()
