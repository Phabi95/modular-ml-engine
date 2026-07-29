from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


def load_and_scale_data():
    breast_cancer = load_breast_cancer(as_frame=True)
    X_df = breast_cancer.data[["worst area", "mean concave points"]]
    y = breast_cancer.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    return X_df, X_scaled, y
