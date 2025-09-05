import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load the Iris dataset
iris = datasets.load_iris()
X = iris.data      # Features: sepal/petal length and width
y = iris.target    # Labels: 0=setosa, 1=versicolor, 2=virginica

# Optional: convert to DataFrame for exploration
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)
print(df.head())

# 2. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train a basic SVM (linear kernel)
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy (linear kernel):", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 5. Optional: Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_

# Evaluate best model
y_best = best_model.predict(X_test)
print("Tuned Accuracy:", accuracy_score(y_test, y_best))
print("\nTuned Classification Report:\n", classification_report(y_test, y_best, target_names=iris.target_names))

# 6. (Optional) Visualize on the first two features
def plot_decision(surface_model, X, y, title):
    # Only use first two features for plotting
    X_plot = X[:, :2]
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    # Train a new SVM on only the first two features
    svc = SVC(kernel='linear', C=1.0, random_state=42)
    svc.fit(X_plot, y)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title(title)
    plt.show()

plot_decision(model, X, y, "SVM with Linear Kernel (first two features)")