# Iris Species Classification with SVM

This project demonstrates how to classify iris flower species using Support Vector Machines (SVM) with the classic Iris dataset. It includes model training, evaluation, hyperparameter tuning, and visualization of decision boundaries.

## Files

- `irisdetection.py` — Main Python script for loading data, training SVM, evaluating, tuning, and plotting.
- (Generated) `iris.png` — Decision boundary plot (if you save the figure).

## Features

- Loads the Iris dataset from scikit-learn.
- Splits data into training and test sets.
- Trains a linear SVM classifier.
- Evaluates accuracy, confusion matrix, and classification report.
- Performs hyperparameter tuning with GridSearchCV.
- Visualizes decision boundaries for the first two features.

## Requirements

- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn

Install dependencies with:

```sh
pip install numpy pandas matplotlib scikit-learn
```

## Usage

1. Clone this repository or download the files.
2. Run the script in your terminal:

   ```sh
   python irisdetection.py
   ```

3. View printed results in the terminal and the decision boundary plot window.

## Customization

- To use different SVM kernels or parameters, modify the `param_grid` in the script.
- To visualize other feature pairs, adjust the slicing in the `plot_decision` function.

## License

This project is provided for educational
