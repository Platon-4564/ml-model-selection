# Regularization and Model Selection: Theory and Experiments

How regularization, feature scaling and the decision threshold affect classical machine learning models. Each question is first answered analytically and then checked on two scikit-learn datasets: Diabetes (regression) and Breast Cancer Wisconsin (classification).

> Coursework project for a machine learning course at MIPT. The notebook and figure labels are in Russian; this README summarizes the work in English.

![Decision boundaries of logistic regression, kNN, RBF SVM and a decision tree on two features; crosses mark validation errors](images/decision_boundaries.png)

## Key results

- **Theory.** Closed-form solutions of OLS, Ridge and Lasso for an orthonormal design (Lasso reduces to soft thresholding). A proof that for any design matrix the norm of the Ridge solution is non-increasing and the training error is non-decreasing in $\lambda$. The cost-optimal decision threshold $t^* = c_{FP}/(c_{FP}+c_{FN})$ and its form for logistic regression.
- **Regularization did not help on Diabetes.** OLS has the lowest validation RMSE, and both Ridge and Lasso only get worse as `alpha` grows: with 10 features and 265 training objects, OLS does not overfit (its validation RMSE is even below the training one). On the test set OLS reaches RMSE 57.0 against 76.9 for the constant baseline (−26%, R² ≈ 0.44).
- **Feature units matter for penalized and distance-based models.** Multiplying `bmi` by 1000 changes the predictions of unstandardized Ridge(`alpha=1000`) by up to 20, and multiplying `mean radius` by 100 flips 6.1% of unstandardized kNN labels. OLS, the decision tree and all standardized models are unaffected.
- **Four classifiers, one overlap zone.** On two features, logistic regression, kNN, an RBF SVM and a decision tree land within 0.86–0.89 balanced accuracy on validation, a difference of a few objects. The selected tree scores 0.877 on the test set against 0.50 for the majority-class baseline; most of its errors are missed malignant tumors (8 false negatives vs 4 false positives).
- **Threshold choice.** For logistic regression, lowering the threshold from 0.5 to 0.2 raises recall from 0.79 to 0.95 at precision 0.71, the better trade-off when a missed malignant tumor costs more than a false alarm.

## What is inside

| Section | Questions |
| --- | --- |
| 1. Regularized linear regression (theory) | Closed-form OLS, Ridge and Lasso; monotonicity in $\lambda$; how rescaling a feature changes the penalty; whether to penalize the intercept |
| 2. Classification metrics and threshold (theory) | Monotonicity of precision, recall and accuracy in the threshold; cost-optimal threshold; threshold for logistic regression |
| 3. Regression on Diabetes | Exploratory analysis, constant baseline vs OLS, sensitivity to feature scale, Ridge and Lasso paths, final test evaluation |
| 4. Classification on Breast Cancer | Majority-class baseline, four classifiers with hyperparameter search, tree depth and overfitting, feature scale, threshold choice, final test evaluation |

## Data and protocol

- **Diabetes** (`sklearn.datasets.load_diabetes`, unscaled): 442 patients, 10 features, disease progression after one year. Train / validation / test: 265 / 88 / 89.
- **Breast Cancer Wisconsin** (`sklearn.datasets.load_breast_cancer`): 569 tumors; only `mean radius` and `mean texture` are used, target 1 = malignant. Stratified train / validation / test: 341 / 114 / 114.

Models and preprocessing are fit on train only, hyperparameters are chosen on validation, and the test set is used once, after the model is fixed. Both datasets ship with scikit-learn, so nothing needs to be downloaded.

## Results

### Regression: Diabetes

| Model | Validation RMSE | Test RMSE |
| --- | --- | --- |
| Constant baseline (train mean) | 76.2 | 76.9 |
| OLS with standardization | 49.2 | 57.0 |

scikit-learn's `alpha` is not the $\lambda$ of the theory: Ridge minimizes $\|y-Xw\|_2^2+\alpha\|w\|_2^2$ and Lasso minimizes $\frac{1}{2n}\|y-Xw\|_2^2+\alpha\|w\|_1$, so $\alpha_{\text{Ridge}} = n\lambda$ while $\alpha_{\text{Lasso}} = \lambda$. Equal values of `alpha` therefore mean very different regularization strength; here Ridge(`alpha=1000`) corresponds to $\lambda \approx 3.8$.

![Ridge and Lasso: train and validation RMSE and standardized coefficients versus alpha](images/regularization_paths.png)

<p align="center">
  <img src="images/feature_scaling_ridge.png" width="560" alt="Ridge(alpha=1000) and OLS predictions before and after multiplying bmi by 1000, with and without standardization">
</p>

### Classification: Breast Cancer, two features

| Model | Selected hyperparameters | Validation balanced accuracy |
| --- | --- | --- |
| Logistic regression | `C=1` | 0.860 |
| kNN | `n_neighbors=9`, uniform weights | 0.879 |
| SVM, RBF kernel | `C=0.1`, `gamma=1` | 0.886 |
| Decision tree | `max_depth=3`, `min_samples_leaf=20` | 0.890 |

On the test set the tree reaches balanced accuracy 0.877 (precision 0.895, recall 0.810) against 0.500 for the majority-class baseline. That baseline has accuracy 0.62 while never detecting a malignant tumor, which is why balanced accuracy is the main metric. An unrestricted tree memorizes the training set (balanced accuracy 1.0 from depth 10), while its validation balanced accuracy stays between 0.81 and 0.87.

![Precision and recall of logistic regression on validation as functions of the threshold](images/precision_recall_threshold.png)

## Limitations

- Validation and test sets are small (88–114 objects). The standard error of test balanced accuracy is about 0.03, so the differences between the four classifiers are within noise, and OLS RMSE moves from 49.2 on validation to 57.0 on test.
- A single train / validation / test split, without cross-validation.
- Classification uses 2 of the 30 available features, and probability calibration is not checked.

## Next steps

- Repeated cross-validation with confidence intervals for model selection.
- All 30 features for classification.
- Probability calibration and a threshold chosen from the costs of errors.

## How to run

```bash
pip install numpy pandas scipy scikit-learn matplotlib jupyter
jupyter notebook regularization_and_model_selection.ipynb
```

The notebook was run with Python 3.13, NumPy 2.1, pandas 2.2, scikit-learn 1.6 and Matplotlib 3.10; the key numbers also reproduce with scikit-learn 1.9.

## Repository structure

```
├── README.md
├── regularization_and_model_selection.ipynb   # theory and experiments
└── images/                                    # figures used in this README
```

## Stack

Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib

## Context

Coursework for a machine learning course at MIPT. Data loading, the train / validation / test splits and the decision-boundary plotting helper come from the course template; the derivations, experiments and analysis are the author's.

Author: Platon Khamatov
