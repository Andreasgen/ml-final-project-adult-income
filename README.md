# Does a Neural Network Beat a Classical Model? — Adult Census Income

**Course:** Machine Learning with Python — Final Project (Option 2)

## 1. Project title and short description

This project compares two approaches to the same binary classification problem — predicting
whether a person's income exceeds $50K/year — using the Adult Census Income dataset. A simple,
transparent logistic regression model is built and evaluated against a small neural network,
to judge whether the added complexity of a neural network earns its place.

## 2. Problem statement

**Scenario:** A public agency wants to use survey data to flag which households are likely to be
in a higher-income bracket, so a benefit or outreach programme can be targeted appropriately.

**Central question:** Does the neural network perform better than logistic regression, and is its
additional complexity worthwhile?

## 3. Data set

- **Source:** Adult Census Income dataset (via `fetch_openml`)
- **Target:** whether an individual's income is `<=50K` or `>50K` per year
- **Features:** a mix of numeric (age, hours-per-week, etc.) and categorical (occupation,
  education, sex, race, native-country, etc.) survey attributes
- **Class balance:** the dataset is imbalanced — most individuals earn `<=50K`, which is why F1
  is used as the primary comparison metric rather than accuracy alone

## 4. Method

**Preprocessing:** built with a `ColumnTransformer` combining two pipelines:
- Numeric features → median imputation, then standard scaling
- Categorical features → most-frequent imputation, then one-hot encoding (unseen categories
  in the test set are safely ignored)

**Model 1 — Logistic Regression:** wrapped with the preprocessor in a single `sklearn` Pipeline,
evaluated first with 5-fold stratified cross-validation on the training set, then fit on the full
training set and tested on the held-out test set.

**Model 2 — Neural Network:** a small dense Keras network (`64 → Dropout(0.3) → 32 → 1 sigmoid`
units), trained with the Adam optimizer, binary cross-entropy loss, and early stopping on
validation loss to avoid overfitting.

Both models were evaluated on the exact same train/test split for a fair comparison.

## 5. Results

**Cross-validation (Logistic Regression, training data only):**
Mean F1 = 0.658, Std = 0.01 across 5 folds — consistent performance across splits.

**Test set comparison:**

| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression | 0.852 | 0.656 |
| Neural Network | 0.857 | 0.676 |

**Confusion matrices** (see notebook) show the neural network correctly identified more true
`>50K` cases (1,455 vs. 1,376) and had fewer false negatives (883 vs. 962), at the cost of
slightly more false positives (511 vs. 480).

*(See the completed notebook for both confusion matrix plots and the model comparison bar chart.)*

## 6. Interpretation

The neural network outperforms logistic regression on both accuracy and F1, but the margin is
small (roughly 0.5–2 percentage points). Accuracy alone would understate the practical
difference here, since the imbalanced classes mean a model can score well overall while still
missing a meaningful share of the higher-income group — exactly the group this system is meant
to identify.

**Limitations:** the dataset contains sensitive attributes (sex, race, native-country) that
correlate with income, so a model trained on it risks reproducing existing societal
inequalities even without those exact features driving the decision directly (via proxies like
occupation or hours worked). Both models miss over a third of true `>50K` cases, which is a
meaningful real-world cost if this were used to gate access to a benefit programme. Given the
network's complexity buys only a marginal F1 gain, the interpretability of logistic regression
is arguably more valuable in a public-sector context where decisions need to be explainable.

## 7. Reflection

**What worked well:** the sklearn preprocessing pipeline made it straightforward to handle mixed
numeric/categorical data consistently across both models, and cross-validation gave confidence
that the logistic regression baseline wasn't a lucky split.

**What was difficult:** understanding *why* F1 mattered more than accuracy for this imbalanced
dataset took some thought, and building the Keras network for the first time (choosing layer
sizes, dropout, and early stopping) required following the guided structure closely.

**What could be improved with more time:** a deeper fairness analysis — e.g. comparing false
negative rates specifically across sex or race subgroups rather than just overall — would give
a much stronger picture of whether the models' errors are evenly distributed. Trying a slightly
deeper or wider network, or tuning hyperparameters more systematically, could also clarify
whether the small NN advantage holds up or is close to noise.

## Files in this repository

- `Final_Project_Option2_Tabular_NN_vs_Classical.ipynb` — completed notebook, all cells run,
  all outputs visible
- `README.md` — this file
