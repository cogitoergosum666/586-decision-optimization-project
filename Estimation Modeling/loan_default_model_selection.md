# Loan Default Probability Model Selection

## Modeling Goal

Predict the probability that an accepted LendingClub loan will become bad:

- `target = 1`: `Charged Off`, `Default`, `Does not meet the credit policy. Status:Charged Off`
- `target = 0`: `Fully Paid`, `Does not meet the credit policy. Status:Fully Paid`
- Exclude `Current`, `In Grace Period`, and late-but-not-final statuses for the main model.

This is a probability-of-default problem, not an approval-rate problem. The dataset only contains accepted loans, so it cannot directly model rejected applications.

## Leakage-Safe Feature Set

Use only fields that are known at origination or underwriting time:

- Loan terms: `loan_amnt`, `term`
- Borrower income/profile: `annual_inc`, `emp_length`, `home_ownership`, `verification_status`
- Credit profile: `dti`, `fico_range_low`, `fico_range_high`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `mort_acc`, `pub_rec_bankruptcies`
- Product/application context: `purpose`, `addr_state`, `application_type`
- Time control: `issue_d` or derived `issue_year`

Avoid post-origination fields such as `total_pymnt`, `recoveries`, `last_pymnt_*`, `last_fico_*`, `out_prncp`, `hardship_*`, `settlement_*`, and `debt_settlement_*`.

Also avoid `grade`, `sub_grade`, and `int_rate` in the main model if the goal is to simulate pre-pricing credit risk. They are useful in a secondary benchmark because they encode LendingClub's own risk assessment.

## Candidate Models

### 1. XGBoost

Recommended main model.

Why:

- Strong default choice for tabular credit-risk data.
- Handles nonlinear interactions better than logistic regression.
- Usually outperforms random forest on structured/tabular problems.
- Produces probability scores that can be calibrated and ranked.

Use it as the primary model if the package is available.

Suggested evaluation:

- ROC-AUC
- PR-AUC
- Log loss
- Brier score
- Calibration curve
- Confusion matrix at selected probability cutoffs

### 2. Random Forest

Recommended benchmark model.

Why:

- Easy to explain as an ensemble tree method.
- Robust and simple to run.
- Useful as a comparison against XGBoost.

Limitations:

- Probability estimates are often less calibrated.
- Can be slower and less accurate than gradient boosting on this dataset.
- Does not extrapolate well and can be memory-heavy with many trees.

### 3. ResNet

Not recommended as the main model for this project.

Reason:

- Standard ResNet is designed for images, not ordinary tabular CSV data.
- A tabular residual neural network can be built, but it needs more preprocessing, tuning, regularization, and compute.
- It is harder to explain in a banking/credit-risk context.

Use only as an optional extension:

- Convert categorical variables to embeddings.
- Normalize numeric variables.
- Train a residual MLP / TabResNet.
- Compare against XGBoost and random forest.

Expected result: ResNet is unlikely to clearly beat XGBoost unless carefully tuned.

## Recommended Project Design

Use this structure:

1. Baseline: logistic regression
2. Benchmark: random forest
3. Main model: XGBoost
4. Optional extension: tabular ResNet

Final recommendation:

> Use XGBoost as the main probability-of-default model, random forest as the interpretable tree-ensemble benchmark, and ResNet only as an experimental extension.

## Train/Test Split

Prefer time-based validation instead of random split:

- Train: earlier issue years
- Validation: later issue year
- Test: most recent fully matured issue year

Example:

- Train: 2007-2015
- Validation: 2016
- Test: 2017

Be careful with 2018 because many loans are still `Current` in this file.

