# Loan Default Modeling Results


## Logistic Regression Full Run

Run time: 2026-04-22 23:00:12

Input file: `accepted_2007_to_2018Q4.csv`

Target:

- `1`: Charged Off / Default
- `0`: Fully Paid
- Excluded: Current, In Grace Period, Late statuses, and rows without final loan outcome

Feature policy:

- Uses origination-time borrower, loan, credit-profile, application, and issue-year fields.
- Excludes post-origination repayment, recovery, hardship, settlement, last-payment, and last-FICO fields.
- Excludes `grade`, `sub_grade`, and `int_rate` to avoid using LendingClub's own risk/pricing output.

Training note:

- `sklearn` emitted a `max_iter reached` convergence warning at 500 iterations. The metrics below were still produced successfully, but a later rerun can increase `max_iter` or tune regularization if coefficient-level inference is needed.

Time split:

| Split | Issue years | Rows | Bad rate |
| --- | --- | ---: | ---: |
| Train | <= 2015 | 829,355 | 0.1846 |
| Validation | 2016 | 293,105 | 0.2329 |
| Test | 2017 | 169,321 | 0.2313 |

Metrics:

| split | rows | bad_rate | roc_auc | pr_auc | log_loss | brier_score | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_2016 | 293105 | 0.2329 | 0.6896 | 0.3985 | 0.6237 | 0.2169 | 0.6563 | 0.3558 | 0.5873 | 0.4432 |
| test_2017 | 169321 | 0.2313 | 0.6838 | 0.3772 | 0.6314 | 0.2182 | 0.6555 | 0.3518 | 0.5805 | 0.4381 |

Confusion matrix at threshold 0.5:

| split | threshold | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- |
| validation_2016 | 0.5000 | 152287 | 72566 | 28168 | 40084 |
| test_2017 | 0.5000 | 88258 | 41894 | 16432 | 22737 |

Raw metrics JSON:

```json
[
  {
    "split": "validation_2016",
    "rows": 293105,
    "bad_rate": 0.23285853192541922,
    "roc_auc": 0.689630290104277,
    "pr_auc": 0.39848629957437476,
    "log_loss": 0.6237402697664783,
    "brier_score": 0.21689205674821588,
    "threshold": 0.5,
    "accuracy": 0.6563211135941045,
    "precision": 0.3558277851753218,
    "recall": 0.5872941452265135,
    "f1": 0.4431570684680103,
    "tn": 152287,
    "fp": 72566,
    "fn": 28168,
    "tp": 40084
  },
  {
    "split": "test_2017",
    "rows": 169321,
    "bad_rate": 0.23132984095298278,
    "roc_auc": 0.6837885791043641,
    "pr_auc": 0.37718520462627725,
    "log_loss": 0.6314201798836468,
    "brier_score": 0.21815790431449533,
    "threshold": 0.5,
    "accuracy": 0.6555300287619373,
    "precision": 0.3517971252185484,
    "recall": 0.5804845668768669,
    "f1": 0.43809248554913294,
    "tn": 88258,
    "fp": 41894,
    "fn": 16432,
    "tp": 22737
  }
]
```


## Elastic Net Logistic Regression Full Run

Run time: 2026-04-22 23:14:35

Input file: `accepted_2007_to_2018Q4.csv`

Model configuration:

- `penalty`: elasticnet
- `solver`: saga
- `C`: 1.0
- `l1_ratio`: 0.5
- `class_weight`: balanced
- `max_iter`: 500

Feature policy:

- Uses the shared cleaning, target, feature list, and time split from `loan_default_data.py`.
- Uses origination-time borrower, loan, credit-profile, application, and issue-year fields.
- Excludes post-origination repayment, recovery, hardship, settlement, last-payment, and last-FICO fields.
- Excludes `grade`, `sub_grade`, and `int_rate` to avoid using LendingClub's own risk/pricing output.

Training note:

- `sklearn` emitted a `max_iter reached` convergence warning at 500 iterations. The metrics were still produced successfully.
- With `C=1.0` and `l1_ratio=0.5`, performance is almost identical to the L2 Logistic Regression baseline, so stronger regularization or a small validation search over `C`/`l1_ratio` would be needed if the goal is feature selection.

Time split:

| Split | Issue years | Rows | Bad rate |
| --- | --- | ---: | ---: |
| Train | <= 2015 | 829,355 | 0.1846 |
| Validation | 2016 | 293,105 | 0.2329 |
| Test | 2017 | 169,321 | 0.2313 |

Metrics:

| split | rows | bad_rate | roc_auc | pr_auc | log_loss | brier_score | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_2016 | 293105 | 0.2329 | 0.6896 | 0.3985 | 0.6237 | 0.2169 | 0.6563 | 0.3559 | 0.5873 | 0.4432 |
| test_2017 | 169321 | 0.2313 | 0.6838 | 0.3772 | 0.6314 | 0.2182 | 0.6555 | 0.3518 | 0.5805 | 0.4381 |

Confusion matrix at threshold 0.5:

| split | threshold | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- |
| validation_2016 | 0.5000 | 152294 | 72559 | 28167 | 40085 |
| test_2017 | 0.5000 | 88253 | 41899 | 16433 | 22736 |

Raw metrics JSON:

```json
[
  {
    "split": "validation_2016",
    "rows": 293105,
    "bad_rate": 0.23285853192541922,
    "roc_auc": 0.6896282942311607,
    "pr_auc": 0.3984828024067676,
    "log_loss": 0.6237419381166703,
    "brier_score": 0.21689269977347347,
    "threshold": 0.5,
    "accuracy": 0.6563484075672541,
    "precision": 0.35585561592273,
    "recall": 0.587308796811815,
    "f1": 0.4431828232796745,
    "tn": 152294,
    "fp": 72559,
    "fn": 28167,
    "tp": 40085
  },
  {
    "split": "test_2017",
    "rows": 169321,
    "bad_rate": 0.23132984095298278,
    "roc_auc": 0.6837862815022978,
    "pr_auc": 0.37717927877750024,
    "log_loss": 0.6314272592054835,
    "brier_score": 0.21816074291950324,
    "threshold": 0.5,
    "accuracy": 0.6554945931101281,
    "precision": 0.35175988241664735,
    "recall": 0.580459036482933,
    "f1": 0.4380563369426997,
    "tn": 88253,
    "fp": 41899,
    "fn": 16433,
    "tp": 22736
  }
]
```


## XGBoost Full Run

Run time: 2026-04-22 23:48:17

Input file: `accepted_2007_to_2018Q4.csv`

Model configuration:

- `n_estimators`: 500
- `max_depth`: 4
- `learning_rate`: 0.05
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `min_child_weight`: 10.0
- `reg_lambda`: 1.0
- `tree_method`: hist
- `scale_pos_weight`: 4.4183

Feature policy:

- Uses the shared cleaning, target, feature list, and time split from `loan_default_data.py`.
- Uses origination-time borrower, loan, credit-profile, application, and issue-year fields.
- Excludes post-origination repayment, recovery, hardship, settlement, last-payment, and last-FICO fields.
- Excludes `grade`, `sub_grade`, and `int_rate` to avoid using LendingClub's own risk/pricing output.

Feature preprocessing:

- Numeric features: median imputation.
- Categorical features: most-frequent imputation plus one-hot encoding with rare-category grouping.
- Raw numeric columns: 14
- Raw categorical columns: 7
- Transformed model columns: 102

Time split:

| Split | Issue years | Rows | Bad rate |
| --- | --- | ---: | ---: |
| Train | <= 2015 | 829,355 | 0.1846 |
| Validation | 2016 | 293,105 | 0.2329 |
| Test | 2017 | 169,321 | 0.2313 |

Metrics:

| split | rows | bad_rate | roc_auc | pr_auc | log_loss | brier_score | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_2016 | 293105 | 0.2329 | 0.7011 | 0.4133 | 0.6267 | 0.2186 | 0.6505 | 0.3578 | 0.6300 | 0.4564 |
| test_2017 | 169321 | 0.2313 | 0.6965 | 0.3971 | 0.6254 | 0.2181 | 0.6535 | 0.3559 | 0.6151 | 0.4509 |

Confusion matrix at threshold 0.5:

| split | threshold | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- |
| validation_2016 | 0.5000 | 147669 | 77184 | 25253 | 42999 |
| test_2017 | 0.5000 | 86551 | 43601 | 15075 | 24094 |

Raw metrics JSON:

```json
[
  {
    "split": "validation_2016",
    "rows": 293105,
    "bad_rate": 0.23285853192541922,
    "roc_auc": 0.7010963513672539,
    "pr_auc": 0.4132967111385722,
    "log_loss": 0.6267054101998367,
    "brier_score": 0.2186091496332679,
    "threshold": 0.5,
    "accuracy": 0.6505109090598932,
    "precision": 0.3577793864356856,
    "recall": 0.6300035163804724,
    "f1": 0.45638018414838005,
    "tn": 147669,
    "fp": 77184,
    "fn": 25253,
    "tp": 42999
  },
  {
    "split": "test_2017",
    "rows": 169321,
    "bad_rate": 0.23132984095298278,
    "roc_auc": 0.6965154698879046,
    "pr_auc": 0.3971356597204895,
    "log_loss": 0.6253982669576997,
    "brier_score": 0.21812174690107536,
    "threshold": 0.5,
    "accuracy": 0.6534629490730625,
    "precision": 0.3559199350025851,
    "recall": 0.6151293114452756,
    "f1": 0.4509282826770475,
    "tn": 86551,
    "fp": 43601,
    "fn": 15075,
    "tp": 24094
  }
]
```


## Random Forest Full Run

Run time: 2026-04-23 00:32:48

Input file: `accepted_2007_to_2018Q4.csv`

Model configuration:

- `n_estimators`: 300
- `max_depth`: 18
- `min_samples_leaf`: 50
- `max_features`: sqrt
- `class_weight`: balanced_subsample

Feature policy:

- Uses the shared cleaning, target, feature list, and time split from `loan_default_data.py`.
- Uses origination-time borrower, loan, credit-profile, application, and issue-year fields.
- Excludes post-origination repayment, recovery, hardship, settlement, last-payment, and last-FICO fields.
- Excludes `grade`, `sub_grade`, and `int_rate` to avoid using LendingClub's own risk/pricing output.

Feature preprocessing:

- Numeric features: median imputation.
- Categorical features: most-frequent imputation plus one-hot encoding with rare-category grouping.
- Raw numeric columns: 14
- Raw categorical columns: 7

Time split:

| Split | Issue years | Rows | Bad rate |
| --- | --- | ---: | ---: |
| Train | <= 2015 | 829,355 | 0.1846 |
| Validation | 2016 | 293,105 | 0.2329 |
| Test | 2017 | 169,321 | 0.2313 |

Metrics:

| split | rows | bad_rate | roc_auc | pr_auc | log_loss | brier_score | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_2016 | 293105 | 0.2329 | 0.6896 | 0.3991 | 0.6151 | 0.2132 | 0.6733 | 0.3658 | 0.5492 | 0.4391 |
| test_2017 | 169321 | 0.2313 | 0.6814 | 0.3807 | 0.6168 | 0.2142 | 0.6688 | 0.3564 | 0.5358 | 0.4281 |

Confusion matrix at threshold 0.5:

| split | threshold | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- |
| validation_2016 | 0.5000 | 159879 | 64974 | 30769 | 37483 |
| test_2017 | 0.5000 | 92255 | 37897 | 18181 | 20988 |

Raw metrics JSON:

```json
[
  {
    "split": "validation_2016",
    "rows": 293105,
    "bad_rate": 0.23285853192541922,
    "roc_auc": 0.6895785057655486,
    "pr_auc": 0.39907048779993864,
    "log_loss": 0.6150750636227377,
    "brier_score": 0.21320201075979495,
    "threshold": 0.5,
    "accuracy": 0.6733491410927824,
    "precision": 0.3658412797563856,
    "recall": 0.5491853718572349,
    "f1": 0.43914497771060695,
    "tn": 159879,
    "fp": 64974,
    "fn": 30769,
    "tp": 37483
  },
  {
    "split": "test_2017",
    "rows": 169321,
    "bad_rate": 0.23132984095298278,
    "roc_auc": 0.6814280930052243,
    "pr_auc": 0.38065195023509546,
    "log_loss": 0.616813563533125,
    "brier_score": 0.2142083231284633,
    "threshold": 0.5,
    "accuracy": 0.6688065863064829,
    "precision": 0.3564235374034134,
    "recall": 0.5358319078863387,
    "f1": 0.42809064393089524,
    "tn": 92255,
    "fp": 37897,
    "fn": 18181,
    "tp": 20988
  }
]
```

