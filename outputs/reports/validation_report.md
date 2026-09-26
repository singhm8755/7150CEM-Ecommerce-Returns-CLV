| Check | Status | Detail |
| --- | --- | --- |
| schema | PASS | all 14 required columns present |
| missing_values | PASS | no missing values |
| duplicate_ids | PASS | transaction_id is unique |
| feature_bounds | PASS | 4 features within documented bounds |
| target_binary | PASS | returned is binary |
| class_balance | PASS | positive rate 20.50%, majority:minority 3.88:1 |
| customer_invariants | PASS | customer_segment constant per customer |
| tenure_consistency | PASS | tenure increases with transaction date |
| order_frequency_consistency | PASS | matches observed trailing-12-month counts |
| business_logic | PASS | COD 27.2% vs other 19.6% (as expected); segments First_Time 40.0% > Repeat 26.2% > Wholesale 9.6% (confirmed) |
| dgp_calibration | PASS | observed 20.50% vs DGP expectation 20.62% (gap 0.12%) |
