| Check | Status | Detail |
| --- | --- | --- |
| schema | PASS | all 14 required columns present |
| missing_values | PASS | no missing values |
| duplicate_ids | PASS | transaction_id is unique |
| feature_bounds | PASS | 4 features within documented bounds |
| target_binary | PASS | returned is binary |
| class_balance | PASS | positive rate 29.46%, majority:minority 2.39:1 |
| customer_invariants | PASS | customer_segment constant per customer |
| tenure_consistency | WARN | tenure is constant for all 11,993 repeat customers - stamped per customer, not measured at transaction time |
| order_frequency_consistency | WARN | 93.5% of rows disagree with the observed trailing-12-month order count (median gap 5 orders) - the field is not derivable from this history, so CLV uses observed frequency instead |
| business_logic | PASS | COD 41.1% vs other 28.1% (as expected); segments First_Time 37.6% > Repeat 27.2% > Wholesale 7.8% (confirmed) |
| dgp_calibration | PASS | observed 29.46% vs DGP expectation 29.69% (gap 0.23%) |
