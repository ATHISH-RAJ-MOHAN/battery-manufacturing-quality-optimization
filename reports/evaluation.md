# Verified synthetic experiment

Seed 42; 24000 rows; selected lightgbm on tuning PR-AUC.

```
        model  precision   recall       f1  roc_auc   pr_auc  balanced_accuracy    brier  false_positive_rate  inspection_burden  latency_ms_per_row  calibration
     logistic   0.091716 0.505435 0.155259 0.638131 0.133635           0.617911 0.047040             0.269614           0.281667            0.000755     isotonic
random_forest   0.256705 0.728261 0.379603 0.872006 0.452612           0.807339 0.037044             0.113583           0.145000            0.010601      sigmoid
      xgboost   0.234079 0.739130 0.355556 0.871826 0.485380           0.804431 0.035180             0.130269           0.161389            0.001708 uncalibrated
     lightgbm   0.281879 0.684783 0.399366 0.872639 0.489120           0.795407 0.035170             0.093970           0.124167            0.003760 uncalibrated
```

Anomaly metrics:
```
{'precision': 0.2465753424657534, 'recall': 0.29347826086956524, 'f1': 0.2679900744416873, 'roc_auc': 0.622588076570614, 'pr_auc': 0.10847561379127788, 'balanced_accuracy': 0.622588076570614, 'brier': 0.08194444444444444, 'false_positive_rate': 0.048302107728337235, 'inspection_burden': 0.060833333333333336, 'confusion_matrix': [[3251, 165], [130, 54]]}
```

Independent simulator probability across three search seeds:
```
                 mean       std  count  approx_95ci_halfwidth
method                                                       
baseline     0.039002  0.000000      3               0.000000
constrained  0.009547  0.002516      3               0.006251
random       0.009992  0.003120      3               0.007751
```

Intervals describe search variability for fixed contexts and simulator, not factory uncertainty. Constrained search includes random initialization cost in addition to reported solver time. Oracle evaluates expected probabilities without sampled-label noise; it is separate from the surrogate but shares the training simulator assumptions. No physical quality improvement is established.
