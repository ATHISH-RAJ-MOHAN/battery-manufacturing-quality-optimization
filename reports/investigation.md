# Batch investigation: 104

Synthetic observations only. Associations are hypotheses, not confirmed causes.

Observed defect rate: 10.000%; n=200. Reference rate: 4.824%. Small batch rates are noisy.

Largest absolute standardized mean deviations:

temperature_variation    0.597582
welding_duration         0.574068
welding_energy           0.565447
welding_resistance       0.546164
alignment_deviation      0.543581

Equipment observations (rate, count):

```
               mean  size
equipment                
EQ-0       0.000000    20
EQ-1       0.125000    24
EQ-2       0.074074    27
EQ-3       0.120000    25
EQ-4       0.090909    22
EQ-5       0.074074    27
EQ-6       0.173913    23
EQ-7       0.125000    32
```

Global model SHAP contributors: {'voltage_variation': 0.27579527653802527, 'internal_resistance': 0.22738297204348384, 'welding_resistance': 0.013393445376850601, 'current': 0.014845812364052283, 'pretest_voltage': 0.017092172113604576, 'process_temperature': 0.012095241770268692, 'interface_temperature': 0.2131862117787267, 'temperature_variation': 0.10494708039997061, 'welding_energy': 0.4193306507325406, 'welding_duration': 0.07001713454365333, 'assembly_pressure': 0.3294690605884042, 'alignment_deviation': 0.20582267133345294, 'cycle_time': 0.014928484725723815}. These explain raw model scores, not calibrated probabilities. Batch-specific mean absolute SHAP: {'welding_energy': 0.5818958095772949, 'voltage_variation': 0.33169721640727146, 'assembly_pressure': 0.27713827369900673, 'alignment_deviation': 0.25286543994314153, 'internal_resistance': 0.23833330609045408, 'interface_temperature': 0.16750290943626753, 'temperature_variation': 0.13406378018735995, 'welding_duration': 0.06290910466305936, 'cycle_time': 0.017107027846450188, 'current': 0.015400074558809741, 'process_temperature': 0.01246074581600219, 'pretest_voltage': 0.011352432852566732, 'welding_resistance': 0.009927514545070532}. Inspect sensors, recipe logs and maintenance history; confirm hypotheses with controlled engineering studies before any intervention.
