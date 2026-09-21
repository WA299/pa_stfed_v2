# Predictive Utility Graph Audit

Train-only Ridge diagnostic. This audit makes no causal claim.

| Grid | Load buses | Ordered pairs | Positive edges | Positive fraction | Mean asymmetry | Directed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 39_bus_semi_urban_reference_grid | 28 | 756 | 606 | 0.801587 | 0.0533782 | True |

## 39_bus_semi_urban_reference_grid

| Model | Node MAE | Node RMSE | Node WAPE (%) | Node sMAPE (%) | Aggregate MAE | Aggregate RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| self_only | 0.000405051 | 0.000549197 | 39.0284 | 37.7492 | 0.00282694 | 0.00351783 |
| electrical_top3 | 0.000391617 | 0.000540275 | 37.6522 | 37.0391 | 0.00238635 | 0.00309102 |
| utility_top3 | 0.00038494 | 0.000537846 | 36.9224 | 36.3094 | 0.00244725 | 0.0032137 |
| union | 0.000386264 | 0.000538108 | 37.0306 | 36.5287 | 0.00242917 | 0.00317429 |
| 50_bus_rural_reference_grid | 21 | 420 | 295 | 0.702381 | 0.0527648 | True |

## 50_bus_rural_reference_grid

| Model | Node MAE | Node RMSE | Node WAPE (%) | Node sMAPE (%) | Aggregate MAE | Aggregate RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| self_only | 0.000319449 | 0.000441705 | 40.9609 | 40.7903 | 0.00209181 | 0.00266233 |
| electrical_top3 | 0.000312803 | 0.000437277 | 39.5736 | 41.0833 | 0.00200487 | 0.00256742 |
| utility_top3 | 0.000313613 | 0.000440246 | 39.4082 | 40.5506 | 0.00204423 | 0.00262254 |
| union | 0.000314567 | 0.000440233 | 39.5343 | 40.9487 | 0.00202483 | 0.00260065 |
| 56_bus_semi_urban_reference_grid | 44 | 1892 | 1564 | 0.826638 | 0.0500865 | True |

## 56_bus_semi_urban_reference_grid

| Model | Node MAE | Node RMSE | Node WAPE (%) | Node sMAPE (%) | Aggregate MAE | Aggregate RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| self_only | 0.000443754 | 0.000590122 | 39.3231 | 39.4801 | 0.00428373 | 0.00523902 |
| electrical_top3 | 0.000429823 | 0.000580584 | 38.0672 | 38.7674 | 0.00356575 | 0.0044907 |
| utility_top3 | 0.000418902 | 0.000580237 | 37.0239 | 38.2535 | 0.00432927 | 0.00550098 |
| union | 0.000420371 | 0.000579256 | 37.1866 | 38.3269 | 0.00410986 | 0.0052396 |
| 80_bus_rural_reference_grid | 32 | 992 | 414 | 0.417339 | 1.68989 | True |

## 80_bus_rural_reference_grid

| Model | Node MAE | Node RMSE | Node WAPE (%) | Node sMAPE (%) | Aggregate MAE | Aggregate RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| self_only | 0.000300598 | 0.000441841 | 52.6137 | 50.2274 | 0.00325195 | 0.00422741 |
| electrical_top3 | 0.000295616 | 0.000438478 | 51.1844 | 49.4904 | 0.003248 | 0.00417966 |
| utility_top3 | 0.000290982 | 0.000436325 | 49.939 | 48.5035 | 0.00329994 | 0.00420972 |
| union | 0.000293889 | 0.000436783 | 50.8088 | 49.9115 | 0.00327366 | 0.00418194 |

Graph selection uses fit and selection only; the audit segment is held out until final evaluation.
Validation and test are not accessed.
