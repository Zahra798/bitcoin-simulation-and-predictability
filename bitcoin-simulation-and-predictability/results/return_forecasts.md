| model                   |   RMSE (%) |   R2_oos vs random walk (%) |   R2_oos vs historical mean (%) |   DM stat vs random walk |   DM p-value |   sign accuracy |
|:------------------------|-----------:|----------------------------:|--------------------------------:|-------------------------:|-------------:|----------------:|
| Random walk (predict 0) |      3.004 |                     0       |                         0.06242 |                nan       |  nan         |        nan      |
| Historical mean         |      3.005 |                    -0.06246 |                         0       |                  0.3205  |    0.7486    |          0.5081 |
| Ridge regression        |      3.003 |                     0.01952 |                         0.08194 |                 -0.06482 |    0.9483    |          0.4964 |
| Gradient boosting       |      3.105 |                    -6.892   |                        -6.825   |                  4.393   |    1.175e-05 |          0.4944 |
| Neural network (MLP)    |      3.736 |                   -54.71    |                       -54.62    |                 10.65    |    9.065e-26 |          0.4858 |