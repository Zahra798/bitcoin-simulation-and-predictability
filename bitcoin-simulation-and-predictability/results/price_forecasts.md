| model                                   |     R2 |   RMSE (USD) |   Theil U (vs naive) |   corr with today's price |   corr of predicted vs actual change |
|:----------------------------------------|-------:|-------------:|---------------------:|--------------------------:|-------------------------------------:|
| Naive (tomorrow = today)                | 0.9968 |         1630 |                1     |                    1      |                            nan       |
| Ridge on past prices                    | 0.9968 |         1643 |                1.008 |                    1      |                             -0.02404 |
| Gradient boosting on past prices        | 0.9806 |         4042 |                2.481 |                    0.9921 |                              0.01748 |
| Neural network on min-max scaled prices | 0.988  |         3180 |                1.952 |                    0.9956 |                              0.01772 |