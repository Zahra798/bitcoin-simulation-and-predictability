| dataset                                                        | expected   | model               |   accuracy |    n |    p vs 50 % |   p vs majority rule |
|:---------------------------------------------------------------|:-----------|:--------------------|-----------:|-----:|-------------:|---------------------:|
| Negative control: random walk (unpredictable by construction)  | ≈ 50 %     | Always 'up'         |     0.5003 | 1967 | nan          |          nan         |
| Negative control: random walk (unpredictable by construction)  | ≈ 50 %     | Logistic regression |     0.487  | 1967 |   0.8795     |            0.884     |
| Negative control: random walk (unpredictable by construction)  | ≈ 50 %     | Gradient boosting   |     0.4931 | 1967 |   0.7361     |            0.7434    |
| Positive control: i.i.d. uniform prices (original course data) | > 50 %     | Always 'up'         |     0.4992 | 1967 | nan          |          nan         |
| Positive control: i.i.d. uniform prices (original course data) | > 50 %     | Logistic regression |     0.7361 | 1967 |   2.111e-101 |            8.7e-101  |
| Positive control: i.i.d. uniform prices (original course data) | > 50 %     | Gradient boosting   |     0.725  | 1967 |   7.306e-92  |            2.816e-91 |
| Real Bitcoin, 2021–2026                                        | ?          | Always 'up'         |     0.5081 | 1968 | nan          |          nan         |
| Real Bitcoin, 2021–2026                                        | ?          | Logistic regression |     0.5112 | 1968 |   0.1662     |            0.4021    |
| Real Bitcoin, 2021–2026                                        | ?          | Gradient boosting   |     0.5142 | 1968 |   0.1075     |            0.3021    |