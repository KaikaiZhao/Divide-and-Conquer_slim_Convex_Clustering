# Divide-and-Conquer_slim_Convex_Clustering
This repository contains our demo code and some empirical results for the DC3 algorithm.

## Contributions

1. We only have *n* primal variables in X compared to other convex clustering methods, so our algorithm is faster.
2. When we use kmeans with random initialization in our experiments, we can defeat kmeans through our convex clustering, demonstrating the robustness of convex clustering.
3. We can trace the history of the development of a dataset.

Hyperparameters and experimental results

Different alpha and the corresponding NMI
| 0.01|0.015 | 0.02|0.025| 0.03 |0.035 | 0.04 | 0.05 |0.06|0.07|0.08|
| --- | --- | --- | ---| ---|--|--|--|--|--|--|
|  66.20 |66.20| 70.80 |70.80| 70.80 |70.12 | 61.76| 62.51 | 62.08 | 61.76 | 62.67 |
