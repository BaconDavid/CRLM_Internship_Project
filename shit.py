import numpy as np
from sklearn.model_selection import StratifiedKFold

# Creating dummy data
np.random.seed(43)  # For reproducibility
X = np.random.rand(249, 10)  # 249 samples, 10 features each
y = np.random.randint(0, 2, 249)  # 249 labels, each either 0 or 1

# Setting up the Stratified K-Fold
stratify_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Using the Stratified K-Fold to split the data
for train_index, test_index in stratify_kfold.split(X, y):
    print("Train Indices:", train_index)
    print("Test Indices:", test_index)
    print("Number of training samples:", len(train_index))
    print("Number of testing samples:", len(test_index))
    print("##################")
