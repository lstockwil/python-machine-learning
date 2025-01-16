# supervised_learning.py
## Explore simple supervised ML model performance for identifying flowers
## Models used:
## - Gaussian Naive Bayes Classifer (GaussianNB)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

## Prepare data
df = pd.read_csv("data/iris_flowers.csv")

# Classify data into features and classes
X = df.iloc[:, :-1] #Features
y = df.iloc[:, -1] #Class to predict

# Split the data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.50, random_state=32)

## GaussianNB Classifier
# Assumes likelihood of the features is Gaussian
model = GaussianNB()
model.fit(X_train, y_train) # first fold training
pred_y = model.predict(X_test) # first fold testing

# Calculate confusion matrix and accuracy of testing results. We will use this later for comparision 
cm_gaussian_nb_raw = confusion_matrix(y_test, pred_y)
accuracy_gaussian_nb_raw = accuracy_score(y_true=y_test, y_pred=pred_y)

## What if we scaled the data?
## We will use a StandardScaler to standardize the features (mean = 0 and standard deviation = 1)
# This might improve our accuracy with GaussianNB classifier
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Now let's train our model using the standardized data
model = GaussianNB()
model.fit(X_train, y_train) # first fold training
pred_y = model.predict(X_test) # first fold testing

# Calculate confusion matrix and accuracy to compare with previous raw feature results
cm_gaussian_nb_standardized = confusion_matrix(y_test, pred_y)
accuracy_gaussian_nb_standardized = accuracy_score(y_true=y_test, y_pred=pred_y)

## Compare GaussianNB accuracy with and without standardized features
## Plot confusion matrices back to back

# Labels for the confusion matrix
labels = ['Class 0', 'Class 1']

# Define sub plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # 1 row, 2 columns

# Confusion matrix for Model 1
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_gaussian_nb_raw, display_labels=np.unique(y_test))
disp1.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Confusion Matrix - GaussianNB Raw")

# Confusion matrix for Model 2
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_gaussian_nb_standardized, display_labels=np.unique(y_test))
disp2.plot(ax=axes[1], cmap="Greens", colorbar=False)
axes[1].set_title("Confusion Matrix - GaussianNB Standardized")

# Adjust layout
plt.tight_layout()
plt.show()

print(f"Difference in GaussianNB accuracy between raw features and standardized features\n(raw - standardized): {accuracy_gaussian_nb_raw - accuracy_gaussian_nb_standardized}")
## Conclusion
# Difference in accuracy is negligible (both models produced exactly the same confusion matrix)
# Possibilities: 
# - Larger dataset needed to see a difference (if there is one)
# - Data within each feature does not vary wildly (small variance)
# - Standardization of features does not largely affect the accuracy of GaussianNB