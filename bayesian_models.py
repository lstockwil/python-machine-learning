#supervised_learning.py
## Explore simple supervised ML model performance for identifying flowers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

##Prepare data
df = pd.read_csv("data/iris_flowers.csv")

print(f"Shape: {df.shape}")

#Classify data into features and classes
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print(X)
print(y)

#Split the data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.50, random_state=32)
#Test Different Naive Bayesian Classifier algorithm


## Naive Bayes Gaussian
### Assumes likelihood of the features is Gaussian
model = GaussianNB()
model.fit(X_train, y_train) #first fold training
pred_y = model.predict(X_test) #first fold testing

accuracy = accuracy_score(y_true=y_test, y_pred=pred_y)
cm = confusion_matrix(y_true=y_test, y_pred=pred_y)

print("Accuracy score: " + str(accuracy) + "\n")


print()

print(classification_report(y_true=y_test, y_pred=pred_y))
print("Confusion Matrix plotted in separate window")

disp = ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=pred_y, labels=None)

plt.xlabel("Predicted Flower", fontsize=14)
plt.ylabel("True Flower", fontsize=14)
plt.suptitle("Confusion Matrix", fontsize=16)
#plt.subplots_adjust(left=0.25)
plt.show()