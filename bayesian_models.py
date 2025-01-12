#supervised_learning.py
## Explore simple supervised ML model performance for identifying flowers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

#method for displaying results of each model's performance
#cm_title sets the title for the confusion matrix plot
def display_results(y_test, pred_y, cm_title="Confusion Matrix"):
    accuracy = accuracy_score(y_true=y_test, y_pred=pred_y)
    cm = confusion_matrix(y_true=y_test, y_pred=pred_y)

    print("Accuracy score: " + str(accuracy) + "\n")


    print()

    print(classification_report(y_true=y_test, y_pred=pred_y))
    print("Confusion Matrix plotted in separate window")

    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=pred_y, labels=None)

    plt.xlabel("Predicted Flower", fontsize=14)
    plt.ylabel("True Flower", fontsize=14)
    plt.suptitle(cm_title, fontsize=16)
    plt.tight_layout()
    #plt.subplots_adjust(left=0.25)
    plt.show()

##Prepare data
df = pd.read_csv("data/iris_flowers.csv")

#Classify data into features and classes
X = df.iloc[:, :-1] #Features
y = df.iloc[:, -1] #Class to predict
print(X)
print(y)

#Split the data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.50, random_state=32)

##Test Different Naive Bayesian Classifier algorithm

# Naive Bayes Gaussian
# Assumes likelihood of the features is Gaussian
model = GaussianNB()
model.fit(X_train, y_train) #first fold training
pred_y = model.predict(X_test) #first fold testing

display_results(y_test=y_test, pred_y=pred_y, cm_title="GaussianNB Confusion Matrix")