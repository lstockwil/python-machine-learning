# hello_world.py
## Initial file for Data Science/Machine Learning projects
## Display relationship between features and class using pyplot
## No machine learning models used for this program
import pandas as pd
import matplotlib.pyplot as plt

# Import data
df = pd.read_csv("data/iris_flowers.csv")
sample_fraction = 0.1 # fraction of data to use for taking sample

# Print info about data
data_size = df.shape[0] # size = number of rows in data
print(f"--Data Shape: {df.shape}--")
print(f"(Samples: {df.shape[0]}, Features {df.shape[1] - 1}, Classes to Predict: 1)")


## Scatter plots: create a 2x2 grid to plot each feature with respect to class
# Define sub plots
fig, ax = plt.subplots(2, 2, figsize=(10, 8)) # 2 rows, 2 columns

# First subplot
ax[0, 0].scatter(df['petal-width'], df['class'])
ax[0, 0].set_title('Petal Width')

# Second subplot
ax[0, 1].scatter(df['petal-length'], df['class'])
ax[0, 1].set_title('Petal Length')

# Third subplot
ax[1, 0].scatter(df['sepal-width'], df['class'], color='green')
ax[1, 0].set_title('Sepal Width')

# Fourth subplot
ax[1, 1].scatter(df['sepal-length'], df['class'], color='green')
ax[1, 1].set_title('Sepal Length')

plt.suptitle("Flower Features vs Class") # Set overall title of plots

plt.tight_layout() # Adjust layout
plt.show()