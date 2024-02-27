# mymodel.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Global variable for the trained model
trained_model = None

def load_iris_dataset():
    return sns.load_dataset('iris')

def train_classification_model():
    global trained_model
    iris = load_iris_dataset()

    # Feature columns
    X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

    # Target column
    y = iris['species']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the classification model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Set the global variable to the trained model
    trained_model = model

    return model

# Train the model when this module is imported
train_classification_model()

def classify_input(input_data):
    global trained_model
    if trained_model is not None:
        result = trained_model.predict(input_data)
        return result[0]
    else:
        raise ValueError("Model not trained. Please call train_classification_model first.")
