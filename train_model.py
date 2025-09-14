import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

data = pd.read_csv("C:\\Users\\User\\Desktop\\ML Projects\\Decisiontrees\\student-scores.csv")

print(data.head())