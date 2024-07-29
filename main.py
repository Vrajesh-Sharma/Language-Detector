import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#Load the data
data = pd.read_csv("Languages.csv")

#Display the first few rows of the data
#print(data.head())

#check for missing values
#print(data.isnull().sum())

#Drop rows with missing values
data.dropna(inplace=True)

#Ensure all data in the 'Language' column is in string format
data['language'] = data['language'].astype(str)

#Display thye count of each language
#print(data["Language"].value_counts())

#convert the text data and labels to numpy array
x = np.array(data["Text"])
y = np.array(data["language"])

#Initialize the CountVectorizer
cv = CountVectorizer()

#Transform the text data into feature vectors
X = cv.fit_transform(x)

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Initialize and train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

#Evaluate the model
print("Model Accuracy:", model.score(X_test, y_test))

#Get user input and predict the language
for i in range(7):
    user_input = input("Enter a Text: ")
    user_data = cv.transform([user_input]).toarray()
    output = model.predict(user_data)
    
    #Print and predict the language
    print("Predicted Language:", output[0])