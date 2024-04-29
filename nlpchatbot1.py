#NLP Sentiment Analysis chatbot for First GOP Debate Twitter Sentiment:
#This sentiment analysis dataset consists of around 14,000 labeled tweets that are positive, neutral, and negative about the first GOP debate that happened in 2016.

from tkinter import Tk, Label, Entry, Button, Text, END, Scrollbar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset (replace 'your_dataset.csv' with your actual dataset path)
df = pd.read_csv('Sentiment.csv')

# Preprocess the data
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['text'] = df['text'].apply(preprocess_text)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['sentiment']

# Train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Create the chatbot GUI
def analyze_sentiment():
    text = entry.get()
    text = preprocess_text(text)
    X_test = vectorizer.transform([text])
    sentiment = clf.predict(X_test)[0]
    output_text.config(state='normal')
    output_text.delete(1.0, END)
    output_text.insert(END, f"Sentiment: {sentiment}\n")
    output_text.config(state='disabled')

def reset_output():
    entry.delete(0, 'end')
    output_text.config(state='normal')
    output_text.delete(1.0, END)
    output_text.config(state='disabled')

root = Tk()
root.title("Sentiment Analysis Chatbot")

label = Label(root, text='Enter your text: ')
label.pack()

entry = Entry(root, width=40)
entry.pack()

analyze_button = Button(root, text='Analyze', command=analyze_sentiment)
analyze_button.pack()

reset_button = Button(root, text='Reset', command=reset_output)
reset_button.pack()

output_text = Text(root, wrap='word', height=10, width=50)
output_text.pack()

root.mainloop()
