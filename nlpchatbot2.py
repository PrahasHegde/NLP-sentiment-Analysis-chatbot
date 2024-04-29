##NLP Sentiment Analysis chatbot for Topical chat dataset
#Topical Chat dataset from Amazon! It consists of over 8000 conversations and over 184000 messages!

from tkinter import Tk, Label, Entry, Button, Text, Scrollbar, END, Frame, ttk
from textblob import TextBlob
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_files
from sklearn.model_selection import GridSearchCV

# Download NLTK resources if not already downloaded
nltk.download('punkt')



# Load CSV data
df = pd.read_csv('topical_chat.csv')
X, y = df['message'].tolist(), df['sentiment'].tolist()

# Tokenize text using TextBlob
X = [' '.join(TextBlob(text).words) for text in X]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Define the hyperparameters to tune
parameters = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'vectorizer__max_df': [0.5, 0.75, 1.0],
    'vectorizer__min_df': [1, 2, 5],
    'classifier__alpha': [0.1, 0.5, 1.0]
}

# Perform GridSearchCV
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Use the best model for prediction
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Best parameters: {best_params}")
print(f"Accuracy: {accuracy}")

# Train the model
pipeline.fit(X_train, y_train)

# Function to analyze sentiment using the trained classifier
def analyze_sentiment_ml(text):
    tokens = ' '.join(TextBlob(text).words)
    return pipeline.predict([tokens])[0]

#chatbot GUI
class ChatbotApp:
    def __init__(self, master, dataset):
        self.master = master
        self.dataset = dataset
        self.output_message = ""
        master.title("Sentiment Analysis Chatbot")

        self.frame = Frame(master, padx=20, pady=20)
        self.frame.pack()

        self.label = ttk.Label(self.frame, text='Enter your text: ')
        self.label.grid(row=0, column=0, sticky='w')

        self.entry = ttk.Entry(self.frame, width=40)
        self.entry.grid(row=1, column=0, padx=5, pady=5)

        self.button = ttk.Button(self.frame, text='Analyze', command=self.analyze_text)
        self.button.grid(row=2, column=0, padx=10)

        self.reset_button = ttk.Button(self.frame, text='Reset', command=self.reset_text)
        self.reset_button.grid(row=2, column=1, padx=5)

        self.output_text = Text(self.frame, wrap='word', height=10, width=50)
        self.output_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        self.scrollbar = Scrollbar(self.frame, command=self.output_text.yview)
        self.scrollbar.grid(row=3, column=2, sticky='ns')

        self.output_text.config(yscrollcommand=self.scrollbar.set)

    def fade_in(self, widget, alpha):
        if alpha <= 1.0:
            color = self.blend_color("black", "white", alpha)
            widget.configure(foreground=color)
            alpha += 0.1
            self.master.after(50, self.fade_in, widget, alpha)

    def blend_color(self, c1, c2, alpha):
        def to_rgb(color):
            if color.startswith("#"):
                return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            elif color.lower() == "black":
                return (0, 0, 0)
            elif color.lower() == "white":
                return (255, 255, 255)
            else:
                raise ValueError("Unknown color: {}".format(color))

        def to_hex(color):
            return "#{:02x}{:02x}{:02x}".format(*color)

        c1_rgb = to_rgb(c1)
        c2_rgb = to_rgb(c2)

        blended_rgb = tuple(int(c1_val + (c2_val - c1_val) * alpha) for c1_val, c2_val in zip(c1_rgb, c2_rgb))
        return to_hex(blended_rgb)

    def display_analysis_result(self, output_message):
        self.output_text.config(state='normal')
        self.output_text.insert(END, output_message + '\n\n')
        self.output_text.config(state='disabled')

        # Scroll to the end of the text
        self.output_text.see(END)


    def analyze_text(self):
        user_input = self.entry.get()

        sentiment = analyze_sentiment_ml(user_input)
        unique_sentiments = sorted(df['sentiment'].unique())
        sentiment_index = unique_sentiments.index(sentiment)

        self.output_message = f"Sentiment: {unique_sentiments[sentiment_index]}"
        self.refresh_output()  # Add this line to refresh the output

        self.entry.delete(0, 'end')

    def refresh_output(self):
        self.display_analysis_result(self.output_message)

    def reset_text(self):
        self.entry.delete(0, 'end')
        self.output_message = ""
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, END)
        self.output_text.config(state='disabled')

if __name__ == "__main__":
    root = Tk()
    app = ChatbotApp(root,dataset=df)
    root.mainloop()

