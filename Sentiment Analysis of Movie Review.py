import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Load the dataset
df = pd.read_csv('movie_reviews.csv')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# Apply preprocessing to the 'text' column
df['text'] = df['text'].apply(preprocess_text)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed

# Transform text data into TF-IDF features
X = tfidf_vectorizer.fit_transform(df['text'])
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and train the model (Naive Bayes example)
model = MultinomialNB()
model.fit(X_train, y_train)

# Function to predict sentiment and calculate metrics
def predict_sentiment():
    user_review = user_input.get("1.0", "end-1c")  # Get user input
    user_review = preprocess_text(user_review)  # Preprocess user input
    user_input_features = tfidf_vectorizer.transform([user_review])  # Transform user input
    
    prediction = model.predict(user_input_features)[0]  # Predict sentiment
    
    # Calculate evaluation metrics on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Display the result and metrics
    if prediction == 'positive':
        result_label.config(text="Sentiment: Positive", foreground="green")
    else:
        result_label.config(text="Sentiment: Negative", foreground="red")
    
    metrics_label.config(text=f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

# Create a tkinter window
window = tk.Tk()
window.title("Movie Review Sentiment Analysis")

# Create and configure a text widget for user input
user_input = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=40, height=5)
user_input.grid(row=0, column=0, padx=10, pady=10)

# Create a button to trigger sentiment analysis
analyze_button = ttk.Button(window, text="Analyze Sentiment", command=predict_sentiment)
analyze_button.grid(row=1, column=0, padx=10, pady=5)

# Create a label to display the sentiment result
result_label = ttk.Label(window, text="", font=("Helvetica", 12))
result_label.grid(row=2, column=0, padx=10, pady=5)

# Create a label to display evaluation metrics
metrics_label = ttk.Label(window, text="", font=("Helvetica", 12))
metrics_label.grid(row=3, column=0, padx=10, pady=5)

# Start the tkinter main loop
window.mainloop()
