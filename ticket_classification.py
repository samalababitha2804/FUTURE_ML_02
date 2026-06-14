import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("tickets.csv")

# Features and target
X = df["Ticket"]
y = df["Category"]

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Sample Ticket
ticket = "Payment failed and refund not received"

category = model.predict(vectorizer.transform([ticket]))[0]

# Priority Logic
if "refund" in ticket.lower() or "payment" in ticket.lower():
    priority = "High"
elif "account" in ticket.lower():
    priority = "Medium"
else:
    priority = "Low"

print("Ticket:", ticket)
print("Category:", category)
print("Priority:", priority)