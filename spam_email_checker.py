
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

emails = [
    "Congratulations! You’ve won a $1000 Walmart gift card. Click here to claim now.",
    "Dear user, your account has been compromised. Please reset your password immediately.",
    "Hey, are we still meeting for lunch today?",
    "Don't forget to attend the team meeting at 3 PM.",
    "Free entry in 2 a weekly competition to win FA Cup final tickets. Text FA to 12345",
    "Can you send me the report by tomorrow?",
    "WINNER! You have been selected for a cash prize. Call now!",
    "Let’s catch up sometime next week."
]

labels = [1, 1, 0, 0, 1, 0, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, y_pred))

custom_email = input("Enter an email message to check if it's spam:\n")
custom_vector = vectorizer.transform([custom_email])
prediction = model.predict(custom_vector)

if prediction[0] == 1:
    print("⚠️ This email is likely SPAM.")
else:
    print("✅ This email is NOT spam.")
