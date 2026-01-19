import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Load CSV File
# -------------------------------
data = pd.read_csv(r"C:\Users\USER\Desktop\Data_Collect\DataCollectGit\input.csv")

# -------------------------------
# Define Label Column
# -------------------------------
label_column = "City"     # <-- Your label

# -------------------------------
# Split Features & Label
# -------------------------------
X = data.drop(label_column, axis=1)
y = data[label_column]

# -------------------------------
# Encode Categorical Variables
# -------------------------------
# Encode label (City)
y = LabelEncoder().fit_transform(y)

# One-hot encode categorical features (Name)
X = pd.get_dummies(X)

# -------------------------------
# Train-test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -------------------------------
# Import Models
# -------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(max_iter=1000)
}

# -------------------------------
# Train & Evaluate
# -------------------------------
print("\n----- ACCURACY RESULTS -----\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name:25} : {accuracy_score(y_test, y_pred):.4f}")
