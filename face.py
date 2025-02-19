import numpy as np
import joblib
from sklearn.svm import SVC

# Load the embeddings and labels
embeddings = joblib.load('embeddings.pkl')
labels = joblib.load('labels.pkl')

# Check the unique labels
print(f"Unique labels: {np.unique(labels)}")

# If there is only one class, training won't work
if len(np.unique(labels)) > 1:
    # Train a classifier if there are multiple classes
    clf = SVC(kernel='linear', probability=True)
    clf.fit(embeddings, labels)
    
    # Save the trained model
    joblib.dump(clf, 'face_recognition_model.pkl')
    print("Model trained and saved successfully.")
else:
    print("Insufficient classes in the dataset for training.")
