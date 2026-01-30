import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    words = text.split()
    words = [word for word in words if word not in STOPWORDS]

    return ' '.join(words)

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    df = df[['text', 'spam']]

    df['text'] = df['text'].apply(clean_text)

    X = df['text']
    y = df['spam']

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
