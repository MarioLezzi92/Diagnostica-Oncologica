import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_data(filename='dataset.csv'):
    """Carica il dataset dal percorso specificato."""
    path = os.path.join('Dataset', filename)
    return pd.read_csv(path)


def clean_data(df):
    """Pulisce il dataframe: rimuove colonne inutili e codifica il target."""
    cols_to_drop = ['id', 'Unnamed: 32']
    df_clean = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)

    if df_clean['diagnosis'].dtype == 'object':
        df_clean['diagnosis'] = df_clean['diagnosis'].map({'M': 1, 'B': 0})

    return df_clean


def split_data(df, target_col='diagnosis', test_size=0.2):
    """Divide il dataset in Training e Test set (dati grezzi)."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=42)


def scale_data(X_train, X_test):
    """Applica lo Standard Scaling."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_model(X_train, y_train):
    """Addestra il modello."""
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model


def print_evaluation(model, X_test, y_test):
    """Stampa metriche e visualizza la Matrice di Confusione grafica."""
    predictions = model.predict(X_test)

    # 1. Stampa il report testuale
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # 2. Genera il grafico della Matrice di Confusione
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    # annot=True scrive i numeri dentro, fmt='d' li formatta come interi
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Benigno (0)', 'Maligno (1)'],
                yticklabels=['Benigno (0)', 'Maligno (1)'])

    plt.xlabel('Predetto dal Modello')
    plt.ylabel('Reale (Clinico)')
    plt.title('Matrice di Confusione')
    plt.show()