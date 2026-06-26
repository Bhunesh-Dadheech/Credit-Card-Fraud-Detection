import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tqdm import tqdm
import shap

# ----------------------------- Step 1: Load and Explore Dataset -----------------------------

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    print(df.head())
    print(df.info())
    print(df.describe())
    
    # Visualize class distribution
    sns.countplot(data=df, x='Class')
    plt.title('Class Distribution')
    plt.show()
    
    return df

# -------------------------- Step 2: Feature Engineering --------------------------

def add_features(df):
    # Drop 'id' column as it's not useful
    df = df.drop(['id'], axis=1)

    # Normalize the transaction amount
    df['Normalized_Amount'] = (df['Amount'] - df['Amount'].min()) / (df['Amount'].max() - df['Amount'].min())

    # Add rolling average of amounts (window = 10)
    df['Rolling_Avg_Amount'] = df['Amount'].rolling(window=10, min_periods=1).mean()

    # Add transaction frequency (transactions per value of Class)
    df['Transaction_Freq'] = df.groupby('Class')['Amount'].transform('count')

    return df

# ----------------------------- Step 3: Data Preprocessing -----------------------------

def preprocess_data(df):
    # Separate features and target
    X = df.drop(['Class', 'Amount'], axis=1)  # Drop unnecessary columns
    y = df['Class']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

# ----------------------------- Step 4: Build the Advanced Model -----------------------------

def create_advanced_model(input_dim):
    model = Sequential()

    # Input layer with batch normalization
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Hidden layers with increased neurons and batch normalization
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ----------------------------- Step 5: Train and Evaluate -----------------------------

class TrainingProgress(Callback):
    """Custom Callback for Visual Progress"""
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1}/{self.params['epochs']}:")
        self.progress_bar = tqdm(total=self.params['steps'], unit='batch', leave=False)
    
    def on_batch_end(self, batch, logs=None):
        self.progress_bar.update(1)
        self.progress_bar.set_description(f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.close()
        print(f"End of Epoch {epoch + 1}: val_loss={logs['val_loss']:.4f}, val_accuracy={logs['val_accuracy']:.4f}")

def train_and_evaluate_advanced_model(X_train, X_test, y_train, y_test):
    model = create_advanced_model(input_dim=X_train.shape[1])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Training progress visual in terminal
    progress = TrainingProgress()

    # Train the model
    model.fit(X_train, y_train, 
              epochs=50, 
              batch_size=128, 
              validation_split=0.2, 
              callbacks=[progress, early_stopping],
              verbose=0)  # Disable built-in verbose

    # Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    return model

# ----------------------------- Step 6: Save the Trained Model -----------------------------

def save_model(model):
    """Save the trained model to a file."""
    model.save('fraud_detection_model.h5')
    print("Model saved to 'fraud_detection_model.h5'.")

# ----------------------------- Step 7: Model Explainability -----------------------------

def explain_model(model, X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test)

# ----------------------------- Main Execution -----------------------------

if __name__ == '__main__':
    # Step 1: Load dataset
    file_path = 'creditcard_2023.csv'  # Adjust path if needed
    df = load_dataset(file_path)

    # Step 2: Add new features
    df = add_features(df)

    # Step 3: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Step 4 & 5: Train and evaluate the advanced model
    model = train_and_evaluate_advanced_model(X_train, X_test, y_train, y_test)

    # Step 6: Save the trained model for deployment
    save_model(model)

    # Step 7: Explain the model
    explain_model(model, X_train, X_test)