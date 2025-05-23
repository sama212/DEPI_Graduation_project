import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Function to load and prepare data
def load_data(url):
    print("Loading dataset...")
    df = pd.read_csv(url)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# Function to train and evaluate models
def train_and_evaluate_models(X, y):
    # Split data for initial evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Define configurations to test
    rf_configs = [
        {"n_estimators": 50, "max_depth": 3, "min_samples_split": 5, "max_features": "sqrt", "name": "RF-50-depth3"},
        {"n_estimators": 100, "max_depth": 4, "min_samples_split": 5, "max_features": "sqrt", "name": "RF-100-depth4"},
        {"n_estimators": 100, "max_depth": 5, "min_samples_split": 5, "max_features": "sqrt", "name": "RF-100-depth5"},
        {"n_estimators": 200, "max_depth": 3, "min_samples_split": 10, "max_features": "sqrt", "name": "RF-200-depth3-mss10"}
    ]
    
    best_model = None
    best_score = 0
    best_config = None
    
    # Test each configuration
    for config in rf_configs:
        print(f"\nTesting configuration: {config['name']}")
        
        rf = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            max_features=config["max_features"],
            random_state=42
        )
        
        # Perform cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf, X, y, cv=cv)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"Cross-validation accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # Train on full training set and evaluate on test set
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        # Calculate precision, recall, and F1 score
        report = classification_report(y_test, y_pred, output_dict=True)
        print("Classification Report:")
        print(f"Precision: {report['1']['precision']:.4f}")
        print(f"Recall: {report['1']['recall']:.4f}")
        print(f"F1 Score: {report['1']['f1-score']:.4f}")
        
        # Check if this is the best model so far
        if cv_mean > best_score:
            best_score = cv_mean
            best_config = config
            best_model = rf
    
    return best_model, best_config, best_score

# Function to save the model
def save_model(model, feature_names, config, accuracy, feature_importance):
    # Save with pickle for Python applications
    with open('heart_disease_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata and model configuration as JSON for reference
    model_data = {
        "config": config,
        "featureNames": feature_names,
        "accuracy": float(accuracy),
        "featureImportance": feature_importance,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open('heart_disease_model_metadata.json', 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print("\nModel saved to heart_disease_model.pkl")
    print("Model metadata saved to heart_disease_model_metadata.json")

# Main function
def main():
    # Load data
    url = "cleaned_data.csv"
    data = load_data(url)
    
    # Basic data exploration
    print("\nFirst few rows:")
    print(data.head())
    print("\nColumn names:", data.columns.tolist())
    
    # Extract features and target
    X = data.drop('num', axis=1)
    y = (data['num'] >= 0.5).astype(int)
    
    # Count classes
    class_counts = y.value_counts()
    print("\nClass distribution:")
    print(class_counts)
    
    # Train and evaluate models
    best_model, best_config, best_score = train_and_evaluate_models(X, y)
    
    print("\n=== Best Model ===")
    print(f"Configuration: {best_config['name']}")
    print(f"Cross-validation accuracy: {best_score:.4f}")
    
    # Train best model on full dataset
    best_model = RandomForestClassifier(
        n_estimators=best_config["n_estimators"],
        max_depth=best_config["max_depth"],
        min_samples_split=best_config["min_samples_split"],
        max_features=best_config["max_features"],
        random_state=42
    )
    
    best_model.fit(X, y)
    
    # Get feature importance
    feature_importance = best_model.feature_importances_
    feature_names = X.columns.tolist()
    
    # Create sorted feature importance list
    importance_array = [{"feature": name, "importance": float(importance)} 
                        for name, importance in zip(feature_names, feature_importance)]
    importance_array.sort(key=lambda x: x["importance"], reverse=True)
    
    print("\nFeature Importance:")
    for item in importance_array[:10]:  # Show top 10
        print(f"{item['feature']}: {item['importance']:.4f}")
    
    # Save the model
    save_model(best_model, feature_names, best_config, best_score, importance_array)

if __name__ == "__main__":
    main()