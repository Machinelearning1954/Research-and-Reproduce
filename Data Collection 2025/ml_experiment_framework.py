# ml_experiment_framework.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.datasets import load_iris, load_wine # Example datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- 1.3.1. Implement Data Loading and Preprocessing --- 

def load_and_preprocess_data(dataset_name="iris", test_size=0.2, random_state=42):
    """
    Loads a specified dataset, performs basic preprocessing (train-test split, scaling for some).
    
    Args:
        dataset_name (str): Name of the dataset to load ("iris", "wine").
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test: Split data.
        feature_names: Names of the features.
        target_names: Names of the target classes (for classification).
    """
    if dataset_name == "iris":
        data = load_iris()
        problem_type = "classification"
    elif dataset_name == "wine":
        data = load_wine()
        problem_type = "classification"
    # Add more datasets here if needed (e.g., regression datasets)
    # elif dataset_name == "boston":
    #     data = load_boston() # Deprecated, find alternative if needed for regression
    #     problem_type = "regression"
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose 'iris' or 'wine'.")

    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names if hasattr(data, 'target_names') else None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if problem_type == "classification" else None)
    
    # Scaling is often beneficial, especially for SVM and Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Loaded dataset: {dataset_name}")
    print(f"Problem type: {problem_type}")
    print(f"X_train shape: {X_train_scaled.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test_scaled.shape}, y_test shape: {y_test.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_names, problem_type

# --- Placeholder for future functions from todo.md ---

if __name__ == '__main__':
    # Example usage of data loading
    print("Testing data loading and preprocessing...")
    X_train, X_test, y_train, y_test, features, targets, p_type = load_and_preprocess_data(dataset_name="iris")
    print(f"Features: {features}")
    if targets is not None:
        print(f"Target classes: {targets}")
    
    X_train_w, X_test_w, y_train_w, y_test_w, features_w, targets_w, p_type_w = load_and_preprocess_data(dataset_name="wine")
    print(f"Features (Wine): {features_w}")
    if targets_w is not None:
        print(f"Target classes (Wine): {targets_w}")



# --- 1.3.2. Design Model Training Pipeline --- 
# --- 1.3.3. Implement Cross-Validation --- 
# --- 1.3.4. Define Performance Metrics --- 
# --- 1.3.5. Implement Model Evaluation and Comparison --- 
# --- 1.3.6. Ensure Reproducibility ---

def get_models(random_state=42):
    """
    Returns a dictionary of models to be evaluated.
    """
    models = {
        "Logistic Regression": LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200),
        "SVM": SVC(random_state=random_state, probability=True), # probability=True for roc_auc
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
    }
    return models

def get_performance_metrics(problem_type="classification"):
    """
    Returns a dictionary of performance metrics based on the problem type.
    """
    if problem_type == "classification":
        return {
            "accuracy": accuracy_score,
            "precision_macro": lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro', zero_division=0),
            "recall_macro": lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0),
            "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro', zero_division=0),
            # "roc_auc_ovr": lambda y_true, y_pred_proba: roc_auc_score(y_true, y_pred_proba, multi_class='ovr') # Requires y_pred_proba
        }
    elif problem_type == "regression":
        return {
            "mse": mean_squared_error,
            "r2": r2_score
        }
    else:
        raise ValueError(f"Problem type {problem_type} not supported for metrics.")

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, problem_type="classification", cv_folds=5):
    """
    Trains a single model, evaluates it using cross-validation and on the test set.
    
    Args:
        model: The machine learning model instance.
        X_train, y_train: Training data and labels.
        X_test, y_test: Test data and labels.
        problem_type (str): "classification" or "regression".
        cv_folds (int): Number of folds for cross-validation.
        
    Returns:
        dict: A dictionary containing model name, cross-validation scores, and test set scores.
    """
    model_name = model.__class__.__name__
    start_time = time.time()
    
    # Cross-validation
    cv_results = {}
    scoring_metrics_cv = []
    if problem_type == "classification":
        scoring_metrics_cv = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'] # 'roc_auc_ovr' requires predict_proba
    elif problem_type == "regression":
        scoring_metrics_cv = ['neg_mean_squared_error', 'r2']

    for metric_name in scoring_metrics_cv:
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=metric_name)
            cv_results[f"cv_{metric_name}_mean"] = np.mean(scores)
            cv_results[f"cv_{metric_name}_std"] = np.std(scores)
        except Exception as e:
            print(f"Could not compute CV for {metric_name} on {model_name}: {e}")
            cv_results[f"cv_{metric_name}_mean"] = np.nan
            cv_results[f"cv_{metric_name}_std"] = np.nan

    # Train on full training set and evaluate on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    test_scores = {}
    metrics_to_calculate = get_performance_metrics(problem_type)
    
    for name, metric_func in metrics_to_calculate.items():
        try:
            if name == "roc_auc_ovr" and problem_type == "classification": # Requires predict_proba
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)
                    # Ensure y_pred_proba has the correct shape for multi-class
                    if y_pred_proba.ndim == 1: # Binary case, might need reshaping or specific handling
                         # For binary, roc_auc_score expects probabilities of the positive class
                        if y_pred_proba.shape[0] == X_test.shape[0] and len(np.unique(y_train)) == 2:
                             test_scores[name] = metric_func(y_test, y_pred_proba) if y_pred_proba.ndim == 1 else metric_func(y_test, y_pred_proba[:,1])
                        else: # Fallback for complex multiclass proba structures not directly usable by roc_auc_ovr
                            print(f"Skipping ROC AUC for {model_name} due to y_pred_proba shape issues for multi-class or binary mismatch.")
                            test_scores[name] = np.nan
                    elif y_pred_proba.shape[1] == len(np.unique(y_train)):
                        test_scores[name] = metric_func(y_test, y_pred_proba)
                    else:
                        print(f"Skipping ROC AUC for {model_name} due to y_pred_proba shape issues for multi-class.")
                        test_scores[name] = np.nan
                else:
                    test_scores[name] = np.nan # Model doesn't support predict_proba
            else:
                test_scores[name] = metric_func(y_test, y_pred)
        except Exception as e:
            print(f"Could not compute test metric {name} for {model_name}: {e}")
            test_scores[name] = np.nan
            
    end_time = time.time()
    training_time = end_time - start_time
    
    results = {
        "model_name": model_name,
        "training_time_seconds": training_time,
        **cv_results,
        **test_scores
    }
    
    return results

def run_experiment(X_train, y_train, X_test, y_test, problem_type="classification", random_state=42, cv_folds=5):
    """
    Runs the experiment by training and evaluating multiple models.
    
    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        problem_type (str): "classification" or "regression".
        random_state (int): Random seed for reproducibility.
        cv_folds (int): Number of folds for cross-validation.
        
    Returns:
        pd.DataFrame: A DataFrame containing the performance of all models.
    """
    models = get_models(random_state=random_state)
    all_results = []
    
    print("\nStarting model training and evaluation...")
    for name, model_instance in models.items():
        print(f"\nTraining and evaluating: {name}")
        try:
            model_results = train_and_evaluate_model(model_instance, X_train, y_train, X_test, y_test, 
                                                     problem_type=problem_type, cv_folds=cv_folds)
            all_results.append(model_results)
            print(f"Completed: {name}")
        except Exception as e:
            print(f"Error training/evaluating {name}: {e}")
            all_results.append({"model_name": name, "error": str(e)})
            
    results_df = pd.DataFrame(all_results)
    return results_df.set_index("model_name")

# --- Main execution block for testing the framework ---
if __name__ == '__main__':
    print("Testing data loading and preprocessing...")
    X_train_i, X_test_i, y_train_i, y_test_i, _, _, p_type_i = load_and_preprocess_data(dataset_name="iris", random_state=42)
    
    # Run experiment for Iris dataset
    iris_experiment_results = run_experiment(X_train_i, y_train_i, X_test_i, y_test_i, problem_type=p_type_i, random_state=42)
    print("\n--- Iris Dataset Experiment Results ---")
    print(iris_experiment_results)

    print("\nTesting data loading and preprocessing for Wine dataset...")
    X_train_w, X_test_w, y_train_w, y_test_w, _, _, p_type_w = load_and_preprocess_data(dataset_name="wine", random_state=42)
    
    # Run experiment for Wine dataset
    wine_experiment_results = run_experiment(X_train_w, y_train_w, X_test_w, y_test_w, problem_type=p_type_w, random_state=42)
    print("\n--- Wine Dataset Experiment Results ---")
    print(wine_experiment_results)

    # Example of how to add visualizations (will be expanded in Phase 2 of todo.md)
    if not iris_experiment_results.empty and 'accuracy' in iris_experiment_results.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=iris_experiment_results.index, y='accuracy', data=iris_experiment_results.reset_index())
        plt.title('Model Accuracy Comparison (Iris Dataset)')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.savefig("/home/ubuntu/iris_accuracy_comparison.png") # Save figure if needed
        # print("\nIris accuracy comparison plot generated (not saved by default in framework test).")
        plt.show() # This might not render in a non-interactive environment, consider saving.

