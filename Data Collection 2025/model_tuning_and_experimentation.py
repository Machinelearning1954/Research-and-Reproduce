# model_tuning_and_experimentation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# Import functions from our framework
from ml_experiment_framework import load_and_preprocess_data, get_models, train_and_evaluate_model, run_experiment

# --- Configuration ---
DATASET_NAME = "wine"  # Can be "iris" or "wine"
RANDOM_STATE = 42
CV_FOLDS = 5

# --- 1. Load and Preprocess Data using the framework ---
print(f"--- Loading and Preprocessing {DATASET_NAME} Dataset ---")
X_train, X_test, y_train, y_test, feature_names, target_names, problem_type = \
    load_and_preprocess_data(dataset_name=DATASET_NAME, random_state=RANDOM_STATE)

# --- 2. Run Baseline Experiment (all models with default parameters) using the framework ---
print(f"\n--- Running Baseline Experiment on {DATASET_NAME} Dataset ---")
baseline_results_df = run_experiment(X_train, y_train, X_test, y_test, 
                                     problem_type=problem_type, random_state=RANDOM_STATE, cv_folds=CV_FOLDS)
print("\nBaseline Model Performance:")
print(baseline_results_df)

# --- 3. Hyperparameter Tuning for a Selected Model (e.g., Random Forest) ---
print(f"\n--- Hyperparameter Tuning for Random Forest on {DATASET_NAME} Dataset ---")

# Define the parameter grid for Random Forest
# This is a basic grid, can be expanded for more thorough tuning
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

models = get_models(random_state=RANDOM_STATE)
rf_model_default = models["Random Forest"] # Get the default Random Forest model instance

# Using GridSearchCV for hyperparameter tuning
# We'll use 'f1_macro' as the scoring metric for tuning, common for classification
# If problem_type is regression, this would need to be adjusted (e.g., 'neg_mean_squared_error')
scoring_for_tuning = 'f1_macro' if problem_type == "classification" else 'r2'

grid_search_rf = GridSearchCV(estimator=rf_model_default, 
                              param_grid=rf_param_grid, 
                              cv=CV_FOLDS, 
                              scoring=scoring_for_tuning, 
                              n_jobs=-1,  # Use all available cores
                              verbose=1)

print("Starting GridSearchCV for Random Forest...")
grid_search_rf.fit(X_train, y_train)

print("GridSearchCV for Random Forest completed.")
print(f"Best parameters found for Random Forest: {grid_search_rf.best_params_}")
print(f"Best cross-validation {scoring_for_tuning} score for Random Forest: {grid_search_rf.best_score_:.4f}")

# Get the best Random Forest model from GridSearchCV
best_rf_model = grid_search_rf.best_estimator_

# --- 4. Evaluate the Tuned Model using the framework's evaluation function ---
print("\n--- Evaluating Tuned Random Forest Model ---")
tuned_rf_results = train_and_evaluate_model(best_rf_model, X_train, y_train, X_test, y_test, 
                                            problem_type=problem_type, cv_folds=CV_FOLDS)

tuned_rf_results_df = pd.DataFrame([tuned_rf_results]).set_index("model_name")
print("\nTuned Random Forest Performance:")
print(tuned_rf_results_df)

# --- 5. Compare Tuned Model with Baseline Models ---
print("\n--- Comparison: Baseline vs. Tuned Random Forest ---")

# Add tuned model results to the baseline results for comparison
# Ensure the tuned model's name is distinct if needed, or update the existing one
# For simplicity, let's rename the tuned model for clarity in the comparison table
tuned_rf_results_df_renamed = tuned_rf_results_df.rename(index={'RandomForestClassifier': 'Random Forest (Tuned)'})

comparison_df = pd.concat([baseline_results_df, tuned_rf_results_df_renamed])

# Select key metrics for comparison display
if problem_type == "classification":
    comparison_metrics = ['cv_f1_macro_mean', 'f1_macro', 'cv_accuracy_mean', 'accuracy', 'training_time_seconds']
else: # Regression
    comparison_metrics = ['cv_r2_mean', 'r2', 'cv_neg_mean_squared_error_mean', 'mse', 'training_time_seconds']

# Filter out columns that might not exist for all models (e.g. if some CV metrics failed for a model)
valid_comparison_metrics = [col for col in comparison_metrics if col in comparison_df.columns]

print("\nOverall Model Comparison (including Tuned Random Forest):")
print(comparison_df[valid_comparison_metrics])

# --- 6. Visualization of Results (Example) ---
print("\n--- Generating Performance Visualization ---")

# Plotting F1-score (test set) for classification, or R2 for regression
plot_metric = 'f1_macro' if problem_type == "classification" else 'r2'

if plot_metric in comparison_df.columns:
    plt.figure(figsize=(12, 7))
    # Ensure data is sorted for better visualization if desired, e.g., by the metric
    # plot_data = comparison_df.reset_index().sort_values(by=plot_metric, ascending=False)
    plot_data = comparison_df.reset_index()
    sns.barplot(x='model_name', y=plot_metric, data=plot_data, palette='viridis')
    plt.title(f'Model Comparison: Test Set {plot_metric.replace("_", " ").title()} ({DATASET_NAME} Dataset)')
    plt.ylabel(f'{plot_metric.replace("_", " ").title()}')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    plot_filename = f"/home/ubuntu/{DATASET_NAME}_model_comparison_{plot_metric}.png"
    try:
        plt.savefig(plot_filename)
        print(f"Performance comparison plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show() # plt.show() might not work in all non-interactive environments
else:
    print(f"Metric '{plot_metric}' not found in results for plotting.")

print("\n--- Model Tuning and Experimentation Script Completed ---")

