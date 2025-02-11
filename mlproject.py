# -*- coding: utf-8 -*-
"""MLProject.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11QXtopSOV9TCYEeuwBtf4CD-WV7ehTTq
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv('CarPrice_Assignment.csv')

    # Create feature for car company from CarName
    df['company'] = df['CarName'].apply(lambda x: x.split(' ')[0])

    # Drop original CarName column
    df.drop('CarName', axis=1, inplace=True)

    # Convert categorical variables
    categorical_features = ['fueltype', 'aspiration', 'doornumber', 'carbody',
                          'drivewheel', 'enginelocation', 'enginetype',
                          'cylindernumber', 'fuelsystem', 'company']

    label_encoder = LabelEncoder()
    for feature in categorical_features:
        df[feature] = label_encoder.fit_transform(df[feature])

    # Split features and target
    X = df.drop('price', axis=1)
    y = df['price']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Save the scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return X_scaled, y, df

def train_and_evaluate_models(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }

    # Results dictionary
    results = {}

    for name, model in models.items():
        # Basic model training
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mpe = np.mean((y_test - y_pred) / y_test) * 100

        results[name] = {
            'R2 Score': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MPE': mpe
        }

        # Save the model
        filename = f"models/{name.lower().replace(' ', '_')}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    # Save results to CSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model_performance.csv')

    return results, models

def hyperparameter_tuning(X, y, models):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tuned_results = {}

    param_grids = {
        'Decision Tree': {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }

    for name, model in models.items():
        if name in param_grids:
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='r2')
            grid_search.fit(X_train, y_train)

            y_pred = grid_search.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mpe = np.mean((y_test - y_pred) / y_test) * 100

            tuned_results[name] = {
                'R2 Score': r2,
                'RMSE': rmse,
                'MAE': mae,
                'MPE': mpe,
                'Best Parameters': grid_search.best_params_
            }

            # Save the tuned model
            filename = f"models/{name.lower().replace(' ', '_')}_tuned.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(grid_search.best_estimator_, f)

    return tuned_results

def create_visualizations(df, results, tuned_results, models):  # Added models parameter
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()

    # Model comparison before tuning
    metrics = ['R2 Score', 'RMSE', 'MAE', 'MPE']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        values = [results[model][metric] for model in results.keys()]
        bars = plt.bar(results.keys(), values)
        plt.title(f'{metric} Comparison (Before Tuning)')
        plt.xticks(rotation=45)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'visualizations/{metric.lower().replace(" ", "_")}_comparison.png')
        plt.close()

    # Create combined performance visualization
    plt.figure(figsize=(15, 8))
    x = np.arange(len(metrics))
    width = 0.25

    # Plot bars for each model
    model_names = list(results.keys())
    for i, model_name in enumerate(model_names):
        values = [results[model_name][metric] for metric in metrics]
        plt.bar(x + i*width, values, width, label=model_name)

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Algorithm Performance Comparison')
    plt.xticks(x + width, metrics, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/algorithm_comparison.png')
    plt.close()

    # Before vs After Tuning Comparison
    for model_name in tuned_results.keys():
        plt.figure(figsize=(12, 6))
        metrics_before = [results[model_name][metric] for metric in metrics]
        metrics_after = [tuned_results[model_name][metric] for metric in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        plt.bar(x - width/2, metrics_before, width, label='Before Tuning')
        plt.bar(x + width/2, metrics_after, width, label='After Tuning')

        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'{model_name} - Before vs After Tuning')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()

        # Add value labels
        for i, v in enumerate(metrics_before):
            plt.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom')
        for i, v in enumerate(metrics_after):
            plt.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'visualizations/{model_name.lower().replace(" ", "_")}_tuning_comparison.png')
        plt.close()

    # Feature Importance Plot for Random Forest
    if 'Random Forest' in results and 'Random Forest' in models:  # Check both conditions
        rf_model = models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': df.drop('price', axis=1).columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Most Important Features (Random Forest)')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png')
        plt.close()

def perform_clustering(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters

def main():
    print("Loading and preprocessing data...")
    X, y, df = load_and_preprocess_data()

    print("Training and evaluating models...")
    results, models = train_and_evaluate_models(X, y)

    print("Performing hyperparameter tuning...")
    tuned_results = hyperparameter_tuning(X, y, models)

    print("Creating visualizations...")
    create_visualizations(df, results, tuned_results, models)  # Pass models here

    print("Process completed successfully!")
    return results, tuned_results

if __name__ == "__main__":
    main()