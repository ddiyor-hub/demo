import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import random

# Set initial seed for random number generators
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Import functions from feature_selection.py
from feature_selection import demo_llm_feature_selection

def tune_xgb(X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBRegressor

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 1.5]
    }

    xgb = XGBRegressor(random_state=SEED, use_label_encoder=False, eval_metric='rmse')

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,   # Set to 1 if full determinism is required
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("Best parameters for XGBRegressor:", grid_search.best_params_)
    best_xgb = grid_search.best_estimator_

    return best_xgb

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates 30 different regression models.
    Returns a DataFrame with results and the best models by R² and RMSE.
    """
    from sklearn.linear_model import (
        Ridge, Lasso, ElasticNet, BayesianRidge, ARDRegression, HuberRegressor, 
        TheilSenRegressor, RANSACRegressor, SGDRegressor, PassiveAggressiveRegressor, 
        OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, LinearRegression
    )
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.svm import SVR, NuSVR, LinearSVR
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.neural_network import MLPRegressor

    try:
        from xgboost import XGBRegressor
    except ImportError:
        XGBRegressor = None
        print("The xgboost library is not installed. XGBoost models will not be available.")

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        LGBMRegressor = None
        print("The lightgbm library is not installed. LightGBM models will not be available.")

    try:
        from catboost import CatBoostRegressor
    except ImportError:
        CatBoostRegressor = None
        print("The catboost library is not installed. CatBoost models will not be available.")

    models = [
        ('LinearRegression', LinearRegression()),
        ('Ridge', Ridge(random_state=SEED)),
        ('Lasso', Lasso(random_state=SEED)),
        ('ElasticNet', ElasticNet(random_state=SEED)),
        ('BayesianRidge', BayesianRidge()),
        ('ARDRegression', ARDRegression()),
        ('HuberRegressor', HuberRegressor()),
        ('TheilSenRegressor', TheilSenRegressor(random_state=SEED)),
        ('RANSACRegressor', RANSACRegressor(random_state=SEED)),
        ('KNeighborsRegressor', KNeighborsRegressor()),
        ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=SEED)),
        ('RandomForestRegressor', RandomForestRegressor(random_state=SEED)),
        ('ExtraTreesRegressor', ExtraTreesRegressor(random_state=SEED)),
        ('GradientBoostingRegressor', GradientBoostingRegressor(random_state=SEED)),
        ('AdaBoostRegressor', AdaBoostRegressor(random_state=SEED)),
        ('SVR', SVR()),
        ('NuSVR', NuSVR()),
        ('LinearSVR', LinearSVR(random_state=SEED)),
        ('SGDRegressor', SGDRegressor(random_state=SEED)),
        ('PassiveAggressiveRegressor', PassiveAggressiveRegressor(random_state=SEED)),
        ('OrthogonalMatchingPursuit', OrthogonalMatchingPursuit()),
        ('OrthogonalMatchingPursuitCV', OrthogonalMatchingPursuitCV()),
        ('GaussianProcessRegressor', GaussianProcessRegressor(random_state=SEED)),
        ('KernelRidge', KernelRidge()),
        ('MLPRegressor', MLPRegressor(random_state=SEED)),
    ]

    # Additionally, include various configurations of some models to reach 30
    additional_models = [
        ('Ridge_alpha_1.0', Ridge(alpha=1.0, random_state=SEED)),
        ('Ridge_alpha_10.0', Ridge(alpha=10.0, random_state=SEED)),
        ('Lasso_alpha_0.1', Lasso(alpha=0.1, random_state=SEED)),
        ('Lasso_alpha_1.0', Lasso(alpha=1.0, random_state=SEED)),
        ('ElasticNet_alpha_1.0', ElasticNet(alpha=1.0, random_state=SEED)),
        ('ElasticNet_alpha_0.1', ElasticNet(alpha=0.1, random_state=SEED)),
        ('RandomForestRegressor_n_estimators_100', RandomForestRegressor(n_estimators=100, random_state=SEED)),
        ('RandomForestRegressor_n_estimators_200', RandomForestRegressor(n_estimators=200, random_state=SEED)),
        ('GradientBoostingRegressor_learning_rate_0.1', GradientBoostingRegressor(learning_rate=0.1, random_state=SEED)),
        ('GradientBoostingRegressor_learning_rate_0.05', GradientBoostingRegressor(learning_rate=0.05, random_state=SEED)),
        ('AdaBoostRegressor_n_estimators_50', AdaBoostRegressor(n_estimators=50, random_state=SEED)),
        ('AdaBoostRegressor_n_estimators_100', AdaBoostRegressor(n_estimators=100, random_state=SEED)),
        ('SVR_C_1.0', SVR(C=1.0)),
        ('SVR_C_10.0', SVR(C=10.0)),
        ('RandomForestRegressor_max_depth_10', RandomForestRegressor(max_depth=10, random_state=SEED)),
        ('RandomForestRegressor_max_depth_20', RandomForestRegressor(max_depth=20, random_state=SEED)),
        ('DecisionTreeRegressor_max_depth_10', DecisionTreeRegressor(max_depth=10, random_state=SEED)),
        ('DecisionTreeRegressor_max_depth_20', DecisionTreeRegressor(max_depth=20, random_state=SEED)),
        ('ExtraTreesRegressor_n_estimators_100', ExtraTreesRegressor(n_estimators=100, random_state=SEED)),
        ('ExtraTreesRegressor_n_estimators_200', ExtraTreesRegressor(n_estimators=200, random_state=SEED)),
        ('KNeighborsRegressor_n_neighbors_5', KNeighborsRegressor(n_neighbors=5)),
        ('KNeighborsRegressor_n_neighbors_10', KNeighborsRegressor(n_neighbors=10)),
        ('MLPRegressor_hidden_layer_sizes_100', MLPRegressor(hidden_layer_sizes=(100,), random_state=SEED)),
        ('MLPRegressor_hidden_layer_sizes_50', MLPRegressor(hidden_layer_sizes=(50,), random_state=SEED)),
    ]

    # Add XGBoost, LightGBM, and CatBoost if available
    if XGBRegressor:
        additional_models.extend([
            ('XGBRegressor_default', XGBRegressor(random_state=SEED)),
            ('XGBRegressor_max_depth_5', XGBRegressor(max_depth=5, random_state=SEED)),
            ('XGBRegressor_max_depth_10', XGBRegressor(max_depth=10, random_state=SEED)),
        ])
    if LGBMRegressor:
        additional_models.extend([
            ('LGBMRegressor_default', LGBMRegressor(random_state=SEED)),
            ('LGBMRegressor_num_leaves_31', LGBMRegressor(num_leaves=31, random_state=SEED)),
            ('LGBMRegressor_num_leaves_61', LGBMRegressor(num_leaves=61, random_state=SEED)),
        ])
    if CatBoostRegressor:
        additional_models.extend([
            ('CatBoostRegressor_default', CatBoostRegressor(random_state=SEED, verbose=0)),
            ('CatBoostRegressor_depth_6', CatBoostRegressor(depth=6, random_state=SEED, verbose=0)),
            ('CatBoostRegressor_depth_10', CatBoostRegressor(depth=10, random_state=SEED, verbose=0)),
        ])

    models.extend(additional_models)

    results = []
    for name, model in models:
        print(f"\nTraining model: {name}")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results.append({
                'Model': name,
                'R2': r2,
                'MAE': mae,
                'RMSE': rmse
            })
            print(f"Model {name} trained. R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        except Exception as e:
            print(f"Error training model {name}: {e}")

    # Create a DataFrame with results
    results_df = pd.DataFrame(results)

    # Sort models by R² (descending)
    best_model_r2 = results_df.sort_values(by='R2', ascending=False).iloc[0]
    # Sort models by RMSE (ascending)
    best_model_rmse = results_df.sort_values(by='RMSE').iloc[0]

    print("\n=== Best Models ===")
    print("\nModel with the highest R²:")
    print(best_model_r2)

    print("\nModel with the lowest RMSE:")
    print(best_model_rmse)

    # Return DataFrame with results and best models
    return results_df, best_model_r2, best_model_rmse

if __name__ == "__main__":
    # Path to your CSV file
    csv_path = "data/input/task.csv"       # Specify your CSV file
    target_column = "Delivery_Time_min"    # Replace with your target column
    MODEL_NAME = "llama3.2"                # One of the models listed by `ollama list`

    # Perform feature selection and obtain split data
    X_train, X_test, y_train, y_test, final_cols = demo_llm_feature_selection(
        csv_file=csv_path,
        target_col=target_column,
        model_name=MODEL_NAME
    )

    # STEP 1: Perform one-time training and save results
    results_df, best_model_r2, best_model_rmse = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # STEP 2: If XGBRegressor is in the list, perform precise GridSearchCV tuning
    if 'XGBRegressor_default' in results_df['Model'].unique():
        best_xgb = tune_xgb(X_train, y_train)
        y_pred = best_xgb.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"\nOptimized XGBRegressor R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # Save results to CSV
    results_df.to_csv('model_results.csv', index=False)
    print("\nModel results saved to 'model_results.csv'.")

    # Save best models
    best_model_r2.to_frame().T.to_csv('best_model_r2.csv', index=False)
    best_model_rmse.to_frame().T.to_csv('best_model_rmse.csv', index=False)
    print("Best models saved to 'best_model_r2.csv' and 'best_model_rmse.csv'.")
