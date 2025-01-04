import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Added imports

# Attempt to import ollama
try:
    import ollama
except ImportError:
    ollama = None
    print("The ollama library is not found or not installed.")

def get_data_summary(df: pd.DataFrame, y: pd.Series):
    """
    Collects a simple summary for each feature:
      - Data type (numeric / object)
      - Number of missing values
      - Number of unique values
      - Correlation with the target (if the feature is numeric)
    Returns a list of dictionaries.
    """
    summary = []
    for col in df.columns:
        col_data = df[col]
        dtype_str = str(col_data.dtype)
        missing_count = col_data.isnull().sum()
        unique_count = col_data.nunique()

        corr_with_target = None
        # If the feature is numeric and y is numeric, calculate Pearson correlation
        if pd.api.types.is_numeric_dtype(col_data) and pd.api.types.is_numeric_dtype(y):
            cval = df[col].corr(y)
            corr_with_target = round(float(cval), 4)

        summary.append({
            "column": col,
            "type": dtype_str,
            "missing": int(missing_count),
            "unique": int(unique_count),
            "corr_with_target": corr_with_target
        })
    return summary

def ask_ollama_for_features(summary_info, target_col, model_goal="regression", model_name="llama3.2"):
    """
    Sends a request to Ollama to return a JSON with a list of recommended features.
    Returns a list of strings (column names) or None if parsing fails.
    NOTE: The model name (model_name) should match what you see in 'ollama list'.
    """

    if not ollama:
        print("Ollama SDK not found. Returning None.")
        return None

    # Format the summary as JSON
    summary_text = json.dumps(summary_info, indent=2)

    # Create a prompt. The stricter, the better.
    prompt_text = f"""
Return strictly JSON without extra comments.

I have a dataset where the column '{target_col}' is the target variable (task '{model_goal}').
Below is brief information about other columns (in JSON format).

Please return the list of recommended columns as an array of strings. For example:

["Age", "Weather", "Traffic_Level"]

{summary_text}
"""

    res = ollama.generate(
        model=model_name,
        prompt=prompt_text
    )

    print("Response from Ollama:", res['response']) 

    try:
        recommended_cols = json.loads(res['response'])
        # If elements are dictionaries, extract the necessary field
        if isinstance(recommended_cols, list):
            if recommended_cols and isinstance(recommended_cols[0], dict):
                # Assume the key 'column' or 'name' contains the column name
                # Try both options
                if 'column' in recommended_cols[0]:
                    recommended_cols = [item['column'] for item in recommended_cols if 'column' in item]
                elif 'name' in recommended_cols[0]:
                    recommended_cols = [item['name'] for item in recommended_cols if 'name' in item]
                else:
                    print("Unknown format of dictionaries in recommended columns.")
                    return None
            elif recommended_cols and isinstance(recommended_cols[0], str):
                # List of strings - all good
                pass
            else:
                print("Unknown format of recommended columns.")
                return None
        else:
            print("Recommended columns should be a list.")
            return None
        return recommended_cols
    except json.JSONDecodeError:
        print("Error parsing JSON from Ollama.")
        return None   

def match_columns_with_dummies(df_enc: pd.DataFrame, recommended_cols: list, original_cat_cols: list) -> list:
    """
    Matches recommended columns (which may be either original categorical or specific dummy names)
    with actual names in df_enc.columns.

    Logic:
    - If 'col' is directly in df_enc.columns, take it.
    - If 'col' was an original categorical column (e.g., 'Weather'),
      then take all columns starting with 'Weather_' (dummy variables).
    - Otherwise, ignore it.
    """
    final_columns = set()
    all_enc_cols = set(df_enc.columns)

    for col in recommended_cols:
        # If the recommended column is already in df_enc, take it directly:
        if col in all_enc_cols:
            final_columns.add(col)
            continue

        # If the recommendation is an original category, take all dummy columns starting with 'col_'
        if col in original_cat_cols:
            pattern_prefix = col + "_"  # e.g., "Weather_"
            matching = [c for c in all_enc_cols if c.startswith(pattern_prefix)]
            if matching:
                final_columns.update(matching)

        # If nothing is found, simply skip this feature
        # Optionally print a message:
        # else:
        #     print(f"LLM recommended '{col}', but no such dummy column exists.")
    return list(final_columns)

def demo_llm_feature_selection(csv_file: str, target_col: str, model_name: str = "llama3.2"):
    """
    1) Load CSV
    2) Handle missing values
    3) One-Hot Encoding for all categorical columns
    4) Generate summary for columns
    5) Query Ollama => list of features
    6) Match recommended features with dummy columns
    Returns prepared data for further use.
    """
    print("=== STEP 1. Loading data ===")
    df = pd.read_csv(csv_file)
    print(f"Data loaded. Shape: {df.shape}")

    if target_col not in df.columns:
        raise ValueError(f"The column '{target_col}' is not in the dataset.")

    # Remove rows where target == NaN
    initial_shape = df.shape
    df = df.dropna(subset=[target_col])
    print(f"Removed {initial_shape[0] - df.shape[0]} rows with NaN in the target column.")

    # Split into X and y before encoding
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Remember which columns were originally categorical
    cat_cols = [col for col in X.columns if X[col].dtype == object]
    num_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]

    # === STEP 2. Handle missing values in original X (before One-Hot) ===
    print("\n=== STEP 2. Handling missing values ===")
    # Fill numeric missing values with the median
    for col in num_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"Filled missing values in numeric column '{col}' with median: {median_val}")

    # Fill categorical missing values with the mode
    for col in cat_cols:
        if X[col].isnull().any():
            mode_val = X[col].mode().iloc[0]
            X[col] = X[col].fillna(mode_val)
            print(f"Filled missing values in categorical column '{col}' with mode: {mode_val}")

    # === STEP 3. One-Hot Encoding for all categorical columns ===
    print("\n=== STEP 3. One-Hot Encoding ===")
    df_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    print(f"After One-Hot Encoding: {df_enc.shape[1]} columns.")

    # === STEP 4. Generate column summary ===
    print("\n=== STEP 4. Generating column summary ===")
    summary_info = get_data_summary(df_enc, y)
    print("Summary (first 5):", summary_info[:5])

    # === STEP 5. Query Ollama for feature selection ===
    print("\n=== STEP 5. Querying Ollama for feature selection ===")
    recommended_cols = ask_ollama_for_features(summary_info, target_col, "regression", model_name=model_name)

    if not recommended_cols:
        print("LLM failed to recommend features or parsing failed.")
        recommended_cols = [c for c in df_enc.columns]
    else:
        print("Recommended columns by LLM:", recommended_cols)

    # === STEP 6. Match recommended columns with df_enc.columns ===
    print("\n=== STEP 6. Matching recommended columns ===")
    final_cols = match_columns_with_dummies(df_enc, recommended_cols, cat_cols)
    if not final_cols:
        print("None of the recommended columns matched with dummy columns.")
        # fallback: take everything
        final_cols = list(df_enc.columns)
    else:
        print("Final set of features after matching:", final_cols)

    # Form X_final
    X_final = df_enc[final_cols].copy()

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, random_state=42)

    # Return prepared data
    return X_train, X_test, y_train, y_test, final_cols
