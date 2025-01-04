# feature_selection.py

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Добавленные импорты

# Попытка импорта ollama
try:
    import ollama
except ImportError:
    ollama = None
    print("Библиотека ollama не найдена или не установлена.")


def get_data_summary(df: pd.DataFrame, y: pd.Series):
    """
    Собирает простую сводку по каждому признаку:
      - Тип данных (числовой / object)
      - Кол-во пропусков
      - Число уникальных значений
      - Корреляцию с таргетом (если признак числовой)
    Возвращает список словарей.
    """
    summary = []
    for col in df.columns:
        col_data = df[col]
        dtype_str = str(col_data.dtype)
        missing_count = col_data.isnull().sum()
        unique_count = col_data.nunique()

        corr_with_target = None
        # Если признак числовой и y числовой, посчитаем корреляцию Пирсона
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
    Отправляет запрос к Ollama, чтобы она вернула JSON со списком рекомендуемых признаков.
    Возвращает список строк (названий столбцов) или None, если парсинг не удался.
    ПРИМЕЧАНИЕ: имя модели (model_name) должно совпадать с тем, что вы видите в 'ollama list'.
    """

    if not ollama:
        print("Ollama SDK не найден. Возвращаем None.")
        return None

    # Формируем сводку в JSON
    summary_text = json.dumps(summary_info, indent=2)

    # Формируем prompt. Чем строже, тем лучше.
    prompt_text = f"""
Верни строго JSON без лишних комментариев.

У меня есть датасет, в котором столбец '{target_col}' — это целевая переменная (задача '{model_goal}').
Ниже краткая информация о других столбцах (в JSON-формате).

Пожалуйста, верни список рекомендуемых столбцов как массив строк. Например:

["Age", "Weather", "Traffic_Level"]

{summary_text}
"""

    res = ollama.generate(
        model=model_name,
        prompt=prompt_text
    )

    print("Ответ от Ollama:", res['response']) 

    try:
        recommended_cols = json.loads(res['response'])
        # Проверяем, если элементы являются dict, извлекаем нужное поле
        if isinstance(recommended_cols, list):
            if recommended_cols and isinstance(recommended_cols[0], dict):
                # Предполагаем, что ключ 'column' или 'name' содержит название колонки
                # Попробуем оба варианта
                if 'column' in recommended_cols[0]:
                    recommended_cols = [item['column'] for item in recommended_cols if 'column' in item]
                elif 'name' in recommended_cols[0]:
                    recommended_cols = [item['name'] for item in recommended_cols if 'name' in item]
                else:
                    print("Неизвестный формат словарей в рекомендованных колонках.")
                    return None
            elif recommended_cols and isinstance(recommended_cols[0], str):
                # Список строк — всё в порядке
                pass
            else:
                print("Неизвестный формат рекомендованных колонок.")
                return None
        else:
            print("Рекомендованные колонки должны быть списком.")
            return None
        return recommended_cols
    except json.JSONDecodeError:
        print("Ошибка парсинга JSON от Ollama.")
        return None   


def match_columns_with_dummies(df_enc: pd.DataFrame, recommended_cols: list, original_cat_cols: list) -> list:
    """
    Сопоставляет рекомендованные столбцы (которые могут быть либо исходными категорическими,
    либо конкретными dummy-именами) с реальными именами в df_enc.columns.

    Логика:
    - Если 'col' входит напрямую в df_enc.columns, берём его.
    - Если 'col' был исходной категорической колонкой (например, 'Weather'),
      то берём все столбцы, начинающиеся с 'Weather_' (dummy-переменные).
    - Иначе игнорируем.
    """
    final_columns = set()
    all_enc_cols = set(df_enc.columns)

    for col in recommended_cols:
        # Если рекомендация колонки уже в df_enc, берём напрямую:
        if col in all_enc_cols:
            final_columns.add(col)
            continue

        # Если рекомендация — это исходная категория, берём все dummy-колонки, начинающиеся на 'col_'
        if col in original_cat_cols:
            pattern_prefix = col + "_"  # например, "Weather_"
            matching = [c for c in all_enc_cols if c.startswith(pattern_prefix)]
            if matching:
                final_columns.update(matching)

        # Если ничего не нашли, мы просто пропустим эту фичу
        # Можно вывести сообщение:
        # else:
        #     print(f"LLM рекомендовала '{col}', но такой dummy-колонки нет.")
    return list(final_columns)


def demo_llm_feature_selection(csv_file: str, target_col: str, model_name: str = "llama3.2"):
    """
    1) Загрузка CSV
    2) Обработка пропусков
    3) One-Hot Encoding для всех категориальных колонок
    4) Формирование сводки по столбцам
    5) Запрос к Ollama => список признаков
    6) Сопоставление рекомендованных фич с dummy-колонками
    Возвращает подготовленные данные для дальнейшего использования.
    """
    print("=== ШАГ 1. Загрузка данных ===")
    df = pd.read_csv(csv_file)
    print(f"Данные загружены. Размер: {df.shape}")

    if target_col not in df.columns:
        raise ValueError(f"В данных нет столбца '{target_col}'.")

    # Уберём строки, где target == NaN
    initial_shape = df.shape
    df = df.dropna(subset=[target_col])
    print(f"Удалено {initial_shape[0] - df.shape[0]} строк с NaN в целевом столбце.")

    # Разделяем на X и y до кодирования
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Запомним, какие столбцы изначально были категориальными
    cat_cols = [col for col in X.columns if X[col].dtype == object]
    num_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]

    # === ШАГ 2. Обработка пропусков в исходном X (до One-Hot) ===
    print("\n=== ШАГ 2. Обработка пропусков ===")
    # Заполним числовые пропуски медианой
    for col in num_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"Заполнены пропуски в числовом столбце '{col}' медианой: {median_val}")

    # Заполним категориальные пропуски модой
    for col in cat_cols:
        if X[col].isnull().any():
            mode_val = X[col].mode().iloc[0]
            X[col] = X[col].fillna(mode_val)
            print(f"Заполнены пропуски в категориальном столбце '{col}' модой: {mode_val}")

    # === ШАГ 3. One-Hot Encoding для всех категориальных столбцов ===
    print("\n=== ШАГ 3. One-Hot Encoding ===")
    df_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    print(f"После One-Hot Encoding: {df_enc.shape[1]} столбцов.")

    # === ШАГ 4. Сбор сводки по столбцам ===
    print("\n=== ШАГ 4. Сбор сводки по столбцам ===")
    summary_info = get_data_summary(df_enc, y)
    print("Сводка (первые 5):", summary_info[:5])

    # === ШАГ 5. Запрос к Ollama для выбора признаков ===
    print("\n=== ШАГ 5. Запрос к Ollama для выбора признаков ===")
    recommended_cols = ask_ollama_for_features(summary_info, target_col, "regression", model_name=model_name)

    if not recommended_cols:
        print("LLM не смог посоветовать признаки или парсинг не удался.")
        recommended_cols = [c for c in df_enc.columns]
    else:
        print("Рекомендованные столбцы от LLM:", recommended_cols)

    # === ШАГ 6. Сопоставляем рекомендованные колонки с df_enc.columns ===
    print("\n=== ШАГ 6. Сопоставление рекомендованных колонок ===")
    final_cols = match_columns_with_dummies(df_enc, recommended_cols, cat_cols)
    if not final_cols:
        print("Ни один из рекомендованных столбцов не сопоставился с dummy-колонками.")
        # fallback: берём всё
        final_cols = list(df_enc.columns)
    else:
        print("Итоговый набор признаков после сопоставления:", final_cols)

    # Формируем X_final
    X_final = df_enc[final_cols].copy()

    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, random_state=42)

    # Возвращаем подготовленные данные
    return X_train, X_test, y_train, y_test, final_cols
