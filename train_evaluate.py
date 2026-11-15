import numpy as np
from sklearn.linear_model  import LinearRegression
from sklearn.tree   import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics  import mean_squared_error, mean_absolute_error, r2_score


def train_all_models(X_train, X_test, y_train, selected: list):
 
    model_registry = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }

    models      = {}
    predictions = {}

    for name in selected:
        if name not in model_registry:
            continue
        model = model_registry[name]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        models[name]      = model
        predictions[name] = pred

    return models, predictions


def evaluate_model(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2


def predict_next_day(model, X_full, scaler):

    last_row = X_full.iloc[[-1]]          # shape (1, n_features)
    last_scaled = scaler.transform(last_row)
    prediction  = model.predict(last_scaled)
    return float(prediction[0])
