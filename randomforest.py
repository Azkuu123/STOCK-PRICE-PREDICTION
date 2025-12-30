from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def train_random_forest(stock_data, input_data):
    # Split features and target from stock_data
    X_train = stock_data[['Open', 'High', 'Low', 'Volume']]
    y_train = stock_data['Close']

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Accuracy on training data
    y_pred_train = model.predict(X_train)
    accuracy = r2_score(y_train, y_pred_train)

    # Predict on new input
    final_prediction = model.predict(input_data[['Open', 'High', 'Low', 'Volume']])

    return accuracy, final_prediction
