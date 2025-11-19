import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Hours_Studied": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                      3, 5, 7, 8, 9, 10, 11, 6, 4, 5],
    "Attendance": [60, 65, 70, 75, 80, 85, 90, 92, 95, 98,
                   70, 75, 80, 85, 88, 92, 96, 83, 78, 74],
    "Sleep_Hours": [4, 5, 6, 6, 7, 8, 8, 5, 6, 7,
                    5, 6, 7, 7, 8, 8, 9, 6, 5, 7],
    "Assignments_Completed": [4, 5, 6, 6, 7, 8, 9, 10, 10, 10,
                              5, 6, 7, 8, 9, 10, 10, 8, 6, 7]
}
# Create random number generator
rng = np.random.default_rng(42)  # 42 = random seed

# Final grade is roughly influenced by all 4 features + small randomness
data["Final_Score"] = (
    3 * np.array(data["Hours_Studied"]) +
    0.4 * np.array(data["Attendance"]) +
    2 * np.array(data["Assignments_Completed"]) +
    rng.standard_normal(20) * 3  # noise
)

df = pd.DataFrame(data)
print(df.head())

X = df[["Hours_Studied", "Attendance", "Sleep_Hours", "Assignments_Completed"]]
y = df["Final_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

tree_low = DecisionTreeRegressor(max_depth=2, random_state=42,ccp_alpha=0.0)
tree_low.fit(X_train, y_train)
y_pred_low = tree_low.predict(X_test)

tree_high = DecisionTreeRegressor(max_depth=8, random_state=42,ccp_alpha=0.0)
tree_high.fit(X_train, y_train)
y_pred_high = tree_high.predict(X_test)

models = {
    "Linear Regression": lr,
    "Decision Tree (Low Depth)": tree_low,
    "Decision Tree (High Depth)": tree_high
}

for name, model in models.items():
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    print(f"{name} → Train MSE: {mse_train:.2f}, Test MSE: {mse_test:.2f}, R²: {r2:.2f}")

plt.figure(figsize=(8,5))
plt.plot(y_test.values, label="Actual", color="blue", marker="o")
plt.plot(y_pred_lr, label="Linear Regression", color="green", marker="x")
plt.plot(y_pred_low, label="Tree Low Depth", color="orange", marker="s")
plt.plot(y_pred_high, label="Tree High Depth", color="red", marker="^")
plt.title("Model Comparison: Actual vs Predicted Final Scores")
plt.xlabel("Test Sample Index")
plt.ylabel("Final Score")
plt.legend()
plt.show()

train_scores, test_scores = [], []
depths = range(1, 11)

for d in depths:
    model = DecisionTreeRegressor(max_depth=d, random_state=42,ccp_alpha=0.0)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.plot(depths, train_scores, label="Train R²")
plt.plot(depths, test_scores, label="Test R²")
plt.xlabel("Tree Depth")
plt.ylabel("R² Score")
plt.title("Bias–Variance Trade-off Curve")
plt.legend()
plt.show()
