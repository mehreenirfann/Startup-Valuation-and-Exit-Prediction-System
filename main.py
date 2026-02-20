### M. Irfan, S. Sari
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from scipy.stats import mode
from mpl_toolkits.mplot3d import Axes3D

## loading and preprocessing
data = pd.read_csv('startup_data.csv')
data["Startup Name"] = data["Startup Name"].str.extract(r'(\d+)')

data["Valuation (M USD)"] *= 0.25
data["Revenue (M USD)"] *= 0.5
data["Employees"] = data["Employees"] * 0.3

#display(data.head())

def preprocess_pipeline(df, is_training=True, stats=None):
    df_n = df.copy()

    cont = ["Funding Amount (M USD)", "Employees", "Revenue (M USD)", "Year Founded", "Market Share (%)"]
    cat = ["Region", "Industry"]

    for c in cont + cat:
        if c not in df_n.columns: df_n[c] = 0

    if is_training:
        means = df_n[cont].mean()
        stds = df_n[cont].std().replace(0, 1)
        stats = {'means': means, 'stds': stds}
    else:
        means, stds = stats['means'], stats['stds']

    df_n[cont] = (df_n[cont] - means) / stds

    for c in cat:
      df_n[c] = df_n[c].astype(str)
    df_n = pd.get_dummies(df_n, columns=cat, drop_first=True)

    if is_training:
        features = df_n.drop(["Startup Name", "Valuation (M USD)", "Exit Status"], axis=1, errors='ignore').columns.tolist()
        stats['features'] = features
    else:
        features = stats['features']
        for col in features:
            if col not in df_n.columns: df_n[col] = 0
        df_n = df_n[features]

    return df_n, stats

processed_df, saved_stats = preprocess_pipeline(data, is_training=True)

display(processed_df[saved_stats['features']].head())

X_all = processed_df[saved_stats['features']].values.astype(float)
y_val = processed_df["Valuation (M USD)"].values

labels = sorted(data["Exit Status"].unique())
exit_map = {k: v for v, k in enumerate(labels)}
inv_exit_map = {v: k for k, v in exit_map.items()}
y_exit = processed_df["Exit Status"].map(exit_map).values.astype(int)

## for the matrix (correlation)

finaldf = processed_df[saved_stats['features']].copy()
finaldf['TARGET_Valuation'] = y_val
finaldf['TARGET_Exit_Status'] = y_exit

## for the matrix (correlation)

plt.figure(figsize=(12, 10))
corr_matrix = finaldf.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Dataset Correlation Matrix")
plt.show()

## Helper Functions
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0: return 0.0
    return 1 - (ss_res / ss_tot)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def train_test_split(X, y, test_size=0.2):
    np.random.seed(42)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    return X[indices[:split_idx]], X[indices[split_idx:]], y[indices[:split_idx]], y[indices[split_idx:]]

def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def precision_per_class(cm):
    precisions = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        if tp + fp == 0: precisions.append(0.0)
        else: precisions.append(tp / (tp + fp))
    return np.array(precisions)

def recall_per_class(cm):
    recalls = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        if tp + fn == 0: recalls.append(0.0)
        else: recalls.append(tp / (tp + fn))
    return np.array(recalls)

def f1_per_class(precision, recall):
    f1 = []
    for p, r in zip(precision, recall):
        if p + r == 0: f1.append(0.0)
        else: f1.append(2 * p * r / (p + r))
    return np.array(f1)

def kappa_score(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    n_samples = np.sum(cm)
    po = np.trace(cm) / n_samples
    pe = 0
    for i in range(n_classes):
        row_sum = np.sum(cm[i, :])
        col_sum = np.sum(cm[:, i])
        pe += (row_sum * col_sum)
    pe /= (n_samples * n_samples)
    if pe == 1: return 1.0
    return (po - pe) / (1 - pe)

def macro_average(metric_array):
    return np.mean(metric_array)

## Regressors
class LinearRegression:
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        I = np.eye(X_b.shape[1]) * 1e-6
        self.theta = np.linalg.pinv(X_b.T @ X_b + I) @ X_b.T @ y
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)

class RidgeRegression:
    def __init__(self, lambda_=1.0):
      self.lambda_ = lambda_
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        I = np.eye(X_b.shape[1]); I[0, 0] = 0
        self.theta = np.linalg.inv(X_b.T @ X_b + self.lambda_ * I) @ X_b.T @ y
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)

class LassoRegression:
    def __init__(self, lambda_=0.1, iterations=1000, learning_rate=0.01):
        self.lambda_, self.iter, self.lr = lambda_, iterations, learning_rate
    def _soft_threshold(self, x, lambda_param):
        return np.sign(x) * np.maximum(np.abs(x) - lambda_param, 0)
    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n); self.b = 0
        for _ in range(self.iter):
            pred = X.dot(self.w) + self.b
            residuals = pred - y
            dw_mse = (1/m) * X.T.dot(residuals)
            db = (1/m) * np.sum(residuals)
            w_temp = self.w - self.lr * dw_mse
            self.b -= self.lr * db
            self.w = self._soft_threshold(w_temp, self.lambda_ * self.lr)
    def predict(self, X):
        return X.dot(self.w) + self.b

## Classifier
class TreeEstimator:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth; self.tree = None
    def fit(self, X, y):
      self.tree = self._grow(X, y)
    def predict(self, X):
      return np.array([self._traverse(x, self.tree) for x in X])
    def _grow(self, X, y, depth=0):
        if len(y) == 0: return 0
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(X) < 5:
            return np.bincount(y.astype(int)).argmax()
        n_features = X.shape[1]
        feat_idxs = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feat is None:
          return np.bincount(y.astype(int)).argmax()
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
          return np.bincount(y.astype(int)).argmax()
        return {"f": best_feat, "t": best_thresh, "l": self._grow(X[left_idx], y[left_idx], depth + 1), "r": self._grow(X[right_idx], y[right_idx], depth + 1)}
    def _best_split(self, X, y, feats):
        best_gain = -1; split_feat, split_thresh = None, None
        for f in feats:
            thresholds = np.unique(X[:, f])
            if len(thresholds) > 10: thresholds = np.percentile(thresholds, [25, 50, 75])
            for t in thresholds:
                gain = self._gini_gain(y, X[:, f], t)
                if gain > best_gain: best_gain = gain; split_feat = f; split_thresh = t
        return split_feat, split_thresh
    def _gini_gain(self, y, col, thresh):
        parent_gini = self._gini(y)
        left = y[col <= thresh]; right = y[col > thresh]
        if len(left) == 0 or len(right) == 0: return 0
        return parent_gini - ((len(left) / len(y)) * self._gini(left) + (len(right) / len(y)) * self._gini(right))
    def _gini(self, y):
        probs = np.bincount(y.astype(int)) / len(y)
        return 1 - np.sum(probs ** 2)
    def _traverse(self, x, node):
        if not isinstance(node, dict): return node
        if x[node["f"]] <= node["t"]:
            return self._traverse(x, node["l"])
        else:
            return self._traverse(x, node["r"])

class XGBoost:
    def __init__(self, n_estimators=40, learning_rate=0.05, max_depth=3):
        self.n_estimators = n_estimators; self.lr = learning_rate; self.max_depth = max_depth
        self.trees = []; self.classes = None; self.class_weights = None
    def fit(self, X, y):
        self.classes = np.unique(y); n_samples = X.shape[0]; K = len(self.classes)
        counts = np.bincount(y.astype(int)); total = np.sum(counts)
        self.class_weights = {c: total / (K * counts[c]) for c in self.classes}
        self.trees = [[] for _ in range(K)]
        for k, cls in enumerate(self.classes):
            y_bin = np.where(y == cls, 1, 0); preds = np.zeros(n_samples); weight = self.class_weights[cls]
            for _ in range(self.n_estimators):
                prob = self._sigmoid(preds)
                gradient = weight * (y_bin - prob)
                tree = TreeEstimator(max_depth=self.max_depth)
                tree.fit(X, (gradient > 0).astype(int))
                preds += self.lr * tree.predict(X)
                self.trees[k].append(tree)
    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classes)))
        for k in range(len(self.classes)):
            pred = np.zeros(X.shape[0])
            for tree in self.trees[k]: pred += self.lr * tree.predict(X)
            scores[:, k] = pred
        return self.classes[np.argmax(scores, axis=1)]
    def _sigmoid(self, x):
      return 1 / (1 + np.exp(-x))

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.alpha = None
        self.polarity = 1

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1: predictions[X_column < self.threshold] = -1
        else: predictions[X_column > self.threshold] = -1
        return predictions

class AdaBoost:
    def __init__(self, n_clf=50): self.n_clf = n_clf; self.models = {}
    def fit(self, X, y):
        n_samples, n_features = X.shape; self.classes = np.unique(y)
        for c in self.classes:
            y_binary = np.where(y == c, 1, -1); weights = np.full(n_samples, 1 / n_samples); self.models[c] = []
            for _ in range(self.n_clf):
                clf = DecisionStump(); min_error = float("inf")
                for feature_i in range(n_features):
                    X_column = X[:, feature_i]; thresholds = np.unique(X_column)
                    if len(thresholds) > 10: thresholds = np.percentile(thresholds, [20, 40, 60, 80])
                    for threshold in thresholds:
                        p = 1; predictions = np.ones(n_samples); predictions[X_column < threshold] = -1
                        error = np.sum(weights[y_binary != predictions])
                        if error > 0.5: error = 1 - error; p = -1
                        if error < min_error: min_error = error; clf.polarity = p; clf.threshold = threshold; clf.feature_idx = feature_i
                EPS = 1e-10; clf.alpha = 0.5 * np.log((1 - min_error + EPS) / (min_error + EPS))
                predictions = clf.predict(X)
                weights *= np.exp(-clf.alpha * y_binary * predictions); weights /= np.sum(weights)
                self.models[c].append(clf)
    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classes)))
        for idx, c in enumerate(self.classes):
            clf_sum = np.zeros(X.shape[0])
            for clf in self.models[c]:
              clf_sum += clf.alpha * clf.predict(X)
            scores[:, idx] = clf_sum
        return self.classes[np.argmax(scores, axis=1)]

class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y); self.params = {}
        for c in self.classes:
            X_c = X[y == c]
            self.params[c] = {"mean": X_c.mean(axis=0), "var": X_c.var(axis=0) + 1e-9, "prior": len(X_c) / len(X)}
    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = []
            for c in self.classes:
                mean = self.params[c]["mean"]; var = self.params[c]["var"]; prior = np.log(self.params[c]["prior"])
                likelihood = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((x - mean) ** 2) / var)
                class_scores.append(prior + likelihood)
            predictions.append(self.classes[np.argmax(class_scores)])
        return np.array(predictions)

## train test choose

print(" REGRESSION EVALUATION")

X_train, X_test, y_train, y_test = train_test_split(X_all, y_val)
best_reg_models = {}
best_reg_score = -float('inf')
final_best_regressor = None
final_best_reg_name = ""


ols = LinearRegression()
ols.fit(X_train, y_train)
ols_preds = ols.predict(X_test)
ols_rmse = rmse(y_test, ols_preds)
print(f"OLS RMSE: {ols_rmse:.4f}")


lambdas_ridge = np.logspace(-2, 2, 20)
ridge_rmses = []
for l in lambdas_ridge:
    r = RidgeRegression(lambda_=l)
    r.fit(X_train, y_train)
    p = r.predict(X_test)
    ridge_rmses.append(rmse(y_test, p))

best_lambda_ridge = lambdas_ridge[np.argmin(ridge_rmses)]
print(f"Best Ridge Lambda: {best_lambda_ridge:.4f}, RMSE: {min(ridge_rmses):.4f}")

plt.figure(figsize=(6, 4))
plt.plot(lambdas_ridge, ridge_rmses, marker='o')
plt.xscale('log')
plt.title("Ridge: Lambda vs RMSE")
plt.xlabel("Lambda")
plt.ylabel("RMSE")
plt.show()

best_ridge = RidgeRegression(lambda_=best_lambda_ridge)
best_ridge.fit(X_train, y_train)


lambdas_lasso = np.array([0.01, 0.1, 1.0, 10.0])
lrs_lasso = np.array([0.0001, 0.001, 0.005, 0.01])
lasso_results = []

for l in lambdas_lasso:
    for lr in lrs_lasso:
        la = LassoRegression(lambda_=l, iterations=2000, learning_rate=lr)
        la.fit(X_train, y_train)
        p = la.predict(X_test)
        rmse_l = rmse(y_test, p)
        lasso_results.append((l, lr, rmse_l))

lasso_results = np.array(lasso_results)
best_idx = np.argmin(lasso_results[:, 2])
best_lambda_lasso = lasso_results[best_idx, 0]
best_lr_lasso = lasso_results[best_idx, 1]
print(f"Best Lasso Lambda: {best_lambda_lasso}, LR: {best_lr_lasso}, RMSE: {lasso_results[best_idx, 2]:.4f}")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(lasso_results[:, 0], lasso_results[:, 1], lasso_results[:, 2], c=lasso_results[:, 2], cmap='viridis')
ax.set_xlabel('Lambda')
ax.set_ylabel('Learning Rate')
ax.set_zlabel('RMSE')
ax.set_title('Lasso: Lambda & LR vs RMSE')
plt.show()

best_lasso = LassoRegression(lambda_=best_lambda_lasso, iterations=5000, learning_rate=best_lr_lasso)
best_lasso.fit(X_train, y_train)


regressors = {
    "OLS": ols,
    "Ridge": best_ridge,
    "Lasso": best_lasso
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()

for i, (name, model) in enumerate(regressors.items()):
    preds = model.predict(X_test)
    current_rmse = rmse(y_test, preds)
    r2 = r2_score(y_test, preds)

    if r2 > best_reg_score:
        best_reg_score = r2
        final_best_regressor = model
        final_best_reg_name = name

    min_val = min(min(y_test), min(preds))
    max_val = max(max(y_test), max(preds))
    axes[i].scatter(y_test, preds, alpha=0.5)
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[i].set_title(f"{name}\nR2: {r2:.4f}, RMSE: {current_rmse:.2f}")
    axes[i].set_xlabel("Actual")
    axes[i].set_ylabel("Predicted")

plt.tight_layout()
plt.show()

print(f"BEST REGRESSOR: {final_best_reg_name}")

print(" CLASSIFICATION EVALUATION")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_all, y_exit)
classifiers = {
    "XGBoost": XGBoost(n_estimators=15, learning_rate=0.01, max_depth=5),
    "AdaBoost": AdaBoost(n_clf=15),
    "Gaussian NB": GaussianNB(),
}

best_clf_score = -1
final_best_classifier = None
final_best_clf_name = ""
n_classes = len(np.unique(y_test_c))

for name, model in classifiers.items():
    print(f"\n {name}")
    model.fit(X_train_c, y_train_c)
    preds = model.predict(X_test_c)

    acc = accuracy(y_test_c, preds)
    cm = confusion_matrix(y_test_c, preds, n_classes)
    precision = precision_per_class(cm)
    recall = recall_per_class(cm)
    f1 = f1_per_class(precision, recall)

    macro_f1 = macro_average(f1)
    kappa = kappa_score(y_test_c, preds, n_classes)
    min_class_acc = np.min(recall)

    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Kappa Score     : {kappa:.4f}")
    print(f"Class Accuracies: {np.round(recall, 2)}")

    if min_class_acc < 0.05:
        adjusted_score = macro_f1 * 0.5
    else:
        adjusted_score = macro_f1 + (0.5 * min_class_acc)

    if adjusted_score > best_clf_score:
        best_clf_score = adjusted_score
        final_best_classifier = model
        final_best_clf_name = name

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=exit_map.keys(), yticklabels=exit_map.keys())
    plt.title(f"Confusion Matrix ({name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

print(f"\nBEST CLASSIFIER: {final_best_clf_name}")

## Demo
def predict_new_startup(startup_dict):
    df_new = pd.DataFrame([startup_dict])
    df_proc, _ = preprocess_pipeline(df_new, is_training=False, stats=saved_stats)
    X_new = df_proc.values.astype(float)

    val = final_best_regressor.predict(X_new)[0]
    exit_idx = final_best_classifier.predict(X_new)[0]
    exit_label = inv_exit_map[int(exit_idx)]

    return val, exit_label

new_startup = {
    "Startup Name": "QuantumHealth",
    "Industry": "IoT",
    "Funding Rounds": 4,
    "Funding Amount (M USD)": 249,
    "Revenue (M USD)": 25,
    "Employees": 600,
    "Year Founded": 1997,
    "Region": "Europe",
    "Market Share (%)": 4
}

print(f"DEMO FOR: {new_startup['Startup Name']}")

p_val, p_exit = predict_new_startup(new_startup)

print(f"Predicted Valuation      : ${p_val:,.2f} Million")
print(f"Predicted Exit Status    : {p_exit}")
