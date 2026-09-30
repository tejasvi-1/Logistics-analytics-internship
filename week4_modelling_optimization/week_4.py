"""
Week 4 - Predictive Modeling and Optimization in Logistics Systems
Dataset: cleaned Olist order-level data from the Week 2 script (olist_clean.csv).

Problem: predict delivery lead time (days from purchase to delivery).
  Model A (checkout-time): uses only information known when the order is placed.
  Model B (post-dispatch): Model A + the seller's dispatch delay.

Optimization analyses built on the models:
  1. Delivery-promise calibration (how long should the promise be?)
  2. Seller dispatch SLA what-if (model B)
  3. Regional fulfilment placement what-if (model A + freight model)

Run (after Week 2):  python week4_modeling_optimization.py
This script can take several minutes (random forest + hyperparameter search).

Outputs (saved next to this script):
    week4_modeling_log.txt, model_results.csv
    fig1_model_comparison.png, fig2_actual_vs_predicted.png,
    fig3_feature_importance.png, fig4_error_by_segment.png,
    fig5_promise_optimization.png, fig6_what_if_scenarios.png
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              HistGradientBoostingRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     cross_val_predict, RandomizedSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

HERE = Path(__file__).resolve().parent
LOG_LINES = []
RS = 42
sns.set_theme(style="whitegrid")
BLUE, ORANGE, RED, GREEN = "#2E5597", "#E07B39", "#C0392B", "#2E8B57"
T0 = time.time()


def log(text=""):
    print(text, flush=True)
    LOG_LINES.append(str(text))


def savefig(name):
    plt.tight_layout()
    plt.savefig(HERE / name, dpi=150)
    plt.close()
    log(f"[saved {name}]")


def find_clean():
    for base in [HERE, HERE.parent, Path.cwd()]:
        for sub in ["data", "."]:
            f = base / sub / "olist_clean.csv"
            if f.exists():
                return f
    raise FileNotFoundError(
        "olist_clean.csv not found. Run the Week 2 script first.")


# ===========================================================================
# 1. PROBLEM DEFINITION AND DATA
# ===========================================================================
path = find_clean()
df = pd.read_csv(path, parse_dates=["order_purchase_timestamp",
                                    "order_estimated_delivery_date"])
df["promised_days"] = (df["order_estimated_delivery_date"]
                       - df["order_purchase_timestamp"]).dt.days

log(f"{'=' * 70}\n1. PROBLEM DEFINITION AND DATA\n{'=' * 70}")
log(f"Loaded {path}")
log(f"Orders: {len(df):,}")
log("Target: lead_time_days (days from purchase to delivery)")
log(f"Target mean {df['lead_time_days'].mean():.2f}, median "
    f"{df['lead_time_days'].median():.1f}, std {df['lead_time_days'].std():.2f}")

NUM_A = ["distance_km", "weight_g", "price", "n_items", "same_state",
         "purchase_month", "purchase_dayofweek"]
NUM_B = NUM_A + ["dispatch_delay_days"]
CAT = ["customer_state", "seller_state"]
log("\nModel A features (known at checkout): "
    + ", ".join(NUM_A + CAT))
log("Model B features (A + known after seller ships): dispatch_delay_days")
log("Freight is left out of the lead-time models because it is an outcome of "
    "the same shipping decision.")

y = df["lead_time_days"]
X_all = df[NUM_B + CAT]
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=RS)
XA_train, XA_test = X_train[NUM_A + CAT], X_test[NUM_A + CAT]
XB_train, XB_test = X_train[NUM_B + CAT], X_test[NUM_B + CAT]
log(f"\nTrain rows: {len(X_train):,}   Test rows: {len(X_test):,} (80/20 split)")


def make_pipe(model, kind, num_cols):
    if kind == "linear":
        pre = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    else:
        pre = ColumnTransformer([
            ("num", "passthrough", num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), CAT)])
    return Pipeline([("pre", pre), ("model", model)])


def metrics(y_true, y_pred):
    err = y_pred - y_true
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "R2": r2_score(y_true, y_pred),
            "within_3_days_pct": 100 * float((np.abs(err) <= 3).mean()),
            "bias": float(err.mean())}


# Reference: the promised delivery estimate currently shown to customers
promised_test = df.loc[X_test.index, "promised_days"]
cur = metrics(y_test, promised_test)
cur_on_time = 100 * float((y_test <= promised_test).mean())
log("\nCurrent delivery estimate (reference, test set):")
log(f"  MAE {cur['MAE']:.2f} days, bias {cur['bias']:+.2f} days "
    f"(positive = promise longer than actual), on-time {cur_on_time:.2f}%, "
    f"average promise {promised_test.mean():.2f} days")

# ===========================================================================
# 2. MODEL COMPARISON (Model A)
# ===========================================================================
log(f"\n{'=' * 70}\n2. MODEL COMPARISON (Model A, checkout-time features)\n{'=' * 70}")
candidates = {
    "Mean baseline": (DummyRegressor(), "tree"),
    "Linear Regression": (LinearRegression(), "linear"),
    "Ridge Regression": (Ridge(alpha=10), "linear"),
    "Decision Tree": (DecisionTreeRegressor(max_depth=8, min_samples_leaf=50,
                                            random_state=RS), "tree"),
    "Random Forest": (RandomForestRegressor(n_estimators=100,
                                            min_samples_leaf=5, n_jobs=-1,
                                            random_state=RS), "tree"),
    "Gradient Boosting (default)": (HistGradientBoostingRegressor(
        random_state=RS), "tree"),
}
rows = []
for name, (model, kind) in candidates.items():
    log(f"Training {name} ... ({(time.time() - T0) / 60:.1f} min elapsed)")
    pipe = make_pipe(model, kind, NUM_A)
    cv_mae = -cross_val_score(pipe, XA_train, y_train, cv=5,
                              scoring="neg_mean_absolute_error").mean()
    pipe.fit(XA_train, y_train)
    m = metrics(y_test, pipe.predict(XA_test))
    rows.append({"model": name, "CV_MAE": cv_mae, **m})

# ===========================================================================
# 3. HYPERPARAMETER TUNING
# ===========================================================================
log(f"\n{'=' * 70}\n3. HYPERPARAMETER TUNING (Gradient Boosting)\n{'=' * 70}")
param_dist = {
    "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "model__max_iter": [200, 400, 600],
    "model__max_depth": [None, 4, 6, 8],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__min_samples_leaf": [20, 50, 100],
    "model__l2_regularization": [0.0, 1.0, 10.0],
}
search = RandomizedSearchCV(
    make_pipe(HistGradientBoostingRegressor(random_state=RS), "tree", NUM_A),
    param_distributions=param_dist, n_iter=12, cv=3,
    scoring="neg_mean_absolute_error", random_state=RS, n_jobs=1)
log(f"Running random search: 12 parameter sets x 3-fold CV "
    f"({(time.time() - T0) / 60:.1f} min elapsed)")
search.fit(XA_train, y_train)
best_pipe = search.best_estimator_
best_params = {k.replace("model__", ""): v
               for k, v in search.best_params_.items()}
log(f"Best parameters: {best_params}")
log(f"Best CV MAE: {-search.best_score_:.3f}")

pred_test = best_pipe.predict(XA_test)
m_tuned = metrics(y_test, pred_test)
train_mae = mean_absolute_error(y_train, best_pipe.predict(XA_train))
rows.append({"model": "Gradient Boosting (tuned)",
             "CV_MAE": -search.best_score_, **m_tuned})
log(f"Tuned model: train MAE {train_mae:.3f} vs test MAE {m_tuned['MAE']:.3f}")

results = pd.DataFrame(rows).set_index("model")
log("\nResults on the test set (Model A):")
log(results.round(3).to_string())
results.to_csv(HERE / "model_results.csv")
best_name = results.drop("Mean baseline")["MAE"].idxmin()
log(f"\nBest model by test MAE: {best_name}")
log(f"Improvement over mean baseline: "
    f"{100 * (1 - results.loc['Gradient Boosting (tuned)', 'MAE'] / results.loc['Mean baseline', 'MAE']):.1f}% lower MAE")
log(f"Tuned model MAE {m_tuned['MAE']:.2f} vs current delivery estimate MAE "
    f"{cur['MAE']:.2f}")

# ===========================================================================
# 4. MODEL B (adds dispatch delay) AND TIME-BASED CHECK
# ===========================================================================
log(f"\n{'=' * 70}\n4. MODEL B AND ROBUSTNESS CHECK\n{'=' * 70}")
model_B = make_pipe(HistGradientBoostingRegressor(random_state=RS, **best_params),
                    "tree", NUM_B)
model_B.fit(XB_train, y_train)
m_B = metrics(y_test, model_B.predict(XB_test))
log("Model B (post-dispatch) test metrics:")
log(pd.Series(m_B).round(3).to_string())
log(f"Adding dispatch delay lowers MAE from {m_tuned['MAE']:.3f} to "
    f"{m_B['MAE']:.3f}")

order = df["order_purchase_timestamp"].sort_values().index
cut = int(0.8 * len(order))
tr_idx, te_idx = order[:cut], order[cut:]
pipe_time = clone(best_pipe).fit(df.loc[tr_idx, NUM_A + CAT], y.loc[tr_idx])
m_time = metrics(y.loc[te_idx], pipe_time.predict(df.loc[te_idx, NUM_A + CAT]))
base_time = float(np.abs(y.loc[te_idx] - y.loc[tr_idx].mean()).mean())
log(f"\nTime-based split (train on earliest 80% of orders, test on latest 20%):")
log(f"  tuned model MAE {m_time['MAE']:.3f}, RMSE {m_time['RMSE']:.3f}, "
    f"R2 {m_time['R2']:.3f}; mean-baseline MAE {base_time:.3f}")

# ===========================================================================
# 5. ERROR ANALYSIS
# ===========================================================================
log(f"\n{'=' * 70}\n5. ERROR ANALYSIS (Model A, test set)\n{'=' * 70}")
res = pd.DataFrame({"actual": y_test.values, "pred": pred_test},
                   index=y_test.index)
res["err"] = res["pred"] - res["actual"]
res["abs_err"] = res["err"].abs()
res["distance_km"] = df.loc[res.index, "distance_km"]
res["customer_state"] = df.loc[res.index, "customer_state"]
long_ = res["actual"] > 20
log(f"Orders with actual lead time above 20 days: {long_.sum():,} "
    f"({100 * long_.mean():.1f}%)")
log(f"  MAE on those orders: {res.loc[long_, 'abs_err'].mean():.2f}, "
    f"mean error {res.loc[long_, 'err'].mean():+.2f} days (negative = underpredicted)")
log(f"MAE on the remaining orders: {res.loc[~long_, 'abs_err'].mean():.2f}, "
    f"mean error {res.loc[~long_, 'err'].mean():+.2f}")
bands = pd.cut(res["distance_km"], [0, 200, 500, 1000, 1500, 3000],
               labels=["0-200", "200-500", "500-1000", "1000-1500", "1500+"],
               include_lowest=True)
mae_dist = res.groupby(bands, observed=True)["abs_err"].mean()
log("\nMAE by distance band:")
log(mae_dist.round(2).to_string())
st = res.groupby("customer_state")["abs_err"].agg(["count", "mean"])
st = st[st["count"] >= 300].sort_values("count", ascending=False).head(12)
log("\nMAE for the 12 largest states:")
log(st.rename(columns={"count": "orders", "mean": "MAE"}).round(2).to_string())

# Permutation importance
log("\nPermutation importance (increase in MAE when a feature is shuffled):")
n_imp = min(20000, len(XA_test))
idx_imp = XA_test.sample(n_imp, random_state=RS).index
imp_A = permutation_importance(best_pipe, XA_test.loc[idx_imp],
                               y_test.loc[idx_imp], n_repeats=5,
                               scoring="neg_mean_absolute_error",
                               random_state=RS, n_jobs=1)
imp_B = permutation_importance(model_B, XB_test.loc[idx_imp],
                               y_test.loc[idx_imp], n_repeats=5,
                               scoring="neg_mean_absolute_error",
                               random_state=RS, n_jobs=1)
imp_A = pd.Series(imp_A.importances_mean, index=NUM_A + CAT).sort_values(
    ascending=False)
imp_B = pd.Series(imp_B.importances_mean, index=NUM_B + CAT).sort_values(
    ascending=False)
log("Model A:")
log(imp_A.round(3).to_string())
log("Model B:")
log(imp_B.round(3).to_string())

# ===========================================================================
# 6. OPTIMIZATION 1: DELIVERY-PROMISE CALIBRATION
# ===========================================================================
log(f"\n{'=' * 70}\n6. OPTIMIZATION 1: DELIVERY PROMISE CALIBRATION\n{'=' * 70}")
log("Idea: promise = predicted lead time + a safety buffer. The buffer is "
    "chosen on out-of-fold training predictions, then checked on the test set.")
oof = cross_val_predict(best_pipe, XA_train, y_train, cv=3)
promised_train = df.loc[X_train.index, "promised_days"]
cur_on_time_train = float((y_train <= promised_train).mean())
buffers = np.arange(-2, 15.01, 0.5)


def on_time_avg(actual, pred, b):
    promise = np.ceil(pred + b)
    return float((actual <= promise).mean()), float(promise.mean())


def pick_buffer(target):
    for b in buffers:
        if on_time_avg(y_train.values, oof, b)[0] >= target:
            return float(b)
    return float(buffers[-1])


b_match = pick_buffer(cur_on_time_train)
b_95 = pick_buffer(0.95)
ot_m, avg_m = on_time_avg(y_test.values, pred_test, b_match)
ot_95, avg_95 = on_time_avg(y_test.values, pred_test, b_95)
log(f"Current promise: on-time {cur_on_time:.2f}%, "
    f"average promised {promised_test.mean():.2f} days")
log(f"Model promise matching current on-time rate (buffer {b_match:+.1f} days): "
    f"on-time {100 * ot_m:.2f}%, average promised {avg_m:.2f} days "
    f"({promised_test.mean() - avg_m:+.2f} days shorter)")
log(f"Model promise for a 95% on-time target (buffer {b_95:+.1f} days): "
    f"on-time {100 * ot_95:.2f}%, average promised {avg_95:.2f} days "
    f"({promised_test.mean() - avg_95:+.2f} days shorter than current)")

curve = pd.DataFrame(
    [(b, *on_time_avg(y_test.values, pred_test, b)) for b in buffers],
    columns=["buffer", "on_time", "avg_promise"])
log("\nBuffer curve (test set), selected rows:")
log(curve[curve["buffer"].isin([0, 2, 4, 6, 8, 10])].assign(
    on_time=lambda d: (100 * d["on_time"]).round(2)).round(2).to_string(
        index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(curve["buffer"], 100 * curve["on_time"], color=BLUE, lw=2,
             label="On-time rate (model promise)")
axes[0].axhline(cur_on_time, color=RED, ls="--",
                label=f"Current on-time rate ({cur_on_time:.1f}%)")
axes[0].axvline(b_match, color=GREEN, ls=":", label=f"Buffer {b_match:+.1f} d")
axes[0].set_xlabel("Safety buffer added to predicted lead time (days)")
axes[0].set_ylabel("On-time rate (%)", color=BLUE)
axes[0].set_title("On-Time Rate vs Safety Buffer")
ax2 = axes[0].twinx()
ax2.plot(curve["buffer"], curve["avg_promise"], color=ORANGE, lw=2,
         label="Average promised days")
ax2.set_ylabel("Average promised days", color=ORANGE)
ax2.grid(False)
axes[0].legend(loc="lower right")
labels = ["Current\npromise", "Model promise\n(same on-time)",
          "Model promise\n(95% on-time)"]
vals = [promised_test.mean(), avg_m, avg_95]
ots = [cur_on_time, 100 * ot_m, 100 * ot_95]
bars = axes[1].bar(labels, vals, color=[RED, BLUE, GREEN])
for bar, o in zip(bars, ots):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"on-time {o:.1f}%", ha="center", fontsize=10)
axes[1].set_ylabel("Average promised delivery time (days)")
axes[1].set_title("Average Promise Length")
savefig("fig5_promise_optimization.png")

# ===========================================================================
# 7. OPTIMIZATION 2 AND 3: WHAT-IF SCENARIOS
# ===========================================================================
log(f"\n{'=' * 70}\n7. WHAT-IF SCENARIOS (model-based estimates, not guarantees)\n{'=' * 70}")

# --- 7.1 Seller dispatch SLA ---
log("\n7.1 Seller dispatch SLA (Model B)")
XB_full = df[NUM_B + CAT]
pred_b0 = model_B.predict(XB_full)
sla_rows = []
for t in [1, 2, 3]:
    mask = df["dispatch_delay_days"] > t
    X2 = XB_full.copy()
    X2.loc[mask, "dispatch_delay_days"] = t
    saved = pred_b0 - model_B.predict(X2)
    sla_rows.append({
        "max_dispatch_days": t,
        "orders_affected": int(mask.sum()),
        "share_of_orders_pct": 100 * mask.mean(),
        "avg_days_saved_per_affected": saved[mask.values].mean(),
        "network_avg_lead_reduction": saved.sum() / len(df)})
sla = pd.DataFrame(sla_rows).set_index("max_dispatch_days")
log(sla.round(3).to_string())

# --- 7.2 Regional fulfilment placement ---
log("\n7.2 Regional fulfilment placement (serve cross-state orders from "
    "same-state stock)")
frt_model = make_pipe(HistGradientBoostingRegressor(
    random_state=RS, max_iter=300), "tree", NUM_A)
frt_model.fit(XA_train, df.loc[X_train.index, "freight"])
frt_pred_test = frt_model.predict(XA_test)
log(f"Freight model on test set: R2 {r2_score(df.loc[X_test.index, 'freight'], frt_pred_test):.3f}, "
    f"MAE {mean_absolute_error(df.loc[X_test.index, 'freight'], frt_pred_test):.2f}")

cross = df[df["same_state"] == 0]
Xc = cross[NUM_A + CAT].copy()
ss = df[df["same_state"] == 1].groupby("customer_state")["distance_km"].agg(
    ["median", "count"])
glob_ss = df.loc[df["same_state"] == 1, "distance_km"].median()
new_dist = cross["customer_state"].map(
    ss["median"].where(ss["count"] >= 30)).fillna(glob_ss)
Xc2 = Xc.copy()
Xc2["seller_state"] = cross["customer_state"].values
Xc2["same_state"] = 1
Xc2["distance_km"] = new_dist.values
days_saved = best_pipe.predict(Xc) - best_pipe.predict(Xc2)
frt_saved = frt_model.predict(Xc) - frt_model.predict(Xc2)
sc = pd.DataFrame({"state": cross["customer_state"].values,
                   "days_saved": days_saved, "freight_saved": frt_saved})
by_state = sc.groupby("state").agg(
    cross_state_orders=("days_saved", "count"),
    total_days_saved=("days_saved", "sum"),
    avg_days_saved=("days_saved", "mean"),
    total_freight_saved=("freight_saved", "sum"),
    avg_freight_saved=("freight_saved", "mean")).sort_values(
        "total_days_saved", ascending=False)
by_state["cum_share_pct"] = 100 * by_state["total_days_saved"].cumsum() / \
    by_state["total_days_saved"].sum()
log(f"Cross-state orders: {len(cross):,} ({100 * len(cross) / len(df):.1f}%)")
log(f"If ALL were served from same-state stock: average predicted saving "
    f"{days_saved.mean():.2f} days and {frt_saved.mean():.2f} freight per order")
log("\nTop 10 states to place regional stock (ranked by total days saved):")
log(by_state.head(10).round(2).to_string())
for k in [3, 5, 10]:
    top = by_state.head(k)
    log(f"Top {k} states: {100 * top['total_days_saved'].sum() / by_state['total_days_saved'].sum():.1f}% "
        f"of total predicted days saved, {int(top['cross_state_orders'].sum()):,} orders, "
        f"network-wide average lead time reduction "
        f"{top['total_days_saved'].sum() / len(df):.2f} days")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
top10 = by_state.head(10)
axes[0].bar(top10.index, top10["total_days_saved"], color=BLUE)
axes[0].set_ylabel("Total predicted days saved")
axes[0].set_xlabel("Customer state")
axes[0].set_title("Regional Stock Placement: Benefit by State")
a2 = axes[0].twinx()
a2.plot(top10.index, top10["cum_share_pct"], color=ORANGE, marker="o")
a2.set_ylabel("Cumulative share of total benefit (%)", color=ORANGE)
a2.set_ylim(0, 105)
a2.grid(False)
axes[1].bar([f"<= {t} days" for t in sla.index],
            sla["avg_days_saved_per_affected"], color=[GREEN, BLUE, ORANGE])
for i, (t, r) in enumerate(sla.iterrows()):
    axes[1].text(i, r["avg_days_saved_per_affected"] + 0.03,
                 f"{int(r['orders_affected']):,} orders", ha="center")
axes[1].set_ylabel("Avg predicted days saved per affected order")
axes[1].set_xlabel("Maximum allowed seller dispatch time")
axes[1].set_title("Seller Dispatch SLA: Predicted Saving")
savefig("fig6_what_if_scenarios.png")

# ===========================================================================
# 8. REMAINING CHARTS
# ===========================================================================
# Fig 1: model comparison
order_m = results.sort_values("MAE")
x = np.arange(len(order_m))
plt.figure(figsize=(11, 5.5))
plt.bar(x - 0.2, order_m["MAE"], 0.4, label="MAE", color=BLUE)
plt.bar(x + 0.2, order_m["RMSE"], 0.4, label="RMSE", color=ORANGE)
plt.axhline(cur["MAE"], color=RED, ls="--",
            label=f"Current delivery estimate MAE ({cur['MAE']:.2f})")
plt.xticks(x, order_m.index, rotation=25, ha="right")
plt.ylabel("Error (days)")
plt.title("Model Comparison on the Test Set (lower is better)")
plt.legend()
savefig("fig1_model_comparison.png")

# Fig 2: actual vs predicted and residuals
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
hb = axes[0].hexbin(res["actual"], res["pred"], gridsize=45, bins="log",
                    cmap="Blues", mincnt=1)
lim = [0, max(res["actual"].max(), res["pred"].max())]
axes[0].plot(lim, lim, color=RED, lw=2, label="perfect prediction")
axes[0].set_xlabel("Actual lead time (days)")
axes[0].set_ylabel("Predicted lead time (days)")
axes[0].set_title("Actual vs Predicted (test set)")
axes[0].legend()
fig.colorbar(hb, ax=axes[0], label="orders (log)")
axes[1].hist(res["err"], bins=60, color=BLUE, edgecolor="white")
axes[1].axvline(0, color="black")
axes[1].axvline(res["err"].mean(), color=RED, ls="--",
                label=f"mean error {res['err'].mean():+.2f}")
axes[1].set_xlabel("Prediction error (predicted - actual, days)")
axes[1].set_ylabel("Orders")
axes[1].set_title("Residual Distribution")
axes[1].legend()
savefig("fig2_actual_vs_predicted.png")

# Fig 3: permutation importance
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
imp_A.sort_values().plot(kind="barh", ax=axes[0], color=BLUE)
axes[0].set_title("Model A (checkout-time)")
axes[0].set_xlabel("Increase in MAE when shuffled (days)")
imp_B.sort_values().plot(kind="barh", ax=axes[1], color=ORANGE)
axes[1].set_title("Model B (adds dispatch delay)")
axes[1].set_xlabel("Increase in MAE when shuffled (days)")
savefig("fig3_feature_importance.png")

# Fig 4: error by segment
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(mae_dist.index.astype(str), mae_dist.values, color=BLUE)
axes[0].set_xlabel("Distance band (km)")
axes[0].set_ylabel("Mean absolute error (days)")
axes[0].set_title("Prediction Error by Distance")
axes[1].bar(st.index, st["mean"], color=ORANGE)
axes[1].set_xlabel("Customer state (12 largest)")
axes[1].set_ylabel("Mean absolute error (days)")
axes[1].set_title("Prediction Error by State")
savefig("fig4_error_by_segment.png")

# ===========================================================================
# 9. SUMMARY
# ===========================================================================
log(f"\n{'=' * 70}\n9. SUMMARY\n{'=' * 70}")
log(f"Total run time: {(time.time() - T0) / 60:.1f} minutes")
(HERE / "week4_modeling_log.txt").write_text("\n".join(LOG_LINES),
                                              encoding="utf-8")
print("\nDone. Full log saved as week4_modeling_log.txt")