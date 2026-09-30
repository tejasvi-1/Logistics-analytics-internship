"""
Week 1 - Logistics Data Analyst Internship
Code illustrations for: Reducing Delivery Delays and Shipping Costs
in an E-Commerce Supply Chain Using Python

Dataset: Olist Brazilian E-Commerce Public Dataset (Kaggle)
Place these CSV files in the same folder as this script:
    olist_orders_dataset.csv
    olist_order_items_dataset.csv
    olist_customers_dataset.csv
    olist_sellers_dataset.csv
    olist_geolocation_dataset.csv

Install: pip install pandas numpy seaborn matplotlib scikit-learn
NOTE: written from the documented Olist schema; not yet run on the real files.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


import os

DATA_DIR = "data" if os.path.exists("data") else "."

# ---------------------------------------------------------------------------
# 6.1 Loading and merging data
# ---------------------------------------------------------------------------
orders = pd.read_csv(
    os.path.join(DATA_DIR, "olist_orders_dataset.csv"),
    parse_dates=["order_purchase_timestamp",
                 "order_approved_at",
                 "order_delivered_carrier_date",
                 "order_delivered_customer_date",
                 "order_estimated_delivery_date"])
items = pd.read_csv(os.path.join(DATA_DIR, "olist_order_items_dataset.csv"))
customers = pd.read_csv(os.path.join(DATA_DIR, "olist_customers_dataset.csv"))
sellers = pd.read_csv(os.path.join(DATA_DIR, "olist_sellers_dataset.csv"))

# Aggregate items to one row per order to avoid duplicated rows
item_agg = (items.groupby("order_id")
                 .agg(freight=("freight_value", "sum"),
                      price=("price", "sum"),
                      n_items=("order_item_id", "count"),
                      seller_id=("seller_id", "first"))
                 .reset_index())

df = (orders.merge(item_agg, on="order_id", how="inner")
            .merge(customers[["customer_id", "customer_state",
                              "customer_zip_code_prefix"]],
                   on="customer_id")
            .merge(sellers[["seller_id", "seller_state",
                            "seller_zip_code_prefix"]],
                   on="seller_id"))
print(df.shape)


# ---------------------------------------------------------------------------
# 6.2 Calculating the KPIs
# ---------------------------------------------------------------------------
delivered = df[df["order_status"] == "delivered"].dropna(
    subset=["order_delivered_customer_date"]).copy()

delivered["lead_time_days"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_purchase_timestamp"]).dt.days

delivered["dispatch_delay_days"] = (
    delivered["order_delivered_carrier_date"]
    - delivered["order_approved_at"]).dt.days

delivered["is_late"] = (
    delivered["order_delivered_customer_date"]
    > delivered["order_estimated_delivery_date"]).astype(int)

kpis = {
    "On-Time Delivery Rate (%)": 100 * (1 - delivered["is_late"].mean()),
    "Avg Lead Time (days)":      delivered["lead_time_days"].mean(),
    "Cost per Shipment":         delivered["freight"].mean(),
    "Avg Dispatch Delay (days)": delivered["dispatch_delay_days"].mean(),
}
for name, value in kpis.items():
    print(f"{name}: {value:.2f}")


# ---------------------------------------------------------------------------
# 6.3 Exploratory analysis
# ---------------------------------------------------------------------------
print(delivered[["lead_time_days", "freight", "dispatch_delay_days"]].describe())

corr = delivered[["lead_time_days", "freight", "price",
                  "dispatch_delay_days", "n_items"]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation of Key Logistics Variables")
plt.savefig("correlation_matrix.png", bbox_inches="tight")
plt.close()

# Late-delivery rate by customer state
late_by_state = (delivered.groupby("customer_state")["is_late"]
                          .mean().sort_values(ascending=False))
plt.figure(figsize=(10, 4))
late_by_state.plot(kind="bar")
plt.ylabel("Share of late orders")
plt.title("Late-Delivery Rate by Customer State")
plt.tight_layout()
plt.savefig("late_orders_by_state.png", bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Feature preparation for modeling (added so section 6.4 runs end to end)
# distance_km: haversine distance between seller and customer zip prefixes
# purchase_month: month of the purchase timestamp
# ---------------------------------------------------------------------------
geo = (pd.read_csv(os.path.join(DATA_DIR, "olist_geolocation_dataset.csv"))
         .groupby("geolocation_zip_code_prefix")[["geolocation_lat",
                                                  "geolocation_lng"]]
         .mean())


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


delivered = (delivered
             .join(geo.add_prefix("cust_"), on="customer_zip_code_prefix")
             .join(geo.add_prefix("sell_"), on="seller_zip_code_prefix"))
delivered["distance_km"] = haversine_km(
    delivered["sell_geolocation_lat"], delivered["sell_geolocation_lng"],
    delivered["cust_geolocation_lat"], delivered["cust_geolocation_lng"])
delivered["purchase_month"] = delivered["order_purchase_timestamp"].dt.month

# Drop rows with missing values in modeling columns (proper handling: Week 2)
model_cols = ["distance_km", "freight", "price", "n_items",
              "dispatch_delay_days", "purchase_month", "lead_time_days"]
model_df = delivered.dropna(subset=model_cols)


# ---------------------------------------------------------------------------
# 6.4 Predictive modeling (delivery time)
# ---------------------------------------------------------------------------
features = ["distance_km", "freight", "price", "n_items",
            "dispatch_delay_days", "purchase_month"]
X = model_df[features]
y = model_df["lead_time_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(
              n_estimators=200, random_state=42, n_jobs=-1)}

for name, model in models.items():
    cv_mae = -cross_val_score(model, X_train, y_train, cv=5,
                              scoring="neg_mean_absolute_error").mean()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    print(f"{name}: CV MAE={cv_mae:.2f}, "
          f"Test MAE={mean_absolute_error(y_test, pred):.2f}, RMSE={rmse:.2f}")


# ---------------------------------------------------------------------------
# 6.5 Seller segmentation (clustering)
# ---------------------------------------------------------------------------
seller_stats = (delivered.groupby("seller_id")
                .agg(avg_dispatch=("dispatch_delay_days", "mean"),
                     late_rate=("is_late", "mean"),
                     avg_freight=("freight", "mean"),
                     orders=("order_id", "count"))
                .query("orders >= 20")
                .dropna())

scaled = StandardScaler().fit_transform(
    seller_stats[["avg_dispatch", "late_rate", "avg_freight"]])
seller_stats["cluster"] = KMeans(n_clusters=3, random_state=42,
                                 n_init=10).fit_predict(scaled)
print(seller_stats.groupby("cluster").mean())


# ---------------------------------------------------------------------------
# 6.6 Optimization (pseudocode - to be implemented in Week 4)
# ---------------------------------------------------------------------------
"""
# Minimize total shipping cost, subject to service and capacity limits
DEFINE decision variable x[shipment, carrier] in {0, 1}

MINIMIZE  sum( cost[shipment, carrier] * x[shipment, carrier] )

SUBJECT TO
    for each shipment:  sum over carriers of x[shipment, carrier] = 1
    for each carrier:   sum of weight[shipment] * x[shipment, carrier]
                        <= capacity[carrier]
    for each shipment:  predicted_days[shipment, carrier] * x[shipment, carrier]
                        <= promised_days[shipment]

SOLVE with a mixed-integer solver (OR-Tools or PuLP)
REPORT total cost, on-time rate under new assignment vs current assignment
"""