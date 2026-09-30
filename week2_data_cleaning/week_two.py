"""
Week 2 - Data Collection, Cleaning and Preprocessing for Logistics Analysis
Dataset: Olist Brazilian E-Commerce Public Dataset (Kaggle)

Needs these files (in a folder named 'data' or next to this script):
    olist_orders_dataset.csv, olist_order_items_dataset.csv,
    olist_customers_dataset.csv, olist_sellers_dataset.csv,
    olist_geolocation_dataset.csv, olist_products_dataset.csv

Outputs (saved next to this script):
    week2_cleaning_log.txt        full printed log, easy to copy into the report
    missing_values.png            missing data before cleaning
    outliers_before_after.png     box plots before and after outlier capping
    skew_log_transform.png        effect of the log transform on freight
    before_after_summary.csv      descriptive statistics before vs after
Cleaned dataset is saved as olist_clean.csv in the data folder.

Run:  python week2_data_cleaning.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

HERE = Path(__file__).resolve().parent
LOG_LINES = []


def log(text=""):
    print(text)
    LOG_LINES.append(str(text))


def find_data_dir():
    for base in [HERE, HERE.parent, Path.cwd()]:
        for sub in ["data", "."]:
            d = base / sub
            if (d / "olist_orders_dataset.csv").exists():
                return d
    raise FileNotFoundError(
        "Could not find olist_orders_dataset.csv. Put the Olist CSV files "
        "in a folder named 'data' next to this script or in the project root.")


DATA = find_data_dir()
log(f"Using data folder: {DATA}")

# ===========================================================================
# 1. DATA COLLECTION: load the tables and describe them
# ===========================================================================
log("\n" + "=" * 70 + "\n1. DATA COLLECTION\n" + "=" * 70)

DATE_COLS = ["order_purchase_timestamp", "order_approved_at",
             "order_delivered_carrier_date", "order_delivered_customer_date",
             "order_estimated_delivery_date"]
orders = pd.read_csv(DATA / "olist_orders_dataset.csv", parse_dates=DATE_COLS)
items = pd.read_csv(DATA / "olist_order_items_dataset.csv")
customers = pd.read_csv(DATA / "olist_customers_dataset.csv")
sellers = pd.read_csv(DATA / "olist_sellers_dataset.csv")
products = pd.read_csv(DATA / "olist_products_dataset.csv")
geo_raw = pd.read_csv(DATA / "olist_geolocation_dataset.csv")

tables = {"orders": orders, "order_items": items, "customers": customers,
          "sellers": sellers, "products": products, "geolocation": geo_raw}
for name, t in tables.items():
    log(f"{name:12s} rows={len(t):>9,}  columns={t.shape[1]}")

log(f"\nOrder date range: {orders['order_purchase_timestamp'].min().date()} "
    f"to {orders['order_purchase_timestamp'].max().date()}")
log("\nOrder status counts:")
log(orders["order_status"].value_counts().to_string())

# Geolocation has many rows per zip prefix: average them to one point each
geo = (geo_raw.groupby("geolocation_zip_code_prefix")
       [["geolocation_lat", "geolocation_lng"]].mean())

# Order-level table: one row per order
items_p = items.merge(
    products[["product_id", "product_weight_g", "product_category_name"]],
    on="product_id", how="left")
item_agg = (items_p.groupby("order_id")
            .agg(freight=("freight_value", "sum"),
                 price=("price", "sum"),
                 n_items=("order_item_id", "count"),
                 weight_g=("product_weight_g", lambda s: s.sum(min_count=1)),
                 category=("product_category_name", "first"),
                 seller_id=("seller_id", "first"))
            .reset_index())

raw = (orders.merge(item_agg, on="order_id", how="inner")
       .merge(customers[["customer_id", "customer_state",
                         "customer_zip_code_prefix"]], on="customer_id")
       .merge(sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]],
              on="seller_id"))
raw = (raw.join(geo.add_prefix("cust_"), on="customer_zip_code_prefix")
          .join(geo.add_prefix("sell_"), on="seller_zip_code_prefix"))


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


raw["distance_km"] = haversine_km(
    raw["sell_geolocation_lat"], raw["sell_geolocation_lng"],
    raw["cust_geolocation_lat"], raw["cust_geolocation_lng"])
raw["lead_time_days"] = (raw["order_delivered_customer_date"]
                         - raw["order_purchase_timestamp"]).dt.days
raw["dispatch_delay_days"] = (raw["order_delivered_carrier_date"]
                              - raw["order_approved_at"]).dt.days

log(f"\nMerged order-level table: {raw.shape[0]:,} rows x {raw.shape[1]} columns")

# ===========================================================================
# 2. DATA QUALITY AUDIT
# ===========================================================================
log("\n" + "=" * 70 + "\n2. DATA QUALITY AUDIT (before cleaning)\n" + "=" * 70)

log("\n2.1 Duplicates")
log(f"Duplicate order_id rows: {raw['order_id'].duplicated().sum()}")

log("\n2.2 Missing values (columns with at least one missing value)")
miss = raw.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
miss_tbl = pd.DataFrame({"missing": miss,
                         "pct": (100 * miss / len(raw)).round(2)})
log(miss_tbl.to_string())

log("\n2.3 Logical consistency checks (delivered orders only)")
dlv = raw[(raw["order_status"] == "delivered")
          & raw["order_delivered_customer_date"].notna()]
checks = {
    "Dispatch delay < 0 (carrier pickup before approval)":
        (dlv["dispatch_delay_days"] < 0).sum(),
    "Lead time < 0 (delivered before purchase)":
        (dlv["lead_time_days"] < 0).sum(),
    "Carrier pickup after customer delivery":
        (dlv["order_delivered_carrier_date"]
         > dlv["order_delivered_customer_date"]).sum(),
    "Price <= 0": (dlv["price"] <= 0).sum(),
    "Lead time > 100 days": (dlv["lead_time_days"] > 100).sum(),
}
for k, v in checks.items():
    log(f"{k}: {v}")

NUM_COLS = ["lead_time_days", "freight", "price", "dispatch_delay_days",
            "distance_km", "weight_g"]


def iqr_report(df, cols):
    rows = []
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"column": c, "median": round(s.median(), 2),
                     "max": round(s.max(), 2), "lower_fence": round(lo, 2),
                     "upper_fence": round(hi, 2), "n_outliers": n_out,
                     "pct_outliers": round(100 * n_out / len(s), 2),
                     "skew": round(s.skew(), 2)})
    return pd.DataFrame(rows).set_index("column")


log("\n2.4 Outlier check with the IQR rule (delivered orders)")
log(iqr_report(dlv, NUM_COLS).to_string())

# ===========================================================================
# 3. CLEANING PIPELINE
# ===========================================================================
log("\n" + "=" * 70 + "\n3. CLEANING PIPELINE\n" + "=" * 70)
steps = [("Start: merged order-level table", len(raw))]

df = raw.drop_duplicates(subset="order_id").copy()
steps.append(("Remove duplicate orders", len(df)))

df = df[(df["order_status"] == "delivered")
        & df["order_delivered_customer_date"].notna()].copy()
steps.append(("Keep delivered orders with a delivery date", len(df)))

bad = ((df["dispatch_delay_days"] < 0) | (df["lead_time_days"] < 0)
       | (df["order_delivered_carrier_date"]
          > df["order_delivered_customer_date"]))
df = df[~bad].copy()
steps.append(("Remove logically impossible timestamps", len(df)))

log("\nRow counts after each step:")
prev = None
for name, n in steps:
    change = "" if prev is None else f"  ({n - prev:+,})"
    log(f"  {name:48s} {n:>8,}{change}")
    prev = n

# ---- Missing value handling -------------------------------------------------
log("\n3.1 Missing value imputation (flag columns record what was filled)")

# dispatch delay: overall median (skewed, so median not mean)
m = df["dispatch_delay_days"].isna()
df["dispatch_delay_imputed"] = m.astype(int)
df["dispatch_delay_days"] = df["dispatch_delay_days"].fillna(
    df["dispatch_delay_days"].median())
log(f"dispatch_delay_days: {m.sum()} filled with overall median")

# product weight: median of the same product category, then overall median
m = df["weight_g"].isna()
df["weight_imputed"] = m.astype(int)
cat_median = df.groupby("category")["weight_g"].transform("median")
df["weight_g"] = df["weight_g"].fillna(cat_median).fillna(
    df["weight_g"].median())
log(f"weight_g: {m.sum()} filled with category median (fallback: overall)")

# distance: median for the same seller-state / customer-state pair
m = df["distance_km"].isna()
df["distance_imputed"] = m.astype(int)
pair_median = df.groupby(["seller_state", "customer_state"])[
    "distance_km"].transform("median")
df["distance_km"] = df["distance_km"].fillna(pair_median).fillna(
    df["distance_km"].median())
log(f"distance_km: {m.sum()} filled with state-pair median (fallback: overall)")

df["category"] = df["category"].fillna("unknown")
log(f"Remaining missing in modeling columns: "
    f"{int(df[NUM_COLS].isna().sum().sum())}")

# ---- Outlier handling (winsorizing) -------------------------------------------
log("\n3.2 Outlier handling: cap values at the 1st and 99th percentile")
before = df[NUM_COLS].copy()
for c in NUM_COLS:
    lo, hi = df[c].quantile(0.01), df[c].quantile(0.99)
    n_cap = int(((df[c] < lo) | (df[c] > hi)).sum())
    df[c + "_raw"] = df[c]          # keep the original value for reference
    df[c] = df[c].clip(lo, hi)
    log(f"{c:22s} capped {n_cap:>5,} rows to [{lo:.2f}, {hi:.2f}]")

# ---- Feature engineering ------------------------------------------------------
log("\n3.3 Feature engineering")
df["is_late"] = (df["order_delivered_customer_date"]
                 > df["order_estimated_delivery_date"]).astype(int)
df["purchase_month"] = df["order_purchase_timestamp"].dt.month
df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
df["freight_to_price"] = df["freight"] / df["price"].replace(0, np.nan)
df["freight_to_price"] = df["freight_to_price"].fillna(
    df["freight_to_price"].median())
df["same_state"] = (df["seller_state"] == df["customer_state"]).astype(int)
log("Added: is_late, purchase_month, purchase_dayofweek, "
    "freight_to_price, same_state")

# ---- Transformation and normalization -----------------------------------------
log("\n3.4 Transformation and normalization")
skew_before = df["freight"].skew()
df["freight_log"] = np.log1p(df["freight"])
log(f"Skewness of freight: {skew_before:.2f} -> "
    f"{df['freight_log'].skew():.2f} after log1p")

SCALE_COLS = ["distance_km", "freight", "price", "weight_g",
              "dispatch_delay_days", "n_items"]
mm = MinMaxScaler().fit_transform(df[SCALE_COLS])
zs = StandardScaler().fit_transform(df[SCALE_COLS])
for i, c in enumerate(SCALE_COLS):
    df[c + "_minmax"] = mm[:, i]
    df[c + "_z"] = zs[:, i]
log("Min-max (0 to 1) and z-score versions created for: "
    + ", ".join(SCALE_COLS))
log("\nCheck of scaled columns (min, max, mean, std):")
chk = pd.DataFrame({
    "minmax_min": df[[c + "_minmax" for c in SCALE_COLS]].min().values,
    "minmax_max": df[[c + "_minmax" for c in SCALE_COLS]].max().values,
    "z_mean": df[[c + "_z" for c in SCALE_COLS]].mean().values.round(4),
    "z_std": df[[c + "_z" for c in SCALE_COLS]].std().values.round(4),
}, index=SCALE_COLS)
log(chk.round(3).to_string())

# ===========================================================================
# 4. BEFORE / AFTER COMPARISON AND OUTPUTS
# ===========================================================================
log("\n" + "=" * 70 + "\n4. BEFORE VS AFTER\n" + "=" * 70)
after = df[NUM_COLS]
summary = pd.concat(
    {"before": before.describe().T[["mean", "std", "min", "50%", "max"]],
     "after": after.describe().T[["mean", "std", "min", "50%", "max"]]},
    axis=1).round(2)
log(summary.to_string())
summary.to_csv(HERE / "before_after_summary.csv")

log(f"\nFinal clean dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
log(f"Rows retained from start: {100 * len(df) / len(raw):.1f}%")
log("\nKPIs on the cleaned data:")
log(f"  On-time delivery rate: {100 * (1 - df['is_late'].mean()):.2f}%")
log(f"  Average lead time:     {df['lead_time_days'].mean():.2f} days")
log(f"  Average freight:       {df['freight'].mean():.2f}")
log(f"  Average dispatch delay:{df['dispatch_delay_days'].mean():.2f} days")

# ---- Charts --------------------------------------------------------------------
# 1. Missing values
if len(miss_tbl):
    ax = miss_tbl["pct"].sort_values().plot(kind="barh", figsize=(8, 5),
                                            color="#2E5597")
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Missing Data Before Cleaning (merged order table)")
    plt.tight_layout()
    plt.savefig(HERE / "missing_values.png", dpi=150)
    plt.close()

# 2. Outliers before and after
show = ["freight", "lead_time_days", "dispatch_delay_days"]
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for j, c in enumerate(show):
    axes[0, j].boxplot(before[c].dropna(), vert=True)
    axes[0, j].set_title(f"{c} (before)")
    axes[1, j].boxplot(after[c].dropna(), vert=True)
    axes[1, j].set_title(f"{c} (after capping)")
plt.tight_layout()
plt.savefig(HERE / "outliers_before_after.png", dpi=150)
plt.close()

# 3. Skew and log transform
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(df["freight"], bins=50, color="#2E5597")
axes[0].set_title("Freight (capped)")
axes[1].hist(df["freight_log"], bins=50, color="#2E5597")
axes[1].set_title("Freight after log1p")
plt.tight_layout()
plt.savefig(HERE / "skew_log_transform.png", dpi=150)
plt.close()

# ---- Save ----------------------------------------------------------------------
df.to_csv(DATA / "olist_clean.csv", index=False)
log(f"\nSaved cleaned data to {DATA / 'olist_clean.csv'}")
log("Saved charts and before_after_summary.csv next to this script.")

(HERE / "week2_cleaning_log.txt").write_text("\n".join(LOG_LINES),
                                              encoding="utf-8")
print("\nDone. Full log saved as week2_cleaning_log.txt")