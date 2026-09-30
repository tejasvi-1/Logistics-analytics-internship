"""
Week 3 - Advanced Data Analysis and Visualization in Logistics
Dataset: cleaned Olist order-level data produced by the Week 2 script
         (olist_clean.csv, saved in the 'data' folder).

Run (after Week 2):  python week3_eda_visualization.py

Outputs (saved next to this script):
    week3_eda_log.txt               all printed statistics, easy to copy
    fig1_distributions.png          distributions of key variables
    fig2_correlation_heatmap.png    correlation matrix
    fig3_monthly_trends.png         orders, on-time rate and lead time by month
    fig4_state_performance.png      late rate and lead time by customer state
    fig5_distance_relationships.png distance vs lead time, freight and lateness
    fig6_stage_breakdown.png        where time is spent, and dispatch vs lateness
    fig7_cost_drivers.png           freight by weight, distance and same-state
    fig8_category_freight.png       most expensive product categories to ship
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
LOG_LINES = []
sns.set_theme(style="whitegrid")
BLUE, ORANGE, RED = "#2E5597", "#E07B39", "#C0392B"


def log(text=""):
    print(text)
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
        "olist_clean.csv not found. Run the Week 2 script first; it saves "
        "the file in the 'data' folder.")


# ===========================================================================
# 1. LOAD AND DESCRIBE THE DATA
# ===========================================================================
path = find_clean()
df = pd.read_csv(path, parse_dates=[
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date"])
log(f"Loaded {path}")
# minimum group sizes (scaled down automatically for small datasets)
MIN_GROUP = min(200, max(5, len(df) // 200))
MIN_CAT = min(500, max(5, len(df) // 100))
log(f"\n{'=' * 70}\n1. DATASET STRUCTURE\n{'=' * 70}")
log(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
log(f"Period: {df['order_purchase_timestamp'].min().date()} to "
    f"{df['order_purchase_timestamp'].max().date()}")
log(f"Customer states: {df['customer_state'].nunique()}   "
    f"Sellers: {df['seller_id'].nunique():,}   "
    f"Categories: {df['category'].nunique()}")

KEY = ["lead_time_days", "dispatch_delay_days", "freight", "price",
       "distance_km", "weight_g", "n_items", "freight_to_price",
       "is_late", "same_state"]
log("\nKey variables:")
log(pd.DataFrame({"dtype": df[KEY].dtypes.astype(str),
                  "non_null": df[KEY].notna().sum()}).to_string())

# ===========================================================================
# 2. DESCRIPTIVE STATISTICS
# ===========================================================================
log(f"\n{'=' * 70}\n2. DESCRIPTIVE STATISTICS\n{'=' * 70}")
NUM = ["lead_time_days", "dispatch_delay_days", "freight", "price",
       "distance_km", "weight_g"]
desc = df[NUM].agg(["mean", "median", "std", "skew"]).T
desc["p90"] = df[NUM].quantile(0.90)
desc["p95"] = df[NUM].quantile(0.95)
desc["mode"] = [df[c].round(0).mode().iloc[0] for c in NUM]
log(desc.round(2).to_string())
log(f"\nOn-time delivery rate: {100 * (1 - df['is_late'].mean()):.2f}%")
log(f"Late orders: {int(df['is_late'].sum()):,} of {len(df):,}")
log(f"Share of orders shipped within the same state: "
    f"{100 * df['same_state'].mean():.1f}%")

# Figure 1: distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col, title, color in [
        (axes[0, 0], "lead_time_days", "Delivery lead time (days)", BLUE),
        (axes[0, 1], "freight", "Freight cost per order", ORANGE),
        (axes[1, 0], "distance_km", "Seller-to-customer distance (km)", BLUE),
        (axes[1, 1], "dispatch_delay_days", "Seller dispatch delay (days)",
         ORANGE)]:
    bins = (range(0, int(df[col].max()) + 2)
            if col == "dispatch_delay_days" else 50)
    ax.hist(df[col], bins=bins, color=color, edgecolor="white")
    ax.axvline(df[col].mean(), color="black", ls="--",
               label=f"mean {df[col].mean():.1f}")
    ax.axvline(df[col].median(), color=RED, ls="-",
               label=f"median {df[col].median():.1f}")
    ax.set_title(title)
    ax.set_ylabel("Orders")
    ax.legend()
fig.suptitle("Distributions of Key Logistics Variables", fontsize=14)
savefig("fig1_distributions.png")

# ===========================================================================
# 3. CORRELATIONS
# ===========================================================================
log(f"\n{'=' * 70}\n3. CORRELATIONS\n{'=' * 70}")
CORR = ["lead_time_days", "freight", "price", "weight_g", "distance_km",
        "dispatch_delay_days", "n_items", "freight_to_price", "is_late",
        "same_state"]
pear = df[CORR].corr()
spear = df[CORR].corr(method="spearman")
log("Pearson correlation with lead_time_days:")
log(pear["lead_time_days"].drop("lead_time_days").sort_values(
    ascending=False).round(3).to_string())
log("\nSpearman correlation with lead_time_days:")
log(spear["lead_time_days"].drop("lead_time_days").sort_values(
    ascending=False).round(3).to_string())
log("\nPearson correlation with freight:")
log(pear["freight"].drop("freight").sort_values(
    ascending=False).round(3).to_string())
log("\nPearson correlation with is_late:")
log(pear["is_late"].drop("is_late").sort_values(
    ascending=False).round(3).to_string())

plt.figure(figsize=(10, 8))
sns.heatmap(pear, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix of Logistics Variables (Pearson)")
savefig("fig2_correlation_heatmap.png")

# ===========================================================================
# 4. TRENDS OVER TIME
# ===========================================================================
log(f"\n{'=' * 70}\n4. MONTHLY TRENDS\n{'=' * 70}")
df["month"] = df["order_purchase_timestamp"].dt.to_period("M")
monthly = df.groupby("month").agg(
    orders=("order_id", "count"),
    on_time_pct=("is_late", lambda s: 100 * (1 - s.mean())),
    avg_lead=("lead_time_days", "mean"),
    avg_freight=("freight", "mean"))
excluded = int((monthly["orders"] < MIN_GROUP).sum())
monthly = monthly[monthly["orders"] >= MIN_GROUP]
log(f"(months with fewer than {MIN_GROUP} orders excluded: {excluded})")
log(monthly.round(2).to_string())
worst = monthly["on_time_pct"].idxmin()
best = monthly["on_time_pct"].idxmax()
log(f"\nLowest on-time rate: {worst} ({monthly.loc[worst, 'on_time_pct']:.1f}%)")
log(f"Highest on-time rate: {best} ({monthly.loc[best, 'on_time_pct']:.1f}%)")

fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
x = monthly.index.astype(str)
a1.bar(x, monthly["orders"], color=BLUE)
a1.set_ylabel("Orders per month")
a1.set_title("Order Volume by Month")
a2.plot(x, monthly["on_time_pct"], color=BLUE, marker="o",
        label="On-time delivery rate (%)")
a2.set_ylabel("On-time rate (%)", color=BLUE)
a3 = a2.twinx()
a3.plot(x, monthly["avg_lead"], color=ORANGE, marker="s",
        label="Average lead time (days)")
a3.set_ylabel("Avg lead time (days)", color=ORANGE)
a3.grid(False)
a2.set_title("On-Time Rate and Lead Time by Month")
a2.tick_params(axis="x", rotation=60)
savefig("fig3_monthly_trends.png")

# ===========================================================================
# 5. GEOGRAPHY
# ===========================================================================
log(f"\n{'=' * 70}\n5. PERFORMANCE BY CUSTOMER STATE\n{'=' * 70}")
state = df.groupby("customer_state").agg(
    orders=("order_id", "count"),
    late_pct=("is_late", lambda s: 100 * s.mean()),
    avg_lead=("lead_time_days", "mean"),
    avg_freight=("freight", "mean"),
    avg_distance=("distance_km", "mean"))
small = int((state["orders"] < MIN_GROUP).sum())
state = state[state["orders"] >= MIN_GROUP].sort_values("late_pct",
                                                        ascending=False)
log(f"(states with fewer than {MIN_GROUP} orders excluded: {small})")
log("\nHighest late-delivery rates:")
log(state.head(5).round(2).to_string())
log("\nLowest late-delivery rates:")
log(state.tail(5).round(2).to_string())
top_share = (df["customer_state"].value_counts(normalize=True).head(3) * 100)
log("\nShare of orders from the top 3 states:")
log(top_share.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 7))
s1 = state.sort_values("late_pct")
axes[0].barh(s1.index, s1["late_pct"], color=RED)
axes[0].set_xlabel("Late deliveries (%)")
axes[0].set_title("Late-Delivery Rate by State")
s2 = state.sort_values("avg_lead")
axes[1].barh(s2.index, s2["avg_lead"], color=BLUE)
axes[1].set_xlabel("Average lead time (days)")
axes[1].set_title("Average Lead Time by State")
savefig("fig4_state_performance.png")

# ===========================================================================
# 6. DISTANCE RELATIONSHIPS
# ===========================================================================
log(f"\n{'=' * 70}\n6. DISTANCE AND LEAD TIME / COST / LATENESS\n{'=' * 70}")
dist_bins = [0, 200, 500, 1000, 1500, 3000]
dist_labels = ["0-200", "200-500", "500-1000", "1000-1500", "1500+"]
df["dist_bucket"] = pd.cut(df["distance_km"], dist_bins, labels=dist_labels,
                           include_lowest=True)
by_dist = df.groupby("dist_bucket", observed=True).agg(
    orders=("order_id", "count"),
    avg_lead=("lead_time_days", "mean"),
    avg_freight=("freight", "mean"),
    late_pct=("is_late", lambda s: 100 * s.mean()))
log(by_dist.round(2).to_string())
slope, intercept = np.polyfit(df["distance_km"], df["lead_time_days"], 1)
log(f"\nLinear trend: each extra 1,000 km adds about {slope * 1000:.1f} days "
    f"of lead time")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
hb = axes[0].hexbin(df["distance_km"], df["lead_time_days"], gridsize=45,
                    bins="log", cmap="Blues", mincnt=1)
xs = np.linspace(df["distance_km"].min(), df["distance_km"].max(), 50)
axes[0].plot(xs, slope * xs + intercept, color=RED, lw=2, label="linear trend")
axes[0].set_xlabel("Distance (km)")
axes[0].set_ylabel("Lead time (days)")
axes[0].set_title("Distance vs Lead Time")
axes[0].legend()
fig.colorbar(hb, ax=axes[0], label="orders (log)")
hb2 = axes[1].hexbin(df["distance_km"], df["freight"], gridsize=45,
                     bins="log", cmap="Oranges", mincnt=1)
axes[1].set_xlabel("Distance (km)")
axes[1].set_ylabel("Freight")
axes[1].set_title("Distance vs Freight")
fig.colorbar(hb2, ax=axes[1], label="orders (log)")
axes[2].bar(by_dist.index.astype(str), by_dist["late_pct"], color=RED)
axes[2].set_xlabel("Distance band (km)")
axes[2].set_ylabel("Late deliveries (%)")
axes[2].set_title("Late Rate by Distance Band")
savefig("fig5_distance_relationships.png")

# ===========================================================================
# 7. WHERE IS TIME SPENT? (BOTTLENECKS)
# ===========================================================================
log(f"\n{'=' * 70}\n7. DELIVERY STAGES AND DISPATCH DELAY\n{'=' * 70}")
day = 86400
df["stage_approval"] = (df["order_approved_at"]
                        - df["order_purchase_timestamp"]).dt.total_seconds() / day
df["stage_dispatch"] = (df["order_delivered_carrier_date"]
                        - df["order_approved_at"]).dt.total_seconds() / day
df["stage_transit"] = (df["order_delivered_customer_date"]
                       - df["order_delivered_carrier_date"]).dt.total_seconds() / day
STAGES = ["stage_approval", "stage_dispatch", "stage_transit"]
stage = df.groupby("is_late")[STAGES].mean().rename(index={0: "on time", 1: "late"})
stage.columns = ["approval", "seller dispatch", "carrier transit"]
stage.loc["all orders"] = df[STAGES].mean().values
log("Average days spent in each stage:")
log(stage.round(2).to_string())
late_minus_ontime = stage.loc["late"] - stage.loc["on time"]
log("\nExtra days in late orders vs on-time orders:")
log(late_minus_ontime.round(2).to_string())

disp_bins = [-1, 0, 1, 2, 4, 7, 50]
disp_labels = ["0", "1", "2", "3-4", "5-7", "8+"]
df["disp_bucket"] = pd.cut(df["dispatch_delay_days"], disp_bins,
                           labels=disp_labels)
by_disp = df.groupby("disp_bucket", observed=True).agg(
    orders=("order_id", "count"),
    late_pct=("is_late", lambda s: 100 * s.mean()),
    avg_lead=("lead_time_days", "mean"))
log("\nLate rate by seller dispatch delay (days):")
log(by_disp.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
stage.loc[["on time", "late"]].plot(kind="bar", stacked=True, ax=axes[0],
                                    color=[BLUE, ORANGE, RED], rot=0)
axes[0].set_xlabel("")
axes[0].set_ylabel("Average days")
axes[0].set_title("Time Spent per Stage: On-Time vs Late Orders")
axes[0].legend(title="Stage")
axes[1].bar(by_disp.index.astype(str), by_disp["late_pct"], color=RED)
axes[1].set_xlabel("Seller dispatch delay (days)")
axes[1].set_ylabel("Late deliveries (%)")
axes[1].set_title("Late Rate by Seller Dispatch Delay")
savefig("fig6_stage_breakdown.png")

# ===========================================================================
# 8. COST DRIVERS
# ===========================================================================
log(f"\n{'=' * 70}\n8. FREIGHT COST DRIVERS\n{'=' * 70}")
w_bins = [0, 500, 1000, 2000, 5000, 30000]
w_labels = ["<500 g", "500-1000 g", "1-2 kg", "2-5 kg", "5 kg+"]
df["weight_bucket"] = pd.cut(df["weight_g"], w_bins, labels=w_labels,
                             include_lowest=True)
by_w = df.groupby("weight_bucket", observed=True).agg(
    orders=("order_id", "count"), avg_freight=("freight", "mean"))
log("Average freight by product weight:")
log(by_w.round(2).to_string())
log("\nAverage freight by distance band:")
log(by_dist[["orders", "avg_freight"]].round(2).to_string())
same = df.groupby("same_state").agg(
    orders=("order_id", "count"), avg_freight=("freight", "mean"),
    avg_lead=("lead_time_days", "mean"),
    late_pct=("is_late", lambda s: 100 * s.mean())).rename(
        index={0: "different state", 1: "same state"})
log("\nSame-state vs cross-state deliveries:")
log(same.round(2).to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].bar(by_w.index.astype(str), by_w["avg_freight"], color=ORANGE)
axes[0].set_xlabel("Product weight")
axes[0].set_ylabel("Average freight")
axes[0].set_title("Freight by Product Weight")
axes[1].bar(by_dist.index.astype(str), by_dist["avg_freight"], color=BLUE)
axes[1].set_xlabel("Distance band (km)")
axes[1].set_ylabel("Average freight")
axes[1].set_title("Freight by Distance")
plot_df = df.assign(route=df["same_state"].map(
    {0: "different state", 1: "same state"}))
sns.boxplot(data=plot_df, x="route", y="freight", showfliers=False,
            palette=[BLUE, ORANGE], hue="route", legend=False, ax=axes[2])
axes[2].set_xlabel("")
axes[2].set_ylabel("Freight")
axes[2].set_title("Freight: Same-State vs Cross-State")
savefig("fig7_cost_drivers.png")

cat = df.groupby("category").agg(
    orders=("order_id", "count"), avg_freight=("freight", "mean"),
    avg_weight=("weight_g", "mean"))
cat = cat[cat["orders"] >= MIN_CAT].sort_values("avg_freight", ascending=False)
log(f"\nCategories ({MIN_CAT}+ orders) with the highest average freight:")
log(cat.head(10).round(2).to_string())
log(f"\nCategories ({MIN_CAT}+ orders) with the lowest average freight:")
log(cat.tail(5).round(2).to_string())
top = cat.head(10).sort_values("avg_freight")
plt.figure(figsize=(9, 6))
plt.barh(top.index, top["avg_freight"], color=ORANGE)
plt.xlabel("Average freight per order")
plt.title("Product Categories with the Highest Shipping Cost")
savefig("fig8_category_freight.png")

# ===========================================================================
# 9. KEY NUMBERS FOR THE REPORT
# ===========================================================================
log(f"\n{'=' * 70}\n9. KEY NUMBERS\n{'=' * 70}")
log(f"corr(distance, lead time)  = {pear.loc['distance_km', 'lead_time_days']:.3f}")
log(f"corr(distance, freight)    = {pear.loc['distance_km', 'freight']:.3f}")
log(f"corr(weight, freight)      = {pear.loc['weight_g', 'freight']:.3f}")
log(f"corr(dispatch, is_late)    = {pear.loc['dispatch_delay_days', 'is_late']:.3f}")
log(f"corr(distance, is_late)    = {pear.loc['distance_km', 'is_late']:.3f}")

(HERE / "week3_eda_log.txt").write_text("\n".join(LOG_LINES),
                                         encoding="utf-8")
print("\nDone. Full log saved as week3_eda_log.txt")