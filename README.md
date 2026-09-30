# Logistics Analytics Internship

This is A Data analysis project for the Logistics Data Analyst Internship. The scenario is an e-commerce retailer facing late deliveries and rising shipping costs. The project builds from planning to data cleaning, analysis and predictive modeling in Python.

## Dataset

This project uses the **Brazilian E-Commerce Public Dataset by Olist** (about 100,000 orders from 2016 to 2018 across several linked tables).

- Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- The data files are not included in this repository.

## How to run the code

1. Download the dataset from the Kaggle link above and unzip it.
2. Create a folder named `data` in the project root and place the CSV files in it.
3. Install the libraries: `pip install -r requirements.txt`
4. Run: `python week1_strategic_planning/week_one.py`
5. For Week 2: `python week2_data_cleaning/week_two.py`
6. for week 3: `python week3_eda_visualization/week_3.py`
7. for week 4:  `week4_modelling_optimization/week_4.py` 

Files used: orders, order items, customers, sellers, products and geolocation.

## Project progress

| Week | Topic | Status |
|------|-------|--------|
| 1 | Strategic planning and data exploration | Done |
| 2 | Data collection, cleaning and preprocessing | Done |
| 3 | Advanced analysis and visualization | Done |
| 4 | Predictive modeling and optimization | Done |

## Week 1 results (preliminary)

- On-time delivery rate: 91.89%
- Average delivery lead time: 12.09 days
- Cost per shipment: 22.79 BRL
- Random forest delivery time model: test MAE 4.57 days (linear regression baseline: 4.86)
- Sellers grouped into three performance clusters with K-Means

## Week 2 results (data cleaning)

Script: `week2_data_cleaning/week_two.py`

- Merged 6 Olist tables into one order-level table (98,666 orders)
- Kept delivered orders only and removed 1,373 orders with impossible timestamps (for example carrier pickup before order approval)
- Filled small gaps with medians (weight, dispatch delay, distance) and added flag columns marking filled values
- Capped outliers at the 1st and 99th percentile (for example maximum freight went from 1,794.96 to 104.26)
- Applied a log transform to freight (skewness 2.74 to 0.89) and created min-max and z-score scaled features
- Final clean dataset: 95,097 orders (98.6% of delivered orders)
- KPIs stayed stable after cleaning (on-time delivery 91.89% to 91.81%)

The pipeline saves a cleaned dataset (`olist_clean.csv`) into the `data` folder, which is not included in this repository.

## Week 3 results (analysis and visualization)

Script: `week3_eda_visualization/week_3.py` (reads the cleaned data from Week 2)

- Descriptive statistics, distributions and correlation analysis of 95,097 delivered orders
- Eight charts: distributions, correlation heatmap, monthly trends, state performance, distance relationships, delivery stage breakdown, cost drivers, category freight
- Carrier transit is the main bottleneck: late orders spend 25.68 days in transit vs 7.91 for on-time orders
- Late rate rises from 4.83% to 28.36% as seller dispatch delay grows from 0 to 8+ days
- Northeast states (AL, MA, PI, CE, SE) have late rates of 15% to 24%, vs about 6% in SP, MG and PR
- Freight is driven mainly by weight (correlation 0.61) and distance; same-state deliveries cost about 41% less and arrive in half the time

## Week 4 results (predictive modeling and optimization)

Script:  `week4_modelling_optimization/week_4.py` 

- Built predictive models for delivery lead time using 95,097 orders, with an 80/20 train-test split (76,077 training rows and 19,020 test rows)
- Model A predicts delivery lead time using information available at checkout: distance, weight, price, number of items, same-state indicator, purchase month/day of week, customer state and seller state
- Gradient Boosting with hyperparameter tuning was the best Model A by test MAE: 4.70 days, compared with 6.26 days for the mean baseline
- The tuned Model A achieved RMSE 6.73 days, R² 0.374, and 44.88% of predictions within 3 days
- The tuned model reduced MAE by approximately 25.0% compared with the mean baseline
- The current delivery estimate used as a reference had 12.69 days MAE on the test set; the tuned Model A had 4.70 days MAE
- Model B added seller dispatch delay, an input available after dispatch. Its test MAE was 4.03 days, RMSE 6.01 days, R² 0.502, with 54.21% of predictions within 3 days
- A time-based robustness check, training on the earliest 80% and testing on the latest 20% of orders, produced 4.09 days MAE, 5.32 days RMSE, and 0.139 R²
- Error analysis showed that orders taking more than 20 days were harder to predict: MAE was 11.73 days for these orders versus 3.62 days for the remaining orders
- Prediction error increased with distance: MAE ranged from 3.30 days for 0–200 km to 6.85 days for 1,500+ km
- Model A permutation importance identified purchase month, distance, and same-state delivery as the largest contributors among the tested features
- In Model B, dispatch delay was the largest contributor, followed by distance and purchase month
- Delivery-promise calibration showed that a +9 day safety buffer produced a model promise with 92.78% on-time delivery, close to the current 92.09% reference rate, with an average promise of 21.54 days
- A +12 day buffer produced 95.08% on-time delivery on the test set, with an average promised time of 24.54 days
- Seller dispatch what-if analysis estimated network-wide average lead-time reductions of 1.51 days for a maximum 1-day dispatch SLA, 1.11 days for 2 days, and 0.86 days for 3 days
- egional fulfilment what-if analysis covered 60,912 cross-state orders (64.1%). Serving these orders from same-state stock was estimated to save 7.02 delivery days and 10.10 freight units per affected order on average
- The regional scenario estimated that the top 10 states accounted for 76.4% of total predicted days saved, while the top 5 accounted for 55.0%
- Week 4 generated model comparison, actual-vs-predicted, feature importance, error segmentation, promise calibration and what-if scenario visualizations

Note: The seller SLA and regional fulfilment figures are model-based what-if estimates from the Week 4 analysis and should not be interpreted as guaranteed operational outcomes.
