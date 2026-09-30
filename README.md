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

Files used: orders, order items, customers, sellers and geolocation.

## Project progress

| Week | Topic | Status |
|------|-------|--------|
| 1 | Strategic planning and data exploration | Done |
| 2 | Data collection, cleaning and preprocessing | Done |
| 3 | Advanced analysis and visualization | In progress |
| 4 | Predictive modeling and optimization | Upcoming |

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
