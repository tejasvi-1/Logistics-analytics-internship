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
| 2 | Data collection, cleaning and preprocessing | In progress |
| 3 | Advanced analysis and visualization | Upcoming |
| 4 | Predictive modeling and optimization | Upcoming |

## Week 1 results (preliminary)

- On-time delivery rate: 91.89%
- Average delivery lead time: 12.09 days
- Cost per shipment: 22.79 BRL
- Random forest delivery time model: test MAE 4.57 days (linear regression baseline: 4.86)
- Sellers grouped into three performance clusters with K-Means
