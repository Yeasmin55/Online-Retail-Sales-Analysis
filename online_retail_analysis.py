# -*- coding: utf-8 -*-
"""
Online Retail Sales Analysis
Author: Yeasmin Akter

Business problem: a UK-based non-store online retail brand selling
all-occasion gifts has inconsistent sales patterns and unclear product
performance across countries and customer segments. This script explores
the dataset to surface top/least-performing products, monthly sales
trends, and country-level performance to support inventory, marketing,
and customer-targeting decisions.

Dataset: Online Retail Data Set (UCI Machine Learning Repository)
https://archive.ics.uci.edu/dataset/352/online+retail
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# ============================================================
# 1. Load data
# ============================================================
df = pd.read_csv('Online Retail Data.csv', parse_dates=['InvoiceDate'])

print(df.head())
print(df.info())
print(df.isna().sum())

# ============================================================
# 2. Clean and prepare
# ============================================================
df_clean = df.copy()

# Split date/time and extract month for trend analysis
df_clean['Invoice_Date'] = df_clean['InvoiceDate'].dt.date
df_clean['InvoiceTime'] = df_clean['InvoiceDate'].dt.time
df_clean['Month'] = df_clean['InvoiceDate'].dt.month
df_clean.drop(columns=['InvoiceDate'], inplace=True)

# Missing descriptions -> 'Unknown' (rest of the row is still usable)
df_clean['Description'] = df_clean['Description'].fillna('Unknown')

# Missing CustomerID -> drop (customer-level analysis needs a valid ID)
df_clean.dropna(subset=['CustomerID'], inplace=True)

# Keep only completed sales (positive quantity and price);
# negative quantities represent returns/cancellations, excluded here
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]

# Derived field
df_clean['TotalCost'] = df_clean['Quantity'] * df_clean['UnitPrice']

# ============================================================
# 3. Country-level sales
# ============================================================
quantity_by_country = df_clean.pivot_table(values='Quantity', index='Country', aggfunc='sum')
quantity_by_country.drop(index='Unspecified', inplace=True, errors='ignore')
quantity_by_country.plot(kind='bar', legend=False, title='Total Quantity Sold by Country')
plt.ylabel('Quantity')
plt.tight_layout()
plt.show()

# Top 10 countries by number of purchases, excluding the UK (dominant market)
top_countries = df_clean[df_clean['Country'] != 'United Kingdom']['Country'].value_counts().head(10)
sns.barplot(x=top_countries.index, y=top_countries.values)
plt.title('Top 10 Countries by Number of Purchases (Excluding UK)')
plt.xlabel('Country')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================================================
# 4. Product performance
# ============================================================
top_products = df_clean['Description'].value_counts().head(10)
sns.barplot(x=top_products.values, y=top_products.index)
plt.title('Top 10 Selling Products')
plt.xlabel('Number Sold')
plt.ylabel('Product')
plt.tight_layout()
plt.show()

# Least-sold products by total quantity
product_sales = df_clean.groupby('Description')['Quantity'].sum()
least_sold = product_sales.sort_values().head(10)
print('\nLeast-sold products:\n', least_sold)

# ============================================================
# 5. Sales trend over time
# ============================================================
sales_by_month = df_clean.groupby('Month')['TotalCost'].sum()
plt.figure(figsize=(10, 6))
sales_by_month.plot(kind='line', marker='o')
plt.title('Total Sales by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.grid()
plt.tight_layout()
plt.show()

# ============================================================
# 6. Distribution and outlier analysis
# ============================================================
plt.figure(figsize=(10, 5))
sns.histplot(df_clean['Quantity'], bins=100, kde=True)
plt.title('Distribution of Quantity Purchased')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.xlim(0, 100)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df_clean['UnitPrice'], bins=100, kde=True)
plt.title('Distribution of Unit Price')
plt.xlabel('Unit Price')
plt.ylabel('Frequency')
plt.xlim(0, 100)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(df_clean[['Quantity', 'UnitPrice', 'TotalCost']].corr(), annot=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x=df_clean['Quantity'])
plt.title('Boxplot of Quantity (Outlier Check)')
plt.tight_layout()
plt.show()

# Skewness / kurtosis for Quantity and UnitPrice (on the raw column,
# before the outlier filter, to match the reported diagnostics)
print('\nQuantity  - Skewness:', round(df.Quantity.skew(), 2), '| Kurtosis:', round(df.Quantity.kurt(), 2))
print('UnitPrice - Skewness:', round(df.UnitPrice.skew(), 2), '| Kurtosis:', round(df.UnitPrice.kurt(), 2))

print('\nSkewness test p-value (Quantity):', round(stats.skewtest(df.Quantity.dropna()).pvalue, 4))
print('Kurtosis test p-value (Quantity):', round(stats.kurtosistest(df.Quantity.dropna()).pvalue, 4))

# Values beyond 3 standard deviations from the mean
sd = df.Quantity.std()
outliers = df[(df.Quantity < (-3 * sd)) | (df.Quantity > (3 * sd))]
print(f'\nRows beyond 3 std dev of Quantity: {len(outliers)}')

# ============================================================
# 7. Summary statistics
# ============================================================
print('\n', df_clean.describe())
