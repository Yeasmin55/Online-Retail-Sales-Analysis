# Online-Retail-Sales-Analysis
Exploratory data analysis of a UK-based non-store online retail brand's transaction data, uncovering product performance, sales trends, and country-level patterns to support inventory and marketing decisions.


Business Problem

The business - a UK-based online retailer selling all-occasion gifts was experiencing inconsistent sales patterns and unclear product performance across countries and customer segments, with limited visibility into return behavior. This analysis uncovers:

Top- and least-performing products
Sales trends over time (by month)
Customer return patterns
Geographic performance by country
Dataset

Online Retail Data Set (UCI Machine Learning Repository) - ~542,000 transactions across 8 variables (invoice number, stock code, product description, quantity, invoice date, unit price, customer ID, and country).

Approach
Data cleaning: handled 135K+ missing CustomerIDs (dropped, since customer-level analysis requires a valid ID) and 1,454 missing product descriptions (filled as "Unknown"); filtered out returns/cancellations (negative quantity or price) to isolate completed sales
Feature engineering: split the combined date/time field into date, time, and month components; derived total cost per transaction
Outlier diagnostics: evaluated skewness, kurtosis, and 3-sigma bounds on Quantity and UnitPrice, backed by skewness/kurtosis significance tests
Visualization: country-level sales bar charts, top/least-selling products, monthly sales trend line, quantity and price distributions, a correlation heatmap, and boxplots for outlier detection
Key Findings
Sales peak sharply in October–December, consistent with holiday shopping demand — a clear signal for inventory and staffing planning
Both Quantity and UnitPrice are heavily right-skewed with high kurtosis, meaning most transactions are small and low-value, with a small number of large outliers (bulk orders and high-price items) driving the tails
The UK dominates transaction volume; outside the UK, a small set of European countries account for most international purchases
Tools Used

Python - pandas, NumPy, Matplotlib, Seaborn, SciPy

Files
online_retail_analysis.py - full analysis script (data cleaning, feature engineering, EDA, and visualizations)
DSCI_505_Semester_Project.docx - full project write-up (business problem, dataset description, methodology, findings, and reflection)
