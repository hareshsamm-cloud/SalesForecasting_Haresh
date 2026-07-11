import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(
    page_title="Superstore Sales Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Superstore Sales Forecasting Dashboard")
st.markdown(
    "Interactive dashboard for sales analysis, forecasting, anomaly detection and demand segmentation."
)

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        dayfirst=True
    )

    return df
df = load_data()
st.sidebar.header("Filters")

years = sorted(df["Order Date"].dt.year.unique())

selected_year = st.sidebar.multiselect(
    "Select Year",
    years,
    default=years
)

regions = sorted(df["Region"].unique())

selected_region = st.sidebar.multiselect(
    "Select Region",
    regions,
    default=regions
)

categories = sorted(df["Category"].unique())

selected_category = st.sidebar.multiselect(
    "Select Category",
    categories,
    default=categories
)
filtered_df = df[
    (df["Order Date"].dt.year.isin(selected_year)) &
    (df["Region"].isin(selected_region)) &
    (df["Category"].isin(selected_category))
]
total_sales = filtered_df["Sales"].sum()

total_orders = filtered_df["Order ID"].nunique()

avg_sales = filtered_df["Sales"].mean()

total_customers = filtered_df["Customer ID"].nunique()

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "💰 Total Sales",
    f"${total_sales:,.0f}"
)

c2.metric(
    "🛒 Orders",
    total_orders
)

c3.metric(
    "📦 Avg Sales",
    f"${avg_sales:.2f}"
)

c4.metric(
    "👥 Customers",
    total_customers
)
st.subheader("Dataset Preview")

st.dataframe(filtered_df.head(10))
# Monthly Sales Trend
st.subheader("📈 Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .groupby(
        pd.Grouper(
            key="Order Date",
            freq="MS"
        )
    )["Sales"]
    .sum()
    .reset_index()
)

fig = px.line(
    monthly_sales,
    x="Order Date",
    y="Sales",
    markers=True,
    title="Monthly Sales"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
left, right = st.columns(2)
# Sales by Category
category_sales = (
    filtered_df
    .groupby("Category")["Sales"]
    .sum()
    .reset_index()
)

fig = px.bar(
    category_sales,
    x="Category",
    y="Sales",
    color="Category",
    title="Sales by Category"
)

left.plotly_chart(
    fig,
    use_container_width=True
)
# Sales by Region
region_sales = (
    filtered_df
    .groupby("Region")["Sales"]
    .sum()
    .reset_index()
)

fig = px.pie(
    region_sales,
    names="Region",
    values="Sales",
    title="Regional Sales Distribution"
)

right.plotly_chart(
    fig,
    use_container_width=True
)
# Top Products
st.subheader("🏆 Top 10 Products")

top_products = (
    filtered_df
    .groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig = px.bar(
    top_products,
    x="Product Name",
    y="Sales",
    color="Sales",
    title="Top 10 Products"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# Sales Forecasting
st.subheader("🤖 Sales Forecasting")

forecast_data = (
    filtered_df
    .groupby(
        pd.Grouper(
            key="Order Date",
            freq="MS"
        )
    )["Sales"]
    .sum()
    .reset_index()
)

forecast_data.columns = ["ds", "y"]

# Train the Prophet model
model = Prophet()

model.fit(forecast_data)

# Forecast the next 3 months
future = model.make_future_dataframe(
    periods=3,
    freq="MS"
)

forecast = model.predict(future)
# Plot the forecast
fig = px.line(
    forecast,
    x="ds",
    y="yhat",
    title="Sales Forecast"
)

fig.add_scatter(
    x=forecast_data["ds"],
    y=forecast_data["y"],
    mode="lines",
    name="Actual Sales"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
# Display the next 3 months forecast
st.subheader("Next 3-Month Forecast")

forecast_table = forecast[
    [
        "ds",
        "yhat",
        "yhat_lower",
        "yhat_upper"
    ]
].tail(3)

forecast_table.columns = [
    "Date",
    "Forecast",
    "Lower Bound",
    "Upper Bound"
]

st.dataframe(forecast_table)
# Display the forecast summary
st.success(
    "The Prophet model predicts sales for the next three months based on historical monthly sales patterns."
)
# Anomaly Detection
st.subheader("🚨 Sales Anomaly Detection")

weekly_sales = (
    filtered_df
    .groupby(
        pd.Grouper(
            key="Order Date",
            freq="W"
        )
    )["Sales"]
    .sum()
    .reset_index()
)

# Detect anomalies using Isolation Forest
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

weekly_sales["Anomaly"] = model.fit_predict(
    weekly_sales[["Sales"]]
)

weekly_sales["Anomaly"] = (
    weekly_sales["Anomaly"] == -1
)

# Plot weekly sales and detected anomalies
fig = px.line(
    weekly_sales,
    x="Order Date",
    y="Sales",
    title="Weekly Sales"
)

fig.add_scatter(
    x=weekly_sales[
        weekly_sales["Anomaly"]
    ]["Order Date"],
    y=weekly_sales[
        weekly_sales["Anomaly"]
    ]["Sales"],
    mode="markers",
    marker=dict(
        size=12,
        color="red"
    ),
    name="Anomaly"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
# Display anomalous records
st.subheader("Detected Anomalies")

st.dataframe(
    weekly_sales[
        weekly_sales["Anomaly"]
    ]
)
# Display anomaly statistics
total_anomalies = (
    weekly_sales["Anomaly"]
    .sum()
)

st.metric(
    "Total Anomalies",
    total_anomalies
)
# Display anomaly insights
st.info(
    """
    Red points represent unusual sales behavior.

    These anomalies may occur because of seasonal demand,
    promotional campaigns, holidays, inventory shortages,
    or unexpected business events.
    """
)

# Product Demand Segmentation
st.subheader("🎯 Product Demand Segmentation")

subcategory_data = (
    filtered_df.groupby("Sub-Category")
    .agg(
        Total_Sales=("Sales", "sum"),
        Average_Sales=("Sales", "mean"),
        Order_Count=("Sales", "count")
    )
    .reset_index()
)
# Scale the features
features = subcategory_data[
    [
        "Total_Sales",
        "Average_Sales",
        "Order_Count"
    ]
]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)
# Train the K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

subcategory_data["Cluster"] = (
    kmeans.fit_predict(scaled_features)
)
# Assign demand labels
cluster_labels = {
    0: "Low Demand",
    1: "Medium Demand",
    2: "High Demand",
    3: "Seasonal Demand"
}

subcategory_data["Demand Segment"] = (
    subcategory_data["Cluster"]
    .map(cluster_labels)
)
# Plot the product demand clusters
fig = px.scatter(
    subcategory_data,
    x="Total_Sales",
    y="Average_Sales",
    color="Demand Segment",
    hover_name="Sub-Category",
    size="Order_Count",
    title="Product Demand Segmentation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
# Display clustered products
st.subheader("Clustered Products")

st.dataframe(
    subcategory_data
)
# Display demand segment distribution
segment_count = (
    subcategory_data["Demand Segment"]
    .value_counts()
    .reset_index()
)

segment_count.columns = [
    "Demand Segment",
    "Count"
]

fig = px.bar(
    segment_count,
    x="Demand Segment",
    y="Count",
    color="Demand Segment",
    title="Demand Segment Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
# Display clustering insights
st.success(
    """
    High-demand products require sufficient inventory.

    Seasonal-demand products should be stocked
    before expected peak periods.

    Low-demand products can be managed with
    optimized inventory levels.
    """
)
# Business Insights
st.subheader("💡 Business Insights")

best_category = (
    filtered_df
    .groupby("Category")["Sales"]
    .sum()
    .idxmax()
)

best_region = (
    filtered_df
    .groupby("Region")["Sales"]
    .sum()
    .idxmax()
)

top_product = (
    filtered_df
    .groupby("Product Name")["Sales"]
    .sum()
    .idxmax()
)
st.success(
    f"""
### Key Findings

• Highest Revenue Category : **{best_category}**

• Best Performing Region : **{best_region}**

• Best Selling Product : **{top_product}**

• Prophet forecasting predicts future sales using historical monthly trends.

• Isolation Forest detects unusual sales behaviour.

• K-Means groups products based on demand characteristics.
"""
)
# Summary statistics
st.subheader("📋 Dashboard Summary")

summary = pd.DataFrame({
    "Metric":[
        "Total Sales",
        "Orders",
        "Customers",
        "Categories",
        "Regions"
    ],
    "Value":[
        filtered_df["Sales"].sum(),
        filtered_df["Order ID"].nunique(),
        filtered_df["Customer ID"].nunique(),
        filtered_df["Category"].nunique(),
        filtered_df["Region"].nunique()
    ]
})

st.dataframe(
    summary,
    use_container_width=True
)
# Download filtered dataset
csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Dataset",
    data=csv,
    file_name="filtered_superstore.csv",
    mime="text/csv"
)
st.divider()
st.markdown(
"""
### 🛠 Technologies Used

- Streamlit
- Pandas
- Plotly
- Prophet
- Scikit-learn
- Statsmodels
"""
)
st.markdown(
"""
---
### 📌 Internship Project

**Title:** Sales Forecasting and Anomaly Detection Dashboard

This dashboard was developed using the Superstore Sales Dataset to perform:

- Sales Analysis
- Forecasting
- Anomaly Detection
- Product Demand Segmentation
- Business Insights

Developed using Python and Streamlit.
"""
)