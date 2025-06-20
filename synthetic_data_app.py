import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import make_blobs, make_regression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- Logo and Title ---
st.markdown("""
<div style='display: flex; align-items: center;'>
    <img src='https://cdn-icons-png.flaticon.com/512/2921/2921826.png' width='60' style='margin-right: 20px;'>
    <h1 style='color:#4F8BF9; display:inline;'>MockData Lab</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Welcome to <b>MockData Lab</b>! Generate realistic synthetic datasets for multiple real-world sectors (Sales, Weather, Finance, IoT, Healthcare) or play with generic ML data. Instantly preview, visualize, and download your data!
""", unsafe_allow_html=True)

# Sidebar: Sector selection
with st.sidebar:
    st.header("1. Choose Data Sector 🏷️")
    sector = st.selectbox(
        "Select a sector:",
        [
            "Sales/Retail",
            "Weather",
            "Finance",
            "IoT/Sensor",
            "Healthcare",
            "ML Playground (Generic)"
        ],
        help="Pick a real-world sector to generate synthetic data for."
    )
    n_samples = st.slider("Number of samples", 50, 2000, 500, help="How many rows to generate?")
    n_features = 2
    if sector == "ML Playground (Generic)":
        n_features = st.slider("Number of features (dimensions)", 2, 5, 2, help="Choose 2D, 3D, 4D, or 5D data.")
    st.markdown("---")
    st.info("Tip: Change the sector and sample size to generate different types of synthetic data.")

# --- Data Generation Functions ---
def generate_sales_data(n):
    dates = pd.date_range(datetime.today() - timedelta(days=n), periods=n).date
    products = [f"Product_{i}" for i in range(1, 6)]
    regions = ["North", "South", "East", "West"]
    data = []
    for i in range(n):
        product = np.random.choice(products)
        region = np.random.choice(regions)
        units = np.random.poisson(20) + np.random.randint(0, 10)
        price = np.round(np.random.uniform(10, 100), 2)
        promo = np.random.choice([0, 1], p=[0.85, 0.15])
        revenue = units * price * (0.8 if promo else 1.0)
        data.append([dates[i], product, region, units, price, revenue, promo])
    return pd.DataFrame(data, columns=["Date", "Product", "Region", "Units_Sold", "Price", "Revenue", "Promotion"])

def generate_weather_data(n):
    dates = pd.date_range(datetime.today() - timedelta(days=n), periods=n).date
    cities = ["New York", "London", "Delhi", "Tokyo", "Sydney"]
    data = []
    for i in range(n):
        city = np.random.choice(cities)
        temp = 15 + 10 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 5)
        humidity = np.clip(60 + 20 * np.sin(2 * np.pi * i / 30) + np.random.normal(0, 10), 20, 100)
        rain = np.random.choice([0, np.random.uniform(0, 50)], p=[0.8, 0.2])
        wind = np.abs(np.random.normal(10, 5))
        data.append([dates[i], city, round(temp, 1), round(humidity, 1), round(rain, 1), round(wind, 1)])
    return pd.DataFrame(data, columns=["Date", "City", "Temperature_C", "Humidity_%", "Rain_mm", "Wind_kmh"])

def generate_finance_data(n):
    dates = pd.date_range(datetime.today() - timedelta(days=n), periods=n).date
    stocks = ["AAPL", "GOOG", "TSLA", "AMZN", "MSFT"]
    data = []
    for i in range(n):
        stock = np.random.choice(stocks)
        open_ = np.random.uniform(100, 500)
        close = open_ + np.random.normal(0, 10)
        high = max(open_, close) + np.random.uniform(0, 5)
        low = min(open_, close) - np.random.uniform(0, 5)
        volume = int(np.random.uniform(1e5, 1e6))
        data.append([dates[i], stock, round(open_,2), round(close,2), round(high,2), round(low,2), volume])
    return pd.DataFrame(data, columns=["Date", "Stock", "Open", "Close", "High", "Low", "Volume"])

def generate_iot_data(n):
    timestamps = [datetime.now() - timedelta(minutes=5*i) for i in range(n)][::-1]
    devices = [f"Device_{i}" for i in range(1, 6)]
    data = []
    for i in range(n):
        device = np.random.choice(devices)
        value = np.random.normal(50, 10)
        status = np.random.choice(["OK", "WARN", "FAIL"], p=[0.92, 0.06, 0.02])
        data.append([timestamps[i], device, round(value,2), status])
    return pd.DataFrame(data, columns=["Timestamp", "Device_ID", "Sensor_Value", "Status"])

def generate_healthcare_data(n):
    patients = [f"P{str(i).zfill(4)}" for i in range(1, 21)]
    dates = pd.date_range(datetime.today() - timedelta(days=n), periods=n).date
    data = []
    for i in range(n):
        pid = np.random.choice(patients)
        hr = int(np.random.normal(75, 10))
        bp_sys = int(np.random.normal(120, 15))
        bp_dia = int(np.random.normal(80, 10))
        temp = round(np.random.normal(36.8, 0.5), 1)
        data.append([pid, dates[i], hr, bp_sys, bp_dia, temp])
    return pd.DataFrame(data, columns=["Patient_ID", "Date", "Heart_Rate", "BP_Systolic", "BP_Diastolic", "Temperature_C"])

# --- Data Generation Logic ---
if sector == "Sales/Retail":
    df = generate_sales_data(n_samples)
    graph_types = ["Line", "Bar", "Area", "Box", "Scatter", "Histogram"]
    y_options = ["Revenue", "Units_Sold", "Price"]
    group_options = ["Product", "Region", "Promotion"]
    with st.container():
        st.subheader(f"📊 Visualization: {sector}")
        graph_type = st.selectbox("Graph Type", graph_types, key="sales_graph")
        y_axis = st.selectbox("Y-axis", y_options, key="sales_y")
        group_by = st.selectbox("Group by", [None] + group_options, key="sales_group")
        if graph_type == "Line":
            chart = px.line(df, x="Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Over Time")
        elif graph_type == "Bar":
            chart = px.bar(df, x="Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} by Date")
        elif graph_type == "Area":
            chart = px.area(df, x="Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Area Chart")
        elif graph_type == "Box":
            chart = px.box(df, x=group_by if group_by else "Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Boxplot")
        elif graph_type == "Scatter":
            chart = px.scatter(df, x="Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Scatterplot")
        else:
            chart = px.histogram(df, x=y_axis, color=group_by if group_by else None, title=f"{y_axis} Histogram")
elif sector == "Weather":
    df = generate_weather_data(n_samples)
    graph_types = ["Line", "Bar", "Scatter", "Box", "Histogram"]
    y_options = ["Temperature_C", "Humidity_%", "Rain_mm", "Wind_kmh"]
    group_options = ["City"]
    with st.container():
        st.subheader(f"📊 Visualization: {sector}")
        graph_type = st.selectbox("Graph Type", graph_types, key="weather_graph")
        y_axis = st.selectbox("Y-axis", y_options, key="weather_y")
        group_by = st.selectbox("Group by", [None] + group_options, key="weather_group")
        if graph_type == "Line":
            chart = px.line(df, x="Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Over Time")
        elif graph_type == "Bar":
            chart = px.bar(df, x="Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} by Date")
        elif graph_type == "Scatter":
            chart = px.scatter(df, x="Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Scatterplot")
        elif graph_type == "Box":
            chart = px.box(df, x=group_by if group_by else "Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Boxplot")
        else:
            chart = px.histogram(df, x=y_axis, color=group_by if group_by else None, title=f"{y_axis} Histogram")
elif sector == "Finance":
    df = generate_finance_data(n_samples)
    graph_types = ["Line", "Bar", "Scatter", "Box", "Histogram"]
    y_options = ["Open", "Close", "High", "Low", "Volume"]
    group_options = ["Stock"]
    with st.container():
        st.subheader(f"📊 Visualization: {sector}")
        graph_type = st.selectbox("Graph Type", graph_types, key="finance_graph")
        y_axis = st.selectbox("Y-axis", y_options, key="finance_y")
        group_by = st.selectbox("Group by", [None] + group_options, key="finance_group")
        if graph_type == "Line":
            chart = px.line(df, x="Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Over Time")
        elif graph_type == "Bar":
            chart = px.bar(df, x="Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} by Date")
        elif graph_type == "Scatter":
            chart = px.scatter(df, x="Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Scatterplot")
        elif graph_type == "Box":
            chart = px.box(df, x=group_by if group_by else "Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Boxplot")
        else:
            chart = px.histogram(df, x=y_axis, color=group_by if group_by else None, title=f"{y_axis} Histogram")
elif sector == "IoT/Sensor":
    df = generate_iot_data(n_samples)
    graph_types = ["Line", "Scatter", "Box", "Histogram"]
    y_options = ["Sensor_Value"]
    group_options = ["Device_ID", "Status"]
    with st.container():
        st.subheader(f"📊 Visualization: {sector}")
        graph_type = st.selectbox("Graph Type", graph_types, key="iot_graph")
        y_axis = st.selectbox("Y-axis", y_options, key="iot_y")
        group_by = st.selectbox("Group by", [None] + group_options, key="iot_group")
        if graph_type == "Line":
            chart = px.line(df, x="Timestamp", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Over Time")
        elif graph_type == "Scatter":
            chart = px.scatter(df, x="Timestamp", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Scatterplot")
        elif graph_type == "Box":
            chart = px.box(df, x=group_by if group_by else "Timestamp", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Boxplot")
        else:
            chart = px.histogram(df, x=y_axis, color=group_by if group_by else None, title=f"{y_axis} Histogram")
elif sector == "Healthcare":
    df = generate_healthcare_data(n_samples)
    graph_types = ["Line", "Scatter", "Box", "Histogram"]
    y_options = ["Heart_Rate", "BP_Systolic", "BP_Diastolic", "Temperature_C"]
    group_options = ["Patient_ID"]
    with st.container():
        st.subheader(f"📊 Visualization: {sector}")
        graph_type = st.selectbox("Graph Type", graph_types, key="health_graph")
        y_axis = st.selectbox("Y-axis", y_options, key="health_y")
        group_by = st.selectbox("Group by", [None] + group_options, key="health_group")
        if graph_type == "Line":
            chart = px.line(df, x="Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Over Time")
        elif graph_type == "Scatter":
            chart = px.scatter(df, x="Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Scatterplot")
        elif graph_type == "Box":
            chart = px.box(df, x=group_by if group_by else "Date", y=y_axis, color=group_by if group_by else None, title=f"{y_axis} Boxplot")
        else:
            chart = px.histogram(df, x=y_axis, color=group_by if group_by else None, title=f"{y_axis} Histogram")
else:
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n_samples, centers=3, n_features=n_features, random_state=42)
    feature_cols = [f"f{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["cluster"] = y
    with st.container():
        st.subheader(f"📊 Visualization: {sector}")
        if n_features == 2:
            graph_types = ["2D Scatter", "Histogram"]
            graph_type = st.selectbox("Graph Type", graph_types, key="ml2d_graph")
            x_axis = st.selectbox("X-axis", feature_cols, key="ml2d_x")
            y_axis = st.selectbox("Y-axis", feature_cols, index=1, key="ml2d_y")
            if graph_type == "2D Scatter":
                chart = px.scatter(df, x=x_axis, y=y_axis, color="cluster", title="2D Clusters", color_continuous_scale="Viridis")
            else:
                chart = px.histogram(df, x=x_axis, color="cluster", title=f"Histogram of {x_axis}")
        elif n_features == 3:
            graph_types = ["3D Scatter", "Histogram"]
            graph_type = st.selectbox("Graph Type", graph_types, key="ml3d_graph")
            x_axis = st.selectbox("X-axis", feature_cols, key="ml3d_x")
            y_axis = st.selectbox("Y-axis", feature_cols, index=1, key="ml3d_y")
            z_axis = st.selectbox("Z-axis", feature_cols, index=2, key="ml3d_z")
            if graph_type == "3D Scatter":
                chart = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color="cluster", title="3D Clusters", color_continuous_scale="Viridis")
            else:
                chart = px.histogram(df, x=x_axis, color="cluster", title=f"Histogram of {x_axis}")
        else:
            graph_types = ["Pairplot", "Histogram"]
            graph_type = st.selectbox("Graph Type", graph_types, key="mlnd_graph")
            if graph_type == "Pairplot":
                chart = px.scatter_matrix(df, dimensions=feature_cols, color="cluster", title=f"{n_features}D Pairplot (Scatter Matrix)")
            else:
                x_axis = st.selectbox("X-axis", feature_cols, key="mlnd_x")
                chart = px.histogram(df, x=x_axis, color="cluster", title=f"Histogram of {x_axis}")

# --- Big Screen Graph Toggle ---
if "big_screen" not in st.session_state:
    st.session_state.big_screen = False

# --- Main Layout ---
if st.session_state.big_screen:
    st.markdown("<div style='text-align:right'>", unsafe_allow_html=True)
    if st.button("❌", help="Close big screen", key="close_big"):
        st.session_state.big_screen = False
    st.markdown("</div>", unsafe_allow_html=True)
    st.plotly_chart(chart, use_container_width=True)
    st.info("Click ❌ to return to normal view.")
else:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"🔍 Data Preview: {sector}")
        # Icon for big screen
        icon_col, chart_col = st.columns([0.08, 0.92])
        with icon_col:
            if st.button("⛶", help="Open graph in big screen", key="open_big"):
                st.session_state.big_screen = True
        with chart_col:
            st.plotly_chart(chart, use_container_width=True)
        st.dataframe(df.head(20), height=350)
        with st.expander("Show summary statistics"):
            st.write(df.describe(include='all'))
    with col2:
        st.subheader("⬇️ Download Data")
        st.download_button("Download as CSV", data=df.to_csv(index=False), file_name=f"synthetic_{sector.lower().replace('/', '_')}.csv")
        st.success("Data ready! Adjust options in the sidebar and download instantly.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>Made with ❤️ in MockData Lab for ML learning, testing, and demos. No real data required!</div>", unsafe_allow_html=True) 