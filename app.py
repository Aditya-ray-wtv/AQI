"""
=============================================================================
AQI INDEX INDIA — Interactive Dashboard (Streamlit)
=============================================================================
Topic: Air Quality Index (AQI) Analysis — India
Built using Streamlit (same approach as Dashboard.py reference)

Features:
    • Sidebar filters (City, Year, Pollutant selection)
    • Summary metrics (KPI cards)
    • Interactive charts (Bar, Line, Heatmap, Scatter, Pie)
    • Data table view with filtered data
=============================================================================
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌫️ AQI India Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Cleaned Dataset ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/cleaned/aqi_india_cleaned.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

df = load_data()

# ─── Dashboard Title ──────────────────────────────────────────────────────────
st.title("🌫️ Air Quality Index (AQI) Dashboard — India")
st.write("Interactive Dashboard for AQI Analysis across 10 Major Indian Cities")
st.markdown("---")

# ─── Sidebar Filters ──────────────────────────────────────────────────────────
# (Same sidebar approach as Dashboard.py)
st.sidebar.header("🔍 Filter Data")

# Select City
city_options = ["All"] + sorted(df['City'].unique().tolist())
selected_city = st.sidebar.selectbox("Select City", city_options)

# Select Year
year_options = ["All"] + sorted(df['Year'].unique().tolist())
selected_year = st.sidebar.selectbox("Select Year", year_options)

# Select Month
month_options = ["All"] + sorted(df['Month_Name'].unique().tolist(),
                                  key=lambda x: pd.to_datetime(x, format='%B').month)
selected_month = st.sidebar.selectbox("Select Month", month_options)

# Select Pollutant for detailed view
pollutant_options = ['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'O3']
selected_pollutant = st.sidebar.selectbox("Select Pollutant", pollutant_options)

# AQI Category Filter
aqi_categories = ["All"] + sorted(df['AQI_Category'].unique().tolist())
selected_category = st.sidebar.selectbox("Select AQI Category", aqi_categories)

# ─── Apply Filters ────────────────────────────────────────────────────────────
filtered_df = df.copy()

if selected_city != "All":
    filtered_df = filtered_df[filtered_df['City'] == selected_city]

if selected_year != "All":
    filtered_df = filtered_df[filtered_df['Year'] == selected_year]

if selected_month != "All":
    filtered_df = filtered_df[filtered_df['Month_Name'] == selected_month]

if selected_category != "All":
    filtered_df = filtered_df[filtered_df['AQI_Category'] == selected_category]

# ─── Summary Metrics (KPI Cards) ──────────────────────────────────────────────
# (Same metric approach as Dashboard.py — st.columns + st.metric)
st.subheader("📈 Summary Statistics")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "📊 Total Records",
    f"{len(filtered_df):,}"
)
col2.metric(
    "🏙️ Cities",
    filtered_df['City'].nunique()
)
col3.metric(
    "🌡️ Avg AQI",
    f"{filtered_df['AQI'].mean():.1f}" if len(filtered_df) > 0 else "N/A"
)
col4.metric(
    "⬆️ Max AQI",
    f"{filtered_df['AQI'].max():.0f}" if len(filtered_df) > 0 else "N/A"
)
col5.metric(
    "⬇️ Min AQI",
    f"{filtered_df['AQI'].min():.0f}" if len(filtered_df) > 0 else "N/A"
)

st.markdown("---")

# ─── Row 1: City-wise AQI Bar Chart + AQI Category Pie Chart ─────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("🏙️ City-wise Average AQI")
    city_aqi = filtered_df.groupby('City')['AQI'].mean().sort_values(ascending=False)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette('RdYlGn_r', len(city_aqi))
    bars = ax1.bar(city_aqi.index, city_aqi.values, color=colors)
    ax1.set_xlabel("City", fontsize=12)
    ax1.set_ylabel("Average AQI", fontsize=12)
    ax1.set_title("Average AQI by City (Higher = Worse)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, city_aqi.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig1)

with chart_col2:
    st.subheader("🎨 AQI Category Distribution")
    cat_counts = filtered_df['AQI_Category'].value_counts()
    cat_colors_map = {
        'Good': '#00b050', 'Satisfactory': '#92d050', 'Moderate': '#ffff00',
        'Poor': '#ff9900', 'Very Poor': '#ff0000', 'Severe': '#c00000', 'Unknown': '#808080'
    }
    pie_colors = [cat_colors_map.get(c, '#808080') for c in cat_counts.index]

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.pie(cat_counts.values, labels=cat_counts.index, colors=pie_colors,
            autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
    ax2.set_title("Distribution of AQI Categories", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("---")

# ─── Row 2: Monthly AQI Trend Line Chart ──────────────────────────────────────
st.subheader("📉 Monthly AQI Trend")

if selected_city != "All":
    # Show trend for selected city
    monthly_data = filtered_df.groupby(['Year', 'Month'])['AQI'].mean().reset_index()
    monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(Day=1))
    monthly_data = monthly_data.sort_values('Date')

    fig3, ax3 = plt.subplots(figsize=(14, 5))
    ax3.plot(monthly_data['Date'], monthly_data['AQI'], color='#e74c3c',
             linewidth=2, marker='o', markersize=4, label=selected_city)
    ax3.fill_between(monthly_data['Date'], monthly_data['AQI'], alpha=0.15, color='#e74c3c')
    ax3.set_xlabel("Date", fontsize=12)
    ax3.set_ylabel("Average AQI", fontsize=12)
    ax3.set_title(f"Monthly AQI Trend — {selected_city}", fontsize=14)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig3)
else:
    # Show trend for top 5 cities
    top_cities = df.groupby('City')['AQI'].mean().nlargest(5).index.tolist()
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    for city in top_cities:
        city_data = df[df['City'] == city].groupby(['Year', 'Month'])['AQI'].mean().reset_index()
        city_data['Date'] = pd.to_datetime(city_data[['Year', 'Month']].assign(Day=1))
        city_data = city_data.sort_values('Date')
        ax3.plot(city_data['Date'], city_data['AQI'], linewidth=1.5, marker='o',
                 markersize=3, label=city, alpha=0.8)
    ax3.set_xlabel("Date", fontsize=12)
    ax3.set_ylabel("Average AQI", fontsize=12)
    ax3.set_title("Monthly AQI Trend — Top 5 Polluted Cities", fontsize=14)
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig3)

st.markdown("---")

# ─── Row 3: Pollutant Analysis + Correlation Heatmap ─────────────────────────
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader(f"📊 {selected_pollutant} Levels by City")
    pollutant_by_city = filtered_df.groupby('City')[selected_pollutant].mean().sort_values(ascending=False)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette('YlOrRd', len(pollutant_by_city))
    ax4.barh(pollutant_by_city.index, pollutant_by_city.values, color=palette)
    ax4.set_xlabel(f"Average {selected_pollutant} (µg/m³)", fontsize=12)
    ax4.set_ylabel("City", fontsize=12)
    ax4.set_title(f"Average {selected_pollutant} Concentration by City", fontsize=14)
    ax4.invert_yaxis()
    for i, val in enumerate(pollutant_by_city.values):
        ax4.text(val + 0.5, i, f'{val:.1f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig4)

with chart_col4:
    st.subheader("🔥 Pollutant Correlation Heatmap")
    corr_cols = ['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'O3', 'AQI']
    corr_data = filtered_df[corr_cols].corr()

    fig5, ax5 = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt='.2f',
                linewidths=0.5, vmin=-1, vmax=1, ax=ax5)
    ax5.set_title("Correlation Heatmap — Pollutants & AQI", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig5)

st.markdown("---")

# ─── Row 4: Scatter Plot + Box Plot ───────────────────────────────────────────
chart_col5, chart_col6 = st.columns(2)

with chart_col5:
    st.subheader("🔬 PM2.5 vs PM10 Scatter Plot")
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    scatter = ax6.scatter(filtered_df['PM2.5'], filtered_df['PM10'],
                          c=filtered_df['AQI'], cmap='RdYlGn_r', alpha=0.5, s=10)
    plt.colorbar(scatter, ax=ax6, label='AQI')
    ax6.set_xlabel("PM2.5 (µg/m³)", fontsize=12)
    ax6.set_ylabel("PM10 (µg/m³)", fontsize=12)
    ax6.set_title("PM2.5 vs PM10 (colored by AQI)", fontsize=14)
    ax6.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig6)

with chart_col6:
    st.subheader(f"📦 {selected_pollutant} Distribution by City")
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    city_order = filtered_df.groupby('City')[selected_pollutant].median().sort_values(ascending=False).index
    sns.boxplot(data=filtered_df, x='City', y=selected_pollutant,
                order=city_order, palette='Set2', ax=ax7)
    ax7.set_xlabel("City", fontsize=12)
    ax7.set_ylabel(f"{selected_pollutant} (µg/m³)", fontsize=12)
    ax7.set_title(f"{selected_pollutant} Box Plot by City", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig7)

st.markdown("---")

# ─── Filtered Data Table ──────────────────────────────────────────────────────
# (Same approach as Dashboard.py — display filtered dataframe)
st.subheader("📋 Filtered Data")
display_cols = ['Timestamp', 'City', 'Location', 'PM2.5', 'PM10', 'NO2', 'NH3',
                'SO2', 'CO', 'O3', 'AQI', 'AQI_Category', 'Year', 'Month_Name']
st.dataframe(filtered_df[display_cols], use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 14px;'>
        📊 AQI India Dashboard | Data Analysis and Visualization Using Python (CUTM1018)<br>
        Dataset: India AQI Data (10 Cities) | Built with Streamlit + Matplotlib + Seaborn
    </div>
    """,
    unsafe_allow_html=True
)
