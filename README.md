# 🌫️ Air Quality Index (AQI) Analysis — India

## Data Analysis and Visualization Using Python (CUTM1018)

### 📋 Problem Statement
Analyze PM2.5, PM10, NO2, NH3, SO2, CO, and O3 levels across 10 major Indian cities to identify air quality trends, seasonal patterns, and city-wise pollution comparisons. Build an interactive dashboard to visualize AQI insights.

### 🏙️ Cities Covered
Bengaluru, Chennai, Delhi, Gwalior, Hyderabad, Jaipur, Kolkata, Lucknow, Mumbai, Visakhapatnam

### 🗂️ Project Structure
```
Python DAVP/
├── archive/                    # Raw dataset (10 city CSV files)
├── data/
│   └── cleaned/
│       └── aqi_india_cleaned.csv   # Cleaned & processed dataset
├── reports/                    # Exported chart images (PNG, 150 DPI)
├── data_cleaning.py            # Data cleaning script
├── app.py                      # Streamlit Dashboard
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### 🛠️ Tools & Libraries
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Streamlit
- SciPy (for Z-score outlier detection)

### ⚙️ Installation & Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Run data cleaning
python data_cleaning.py

# Step 2: Launch dashboard
streamlit run app.py
```

### 📊 Data Cleaning Steps
1. Load all 10 city CSV files and combine
2. Replace 'NA' strings with NaN
3. Handle missing values (median for numeric, mode for categorical, ffill for dates)
4. Fix inconsistent formats (strip, title case, datetime parsing)
5. Remove duplicates
6. Correct data types
7. Calculate AQI using Indian CPCB breakpoints
8. Outlier detection & removal using Z-scores

### 📈 Dashboard Features
- Sidebar filters: City, Year, Month, Pollutant, AQI Category
- KPI cards: Total records, cities, avg/max/min AQI
- City-wise average AQI bar chart
- AQI category pie chart
- Monthly AQI trend line chart
- Pollutant concentration comparison
- Correlation heatmap
- PM2.5 vs PM10 scatter plot
- Box plot distribution by city
- Filtered data table

### 📌 CO Mapping
- CO3: Exploratory Data Analysis
- CO4: Visualization
- CO5: Deployment

### 📚 Dataset Source
- [Kaggle India AQI Dataset](https://www.kaggle.com/datasets)
- [data.gov.in](https://data.gov.in)
