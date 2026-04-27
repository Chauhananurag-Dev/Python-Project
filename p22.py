
# ==========================================
# WATER POTABILITY ANALYSIS (FIXED VERSION)
# ==========================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 2. LOAD DATASET
data = pd.read_csv(r"C:\Users\HP\Downloads\water_potability.csv")

# CHECK COLUMN NAMES
print("Columns:", data.columns)

# 3. DATA CLEANING
print("\nMissing Values:\n", data.isnull().sum())

# Fill missing values
data = data.fillna(data.mean(numeric_only=True))

# 4. DATA INFO & STATS
print("\nDataset Info:")
data.info()

print("\nStatistical Summary:")
print(data.describe())

# ==========================================
# OBJECTIVE 1: HISTOGRAMS
# ==========================================

data.hist(figsize=(12,10))
plt.suptitle("Histograms of All Features")
plt.show()

# ==========================================
# OBJECTIVE 2: SCATTER PLOT (ph vs Hardness)
# ==========================================

plt.scatter(data['ph'], data['Hardness'], color='blue')
plt.xlabel("pH")
plt.ylabel("Hardness")
plt.title("pH vs Hardness")
plt.show()

# ==========================================
# OBJECTIVE 3: LINEAR REGRESSION
# ==========================================

X = data[['ph']]
y = data['Potability']

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

print("\nMSE:", mean_squared_error(y, pred))

plt.scatter(X, y, color='green')
plt.plot(X, pred, color='red')
plt.xlabel("pH")
plt.ylabel("Potability")
plt.title("Linear Regression: pH vs Potability")
plt.show()

# ==========================================
# OBJECTIVE 4: TREND GRAPH
# ==========================================

data['Potability'].rolling(50).mean().plot()
plt.title("Rolling Mean of Potability")
plt.xlabel("Index")
plt.ylabel("Trend")
plt.grid()
plt.show()

# ==========================================
# OBJECTIVE 5: BOXPLOT
# ==========================================

plt.figure(figsize=(12,6))
sns.boxplot(data=data)
plt.xticks(rotation=90)
plt.title("Boxplot for Outliers")
plt.show()

# ==========================================
# OBJECTIVE 6: HEATMAP
# ==========================================

corr = data.corr(numeric_only=True)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ==========================================
# OBJECTIVE 7: CLASSIFICATION
# ==========================================

def water_quality(ph):
    if ph < 6.5:
        return "Acidic"
    elif ph <= 8.5:
        return "Normal"
    else:
        return "Alkaline"

data['Water_Quality'] = data['ph'].apply(water_quality)

print("\nWater Quality Count:")
print(data['Water_Quality'].value_counts())

data['Water_Quality'].value_counts().plot(kind='bar', color='orange')
plt.title("Water Quality Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# ==========================================
# BONUS: POTABILITY DISTRIBUTION
# ==========================================

sns.countplot(x='Potability', data=data)
plt.title("Potable vs Non-Potable Water")
plt.show()

# ==========================================
# END
# ==========================================
