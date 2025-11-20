# 📘 Machine Learning Notes: Clustering, Scaling & Skewness

This document summarizes all the concepts you’ve learned so far — clustering fundamentals, scaling, skewness, and how to handle skewed data — in a clean and structured Markdown format.

---

# #1️⃣ Why Scaling Matters in Machine Learning

Many ML algorithms rely on **distance**, **dot products**, or **gradient descent**. If features have different scales (e.g., Income = 100000, SpendingScore = 70), then large-scale features dominate.

Scaling ensures **all features contribute equally**.

---

# #2️⃣ Types of Scalers

We learned three major scalers in detail:

## ## 2.1 StandardScaler (Z‑Score Normalization)

### **Formula**

[
z = \frac{x - \mu}{\sigma}
]
Where:

* **μ** = mean of feature
* **σ** = standard deviation

### **Intuition**

Centers data at 0 and spreads it using standard deviation.

### **Best for:**

* KMeans, PCA
* Linear/Logistic Regression
* SVM
* Data roughly normal without extreme outliers

---

## ## 2.2 MinMaxScaler (Normalization)

### **Formula**

[
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
]
Maps data into the **0 to 1** range.

### **Best for:**

* Neural networks
* KNN
* Gradient‑descent methods
* When **no outliers** are present

### **Avoid when:**

* Heavy outliers exist (they compress the entire feature)

---

## ## 2.3 RobustScaler

### **Formula**

[
x' = \frac{x - \text{median}}{IQR}
]
Where:

* **IQR = Q3 - Q1**

### **Intuition**

Median + IQR are stable against outliers.

### **Best for:**

* Outlier‑heavy data
* KMeans, PCA, KNN

---

# #3️⃣ Understanding Skewness

Skewness measures **asymmetry** in a distribution.

## ## 3.1 Types of Skewness

### **Right‑Skewed (Positive Skew)**

* Tail on **right** side
* Many small values, few large ones
* Example → income, sales amounts

### **Left‑Skewed (Negative Skew)**

* Tail on **left** side
* Many large values, few small ones
* Example → exam scores in easy tests

### **Normal Distribution**

* Symmetrical
* Mean ≈ Median ≈ Mode
* Skewness ≈ 0

---

# #4️⃣ How to Check Skewness

### **1. Visual methods:**

* Histogram
* KDE plot
* Boxplot

### **2. Mathematical method:**

Use skewness coefficient:
[
\text{Skewness} = \frac{1}{n} \sum \left( \frac{x - \mu}{\sigma} \right)^3
]

### **Interpretation:**

| Skewness value | Interpretation      |
| -------------- | ------------------- |
| 0              | Perfectly symmetric |
| 0.1 to 0.5     | Slightly skewed     |
| 0.5 to 1       | Moderately skewed   |
| > 1            | Highly skewed       |

---

# #5️⃣ How to Handle Highly Skewed Features

If a feature is extremely skewed, you can transform it.

## ## 5.1 Log Transform

Good for **right‑skewed**, positive data.

[
x' = \ln(x + \epsilon)
]

## ## 5.2 Square‑root Transform

Mild skew.

---

## ## 5.3 Box‑Cox Transform

Only for **positive data**.

[
x' = \frac{x^\lambda - 1}{\lambda}
]

---

## ## 5.4 Yeo‑Johnson Transform

Works for **both positive and negative values**.

Automatically chooses the best λ.

---

## ## 5.5 When to Transform Skewed Data

Transform when:

* Feature influences model by magnitude
* Using KMeans, PCA, linear models
* Feature distribution is extremely asymmetric
* There are extreme right‑tail outliers

Do **not** transform when:

* Using tree‑based models (RF, XGBoost)
* The skew carries meaningful business signal

---

# #6️⃣ Visual Example of Skewness Fixing

We plotted:

* **Original skewed data** → exponential distribution
* **Log transformed** → closer to normal
* **Yeo‑Johnson transformed** → nearly symmetrical

(See the plots in the main chat.)

---

# #7️⃣ Summary Table

| Problem                 | Best Fix                  |
| ----------------------- | ------------------------- |
| Highly right‑skewed     | Log, Box‑Cox, Yeo‑Johnson |
| Negative values present | Yeo‑Johnson               |
| Heavy outliers          | RobustScaler              |
| Data must be 0 to 1     | MinMaxScaler              |
| Normal distributed data | StandardScaler            |

---

# ✔ What’s Next?

Possible next learning steps:

* PCA + scaling
* Clustering (KMeans) with scaling
* Using transformations in pipelines
* Business use cases for clustering

Let me know what you want to do next!

# 📘 Machine Learning Notes — Scaling, Skewness & Normality

A clean and practical Markdown reference for everything we've learned: **scalers, their math, skewness, transformations, and visuals**. This is structured for practical ML + business use-case clarity.

---

# ## 1. Why Scaling Matters in Machine Learning

Many ML models depend on **distance** or **gradient-based optimization**. If features are not on similar scales:

* Larger-valued features dominate
* Distance-based models behave incorrectly
* Gradient descent becomes unstable
* Clustering picks wrong centroids

### 🔹 Algorithms *affected* by scaling

* K-Means, KNN (distance-based)
* Logistic/Linear Regression (gradient-based)
* Neural Networks
* PCA (variance-based)

### 🔹 Algorithms *less affected*

* Tree-based models (Random Forest, XGBoost)

Scaling is essential **before clustering**.

---

# ## 2. Types of Scalers & Their Math

Below are the most important scalers with formulas + use-cases.

---

# ### **2.1 StandardScaler (Z-score Normalization)**

### Formula

```
z = (x - μ) / σ
```

Where:

* **μ** = mean of the feature
* **σ** = standard deviation

### When to use

* Data is **roughly normal** (bell-shaped)
* No extreme outliers
* Needed for PCA, KMeans

### Example

If a feature has:

* Mean = 50
* Std = 10
* Value = 70

Then

```
z = (70 - 50) / 10 = 2
```

---

# ### **2.2 MinMaxScaler (Normalization 0–1)**

### Formula

```
x_scaled = (x - x_min) / (x_max - x_min)
```

### When to use

* Neural networks (0–1 improves gradient flow)
* When the **distribution is NOT normal**
* No major outliers

### Danger

* Extremely sensitive to outliers

---

# ### **2.3 RobustScaler (Outlier-Resistant Scaling)**

### Formula (uses IQR)

```
x_scaled = (x - median) / IQR
```

Where:

```
IQR = Q3 - Q1
```

### When to use

* Feature has **outliers**
* Data is **not normal**
* Works well before clustering

### Business use-case

* Customer income distribution (usually skewed)
* Delivery times with delays

---

# ## 3. Understanding Skewness

Skewness tells us how **symmetrical** a distribution is.

### 📌 Formula (Fisher’s skewness)

```
Skew = (1/n) Σ ((x - μ)/σ)^3
```

### Interpretation

| Skewness  | Meaning              |
| --------- | -------------------- |
| 0         | Perfectly symmetric  |
| 0.0 – 0.5 | Approximately normal |
| 0.5 – 1.0 | Moderately skewed    |
| > 1.0     | Highly skewed        |
| < -1.0    | Highly left-skewed   |

### How to check skewness (Python)

```python
import pandas as pd
x.skew()
```

### Visual check

* Histogram
* KDE plot
* Q–Q plot

---

# ## 4. When Can We Say a Distribution Is "Roughly Normal"?

A feature is **approximately normal** if:

### ✔ Histogram looks bell-shaped

### ✔ Q–Q plot follows the diagonal

### ✔ Skewness is between -0.5 and +0.5

### ✔ No extreme outliers

### ✔ Mean ≈ Median ≈ Mode

This matters because StandardScaler assumes roughly normal shape.

---

# ## 5. How to Handle Highly Skewed Data

If a feature has **skewness > 1** (e.g., income, sales, time-to-deliver), use **transformation**.

### Methods to fix skewness

1. **Log Transform** (for right-skew)

```
x_new = log(x + c)
```

2. **Square root transform**
3. **Box-Cox Transform** (requires positive values)
4. **Yeo-Johnson Transform** (works with negative & zero)
5. **Clipping outliers**
6. **Winsorizing (capping)**
7. **Binning** (convert continuous → categories)

---

# ## 6. Visual Demonstration (Log & Yeo-Johnson)

Images generated earlier show:

* Original highly skewed data → long right tail
* Log transform → reduces skewness
* Yeo–Johnson → even better normalization

These are extremely useful **before KMeans or distance-based ML**.

---

# ## 7. Business Examples of Skewed Data

| Feature             | Reason of Skew     | Fix                        |
| ------------------- | ------------------ | -------------------------- |
| Customer income     | Few rich customers | Log transform              |
| Order delivery time | Rare delays        | RobustScaler + Yeo-Johnson |
| Product sales       | Few best-sellers   | Log + MinMax               |
| Hospital wait times | Rare long waits    | Robust + YJ                |

---

# ## 8. Quick Decision Guide

### ✔ If data is normal → **StandardScaler**

### ✔ If data is skewed → **Log / BoxCox / Yeo-Johnson**

### ✔ If data has outliers → **RobustScaler**

### ✔ If neural network → **MinMaxScaler**

### ✔ If clustering → **Standard or Robust** depending on outliers

---

Let me know when you're ready — I can add:
✅ PCA notes
✅ K-Means math (centroid updates & inertia)
✅ Full clustering tutorial in the same markdown document
