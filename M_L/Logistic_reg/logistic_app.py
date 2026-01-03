import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------------------------------
# Page Config
st.set_page_config("Logistic Regression", layout="centered")

# --------------------------------------------------
# Load CSS (optional)
def load_css(filename):
    css_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --------------------------------------------------
# Title
st.markdown("""
<div class="card">
<h1> Logistic Regression </h1>
<p> Classify <b>Tip</b> as High or Low </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# Binary Target
df['tip_class'] = (df['tip'] >= 3).astype(int)

# --------------------------------------------------
# Dataset Preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df[['total_bill', 'tip', 'tip_class']].head())
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Prepare Data
X = df[['total_bill']]
y = df['tip_class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# Train Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# --------------------------------------------------
# Metrics
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# --------------------------------------------------
# HEATMAP Visualization
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Decision Boundary Heatmap")

# Grid along total_bill
x_vals = np.linspace(
    df["total_bill"].min(),
    df["total_bill"].max(),
    400
).reshape(-1, 1)

# Predict probability
x_scaled = scaler.transform(x_vals)
probs = model.predict_proba(x_scaled)[:, 1]

# Create heatmap matrix (repeat vertically for thickness)
heatmap = np.tile(probs, (50, 1))

fig, ax = plt.subplots()

im = ax.imshow(
    heatmap,
    extent=[
        df["total_bill"].min(),
        df["total_bill"].max(),
        0,
        1
    ],
    aspect="auto",
    origin="lower"
)

# Scatter actual data
ax.scatter(
    df["total_bill"],
    df["tip_class"],
    color="black",
    alpha=0.6,
    s=15
)

ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Class / Probability")

plt.colorbar(im, ax=ax, label="Probability of High Tip")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")
st.metric("Accuracy", f"{acc:.2f}")
st.write("Confusion Matrix:")
st.write(cm)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Model Details
st.markdown(f"""
<div class="card">
<h3> Model Parameters </h3>
<p>
<b>Coefficient:</b> {model.coef_[0][0]:.3f}<br>
<b>Intercept:</b> {model.intercept_[0]:.3f}
</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Category")

bill = st.slider(
    "Total Bill ($)",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

pred = model.predict(scaler.transform([[bill]]))[0]
label = "High Tip" if pred == 1 else "Low Tip"

st.markdown(
    f'<div class="predict-box">{label}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
