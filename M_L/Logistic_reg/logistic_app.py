import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config("Logistic Regression", layout="centered")

# --------------------------------------------------
# Load CSS
def load_css(filename):
    css_path = os.path.join(os.path.dirname(__file__), filename)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --------------------------------------------------
# Title
st.markdown("""
<div class="card">
<h1> Logistic Regression </h1>
<p> Classify <b> Tip </b> as High or Low </p>
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

# Dataset Preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df[['total_bill', 'tip', 'tip_class']].head())
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Prepare Data
x = df[['total_bill']]
y = df['tip_class']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# --------------------------------------------------
# Train Model
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# --------------------------------------------------
# Metrics
acc = accuracy_score(y_test, y_pred)

# --------------------------------------------------
# Visualization
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Decision Boundary")

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip_class"], alpha=0.5)

x_vals = np.linspace(df.total_bill.min(), df.total_bill.max(), 100).reshape(-1, 1)
y_vals = model.predict(scaler.transform(x_vals))
ax.plot(x_vals, y_vals, color="red")

ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Class")
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")
st.metric("Accuracy", f"{acc:.2f}")
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Model Details
st.markdown(f"""
<div class="card">
<h3> Model Intercept & Co-efficient </h3>
<p>
<b>Co-efficient:</b> {model.coef_[0][0]:.3f}<br>
<b>Intercept:</b> {model.intercept_[0]:.3f}
</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Category")

bill = st.slider(
    "Total Bill $",
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
