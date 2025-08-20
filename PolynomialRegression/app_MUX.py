import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="ML 4x1 MUX via Polynomial Regression", layout="wide")
st.title("ML replication of a 4×1 Multiplexer with Polynomial Regression")
st.markdown(
    """
This app explores whether supervised learning can replicate a digital **4×1 MUX**.
We synthesize continuous input lines, construct selection bits, train a polynomial regression model,
inspect the learned equation, and test generalization on arbitrary inputs.
"""
)

# ---------------------------
# Sidebar: controls
# ---------------------------
st.sidebar.header("Configuration")

n_samples = st.sidebar.number_input("Samples", min_value=4000, max_value=200000, value=30000, step=1000)
test_size = st.sidebar.slider("Test size (fraction)", 0.1, 0.9, 0.5, 0.05)
degree = st.sidebar.selectbox("Polynomial degree", [1, 2, 3], index=2)
interaction_only = st.sidebar.checkbox("Interaction only", value=True)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

st.sidebar.caption(
    "Tip: Degree 3 with interaction-only terms typically learns the exact symbolic structure."
)

# ---------------------------
# Data generation
# ---------------------------
def make_mux_dataframe(n_samples: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    # Four continuous input lines
    line_vals = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 4))
    df = pd.DataFrame(line_vals, columns=["Line_0", "Line_1", "Line_2", "Line_3"])

    # Selection bits arranged in four equal blocks over the dataset:
    # 00 -> choose Line_0, 01 -> Line_1, 10 -> Line_2, 11 -> Line_3
    quarter = n_samples // 4
    s1 = np.zeros(n_samples, dtype=int)
    s0 = np.zeros(n_samples, dtype=int)
    # [0:Q) -> 00, [Q:2Q) -> 01, [2Q:3Q) -> 10, [3Q:4Q) -> 11; leftover rows cycle
    for i in range(n_samples):
        bucket = (i // quarter) % 4 if quarter > 0 else 0
        if bucket == 0:
            s1[i], s0[i] = 0, 0
        elif bucket == 1:
            s1[i], s0[i] = 0, 1
        elif bucket == 2:
            s1[i], s0[i] = 1, 0
        else:
            s1[i], s0[i] = 1, 1

    df["Selection_Stream_1"] = s1
    df["Selection_Stream_0"] = s0

    # Truth MUX output
    out = np.empty(n_samples, dtype=float)
    mask00 = (s1 == 0) & (s0 == 0)
    mask01 = (s1 == 0) & (s0 == 1)
    mask10 = (s1 == 1) & (s0 == 0)
    mask11 = (s1 == 1) & (s0 == 1)
    out[mask00] = df.loc[mask00, "Line_0"]
    out[mask01] = df.loc[mask01, "Line_1"]
    out[mask10] = df.loc[mask10, "Line_2"]
    out[mask11] = df.loc[mask11, "Line_3"]
    df["OutPut"] = out
    return df

df = make_mux_dataframe(n_samples, random_state)

with st.expander("Preview synthesized dataset", expanded=False):
    st.dataframe(df.head(12), use_container_width=True)

# ---------------------------
# Train model
# ---------------------------
X_cols = ["Line_0", "Line_1", "Line_2", "Line_3", "Selection_Stream_1", "Selection_Stream_0"]
X = df[X_cols].values
y = df["OutPut"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, shuffle=True
)

poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)

st.subheader("Model performance")
m1, m2 = st.columns(2)
m1.metric("Mean Squared Error", f"{mse:.3e}")
m2.metric("R² on test", f"{model.score(X_test_poly, y_test):.6f}")

# ---------------------------
# Show learned equation (coefficients with feature names)
# ---------------------------
st.subheader("Learned equation (non-zero / salient coefficients)")

feature_names = poly.get_feature_names_out(X_cols)
coefs = model.coef_
intercept = model.intercept_

coef_df = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coeff": np.abs(coefs),
        "rounded": np.round(coefs),
    }
).sort_values("abs_coeff", ascending=False)

# Show top terms and those close to ±1
threshold_small = 1e-6
near_one_mask = np.isclose(coefs, 1.0, atol=1e-8) | np.isclose(coefs, -1.0, atol=1e-8)
salient = pd.concat(
    [
        coef_df.query("abs_coeff > @threshold_small").head(20),
        coef_df[near_one_mask],
    ]
).drop_duplicates(subset=["feature"]).reset_index(drop=True)

st.write(f"Intercept: {intercept:.3e}")
st.dataframe(salient[["feature", "coefficient"]], use_container_width=True)

with st.expander("Show all coefficients", expanded=False):
    st.dataframe(coef_df[["feature", "coefficient"]], use_container_width=True)

# ---------------------------
# Interactive test bench
# ---------------------------
st.subheader("Interactive test of generalization")

c1, c2, c3 = st.columns(3)
with c1:
    l0 = st.number_input("Line_0", value=69.0, step=1.0, format="%.6f")
    l1 = st.number_input("Line_1", value=100.0, step=1.0, format="%.6f")
with c2:
    l2 = st.number_input("Line_2", value=9.0, step=1.0, format="%.6f")
    l3 = st.number_input("Line_3", value=500.0, step=1.0, format="%.6f")
with c3:
    s1 = st.selectbox("Selection_Stream_1", [0, 1], index=0)
    s0 = st.selectbox("Selection_Stream_0", [0, 1], index=0)

test_point = np.array([[l0, l1, l2, l3, int(s1), int(s0)]], dtype=float)
test_point_poly = poly.transform(test_point)
pred = model.predict(test_point_poly)[0]

# Theoretical MUX output for comparison
truth = (
    (1 - s1) * (1 - s0) * l0
    + (1 - s1) * s0 * l1
    + s1 * (1 - s0) * l2
    + s1 * s0 * l3
)

cA, cB, cC = st.columns(3)
cA.metric("Predicted output", f"{pred:.6f}")
cB.metric("MUX truth output", f"{truth:.6f}")
cC.metric("Absolute error", f"{abs(pred - truth):.3e}")

# Quick sanity table over all four select combinations for the same lines
st.markdown("Quick check for all select combinations with your current line values")
grid = []
for S1 in [0, 1]:
    for S0 in [0, 1]:
        pt = np.array([[l0, l1, l2, l3, S1, S0]], dtype=float)
        yhat = model.predict(poly.transform(pt))[0]
        ytruth = (1 - S1) * (1 - S0) * l0 + (1 - S1) * S0 * l1 + S1 * (1 - S0) * l2 + S1 * S0 * l3
        grid.append({"S1": S1, "S0": S0, "Pred": yhat, "Truth": ytruth, "AbsErr": abs(yhat - ytruth)})
grid_df = pd.DataFrame(grid)
st.dataframe(grid_df, use_container_width=True, hide_index=True)

# ---------------------------
# Notes
# ---------------------------
with st.expander("Notes and interpretation", expanded=False):
    st.markdown(
        f"""
- With degree={degree} and interaction_only={interaction_only}, the model typically recovers a symbolic equation
  equivalent to the MUX boolean-structured expression:
  
  y = (1−S1)(1−S0)·Line_0 + (1−S1)S0·Line_1 + S1(1−S0)·Line_2 + S1·S0·Line_3

- Very small MSE indicates the learned linear model over polynomial features matches the exact MUX behavior.
- Changing degree or interaction settings will affect which terms the model can express.
"""
    )
