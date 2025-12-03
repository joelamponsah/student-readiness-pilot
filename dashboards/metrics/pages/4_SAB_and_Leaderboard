import streamlit as st
import pandas as pd
from utils.metrics import compute_SAB

st.title("🏅 SAB Index & Leaderboard")

df = pd.read_csv("data/verify_df_fixed.csv")
sab = compute_SAB(df)

sab["robust_SAB_scaled"] = 100 * (
    sab["robust_SAB_index"] - sab["robust_SAB_index"].min()
) / (
    sab["robust_SAB_index"].max() - sab["robust_SAB_index"].min()
)

st.subheader("Leaderboard (Top 20)")
st.dataframe(sab.sort_values("robust_SAB_scaled", ascending=False).head(20))
