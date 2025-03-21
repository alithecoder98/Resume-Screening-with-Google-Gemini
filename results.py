import streamlit as st
import pandas as pd

st.set_page_config(page_title="Resume Ranking", page_icon="🏆", layout="wide")

st.title("🏆 Resume Ranking & Match Results")

if "results" not in st.session_state or not st.session_state.results:
    st.warning("⚠️ No results found! Please upload resumes first.")
    st.stop()

df = pd.DataFrame(st.session_state.results)
st.dataframe(df, use_container_width=True)

st.download_button("📥 Download CSV", df.to_csv(index=False), "resume_rankings.csv", "text/csv")

if st.button("🔙 Go Back"):
    st.switch_page("app.py")
