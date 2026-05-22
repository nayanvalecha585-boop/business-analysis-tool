import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

st.set_page_config(
    page_title="Business Analysis Tool",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Automated Business Analysis Tool")
st.markdown("Upload your business CSV file and get instant AI-powered insights.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head(10))

    st.subheader("📐 Dataset Info")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("🔍 Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.dataframe(missing.rename("Missing Count"))
    else:
        st.success("✅ No missing values found!")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    st.subheader("📈 Visual Analysis")
    
    # Ensure we have enough numeric columns to plot
    if len(numeric_cols) >= 2:
        
        # 1. SCATTER PLOT SECTION
        with st.expander("🔗 1. Scatter Plot (Relationship Analysis)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                scatter_x = st.selectbox("Select X axis (Scatter)", numeric_cols, key="scatter_x")
            with col2:
                scatter_y = st.selectbox("Select Y axis (Scatter)", numeric_cols, key="scatter_y")
            
            fig1 = px.scatter(df, x=scatter_x, y=scatter_y, title=f"Scatter Plot: {scatter_x} vs {scatter_y}")
            st.plotly_chart(fig1, use_container_width=True)
            
        # 2. HISTOGRAM SECTION
        with st.expander("📊 2. Histogram (Distribution Analysis)", expanded=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                hist_x = st.selectbox("Select Column to view distribution", numeric_cols, key="hist_x")
            
            fig2 = px.histogram(df, x=hist_x, title=f"Distribution of {hist_x}")
            st.plotly_chart(fig2, use_container_width=True)
            
       # 3. BOX PLOT SECTION
        with st.expander("📦 3. Box Plot (Spread & Outlier Analysis)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                box_y = st.selectbox("Select numeric column", numeric_cols, key="box_y")
            with col2:
                cat_cols = df.select_dtypes(include='object').columns.tolist()
                box_color = st.selectbox("Group by category (optional)", ["None"] + cat_cols, key="box_color")
            
            if box_color == "None":
                fig3 = px.box(df, y=box_y, title=f"Box Plot of {box_y}")
            else:
                fig3 = px.box(df, x=box_color, y=box_y, title=f"{box_y} by {box_color}")
            
            st.plotly_chart(fig3, use_container_width=True)
            
    else:
        st.warning("⚠️ Need at least 2 numeric columns in the dataset to generate visual charts.")

    # AI Section Header
    st.subheader("🤖 AI-Generated Business Insights")
    
    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing your dataset and finding trends..."):
            summary_stats = df.describe().to_string()
            columns_info = str(df.dtypes)
            sample_data = df.head(5).to_string()
            
            prompt = f"""
            You are a senior business analyst. Analyze this dataset and provide:
            1. Key business insights (3-5 high-impact bullet points)
            2. Potential risks, anomalies, or outliers
            3. Actionable strategic recommendations
            
            Dataset columns and data types:
            {columns_info}
            
            Statistical summary:
            {summary_stats}
            
            Sample data rows:
            {sample_data}
            
            Write your response in clear, professional business language with bold key terms.
            """
            
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                st.markdown(response.text)
            except Exception as e:
                st.error(f"❌ Gemini API Error: {e}")