import streamlit as st

def show_project():
    with st.container():
        cols = st.columns([3, 1])
        with cols[0]:
            st.subheader("Regression Model Comparison – House Price Prediction")
        with cols[1]:
            st.caption("*4/2024*")

        st.markdown(
            "- Performed **Exploratory Data Analysis (EDA)** on Housing dataset (79 features).\n"
            "- Conducted **feature engineering**: handling missing values, encoding categorical variables, scaling numerical features.\n"
            "- Implemented multiple regression models (**Linear Regression, Ridge Regression, Lasso Regression, Elastic Regression**).\n"
            "- Evaluated performance using **cross-validation** and metrics such as RMSE and R² and achieved an errors approximately 0.13 on Kaggle leaderborad.\n"
        )

        col1, col2 = st.columns([3, 7])
        with col1:
            col11, col12 = st.columns(2)
            with col11:
                st.link_button("GitHub", "https://github.com/honggquan24/regression_comparison")
            # with col12:
            #     st.link_button("Demo", "https://your-streamlit-demo-url")
