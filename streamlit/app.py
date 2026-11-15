# ==========================================================
# EV CHARGING DEMAND PREDICTION AND USAGE PATTERN ANALYSIS
# STREAMLIT APPLICATION
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# ----------------------------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------------------------

st.set_page_config(
    page_title="EV Charging Analytics",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ EV Charging Demand Prediction")
st.markdown(
    "### Usage Pattern Analysis and Machine Learning Dashboard"
)

st.markdown("---")

# ----------------------------------------------------------
# 2. PROJECT PATHS
# ----------------------------------------------------------

from pathlib import Path

# app.py is inside the streamlit folder.
# Its parent folder is the main EV_Charging_Demand_Project folder.
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

# Use absolute paths so Streamlit can find files regardless of launch location.
BASE_DIR = PROJECT_DIR
DATA_PATH = BASE_DIR / "data" / "charging_records_featured.csv"
MODEL_PATH = BASE_DIR / "models"
EXPLAIN_PATH = BASE_DIR / "explainability"

print("Project folder:", BASE_DIR)
print("Dataset path:", DATA_PATH)
print("Dataset exists:", DATA_PATH.exists())

FEATURES = [
    "ChargerCompany",
    "Location",
    "ChargerType",
    "Hour",
    "DayOfWeek",
    "Month",
    "IsWeekend"
]

# ----------------------------------------------------------
# 3. LOAD DATASET
# ----------------------------------------------------------

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    data["StartDatetime"] = pd.to_datetime(
        data["StartDatetime"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["StartDatetime", "Demand"]
    )

    return data


# Load dataset
if DATA_PATH.exists():

    df = load_data()

    st.success("Dataset loaded successfully!")

    st.write("Dataset shape:", df.shape)

else:

    st.error(f"Dataset not found: {DATA_PATH}")

    st.stop()


# ----------------------------------------------------------
# 4. LOAD MACHINE LEARNING MODELS
# ----------------------------------------------------------

@st.cache_resource
def load_model(path):

    return joblib.load(path)


# Load regression model and model comparison CSV files

REGRESSION_MODEL_PATH = MODEL_PATH / "regression_pipeline.joblib"

if REGRESSION_MODEL_PATH.exists():
    regression_model = load_model(REGRESSION_MODEL_PATH)
    st.success("Regression model loaded successfully!")
else:
    regression_model = None
    st.warning(
        f"Regression model not found: {REGRESSION_MODEL_PATH}"
    )

# Load classification model comparison results.
# These results are used by the Classification Prediction and Model Performance pages.
CLASSIFICATION_RESULTS_PATH = (
    MODEL_PATH / "classification_model_comparison.csv"
)

if CLASSIFICATION_RESULTS_PATH.exists():
    classification_results = pd.read_csv(CLASSIFICATION_RESULTS_PATH)
    st.success("Classification model comparison results loaded successfully!")
else:
    classification_results = pd.DataFrame()
    st.warning(
        f"Classification comparison CSV not found: "
        f"{CLASSIFICATION_RESULTS_PATH}"
    )

# Load regression model comparison results for the Model Performance page.
REGRESSION_RESULTS_PATH = (
    MODEL_PATH / "regression_model_comparison.csv"
)

if REGRESSION_RESULTS_PATH.exists():
    regression_results = pd.read_csv(REGRESSION_RESULTS_PATH)
else:
    regression_results = pd.DataFrame()
    st.info(
        f"Regression comparison CSV not found: {REGRESSION_RESULTS_PATH}"
    )


# Load classification threshold

try:

    demand_threshold = load_model(
        MODEL_PATH / "demand_threshold.joblib"
    )

except Exception:

    demand_threshold = None


# ----------------------------------------------------------
# 5. SIDEBAR FILTERS AND NAVIGATION
# ----------------------------------------------------------

st.sidebar.title("⚡ EV Charging Dashboard")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "EV Charging Overview",
        "Usage Pattern Analysis",
        "Regression Prediction",
        "Classification Prediction",
        "Model Performance",
        "Model Explainability"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Dashboard Filters")

min_date = df["StartDatetime"].min().date()

max_date = df["StartDatetime"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

locations = sorted(
    df["Location"].dropna().unique().tolist(),
    key=str
)

companies = sorted(
    df["ChargerCompany"].dropna().unique().tolist(),
    key=str
)

charger_types = sorted(
    df["ChargerType"].dropna().unique().tolist(),
    key=str
)

selected_locations = st.sidebar.multiselect(
    "Select Location",
    locations,
    default=locations
)

selected_companies = st.sidebar.multiselect(
    "Select Charger Company",
    companies,
    default=companies
)

selected_charger_types = st.sidebar.multiselect(
    "Select Charger Type",
    charger_types,
    default=charger_types
)


# ----------------------------------------------------------
# 6. APPLY FILTERS
# ----------------------------------------------------------

filtered_df = df.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

    start_date, end_date = date_range

    filtered_df = filtered_df[
        (filtered_df["StartDatetime"] >= pd.Timestamp(start_date))
        &
        (filtered_df["StartDatetime"] < (
            pd.Timestamp(end_date) + pd.Timedelta(days=1)
        ))
    ]


filtered_df = filtered_df[
    filtered_df["Location"].isin(selected_locations)
    &
    filtered_df["ChargerCompany"].isin(selected_companies)
    &
    filtered_df["ChargerType"].isin(selected_charger_types)
]


# ----------------------------------------------------------
# PAGE 1 - EV CHARGING OVERVIEW
# ----------------------------------------------------------

if page == "EV Charging Overview":

    st.header("EV Charging Overview")

    if filtered_df.empty:

        st.warning("No records available for the selected filters.")

    else:

        # KPI calculations

        total_sessions = len(filtered_df)

        total_demand = filtered_df["Demand"].sum()

        average_demand = filtered_df["Demand"].mean()

        total_locations = filtered_df["Location"].nunique()

        # KPI cards

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Charging Sessions",
            f"{total_sessions:,}"
        )

        col2.metric(
            "Total Demand",
            f"{total_demand:,.2f}"
        )

        col3.metric(
            "Average Demand",
            f"{average_demand:,.2f}"
        )

        col4.metric(
            "Total Locations",
            total_locations
        )

        st.markdown("---")

        # Demand by location

        col1, col2 = st.columns(2)

        with col1:

            location_demand = (
                filtered_df
                .groupby("Location")["Demand"]
                .sum()
                .reset_index()
            )

            fig = px.bar(
                location_demand,
                x="Location",
                y="Demand",
                title="Charging Demand by Location"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            charger_demand = (
                filtered_df
                .groupby("ChargerType")["Demand"]
                .sum()
                .reset_index()
            )

            fig = px.bar(
                charger_demand,
                x="ChargerType",
                y="Demand",
                title="Charging Demand by Charger Type"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # Daily demand trend

        daily_demand = (
            filtered_df
            .groupby(
                filtered_df["StartDatetime"].dt.date
            )["Demand"]
            .sum()
            .reset_index()
        )

        daily_demand.columns = [
            "Date",
            "Demand"
        ]

        fig = px.line(
            daily_demand,
            x="Date",
            y="Demand",
            title="Daily Charging Demand Trend"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ----------------------------------------------------------
# PAGE 2 - USAGE PATTERN ANALYSIS
# ----------------------------------------------------------

elif page == "Usage Pattern Analysis":

    st.header("EV Charging Usage Pattern Analysis")

    if filtered_df.empty:

        st.warning("No records available for the selected filters.")

    else:

        col1, col2 = st.columns(2)

        # Hourly demand

        with col1:

            hourly = (
                filtered_df
                .groupby("Hour")["Demand"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                hourly,
                x="Hour",
                y="Demand",
                title="Average Demand by Hour"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # Day of week demand

        with col2:

            weekday = (
                filtered_df
                .groupby("DayOfWeek")["Demand"]
                .mean()
                .reset_index()
            )

            day_names = {
                0: "Monday",
                1: "Tuesday",
                2: "Wednesday",
                3: "Thursday",
                4: "Friday",
                5: "Saturday",
                6: "Sunday"
            }

            weekday["Day"] = weekday["DayOfWeek"].map(
                day_names
            )

            fig = px.bar(
                weekday,
                x="Day",
                y="Demand",
                title="Average Demand by Weekday"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # Monthly demand

        monthly = (
            filtered_df
            .groupby("Month")["Demand"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            monthly,
            x="Month",
            y="Demand",
            markers=True,
            title="Monthly Charging Demand"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Company analysis

        company_demand = (
            filtered_df
            .groupby("ChargerCompany")["Demand"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            company_demand,
            x="ChargerCompany",
            y="Demand",
            title="Charging Demand by Company"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ----------------------------------------------------------
# PAGE 3 - REGRESSION PREDICTION
# ----------------------------------------------------------

elif page == "Regression Prediction":

    st.header("EV Charging Demand Prediction")

    st.write(
        "Enter charging details to predict numerical demand."
    )

    if regression_model is None:

        st.error(
            "Regression model not found. "
            "Check models/regression_pipeline.joblib."
        )

    else:

        with st.form("regression_form"):

            col1, col2 = st.columns(2)

            with col1:

                company = st.selectbox(
                    "Charger Company",
                    companies
                )

                location = st.selectbox(
                    "Location",
                    locations
                )

                charger_type = st.selectbox(
                    "Charger Type",
                    charger_types
                )

                charging_date = st.date_input(
                    "Charging Date",
                    value=max_date
                )

            with col2:

                charging_time = st.time_input(
                    "Charging Start Time"
                )

                st.write(
                    "Select the charging date and start time."
                )

            submitted = st.form_submit_button(
                "Predict Charging Demand"
            )

        if submitted:

            selected_datetime = pd.Timestamp.combine(
                charging_date,
                charging_time
            )

            prediction_input = pd.DataFrame([{

                "ChargerCompany": company,

                "Location": location,

                "ChargerType": charger_type,

                "Hour": selected_datetime.hour,

                "DayOfWeek": selected_datetime.dayofweek,

                "Month": selected_datetime.month,

                "IsWeekend": int(
                    selected_datetime.dayofweek >= 5
                )

            }])

            try:

                prediction = regression_model.predict(
                    prediction_input
                )[0]

                st.success(
                    "Demand prediction generated successfully!"
                )

                st.metric(
                    "Predicted Charging Demand",
                    f"{prediction:.2f}"
                )

                st.write("Input details:")

                st.dataframe(
                    prediction_input,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# ----------------------------------------------------------
# PAGE 4 - CLASSIFICATION PREDICTION
# ----------------------------------------------------------

elif page == "Classification Prediction":

    st.header("High Demand / Normal Demand Prediction")

    st.write(
        "Predict whether a charging session belongs "
        "to High Demand or Normal Demand."
    )

    if classification_results.empty:

        st.error(
            "Classification comparison file not found."
        )

    else:

        available_models = []

        for model_name in classification_results["Model"]:

            filename = (
                model_name.lower().replace(" ", "_")
                + "_pipeline.joblib"
            )

            if os.path.exists(
                os.path.join(MODEL_PATH, filename)
            ):

                available_models.append(model_name)

        if not available_models:

            st.error(
                "No saved classification model was found."
            )

        else:

            sorted_models = (
                classification_results
                .sort_values(
                    "F1 Score",
                    ascending=False
                )["Model"]
                .tolist()
            )

            available_models = [
                name for name in sorted_models
                if name in available_models
            ]

            selected_model_name = st.selectbox(
                "Select Classification Model",
                available_models
            )

            filename = (
                selected_model_name.lower().replace(" ", "_")
                + "_pipeline.joblib"
            )

            selected_model = load_model(
                os.path.join(MODEL_PATH, filename)
            )

            with st.form("classification_form"):

                col1, col2 = st.columns(2)

                with col1:

                    company = st.selectbox(
                        "Charger Company",
                        companies,
                        key="class_company"
                    )

                    location = st.selectbox(
                        "Location",
                        locations,
                        key="class_location"
                    )

                    charger_type = st.selectbox(
                        "Charger Type",
                        charger_types,
                        key="class_charger"
                    )

                with col2:

                    charging_date = st.date_input(
                        "Charging Date",
                        value=max_date,
                        key="class_date"
                    )

                    charging_time = st.time_input(
                        "Charging Start Time",
                        key="class_time"
                    )

                submitted = st.form_submit_button(
                    "Predict Demand Category"
                )

            if submitted:

                selected_datetime = pd.Timestamp.combine(
                    charging_date,
                    charging_time
                )

                prediction_input = pd.DataFrame([{

                    "ChargerCompany": company,

                    "Location": location,

                    "ChargerType": charger_type,

                    "Hour": selected_datetime.hour,

                    "DayOfWeek": selected_datetime.dayofweek,

                    "Month": selected_datetime.month,

                    "IsWeekend": int(
                        selected_datetime.dayofweek >= 5
                    )

                }])

                try:

                    prediction = selected_model.predict(
                        prediction_input
                    )[0]

                    if prediction == 1:

                        st.success("Predicted Category: HIGH DEMAND")

                    else:

                        st.info("Predicted Category: NORMAL DEMAND")

                    if hasattr(selected_model, "predict_proba"):

                        probability = (
                            selected_model
                            .predict_proba(prediction_input)[0, 1]
                        )

                        st.metric(
                            "High Demand Probability",
                            f"{probability * 100:.2f}%"
                        )

                    else:

                        st.caption(
                            "This model does not provide calibrated "
                            "prediction probabilities."
                        )

                    st.write("Prediction input:")

                    st.dataframe(
                        prediction_input,
                        use_container_width=True
                    )

                except Exception as e:

                    st.error(
                        f"Classification error: {e}"
                    )


# ----------------------------------------------------------
# PAGE 5 - MODEL PERFORMANCE COMPARISON
# ----------------------------------------------------------

elif page == "Model Performance":

    st.header("Machine Learning Model Performance")

    st.subheader("Classification Model Comparison")

    if classification_results.empty:

        st.warning(
            "Classification comparison results are unavailable."
        )

    else:

        st.dataframe(
            classification_results.sort_values(
                "F1 Score",
                ascending=False
            ),
            use_container_width=True
        )

        fig = px.bar(
            classification_results,
            x="Model",
            y="Accuracy",
            title="Classification Accuracy Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig = px.bar(
            classification_results,
            x="Model",
            y="F1 Score",
            title="Classification F1 Score Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Optional regression comparison results

    regression_results_path = REGRESSION_RESULTS_PATH

    if regression_results_path.exists():

        st.markdown("---")

        st.subheader("Regression Model Comparison")

        regression_results = pd.read_csv(
            regression_results_path
        )

        st.dataframe(
            regression_results,
            use_container_width=True
        )

    else:

        st.info(
            "Regression comparison CSV is not yet available. "
            "You can add it from your Stage 5 notebook."
        )


# ----------------------------------------------------------
# PAGE 6 - MODEL EXPLAINABILITY
# ----------------------------------------------------------

elif page == "Model Explainability":

    st.header("Machine Learning Model Explainability")

    # Feature Importance

    feature_path = EXPLAIN_PATH / "feature_importance.csv"

    if os.path.exists(feature_path):

        st.subheader("Feature Importance")

        feature_df = pd.read_csv(
            feature_path
        )

        fig = px.bar(
            feature_df.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance Analysis"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Feature importance CSV not found."
        )

    # SHAP Chart

    shap_path = EXPLAIN_PATH / "shap_feature_importance.png"

    if os.path.exists(shap_path):

        st.subheader("SHAP Feature Importance")

        st.image(
            str(shap_path),
            use_container_width=True
        )

    else:

        st.warning(
            "SHAP chart not found."
        )

    # LIME Explanation

    lime_path = EXPLAIN_PATH / "lime_explanation.csv"

    if os.path.exists(lime_path):

        st.subheader("LIME Individual Prediction Explanation")

        lime_df = pd.read_csv(
            lime_path
        )

        st.dataframe(
            lime_df,
            use_container_width=True
        )

        fig = px.bar(
            lime_df,
            x="Contribution",
            y="Feature",
            orientation="h",
            title="LIME Feature Contributions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "LIME explanation file not found."
        )


# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------

st.markdown("---")

st.caption(
    "EV Charging Demand Prediction and Usage Pattern Analysis | "
    "Machine Learning Project"
)