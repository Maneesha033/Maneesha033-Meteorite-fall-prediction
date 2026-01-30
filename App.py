import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
import plotly.express as px

# Load models
lr_model = pickle.load(open("logistic_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))


def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("Meteorite Image 2026-01-26 at 10.42.08 PM.jpeg")


st.title("Meteorite Fall Prediction")

st.write("Predict whether a meteorite was **Fell** or **Found**")

# User inputs
mass = st.number_input("Meteorite Mass (grams)", min_value=0.0)
year = st.number_input("Year", min_value=1000, max_value=2025)
lat = st.number_input("Latitude")
long = st.number_input("Longitude")

meteor_class = st.selectbox(
    "Meteorite Class",
    ['H4','H4/5','H5','H6','L4','L5','L6','LL5','LL6','Other']
)

def prepare_input():
    data = dict.fromkeys(feature_columns, 0)

    data['mass'] = np.log1p(mass)
    data['year'] = year
    data['lat'] = lat
    data['long'] = long

    class_col = f'class_{meteor_class}'
    if class_col in data:
        data[class_col] = 1
    else:
        data['class_Other'] = 1

    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)

    return df, df_scaled




if st.button("Predict"):
    df_input, df_scaled = prepare_input()
    
    # 1. Get predictions and probability
    lr_pred = lr_model.predict(df_scaled)[0]
    rf_pred = rf_model.predict(df_input)[0]
    # Get confidence from Random Forest
    rf_prob = rf_model.predict_proba(df_input)[0].max() 

    st.divider()

    # 2. Main Result Section
    st.subheader("🚀 Final Prediction")
    if rf_pred == 1:
        st.success("### Result: ☄️ This Meteorite FELL")
    else:
        st.info("### Result: 🌍 This Meteorite was FOUND")
    
    st.metric(label="Prediction Confidence", value=f"{rf_prob * 100:.2f}%")

    # 3. Tabs for Map, Analysis, and Comparison
    tab1, tab2, tab3 = st.tabs(["📍 Location Map", "📊 Importance Analysis", "🤖 Model Comparison"])



    # with tab1:
    #     st.write("### 📍 Predicted Impact Location")
    #     if lat == 0.0 and long == 0.0:
    #         st.warning("Please enter valid Latitude and Longitude to see the location on map.")
    #     else:         # Map-il oru label varaan vendi dataframe-il column add cheyyunnu
    #         map_data = pd.DataFrame({
    #         'lat': [lat],
    #         'lon': [long],
    #         'name': ['Predicted Meteorite Location']
    #         })
    #         st.map(map_data)
    #         st.caption(f"Showing coordinates: {lat}, {long}")



    
    with tab1:
        st.write("### 📍 Impact Site Visualization")
    
    # Create the map using a professional dark theme
        fig = px.scatter_mapbox(
            lat=[lat], 
            lon=[long], 
            zoom=1, 
            height=400
        )
    
    # This makes it look like a high-tech dark map
        fig.update_layout(
             mapbox_style="carto-darkmatter", 
             margin={"r":0,"t":0,"l":0,"b":0}
        )
    
    # Customizing the marker to look like a glowing impact point
        fig.update_traces(marker=dict(size=15, color="cyan"))
    
        st.plotly_chart(fig, use_container_width=True)



    with tab2:
        st.write("### What influenced the Random Forest?")
        # Show which features were important
        # Ensure 'feature_columns' is the list of your column names
        importances = rf_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importances})
        # Sorting to show top features
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
        st.bar_chart(importance_df.set_index('Feature'))
        

    with tab3:
        st.write("Comparison between the two models used:")
        col1, col2 = st.columns(2)
        col1.metric("Logistic Regression", "Fell" if lr_pred == 1 else "Found")
        col2.metric("Random Forest", "Fell" if rf_pred == 1 else "Found")
        
