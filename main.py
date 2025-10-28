
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score, mean_absolute_error
# import joblib
# import streamlit as st 

# def main():
#     print("Hello from country-wise-prediction!")
#     # ======================================
# # 🧠 PRICE PREDICTION USING REGRESSION
# # ======================================

  
#     # =======================
#     # 1️⃣ Load Dataset
#     # =======================
#     df = pd.read_csv("updated_inventory_with_country.csv")

#     countries = df["country"].unique()
#     countries

#     country_based = df[df["country"]==countries[0]]


#     print("✅ Dataset loaded successfully!")
#     print("Shape:", country_based.shape)

#     # =======================
#     # 2️⃣ Define Features & Target
#     # =======================
#     # Target variable (dependent)
#     selected_product = country_based[country_based["product_id"]=="P010"]

#     y = selected_product["price"]

#     # Independent variables (features)
#     X =selected_product.drop(columns=["price", "date", "product_id","selling price","predicted price"],axis=1)

#     # =======================
#     # 3️⃣ Encode Categorical Columns
#     # =======================
#     categorical_features = ["category", "region", "country"]

#     ct = ColumnTransformer(
#         transformers=[
#             ("encoder", OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ],
#         remainder="passthrough"
#     )

#     X_encoded = ct.fit_transform(X)

#     # =======================
#     # 4️⃣ Split Dataset
#     # =======================
#     x_train, x_test, y_train, y_test = train_test_split(
#         X_encoded, y, test_size=0.2, random_state=42
#     )

#     # =======================
#     # 5️⃣ Train Regression Model
#     # =======================
#     model = RandomForestRegressor(n_estimators=200, random_state=42)
#     model.fit(x_train, y_train)

#     # =======================
#     # 6️⃣ Predict & Evaluate
#     # =======================
#     y_pred = model.predict(x_test)

#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)

#     print(f"\n✅ Model Evaluation:")
#     print(f"R² Score: {r2:.3f}")
#     print(f"Mean Absolute Error: {mae:.3f}")

#     # =======================
#     # 7️⃣ Save Model
#     # =======================
#     joblib.dump({"model": model, "encoder": ct}, "price_prediction_model.joblib")
#     print("\n💾 Model saved as 'price_prediction_model.joblib'")

#     # =======================
#     # 8️⃣ Example Prediction
#     # =======================
#     new_data = pd.DataFrame([{
#         "sales": 120,
#         "category": "B",
#         "region": "East",
#         "discount": 5.0,
#         "rating": 4.2,
#         "views": 340,
#         "clicks": 25,
#         "cost_price": 55.0,
#         "country": "India"
#     }])

#     print(f"\nCountry: {country_based['country'].values.tolist()[0]}")
#     new_data_encoded = ct.transform(country_based)
#     predicted_price = model.predict(new_data_encoded)
#     print(f"\n Actual Price: {y_test.tolist()[0]}")
#     print(f"\n💰 Predicted Price for input: {predicted_price[0]:.2f}")



#     st.header("Prediction")

#     st.sidebar.write(countries[0])
    







# if __name__ == "__main__":
#     main()





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import streamlit as st

# ======================================
# STREAMLIT DASHBOARD
# ======================================

st.set_page_config(page_title="Country-wise Price Prediction", layout="wide")
st.title("🌍 Country Based Price Prediction Dashboard")

# =======================
# LOAD DATA
# =======================
@st.cache_data
def load_data():
    df = pd.read_csv("updated_inventory_with_country.csv")
    return df

df = load_data()

st.success("✅ Dataset Loaded Successfully")
st.write(f"📊 Total Rows: {len(df)}")

# =======================
# SIDEBAR FILTERS
# =======================
st.sidebar.header("🔧 Filters")

# country dropdown
country_list = sorted(df["country"].unique())
selected_country = st.sidebar.selectbox("Select Country", country_list)

# filter by country
country_data = df[df["country"] == selected_country]

# product dropdown
product_list = sorted(country_data["product_id"].unique())
selected_product = st.sidebar.selectbox("Select Product ID", product_list)

# Filter by country & product
product_data = country_data[country_data["product_id"] == selected_product]

# region dropdown
region_list = sorted(product_data["region"].unique())
selected_region = st.sidebar.selectbox("Select Region", region_list)

# final filtered dataset
filtered_data = product_data[product_data["region"] == selected_region]

st.write(f"✅ Rows Matching Selection: {len(filtered_data)}")

if len(filtered_data) < 5:
    st.warning("⚠ Not enough data for training. Please change filter options.")
else:
    # =======================
    # TRAIN MODEL
    # =======================
    st.subheader("📚 Model Training & Evaluation")

    y = filtered_data["price"]
    X = filtered_data.drop(columns=["price", "date", "product_id", "selling price", "predicted price"])

    categorical_features = ["category", "region", "country"]

    ct = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder="passthrough"
    )

    X_encoded = ct.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)

    # Evaluate
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write(f"✅ R² Score: **{r2:.3f}**")
    st.write(f"✅ MAE: **{mae:.3f}**")

    # =======================
    # PREDICTION SECTION
    # =======================
    st.subheader("💰 Price Prediction")

    sales = st.slider("Sales", 0, 200, 50)
    discount = st.slider("Discount (%)", 0, 60, 10)
    rating = st.slider("Rating", 1.0, 5.0, 4.0, step=0.1)
    views = st.slider("Views", 0, 5000, 200)
    clicks = st.slider("Clicks", 0, 1000, 40)
    cost_price = st.number_input("Cost Price", min_value=1.0, value=50.0)

    if st.button("Predict Price"):
        input_data = pd.DataFrame([{
            "sales": sales,
            "category": filtered_data.iloc[0]["category"],
            "region": selected_region,
            "discount": discount,
            "rating": rating,
            "views": views,
            "clicks": clicks,
            "cost_price": cost_price,
            "country": selected_country
        }])

        encoded_input = ct.transform(input_data)
        predicted_price = model.predict(encoded_input)[0]

        st.success(f"✅ **Predicted Price: ₹ {predicted_price:.2f}**")

    # =======================
    # SAVE MODEL
    # =======================
    joblib.dump({"model": model, "encoder": ct}, "price_prediction_model.joblib")
    st.info("💾 Model saved as price_prediction_model.joblib")
