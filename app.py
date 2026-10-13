import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="PersonaIQ | Customer Segmentation",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, #f0eaff 0%, transparent 35%),
        radial-gradient(ellipse at 95% 10%, #ffe8f2 0%, transparent 32%),
        #f8f9ff;
    color: #242044;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    color: #28204e;
}

.hero {
    padding: 38px 42px;
    border-radius: 28px;
    background: linear-gradient(120deg, #5428c7 0%, #8653ed 48%, #ed69b8 100%);
    color: white;
    box-shadow: 0 16px 38px rgba(104, 64, 190, 0.22);
    margin-bottom: 25px;
    position: relative;
    overflow: hidden;
}

.hero:after {
    content: "✦";
    position: absolute;
    right: 42px;
    top: 12px;
    font-size: 115px;
    color: rgba(255,255,255,0.15);
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 6px 12px;
    border-radius: 30px;
    font-size: 12px;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin-bottom: 15px;
}

.hero h1 {
    color: white !important;
    font-size: 42px;
    line-height: 1.12;
    margin: 0;
    font-weight: 800;
}

.hero p {
    color: #f6edff;
    font-size: 16px;
    margin-top: 12px;
    max-width: 700px;
    line-height: 1.7;
}

.section-heading {
    font-family: 'Outfit', sans-serif;
    font-size: 23px;
    font-weight: 700;
    color: #30245b;
    margin: 18px 0 6px;
}

.section-subtitle {
    color: #77728f;
    font-size: 14px;
    margin-bottom: 20px;
}

div[data-testid="stForm"] {
    background: rgba(255,255,255,0.92);
    padding: 26px;
    border: 1px solid #e9e3fa;
    border-radius: 24px;
    box-shadow: 0 8px 28px rgba(65, 44, 116, 0.06);
}

div[data-testid="stNumberInput"] label {
    color: #51476f;
    font-size: 13px;
    font-weight: 600;
}

div[data-testid="stNumberInput"] input {
    border-radius: 12px;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    min-height: 50px;
    border: none;
    border-radius: 14px;
    color: white;
    font-weight: 700;
    font-size: 15px;
    background: linear-gradient(100deg, #6536d7, #a44ce0, #e45fae);
    box-shadow: 0 8px 18px rgba(116, 66, 202, 0.2);
    transition: all 0.2s ease;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 12px 23px rgba(116, 66, 202, 0.3);
}

.metric-card {
    padding: 23px;
    border-radius: 22px;
    background: white;
    border: 1px solid #ece6fa;
    box-shadow: 0 8px 24px rgba(65, 44, 116, 0.055);
    min-height: 130px;
}

.metric-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #88809f;
    font-weight: 700;
}

.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 30px;
    font-weight: 800;
    color: #5c32c8;
    margin-top: 8px;
}

.metric-note {
    font-size: 12px;
    color: #928aa8;
    margin-top: 4px;
}

.result-panel {
    padding: 30px;
    border-radius: 25px;
    background: linear-gradient(130deg, #eee7ff, #fff0f7);
    border: 1px solid #e3d6ff;
    margin-top: 22px;
    margin-bottom: 20px;
}

.result-label {
    font-size: 12px;
    font-weight: 700;
    color: #7655ba;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.result-title {
    font-family: 'Outfit', sans-serif;
    color: #4c25b2;
    font-size: 34px;
    font-weight: 800;
    margin: 5px 0;
}

.result-description {
    color: #62577e;
    font-size: 14px;
    line-height: 1.7;
}

.pill {
    display: inline-block;
    border-radius: 30px;
    padding: 7px 12px;
    background: #e4d9ff;
    color: #5c32c8;
    font-size: 12px;
    font-weight: 700;
    margin-top: 12px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #e9e3fa;
    border-radius: 15px;
    overflow: hidden;
}

.footer {
    text-align: center;
    padding: 28px 10px 8px;
    color: #9891ae;
    font-size: 12px;
}

hr {
    border-color: #e9e3fa;
}

@media (max-width: 768px) {
    .hero { padding: 27px 23px; }
    .hero h1 { font-size: 31px; }
    .result-title { font-size: 27px; }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD TRAINED COMPONENTS
# =====================================================
@st.cache_resource
def load_artifacts():
    return joblib.load("deployment/hierarchical_artifacts.pkl")


try:
    artifacts = load_artifacts()
    scaler = artifacts["scaler"]
    features = artifacts["features"]
    cluster_centers = artifacts["cluster_centers"]

    if "income_upper_limit" not in artifacts:
        st.error(
            "The saved model artifacts do not contain "
            "'income_upper_limit'. Please rerun the artifact-saving "
            "cell in your Jupyter Notebook."
        )
        st.stop()

    income_upper_limit = artifacts["income_upper_limit"]

except Exception as error:
    st.error(
        f"Unable to load model artifacts: {error}. "
        "Check deployment/hierarchical_artifacts.pkl."
    )
    st.stop()


# =====================================================
# HERO HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ CUSTOMER INTELLIGENCE PLATFORM</div>
    <h1>Meet your customers<br>on a whole new level.</h1>
    <p>
        Discover meaningful customer segments with hierarchical
        clustering. Explore purchasing patterns and turn customer
        data into smarter marketing decisions.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-heading">✨ Customer Persona Explorer</div>
<div class="section-subtitle">
Enter customer characteristics to find the closest existing segment.
</div>
""", unsafe_allow_html=True)


# =====================================================
# CUSTOMER INPUT FORM
# =====================================================
with st.form("customer_form"):
    st.markdown("### 👤 Demographics & Lifestyle")
    st.caption("Tell us a little about this customer.")

    col1, col2, col3 = st.columns(3)

    with col1:
        income = st.number_input(
            "💰 Annual Income",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        age = st.number_input(
            "🎂 Customer Age",
            min_value=18,
            max_value=90,
            value=40
        )

        total_children = st.number_input(
            "👨‍👩‍👧 Children at Home",
            min_value=0,
            max_value=10,
            value=1
        )

        recency = st.number_input(
            "🗓️ Days Since Last Purchase",
            min_value=0,
            max_value=365,
            value=30
        )

    with col2:
        st.markdown("### 🛍️ Shopping Behavior")

        total_spending = st.number_input(
            "💳 Total Product Spending",
            min_value=0.0,
            value=500.0,
            step=50.0
        )

        total_purchases = st.number_input(
            "🧾 Total Purchases",
            min_value=0,
            value=15
        )

        deals_purchases = st.number_input(
            "🏷️ Purchases Using Deals",
            min_value=0,
            value=2
        )

        web_purchases = st.number_input(
            "🌐 Web Purchases",
            min_value=0,
            value=5
        )

    with col3:
        st.markdown("### 📣 Channels & Engagement")

        catalog_purchases = st.number_input(
            "📚 Catalog Purchases",
            min_value=0,
            value=3
        )

        store_purchases = st.number_input(
            "🏬 Store Purchases",
            min_value=0,
            value=7
        )

        web_visits = st.number_input(
            "👀 Website Visits per Month",
            min_value=0,
            value=5
        )

        campaign_acceptance = st.number_input(
            "💌 Campaigns Accepted (Including Latest Response)",
            min_value=0,
            max_value=6,
            value=1
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "✨ Discover Customer Segment",
        type="primary",
        use_container_width=True
    )


# =====================================================
# SEGMENT ASSIGNMENT
# =====================================================
if submitted:

    customer = pd.DataFrame([{
        "Income": income,
        "Age": age,
        "Total_Children": total_children,
        "Recency": recency,
        "Total_Spending": total_spending,
        "Total_Purchases": total_purchases,
        "NumDealsPurchases": deals_purchases,
        "NumWebPurchases": web_purchases,
        "NumCatalogPurchases": catalog_purchases,
        "NumStorePurchases": store_purchases,
        "NumWebVisitsMonth": web_visits,
        "Total_Campaign_Acceptance": campaign_acceptance
    }])[features]

    # ---------------------------------------------
    # APPLY SAME PREPROCESSING AS TRAINING
    # ---------------------------------------------
    customer["Income"] = customer["Income"].clip(
        upper=artifacts["income_upper_limit"]
    )

    customer["Age"] = customer["Age"].clip(upper=90)

    # ---------------------------------------------
    # SCALE INPUT FEATURES
    # ---------------------------------------------
    customer_scaled = scaler.transform(customer)

    # ---------------------------------------------
    # FIND NEAREST CLUSTER REPRESENTATIVE
    # ---------------------------------------------
    centers = cluster_centers[features].to_numpy()

    distances = np.linalg.norm(
        centers - customer_scaled[0],
        axis=1
    )

    closest_position = int(np.argmin(distances))
    segment = cluster_centers.index[closest_position]
    distance = distances[closest_position]

    # ---------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------
    st.markdown("---")

    st.markdown("""
    <div class="section-heading">🎯 Your Customer Insight</div>
    <div class="section-subtitle">
    Here's the segment most similar to the customer you entered.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-panel">
        <div class="result-label">BEST MATCHING SEGMENT</div>
        <div class="result-title">Customer Segment {segment}</div>
        <div class="result-description">
            This customer is closest to the representative of
            Segment {segment}, based on the selected demographic
            and purchasing features after scaling.
        </div>
        <div class="pill">✦ Hierarchical Clustering · K = 2</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👥 Assigned Segment</div>
            <div class="metric-value">{segment}</div>
            <div class="metric-note">Closest cluster representative</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💰 Total Spending</div>
            <div class="metric-value">${total_spending:,.0f}</div>
            <div class="metric-note">Customer-entered spending</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📏 Representative Distance</div>
            <div class="metric-value">{distance:.2f}</div>
            <div class="metric-note">Lower means closer, not more certain</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Customer Profile")
    st.dataframe(
        customer,
        use_container_width=True,
        hide_index=True
    )

    with st.expander("🔬 How was this segment selected?"):
        st.write(
            "The input values were preprocessed and transformed using "
            "the same fitted scaler used during model development. "
            "Euclidean distance was calculated between the scaled "
            "customer and each cluster's average scaled position. "
            "The nearest representative determines the assigned segment."
        )

        distance_table = pd.DataFrame({
            "Segment": cluster_centers.index,
            "Distance": distances
        }).sort_values("Distance")

        st.dataframe(
            distance_table,
            use_container_width=True,
            hide_index=True
        )

else:
    st.markdown("""
    <div class="result-panel">
        <div class="result-label">YOUR NEXT DISCOVERY</div>
        <div class="result-title">Every customer tells a story. ✨</div>
        <div class="result-description">
            Complete the form above and select
            <b>Discover Customer Segment</b> to reveal the closest segment.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    PERSONAIQ · CUSTOMER PERSONALITY ANALYSIS<br>
    Powered by Python, Streamlit & Hierarchical Clustering
</div>
""", unsafe_allow_html=True)
