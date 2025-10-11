import os
import json
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv('.env')

# -------------------------------
# 0️⃣ Streamlit page config
# -------------------------------
st.set_page_config(
    layout="centered"
)

# -------------------------------
# 1️⃣ Login setup
# -------------------------------
USER_CREDENTIALS = {
    "main": os.getenv("USER_MAIN"),
    "alice": os.getenv("USER_ALICE")
}

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# Login page using form
if not st.session_state["logged_in"]:
    st.title("Login")
    with st.form("login_form"):
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username_input in USER_CREDENTIALS and USER_CREDENTIALS[username_input] == password_input:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username_input
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    st.stop()  # stop until login

# -------------------------------
# 2️⃣ Main app after login
# -------------------------------
st.success(f"✅ Logged in as {st.session_state['username']}")

# -------------------------------
# 3️⃣ Cache static data
# -------------------------------
@st.cache_data
def load_json_data():
    with open("data/all_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open("data/all_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return chunks, metadata

all_chunks, all_metadata = load_json_data()
metadata_dict = {meta["chunk_id"]: meta for meta in all_metadata}

# -------------------------------
# 4️⃣ Cache heavy resources
# -------------------------------
api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def load_resources():
    X = np.load("data/chunk_embeddings.npy")
    index = faiss.read_index("data/faiss_index.index")
    embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)
    client = OpenAI(api_key=api_key)
    return X, index, embeddings_model, client

X, index, embeddings_model, client = load_resources()

# -------------------------------
# 5️⃣ Main UI
# -------------------------------
st.title("Singapore Food Licence AI 🍽️")
st.subheader("Tell us about your food business 💡")

st.markdown(
    """
    Fill in your business details below and get guidance on the approvals / licences you need.  
    """
)

# Wrap the main form for “Get Licence Guidance”
with st.form("licence_form"):
    # Step 1 & 2: Side-by-side columns
    col1, col2 = st.columns(2)

    with col1:
        business_type = st.selectbox(
            "🍽️ Business Type",
            [
                "Restaurant / Café / Eatery (selling cooked food on-site)",
                "Hawker Stall / Coffeeshop Stall / Food Court Stall",
                "Catering Business (preparing food for delivery or events)",
                "Retail Shop / Supermarket (selling raw or packaged food)",
                "Import / Export / Transhipment of Food Products",
                "Food Manufacturing / Processing Facility",
                "Cold Storage / Warehouse",
                "Farming / Agriculture",
                "Event-based / Temporary Food Booth",
                "Other"
            ]
        )

    with col2:
        food_types = st.multiselect(
            "🥗 Food Types (select all that apply)",
            [
                "Meat or Poultry",
                "Seafood",
                "Fruits and Vegetables",
                "Baked Goods / Pastries",
                "Beverages (Non-alcoholic)",
                "Alcoholic Drinks",
                "Ready-to-eat / Cooked Food",
                "Processed or Packaged Food",
                "Animal Feed",
                "Other"
            ]
        )


    # Step 3: Additional details
    additional_details = st.text_area(
        "Tell us more about your business idea",
        placeholder="E.g. A cafeteria selling zichar dishes in central Singapore, sourcing ingredients from local farms..."
    )

    # Step 4: Dynamic follow-up question
    followup_answer = ""
    if any(keyword in business_type for keyword in ["Restaurant", "Hawker", "Catering", "Home-based", "Retail", "Eatery"]):
        followup_answer = st.text_input(
            "Are the foods raw/cooked, fresh, chilled, frozen, or other? (You can describe multiple types)",
            placeholder="E.g. cooked bentos and chilled drinks, or frozen seafood, etc."
        )

    # Submit button
    submitted = st.form_submit_button("Get Licence Guidance")

# -------------------------------
# 6️⃣ Generate guidance after submit
# -------------------------------
if submitted:
    status_placeholder = st.empty()
    status_placeholder.info("🧠 Understanding your business...")

    # Build query text
    if business_type == "Other":
        query_text = (
            f"The user is starting a food-related business in Singapore. "
            f"The main idea is: {additional_details}. "
            f"They also mentioned these products or ingredients: {', '.join(food_types)}. "
            f"Describe what licences or approvals they may need for this business type."
        )
        query_text = (query_text + " ") * 2
    else:
        query_text = (
            f"Business type: {business_type}. "
            f"Product sold: {', '.join(food_types)}. "
        )
        if followup_answer:
            query_text += f"Food form: {followup_answer}. "
        if additional_details:
            query_text += f"Additional details: {additional_details}. "

    # Embed query
    query_vector = np.array(embeddings_model.embed_query(query_text), dtype=np.float16)

    # Compute similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [cosine_similarity(query_vector, emb) for emb in X]
    top_n = 12
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:top_n]

    # Keyword filtering (same as your original logic)
    if "Supermarket" in business_type or "Retail" in business_type:
        business_keywords = ["supermarket", "grocery", "retail", "food retail", "food shop"]
    elif "Restaurant" in business_type or "Café" in business_type:
        business_keywords = ["restaurant", "cafe", "eatery", "food shop", "dine-in"]
    elif "Hawker" in business_type:
        business_keywords = ["hawker", "food stall", "coffeeshop", "food court"]
    elif "Catering" in business_type:
        business_keywords = ["catering", "central kitchen", "food catering"]
    elif "Home-based" in business_type:
        business_keywords = ["home-based", "home kitchen", "home business", "small scale"]
    elif "Cold Storage" in business_type:
        business_keywords = ["cold store", "cold storage", "warehouse", "frozen"]
    elif "Import" in business_type or "Export" in business_type:
        business_keywords = ["import", "export", "transhipment", "distributor"]
    elif "Manufacturing" in business_type or "Processing" in business_type:
        business_keywords = ["manufacturing", "processing", "production", "factory"]
    elif "Farming" in business_type:
        business_keywords = ["farm", "agriculture", "cultivation"]
    else:
        business_keywords = []

    if business_keywords:
        filtered_indices = [
            i for i in top_indices if any(kw.lower() in all_chunks[i].lower() for kw in business_keywords)
        ]
        if not filtered_indices:
            filtered_indices = top_indices
    else:
        filtered_indices = top_indices

    # Retrieve chunks
    filtered_chunks = [all_chunks[i] for i in filtered_indices]
    combined_context = "\n\n".join(filtered_chunks)

    # Build prompt
    prompt = f"""
You are a knowledgeable assistant for Singapore food business regulations. 
Given the following government-sourced information: {combined_context}

The user wants to open a business with these details:
Business type: {business_type}
Product sold: {additional_details}

Instructions:
- Only list licences, permits, or approvals that are specifically relevant to this type of business and product.
- Ignore general licences that cover unrelated food types or retail formats.
- Provide plain-language explanation of why each licence is required.
- Include step-by-step guidance or application URLs if available.
"""

    # Call GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    summary = response.choices[0].message.content

    st.header("Summary of Required Approvals / Licences")
    st.write(summary)
