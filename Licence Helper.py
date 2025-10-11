import streamlit as st
import os
import json
import numpy as np
import faiss
from langchain_openai.embeddings import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv  

load_dotenv('.env')

# -------------------------------
# 0️⃣ Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="SG Food Business Licence Helper",
    page_icon="🍴",
    layout="centered"
)

# -------------------------------
# 1️⃣ Login setup
# -------------------------------
# Access credentials
USER_CREDENTIALS = {
    "main": os.getenv("USER_MAIN"),
    "alice": os.getenv("USER_ALICE")
}

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# Login page
if not st.session_state["logged_in"]:
    st.title("Login")
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    if st.button("Login"):
        if username_input in USER_CREDENTIALS and USER_CREDENTIALS[username_input] == password_input:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username_input  # save username
            st.rerun()  # rerun so the main app loads immediately
        else:
            st.error("❌ Invalid username or password")
    st.stop()  # stop running the rest of the script until login

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
# 5️⃣ Start of main UI
# -------------------------------
st.title("Singapore Food Licence Helper")
st.header("Step 1: Describe your business")

business_type = st.selectbox(
    "Business Type",
    ["Import/Export/Transhipment", "Sale of Food", "Manufacturing", "Food Cold Storage", "Temporary Stall", "Farming", "Other"]
)

food_types = st.multiselect(
    "Types of Food Involved (select all that apply)",
    ["Meat", "Seafood", "Fruits", "Vegetables", "Processed Food", "Animal Feed", "Other"]
)

additional_details = st.text_area(
    "Give additional details about your business (e.g., setting, countries, event type, scale)"
)

user_profile = {
    "business_type": business_type,
    "food_types": food_types,
    "additional_details": additional_details
}

# -------------------------------
# Step 2: Dynamic follow-up question
# -------------------------------
followup_answer = ""
if business_type == "Sale of Food":
    st.subheader("Step 2: Details about the food forms")
    followup_answer = st.text_input(
        "Are the foods raw/cooked, fresh, chilled, frozen, or other? (You can describe multiple types)"
    )
    if followup_answer:
        user_profile["food_form"] = followup_answer

# -------------------------------
# -------------------------------
# Step 3: Submit and get licence guidance
# -------------------------------
if st.button("Get Licence Guidance"):
    status_placeholder = st.empty()
    status_placeholder.info("🧠 Understanding your business...")

    # -------------------------------
    # Build query text emphasizing business type
    # -------------------------------
    query_text = f"Business type: {business_type}. Product sold: {', '.join(food_types)}"
    if "food_form" in user_profile:
        query_text += f". Food form: {user_profile['food_form']}"
    if additional_details:
        query_text += f". Additional details: {additional_details}"

    # Embed the query
    query_vector = np.array(embeddings_model.embed_query(query_text), dtype=np.float16)

    # -------------------------------
    # Compute similarity to all chunks
    # -------------------------------
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [cosine_similarity(query_vector, emb) for emb in X]

    # -------------------------------
    # Rank top candidates
    # -------------------------------
    top_n = 12
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:top_n]

    # -------------------------------
    # Filter by business type keywords
    # -------------------------------
    business_keywords = ["vending machine", "food shop", "retail outlet"]
    filtered_indices = [
        i for i in top_indices
        if any(kw.lower() in all_chunks[i].lower() for kw in business_keywords)
    ]

    filtered_chunks = [all_chunks[i] for i in filtered_indices]
    filtered_metadata = [all_metadata[i] for i in filtered_indices]

    # -------------------------------
    # Combine filtered chunks into prompt
    # -------------------------------
    combined_context = "\n\n".join(filtered_chunks)

    prompt = f"""
You are a knowledgeable assistant for Singapore food business regulations.

Given the following government-sourced information:

{combined_context}

The user wants to open a business with these details:
Business type: {business_type}
Product sold: {additional_details}

Instructions:
- Only list licences, permits, or approvals that are specifically relevant to this type of business and product.
- Ignore general licences that cover unrelated food types or retail formats.
- Provide plain-language explanation of why each licence is required.
- Include step-by-step guidance or application URLs if available.
"""

    # -------------------------------
    # Get GPT summary
    # -------------------------------
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    summary = response.choices[0].message.content
    st.header("Summary of Required Approvals / Licences")
    st.write(summary)
