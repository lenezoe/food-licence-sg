import streamlit as st
import json
import numpy as np
import faiss
from langchain.embeddings import OpenAIEmbeddings
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
USER_CREDENTIALS = {
    "john": "MySecret123!",
    "alice": "AnotherPass456"
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
@st.cache_resource
def load_resources():
    X = np.load("data/chunk_embeddings.npy")
    index = faiss.read_index("data/faiss_index.index")
    embeddings_model = OpenAIEmbeddings()  # uses OPENAI_API_KEY
    client = OpenAI(api_key=None)          # uses OPENAI_API_KEY
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
# Step 3: Submit and get licence guidance
# -------------------------------
if st.button("Get Licence Guidance"):
    status_placeholder = st.empty()
    status_placeholder.info("🧠 Understanding message...")

    # Build query
    query_text = f"{business_type} business selling {', '.join(food_types)}"
    if "food_form" in user_profile:
        query_text += f" in the form: {user_profile['food_form']}"
    if additional_details:
        query_text += f". Additional details: {additional_details}"

    # Embed the query
    query_vector = np.array(embeddings_model.embed_query(query_text), dtype=np.float16)

    # Retrieve top chunks
    top_n = 15
    D, I = index.search(np.expand_dims(query_vector, axis=0), top_n)
    retrieved_chunks = [all_chunks[i] for i in I[0]]
    retrieved_chunks = [chunk[:1000] for chunk in retrieved_chunks[:8]]

    # Craft prompt
    combined_context = "\n\n".join(retrieved_chunks)
    prompt = f"""
    You are a reliable assistant specialising in Singapore government food-related business licensing...
    (your full prompt continues here exactly as in your original code)
    """

    # Clear status before streaming
    status_placeholder.empty()
    response_box = st.empty()
    response_text = ""

    # Stream GPT output
    for chunk in client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1000,
        stream=True
    ):
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            response_text += delta.content
            response_box.write(response_text)
