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
# st.success(f"✅ Logged in as {st.session_state['username']}")

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
# 5️⃣ Sidebar Navigation
# -------------------------------
st.sidebar.markdown(
    """
    <style>
    /* Sidebar title */
    .sidebar-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }

    /* Nav link styles */
    .nav-link {
        display: block;
        padding: 10px 15px;
        margin-bottom: 5px;
        font-weight: bold;
        color: #555;             /* text color */
        text-decoration: none;
        border-radius: 5px;
        background-color: #f0f0f0;   /* same as sidebar */
        transition: background-color 0.2s, color 0.2s;
    }

    .nav-link:hover {
        background-color: #d9d9d9; /* slightly darker grey on hover */
        color: #333;  /* darker text on hover */
        cursor: pointer;
    }

    .nav-link.active {
        background-color: #bfbfbf; /* active page highlight */
        color: #000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add navigation container with title
st.sidebar.markdown('<div class="sidebar-container">', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)


# Map page names to identifiers
pages = {"Main App": "main", "About Us": "about","Methodology": "methodology"}

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "main"

# Render links as sidebar buttons
for name, key in pages.items():
    if st.sidebar.button(name):
        st.session_state.page = key


# -------------------------------
# 6️⃣ Main App
# -------------------------------
if st.session_state.page == "main":
    st.title("🍽️ Singapore Food Licence AI")
    st.subheader("Tell us about your food business 💡")
    
    st.markdown(
        """
        Fill in your business details below and get guidance on the approvals / licences you need.  
        """
    )

    # Wrap the main form for “Get Licence Guidance”
    with st.form("licence_form"):
        # Type of business
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

# -------------------------------
# 6️⃣ About Us
# -------------------------------
if st.session_state.page == "about":
    st.title("❓About Us")
    st.markdown(
        """
        ### 📌 Overview
        ***Singapore Food Licence AI*** is a web-based application designed to help food entrepreneurs, retailers, and import/export businesses quickly understand which regulatory approvals and licences they may require. 
        
        By leveraging open-source data from government sources, embeddings, and AI-powered synthesis, users receive personalized, actionable guidance in plain language.
        
        ---

        ### 📝 Project Scope
        This application covers a wide range of food-related businesses, including:
        - Restaurants, cafés, and food stalls  
        - Retail shops and supermarkets  
        - Catering and event-based food businesses  
        - Food manufacturing and processing facilities  
        - Import, export, and transhipment of food products and animal feed  

        ---

        ### 🎯 Project Objectives
        - Deliver **accurate, up-to-date guidance** on licences and approvals.  
        - Make regulatory information **accessible and understandable** for small and medium food enterprises.  
        - Enable **efficient discovery** of relevant regulations using AI-powered search and summarization.  
        - Reduce the time and effort required for business owners to **identify required approvals**.

        ---

        ### 📂 Data Sources
        The underlying data comes from information on **licences, permits, and approvals** in **official Singapore government websites**, including:
        - [Singapore Food Agency (SFA)](https://www.sfa.gov.sg/)  
        - [Majlis Ugama Islam Singapura (MUIS)](https://www.muis.gov.sg/)  
        - [National Environment Agency (NEA)](https://www.nea.gov.sg/)  
        - [Singapore Police Force (SPF)](https://www.police.gov.sg/) 
        - [GoBusiness Licensing portal](https://www.gobusiness.gov.sg/)

        The data is **scraped, cleaned, and structured** into JSON chunks for downstream AI processing.

        ---

        ### ⚡ Key Features
        - **Interactive Business Form:** Collects business type, food types, and other relevant details.  
        - **Intelligent Licence Guidance:** Personalized recommendations based on user input.  
        - **AI Summarization:** GPT-4o-mini generates plain-language explanations for licences.  
        - **Search & Filtering:** Keyword-based and vector similarity retrieval for precision.  
        - **Secure Login:** User authentication ensures data privacy.  
        - **Transparent Methodology:** Users can view the methodology page to understand how guidance is generated.

        """
    )
# -------------------------------
# 7️⃣ Methodology
# -------------------------------
elif st.session_state.page == "methodology":
    st.title("🛠️ Methodology: How This Works")

    st.image("App Workflow.JPEG", caption="App Workflow", use_container_width=True)
    st.markdown(
        """
        The application follows a multi-step process:

        1. **Data Collection & Cleaning**
        - Scrape government websites (SFA, MUIS, GoBusiness Licensing, etc) using `requests` and `BeautifulSoup`.  
        - Remove HTML tags, scripts, and navigation elements.  
        - Store meaningful text for processing.

        2. **Data Structuring**
        - Use GPT-4o-mini LLM to parse raw text into structured JSON:
            - `title`, `licence_name`, `requirements`, `application_guidance`, `reason_for_licence`, `other`, `url`.  
        - Prompt enforces strict rules to include **official licences only**.

        3. **Chunking & Embeddings**
        - Split long text into manageable semantic chunks.  
        - Convert each chunk into a vector using `OpenAIEmbeddings`.  
        - Store embeddings in **FAISS** for fast similarity search.

        4. **User Query Processing**
        - User inputs business type, food types, and additional details.  
        - Input is embedded into the same vector space as the data chunks.  
        - Retrieve top-N most similar chunks using cosine similarity.

        5. **Keyword Filtering**
        - Filter chunks based on business category keywords (e.g., "restaurant", "import/export").  
        - This ensures higher precision in the retrieved content.

        6. **AI Synthesis**
        - Combine retrieved chunks into a single context.  
        - Pass context to GPT-4o-mini to generate plain-language licence guidance.  
        - Output is displayed to user in the app interface.

        ---

        ### 🔄 Use Case Flowcharts

        #### **Use Case A: Chat with Information**
        ```text
        User Inputs Business Info
                │
                ▼
        Generate Query Text
                │
                ▼
        Embed Query → Vector Representation
                │
                ▼
        Retrieve Top Chunks via FAISS
                │
                ▼
        Keyword Filtering (Business Type)
                │
                ▼
        Combine Context Chunks
                │
                ▼
        GPT-4o-mini Summarization
                │
                ▼
        Display Licence Guidance to User
        ```
        ---
        #### **Use Case B: Intelligent Search**
        ```text
        User Enters Search Keywords
                │
                ▼
        Embed Search Query
                │
                ▼
        Retrieve Similar Chunks via FAISS
                │
                ▼
        Rank and Filter Chunks by Relevance
                │
                ▼
        Return Top Chunks with Metadata
                │
                ▼
        Display Results with Links & Guidance
        ```
        """
    )
st.markdown("""
    > **Note:** This application provides guidance only and does not replace official regulatory advice. Users should consult the relevant government agencies for final approvals.
    """
)