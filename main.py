import os
import re
import json
import jsonschema
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
    page_title="Singapore Food Licence AI",  
    layout="centered"
)

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "page" not in st.session_state:
    st.session_state.page = "main"

# -------------------------------
# 1️⃣ Login setup
# -------------------------------
USER_CREDENTIALS = {
    "main": os.getenv("USER_MAIN"),
    "alice": os.getenv("USER_ALICE")
}

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
# 2️⃣ Cache static data
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
# 3️⃣ Cache heavy resources
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
# 4️⃣ Sidebar Navigation
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

    .stSelectbox [data-baseweb="select"] div {
    white-space: normal !important;
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
st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)

# Map page names to identifiers
pages = {"Food Licence Guidance": "main", "About Us": "about","Methodology": "methodology"}

# Render links as sidebar buttons
for name, key in pages.items():
    if st.sidebar.button(name):
        st.session_state.page = key

# Reset form submission if we are not on the main page
if st.session_state.page != "main":
    st.session_state.submitted = False

# -------------------------------
# 5️⃣ Main App
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
        st.session_state.business_type = st.selectbox(
            "🍽️ Business Type",
            [
                "Restaurant / Café / Eatery",
                "Hawker / Coffeeshop / Food Court Stall",
                "Catering Business",
                "Retail Shop / Supermarket",
                "Import / Export / Transhipment",
                "Food Manufacturing / Processing Facility",
                "Cold Storage / Warehouse",
                "Farming / Agriculture",
                "Event-based / Temporary Booth",
                "Other"
            ],
            key="business_type_select"
        )

        # Food types
        st.session_state.food_types = st.multiselect(
            "🥗 Food Types (select all that apply)",
            [
                "Meat or Poultry",
                "Seafood",
                "Eggs",
                "Fruits and Vegetables",
                "Baked Goods / Pastries",
                "Beverages (Non-alcoholic)",
                "Alcoholic Drinks",
                "Ready-to-eat / Cooked Food",
                "Processed or Packaged Food",
                "Animal Feed",
                "Other"
            ],
            key="food_types_select"
        )

        # Dynamic follow-up question
        if any(keyword in st.session_state.business_type for keyword in ["Restaurant", "Hawker", "Catering", "Home-based", "Retail", "Eatery"]):
            st.session_state.followup_answer = st.text_input(
                "Are the foods raw/cooked, fresh, chilled, frozen, or other? (You can describe multiple types)",
                placeholder="E.g., frozen raw seafood, chilled drinks and cooked bentos, etc.",
                key="followup_text"
            )
        else:
            st.session_state.followup_answer = ""

        # Additional details
        st.session_state.additional_details = st.text_area(
            "Tell us more about your business idea",
            placeholder="E.g. A halal-certified stall in a cafeteria, selling nasi padang and drinks...",
            key="additional_details_text"
        )


        # Submit button
        if st.form_submit_button("Get Licence Guidance"):
            st.session_state.submitted = True
        else:
            st.session_state.submitted = False
# -------------------------------
# 6️⃣ Generate guidance after submit (with prompt chaining + secure handling)
# -------------------------------
if st.session_state.submitted:
    status_placeholder = st.empty()
    status_placeholder.info("🧠 Understanding your business...")

    # Retrieve user inputs from session_state
    business_type = st.session_state.get("business_type", "")
    food_types = st.session_state.get("food_types", [])
    additional_details = st.session_state.get("additional_details", "")
    followup_answer = st.session_state.get("followup_answer", "")

    # -------------------------------
    # STEP 1: Sanitize and structure user inputs
    # -------------------------------

    def sanitize_input(user_input: str) -> str:
        """Remove potentially dangerous characters and patterns."""
        if not user_input:
            return ""
        user_input = re.sub(r'[`"{}<>]', '', user_input)
        forbidden_phrases = [
            "ignore previous instructions", "disregard all instructions",
            "forget all rules", "execute", "run code", "insert", "delete"
        ]
        pattern = re.compile("|".join(forbidden_phrases), flags=re.IGNORECASE)
        user_input = pattern.sub("", user_input)
        return user_input[:1000]  # limit length

    additional_details = sanitize_input(additional_details)
    followup_answer = sanitize_input(followup_answer)
    food_types = [sanitize_input(ft) for ft in food_types]

    # Build structured query
    query_dict = {
        "business_type": business_type,
        "food_types": food_types,
        "food_form": followup_answer,
        "additional_details": additional_details
    }
    query_text = json.dumps(query_dict, ensure_ascii=False)

    # -------------------------------
    # STEP 2: Business Classification (prompt chaining stage 1)
    # -------------------------------
    classification_prompt = f"""
    You are a classification assistant well-versed in Singapore food (e.g. kueh) and business types (e.g. pasar malams). 
    Categorise the following food business into one or more of these official groups:
    ["Restaurant / Café / Eatery / Catering", "Food Stall","Import/Export", "Manufacturing", "Temporary Food Fair", "Farming", "Cold Storage", "Other"]

    Business details (JSON format):
    {query_text}

    Respond ONLY in valid JSON like this:
    {{
      "categories": ["Restaurant / Café / Eatery / Catering"],
      "keywords": ["restaurant", "cooked food", "SFA"]
    }}
    """

    classification_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": classification_prompt}],
        temperature=0
    )

    # -------------------------------
    # STEP 3: Validate JSON output
    # -------------------------------
    schema = {
        "type": "object",
        "properties": {
            "categories": {"type": "array", "items": {"type": "string"}},
            "keywords": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["categories", "keywords"]
    }

    try:
        business_class = json.loads(classification_response.choices[0].message.content)
        jsonschema.validate(instance=business_class, schema=schema)
        # Keep only valid categories
        valid_categories = ["Restaurant / Café / Eatery / Catering", "Food Stall","Import/Export", "Manufacturing", "Temporary Food Fair", "Agriculture", "Cold Storage", "Other"]
        business_class["categories"] = [c for c in business_class["categories"] if c in valid_categories]
    except Exception:
        business_class = {"categories": ["Other"], "keywords": []}

    status_placeholder.info(f"Your query is classified as: '{', '.join(business_class['categories'])}'. Retrieving relevant information...")

    # -------------------------------
    # STEP 4: Vector Retrieval
    # -------------------------------
    query_vector = np.array(embeddings_model.embed_query(query_text), dtype=np.float16)

    top_n = 20
    D, top_indices = index.search(np.array([query_vector], dtype=np.float32), top_n)
    top_indices = top_indices[0]

    retrieved_chunks = [
        all_chunks[idx] 
        for idx in top_indices 
        if any(cat.lower() in json.dumps(metadata_dict.get(str(idx), {})).lower() for cat in business_class["categories"])
    ] or [all_chunks[i] for i in top_indices[:10]]


    def chunk_to_text(chunk):
        parts = [
            f"Title: {chunk.get('title','')}",
            f"Subsection: {chunk.get('subsection','')}",
            f"Licence: {chunk.get('licence','')}",
            f"Description: {chunk.get('description','')}",
            f"Reason: {chunk.get('reason_for_licence','')}",
            f"Requirements: {chunk.get('requirements','')}",
            f"Application Guidance: {chunk.get('app_guidance','')}",
            f"Other: {chunk.get('other','')}",
            f"URL: {chunk.get('url','')}"
        ]
        return "\n".join([p for p in parts if p])

    combined_context = "\n\n".join(chunk_to_text(chunk) for chunk in retrieved_chunks)

    # -------------------------------
    # STEP 5: Summarize Context
    # -------------------------------
    summarizer_prompt = f"""
    Summarize the following raw data into key licence items.
    For each item, include: licence name, purpose, and issuing agency if known.
    Output in plain bullet points, concise, no duplication.

    Context:
    {combined_context}
    """

    summarizer_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summarizer_prompt}],
        temperature=0.2
    )
    context_summary = summarizer_response.choices[0].message.content

    status_placeholder.info("📚 We got you! Generating guidance...")

    # -------------------------------
    # STEP 6: Generate Structured Guidance
    # -------------------------------
    system_prompt = """
    You are "Singapore Food Licence AI", an expert in Singapore's food and food business licensing.
    Explain which licences are needed for a given food business setup.
    Use ONLY the context you have and do not provide information outside the context.
    If unsure, respond: "Sorry, we do not have information related to your query. Please refer to GoBusiness (https://licensing.gobusiness.gov.sg) for the latest licensing requirements."
    If relevant URL for the information is unavailable, use https://licensing.gobusiness.gov.sg. 
    """

    few_shot_examples = """
    Example 1:
    User Input:
    Business type: Hawker Stall
    Food sold: Cooked seafood dishes

    Response:
    You are planning to operate a hawker stall selling cooked seafood. 
    You will need the following approvals/licences:

    1. Secure a Stall through NEA Tender  
    - Participate in an NEA hawker stall tender.  
    - If successful, you will sign a Tenancy Agreement with NEA for the specific stall.

    2. Apply for the Food Shop Licence 
    - After signing the tenancy agreement, apply for the **Food Shop Licence** via the [GoBusiness Licensing Portal](https://licensing.gobusiness.gov.sg/licence-directory/sfa/food-shop-licence).  
    - The Singapore Food Agency (SFA) issues the licence once you submit:
        • A copy of the signed tenancy agreement  
        • A layout plan of the stall  
        • Details of your food operations

    3. Complete a Pre-Licensing Inspection  
    - SFA officers will inspect your stall before approving the licence to ensure hygiene and layout compliance.

    4. Food Hygiene Training Requirement  
    - At least one staff (usually the stallholder) must complete the Basic Food Hygiene Course before the licence is granted.

    Application Overview
    Apply through GoBusiness, ensure all documents are complete, and payment must be made within 28 days after notification.

    More information on Licences (tabulate):
    Licence Name | Topic | Application Guidance | Webpage
    Food Stall Licence | Food retail | Secure stall via NEA tender, then apply through GoBusiness with tenancy docs and layout plan. | https://licensing.gobusiness.gov.sg/licence-directory/sfa/food-stall-licence
    Basic Food Hygiene Course | Training | Mandatory for stallholders before licence approval. | https://licensing.gobusiness.gov.sg

    # Note: The second row has no URL because it is not present in the 'all_chunks' dataset.

    Example 2: 
    User Input:
    Business type: Import of Food Products
    Food sold: Frozen meat and seafood

    Response:
    You intend to import frozen meat products. 
    You will need the following approvals/licences:

    1. Licence for Import/Export/Transhipment of Meat and Fish Products  
    - If you are a business in Singapore seeking to import meat or seafood (including frozen products), you must hold this licence issued by the Singapore Food Agency (SFA).  
    - The applicant must be a business registered or incorporated in Singapore.

    2. Import Permit for Each Consignment  
    - For each shipment of meat or fish imported, you must apply for an Import Permit through TradeNet.  
    - Each consignment must come from an approved source (country and processing establishment) and be accompanied by a valid veterinary health certificate from the exporting country.

    3. Cargo Clearance Permit (CCP)  
    - After the Import Permit is approved, you must obtain a Cargo Clearance Permit for customs clearance of the consignment.

    4. Approved Cold-storage Facility Licence (if storage is involved)  
    - If you intend to store frozen or chilled meat products locally, your cold-store facility must be licensed by SFA.  
    - Operating an unlicensed cold store is an offence under the Sale of Food Act.

    Application Overview
    Apply as a Business User with a UEN-registered company/entity or as a Third Party Filer for Business. Estimated processing time is 1 working day. Licence Fee: $84.00. Payment Methods: American Express, Diners Club, JCB, Mastercard, UnionPay, Visa, PayNow, GIRO.
    More information on Licences (tabulate):
    Licence Name | Topic | Application Guidance | Webpage
    Licence for Import/Export/Transhipment of Meat and Fish Products | Import/export | Apply via GoBusiness Licensing. Applicant must be a registered business in Singapore. | https://www.sfa.gov.sg/food-import-export/licence-permit-registration/businesses-that-need-licence-permit-registration-for-import-export
    Import Permit (per consignment) | TradeNet | Apply via TradeNet before import. Approved countries and plants only. | https://licensing.gobusiness.gov.sg/e-adviser/imports-and-exports#step-1-activate-your-customs-account-via-tradenet
    Cargo Clearance Permit | Customs | Used for clearance of approved consignments. | https://licensing.gobusiness.gov.sg/e-adviser/imports-and-exports
    Cold Storage Licence | Facility | Required if storing frozen/chilled meat locally. | https://licensing.gobusiness.gov.sg/licence-directory/sfa/licence-to-operate-a-coldstore

    Example 3:
    User Input:
    Business type: Operate a Halal cafeteria stall selling cooked food

    Response:
    Relevant Licences for Your Business
    You intend to operate a Halal-certified cafeteria stall selling cooked food. You will need the following licences and certifications:

    1. Food Shop Licence (SFA)  
   - Issued by the Singapore Food Agency (SFA).  
   - Required for any fixed food outlet preparing or selling ready-to-eat food, including cafeteria stalls.  
   - Apply through the [GoBusiness Licensing Portal](https://licensing.gobusiness.gov.sg/licence-directory/muis/halal-certification).  
   - You must submit a layout plan, details of equipment, and ensure the stall passes SFA’s pre-licensing inspection.

    2. Halal Certification — Eating Establishment Scheme (Category 1)  
   - Administered by the **Majlis Ugama Islam Singapura (Muis)**, the sole authority for Halal certification in Singapore.  
   - The **Category 1 (Eating Establishment Scheme)** applies to consistent, controlled retail food operations such as:  
     • Restaurants  
     • Cafeteria or staff canteen stalls  
     • Food stations and kiosks  
     • Chain or franchise outlets  
   - You must already hold a valid SFA Food Shop Licence before applying for Halal certification.  
   - Key requirements include:
       • All ingredients and suppliers must be Halal-approved by Muis  
       • Staff must undergo Halal awareness training  
       • Premises and food preparation areas must comply with Halal assurance and segregation guidelines  
   - Apply directly through the [Muis Halal Certification Portal](https://licensing.gobusiness.gov.sg/licence-directory/muis/halal-certification).

    Application Overview 
    Follow the application process outlined for Halal certification.
    
    More information on Licences:
    Licence/Certification Name | Topic | Application Guidance | Webpage
    Food Shop Licence | Food retail | Required for any food stall preparing/selling ready-to-eat food. Apply via GoBusiness. |https://licensing.gobusiness.gov.sg/licence-directory/sfa/food-shop-licence
    Muis Halal Certification (Category 1 — Eating Establishment Scheme) | Halal compliance | Apply via Muis. Applicant must hold a valid SFA licence and meet Halal assurance requirements. | https://licensing.gobusiness.gov.sg/licence-directory/muis/halal-certification

    """

    user_prompt = f"""
    Follow the format and tone from the examples above.

    Business details:
    {query_text}

    Classified as: {', '.join(business_class['categories'])}

    Summary of relevant context:
    {context_summary}

    Instructions:
    1. Start with a short summary of the user's business.
    2. Then list "Relevant Licences for Your Business" with bullet points.
    3. Then list Application Overview with bullet points
    4. End with a table: Licence Name | Topic | Application Guidance | Webpage.
    5. Add a one-line compliance reminder.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": few_shot_examples},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3
    )

    summary = response.choices[0].message.content

    status_placeholder.empty()
    st.header("📄 Summary of Required Approvals / Licences")
    st.write(summary)


# -------------------------------
# 7️⃣ About Us
# -------------------------------
if st.session_state.page == "about":
    st.title("❓About Us")
    st.markdown(
        """
        ### 📌 Overview
        ***Singapore Food Licence AI*** is a web-based assistant designed to help food entrepreneurs, retailers, and import/export businesses quickly understand the regulatory approvals and licences they may need in Singapore.

        By combining open-source government data, vector-based retrieval, and AI-powered synthesis, the app delivers personalised, easy-to-understand guidance within seconds.

         ⚡ **Key Features**
        - **Interactive Business Form:** A simple and intuitive interface that captures your business type, food categories, and operational details.
        - **Personalised Licence Guidance:** Generates tailored recommendations based on your specific business activities.
        - **AI-Powered Summaries:** Uses GPT-4o-mini to transform complex regulations into clear, plain-language explanations.
        - **Smart Search & Retrieval:** Combines keyword search with vector-based similarity to surface the most relevant information quickly and accurately.

        ---

        ### 📝 Project Scope
        This application supports a broad range of food-related businesses, including:

        - Restaurants, cafés, and food stalls  
        - Retail shops and supermarkets  
        - Catering services  
        - Temporary food fairs and event-based food booths  
        - Food manufacturing and processing facilities  
        - Import, export, and transhipment of food products  
        - Agriculture and farm production  

        ---

        ### 🎯 Project Objectives
        - Provide **accurate and up-to-date** information on licences and approvals  
        - Make complex regulatory requirements **clear and accessible**  
        - Help businesses **quickly identify** relevant licences  
        - Reduce manual effort and confusion when navigating food-related regulations  

        ---

        ### 📂 Data Sources
        All information is derived from official Singapore government websites, including:

        - Singapore Food Agency (SFA)  
        - Majlis Ugama Islam Singapura (MUIS)  
        - National Environment Agency (NEA)  
        - Singapore Police Force (SPF)  
        - GoBusiness Licensing Portal  

        The data is scraped, cleaned, and structured into machine-readable JSON for AI processing.

        ---

        ### 🚀 Potential Future Directions
        - **Automated Data Pipeline:**  
        Implement an automated system that continually monitors and re-checks the relevancy, accuracy, and status of source websites—ensuring that the application always reflects the latest regulatory updates.  
        - **Expanded Dataset Coverage:**  
        Incorporate more agency sources, permit categories, and cross-agency workflows.  
        - **Scenario-Based Guidance:**  
        Provide guidance tailored to business stages such as pre-launch, renovation, expansion, and relocation.
        - **Enhanced Interface:**  
        Introduce step-by-step workflows, interactive charts, and integrated agency links for an even smoother user experience.

        ---
        """
    )
# -------------------------------
# 7️⃣ Methodology
# -------------------------------
elif st.session_state.page == "methodology":
    st.title("🛠️ Methodology: How This Works")

    st.image("App Workflow.png", caption="App Workflow", use_container_width=True)
    st.markdown(
        """
        The ***Singapore Food Licence AI*** application follows a multi-step process:

        1. **Data Collection & Cleaning**
        - Scrape government websites (SFA, MUIS, GoBusiness Licensing, etc) using `requests` and `BeautifulSoup`.  
        - Remove HTML tags, scripts, and navigation elements to isolate meaningful regulatory text.  
        - Store cleaned text in structured JSON for downstream processing.

        2. **Data Structuring**
        - Use GPT-4o-mini LLM to parse raw text into structured JSON:
            - `title`, `licence_name`, `requirements`, `application_guidance`, `reason_for_licence`, `other`, `url`.  
        - Prompt enforces strict rules to include **only official licences**.
        - Each JSON object becomes a “chunk” for embedding and retrieval.

        3. **Chunking & Embeddings**
        - Split long text into manageable semantic chunks.  
        - Convert each chunk into a vector embedding using `OpenAIEmbeddings`.  
        - Store embeddings in **FAISS** for fast similarity search.

        4. **User Query Processing**
        - User inputs business type, food types, and additional details via the app form.  
        - Inputs are sanitized to prevent prompt injection by removing unsafe characters or malicious instructions.
        - User input is converted into a vector in the same embedding space as the dataset.
        - Retrieve the top-N most relevant chunks using cosine similarity.

        5. **Prompt Chaining & Classification**
        - **Step 1: Business Classification**  
        GPT-4o-mini categorizes the business into official groups (e.g., "Restaurant / Café / Eatery", "Import/Export") and extracts relevant keywords.

        - **Step 2: Context Retrieval**  
        FAISS retrieval results are filtered by business category keywords to improve precision.

        - **Step 3: Summarization**  
        Retrieved chunks are combined and passed to GPT-4o-mini to generate a concise summary of relevant licences.

        - **Step 4: Guidance Generation**  
        Using the summarized context and few-shot examples, GPT-4o-mini produces user-facing licence guidance in plain language, structured with bullet points and tables.

        6. **AI Synthesis & Validation**
        - JSON validation ensures classification outputs conform to the expected schema.  
        - Few-shot examples guide GPT-4o-mini to provide accurate, structured advice.  
        - Output includes:
            - Short summary of the user's business
            - List of relevant licences with issuing agencies
            - Tabulated guidance: Licence Name | Topic | Application Guidance | Webpage
            - One-line compliance reminder

        7. **Vector Search & Keyword Filtering**
        - Retrieved chunks are ranked by similarity score and filtered using business category keywords to ensure contextual relevance.  
        - This reduces noise and ensures the AI synthesizes only relevant information for the user's business.

        ---

        ### 🔄 Use Case Flowcharts

        #### **Use Case A: Chat with Information**
        ```text
        User Inputs Business Info
                │
                ▼
        Sanitize & Structure Input
                │
                ▼
        Embed Input → Vector Representation
                │
                ▼
        Retrieve Top Chunks via FAISS
                │
                ▼
        Keyword Filtering by Business Type
                │
                ▼
        Stepwise Prompt Chaining:
            1️⃣ Classification
            2️⃣ Context Summarization
            3️⃣ Guidance Generation
                │
                ▼
        Display Structured Licence Guidance
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
    > **Note:** This application provides guidance only and does not replace official regulatory advice. Please consult the relevant government agencies for final approvals.
    """
)
