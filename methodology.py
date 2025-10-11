import streamlit as st

st.title("Methodology")

st.markdown("""
Here’s an overview of the methodology behind this app:

### 1. Data Collection
- Web scraping relevant pages using libraries like `requests` and `BeautifulSoup`.
- Extracting the necessary text content while cleaning out HTML tags, ads, and navigation menus.

### 2. Data Chunking
- Splitting large documents into smaller, manageable chunks for easier processing.
- Helps improve performance for embedding and search.

### 3. Vector Indexing
- Converting text chunks into vector representations using embeddings (e.g., OpenAI embeddings).
- Storing them in a vector database (like FAISS, Pinecone, or Chroma) for efficient retrieval.

### 4. Similarity Search
- When a user asks a question, their query is converted into a vector.
- The system retrieves the most similar chunks from the vector database using cosine similarity.

### 5. Response Generation
- Retrieved chunks are fed to a language model (e.g., GPT) to generate a coherent and relevant answer.
- Optional: use prompts to provide context and ensure accurate responses.

### Tools & Libraries
- `requests`, `BeautifulSoup`, `pandas` for scraping & data handling.
- `nltk` or `spacy` for text preprocessing.
- `faiss` or `pinecone` for vector storage and similarity search.
- `streamlit` for the user interface.
""")
