from google import genai
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np


print("=" * 70)
print("               PDF QUESTION ANSWERING SYSTEM (RAG)")
print("=" * 70)
print("Loading PDF, Creating Embeddings and Building FAISS Index...")
print("=" * 70)

try:

    # PDF reader
    reader = PdfReader("sample.pdf")

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    print("\nPDF Loaded Successfully.")

    # Dividing into chunks
    chunks = text.split("\n")

    print(f"\nTotal Chunks Created : {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print("-" * 70)
        print(f"CHUNK : {i}")
        print(chunk)

    print("\n" + "=" * 70)
    print("Loading Embedding Model...")
    print("=" * 70)

    # Embedding model loading
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding Model Loaded Successfully.")

    # Creating embeddings
    embeddings = model.encode(chunks)

    # FAISS accepts float32
    embeddings = np.array(embeddings).astype("float32")
    print(f"TOTAL EMBEDDINGS : {len(embeddings)}")

    print("\nCreating FAISS Index...")

    # Creating FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Storing embeddings
    index.add(embeddings)
    print("Embeddings Stored Successfully.")

except Exception as e:
    print(f"\nError during setup: {e}")
    exit()


print("\n" + "=" * 70)
print("SYSTEM IS READY")
print("=" * 70)
print("Type your question below.")
print("Type 'exit' anytime to close the application.")
print("=" * 70)


# Gemini calling
load_dotenv()

client = genai.Client(
    api_key=os.getenv("API_KEY"),
)


while True:
    try:

        print("\n" + "-" * 70)

        # Convert user questions into embeddings
        query = input("Ask Question : ")

        if query.lower() == "exit":
            print("\nThank you for using PDF Question Answering System.")
            print("Goodbye!")
            break

        print("\nSearching Relevant Context...")

        query_embedding = model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        # Similarity search
        distances, indices = index.search(query_embedding, k=2)

        # Showing relevant chunks
        relevant_chunks = []

        for i in indices[0]:
            relevant_chunks.append(chunks[i])

        context = "\n".join(relevant_chunks)

        print("Generating Answer...\n")

        prompt = f"""Give me the answer according to my questions by following the rules
1. don't use technical jargons,
2. keep it simple and short,
3. answer in max 3 lines
4. qst : {query}
5. context : {context}"""

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        print("=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(response.text)
        print("=" * 70)

    except Exception as e:
        print(f"\nSOMETHING WENT WRONG : {e}")


print("THANK YOU FOR USING OUR SYSTEM")