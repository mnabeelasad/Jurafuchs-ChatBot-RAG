# Full-Stack RAG Chatbot (FastAPI + Azure AI)

This is a full-stack, Retrieval-Augmented Generation (RAG) chatbot. It allows a user to upload a PDF document, which is then processed, chunked, and vectorized into an Azure AI Search index. The user can then have a real-time conversation with the document, with all answers backed by source citations.

The backend is built with **FastAPI** and connects to **Azure OpenAI** and **Azure AI Search**. The frontend is a clean, single-page chat interface built with **HTML and Vanilla JavaScript**.

## 📸 App Screenshot

<img width="808" height="904" alt="image" src="https://github.com/user-attachments/assets/63d11322-a319-49f1-86f9-9077ef475f6e" />

*(Note: You'll need to drag your screenshot file into the same directory as your README.md for this to show up on GitHub)*

---

## ✨ Core Features

* **PDF Document Upload:** Ingests and processes new PDF files on the fly.
* **Dynamic RAG Pipeline:** Creates a new, separate search index for each uploaded document.
* **Citable Answers:** Provides the source page number for every answer.
* **Hybrid Search:** Uses Azure AI Search (`search_type="hybrid"`) to combine keyword and vector search for accurate results.
* **Modern UI:** A clean, dark-mode chat interface with left/right message alignment.
* **Markdown Rendering:** Bot responses are parsed as Markdown to show lists, bolding, etc.
* **Speech-to-Text:** Click the microphone to provide voice input (via the browser's Web Speech API).
* **Text-to-Speech:** Click the speaker icon on any bot message to hear it read aloud.
* **Secure Secrets:** All API keys are loaded from environment variables (`.env` locally, Render Environment Variables in production).

---

## 💻 Tech Stack

| Area | Technology |
| :--- | :--- |
| **Backend** | Python 3.12, FastAPI, Gunicorn |
| **AI / RAG** | LangChain, Azure OpenAI, Azure AI Search |
| **Frontend** | HTML, Tailwind CSS (via CDN), Vanilla JavaScript |
| **Deployment** | Render (Web Service + Static Site) |
| **DevOps** | Git, GitHub (for Push-to-Deploy) |

---

## 🚀 How to Run Locally

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd jurafuchs-app
    ```

2.  **Create Environment & Install Dependencies**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On Mac/Linux:
    source venv/bin/activate

    pip install -r requirements.txt
    ```

3.  **Set Up Secrets**
    Create a file named `.env` in the root folder. Add your 8 secret keys and endpoints:
    ```ini
    AZURE_OPENAI_ENDPOINT=...
    AZURE_OPENAI_API_KEY=...
    AZURE_OPENAI_CHAT_DEPLOYMENT=...
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT=...
    OPENAI_API_VERSION=...
    AZURE_AI_SEARCH_ENDPOINT=...
    AZURE_AI_SEARCH_KEY=...
    PYTHON_VERSION=3.12.0
    ```

4.  **Run the Backend**
    ```bash
    uvicorn main:app --reload
    ```
    The server will be running at `http://127.0.0.1:8000`.

5.  **Run the Frontend**
    No server needed. Just open the `index.html` file directly in your browser. It is already configured to talk to your local backend.

---

## ☁️ Deployment on Render

This project is deployed as two separate services on Render.

### 1. Backend (Web Service)
* **Runtime:** `Python 3`
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app -b 0.0.0.0:$PORT`
* **Environment:**
    * Add all 8 secret variables from your `.env` file to the Render **Environment** tab.

### 2. Frontend (Static Site)
* **Update API URLs:** Before deploying, you **must** change the `uploadApiUrl` and `chatApiUrl` variables in `index.html` to point to your live backend URL.
* **Build Command:** (Leave blank)
* **Publish Directory:** `.` (a single dot)
