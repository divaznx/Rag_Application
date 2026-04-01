🧠 Knowledge Base Manager Project
The Knowledge Base Manager project is a comprehensive tool designed to manage and query a knowledge base using ChromaDB for retrieval and OpenAI for generation. The project provides a command-line interface for ingesting folders, checking the status of the knowledge base, and deleting folders. It also allows users to query the knowledge base interactively or using a single question.

🚀 Features
Ingesting folders and processing files based on their extensions
Extracting text from files and storing it in a ChromaDB database
Providing a command-line interface for managing the knowledge base
Using OpenAI's text-embedding-3-small model for text embedding
Querying the knowledge base using ChromaDB for retrieval
Generating answers using OpenAI's API
🛠️ Tech Stack
Frontend: None
Backend: Python
Database: ChromaDB
AI Tools: OpenAI
Build Tools: None
Libraries:
chromadb for database operations
embedder for text embedding
argparse for command-line argument parsing
pathlib for file path manipulation
json and hashlib for data processing
openai for interacting with OpenAI's API
dotenv for loading environment variables from a .env file
📦 Installation
To install the project, follow these steps:

Clone the repository using git clone
Install the required libraries using pip install -r requirements.txt
Set up the environment variables in a .env file
💻 Usage
To use the project, follow these steps:

Run the kb_manager.py file to ingest folders and manage the knowledge base
Run the rag_query.py file to query the knowledge base interactively or using a single question
📂 Project Structure
.
├── main.py
├── kb_manager.py
├── embedder.py
├── rag_query.py
├── requirements.txt
└── .env
