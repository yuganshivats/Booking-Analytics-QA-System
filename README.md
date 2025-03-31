# Booking-Analytics-QA-System
A system that processes hotel booking data, extracts insights, and enables retrieval-augmented question answering (RAG). The system would provide analytics and answer user queries about the data.
## Features

- **Chat Endpoint (`/chat`)**: Users can ask hotel-related questions (e.g., "What are the most popular hotels?") and receive AI-generated responses powered by RAG.
- **Visualization Endpoint (`/visualize`)**: Retrieves and displays hotel analytics from `hotel_analytics.json` to provide data-driven insights.

## Tech Stack

- **Flask**: Lightweight web framework for API development.
- **LangChain**: Framework for integrating LLMs with external data, used for RAG.
  - **Chroma**: Vector store for document retrieval.
  - **Ollama**: Embedding model (`nomic-embed-text:latest`) for text-to-vector conversion.
  - **Groq**: LLM (`llama-3.1-8b-instant`) for generating responses.
- **dotenv**: Manages environment variables securely.

## Prerequisites

- Python 3.8 or higher (Tested on 3.12.0)
- Git
- Groq API key (available from [Groq Console](https://console.groq.com/keys))
- (Optional) Postman for API testing

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yuganshivats/Hotel_Booking_Analytics.git
   cd Hotel_Booking_Analytics
   ```
2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set Up Environment Variables:**
   Create a `.env` file in the root directory and add:
   ```bash
   GROQ_API_KEY=your_api_key_here
   ```

### Running the Application

1. **Generate Analytics Data:**
   ```bash
   python analysis.py
   ```
2. **Prepare the RAG Model:**
   ```bash
   python rag_bot.py
   ```
3. **Start the Flask Application:**
   ```bash
   python main.py
   ```

## API Endpoints

### `/visualize` (GET)
- **Description:** Fetches analytics data from `hotel_analytics.json`.
- **URL:** `http://localhost:5000/visualize`
- **Response:** JSON data (or 404 error if the file is missing).

### `/chat` (POST)
- **Description:** Accepts user queries related to hotel bookings and returns AI-generated responses.
- **URL:** `http://localhost:5000/chat`
- **Request Body:** JSON object containing a `question` field.
- **Headers:**
  - `Content-Type: application/json`
  - `Authorization: Bearer <GROQ_API_KEY>` (if required)
- **Example Request:**
  ```json
  {
    "question": "What is the cancellation count?"
  }
  ```

## Testing with Postman

1. **Visualize Endpoint:**
   - Method: `GET`
   - URL: `http://localhost:5000/visualize`
   - Expected Response: JSON analytics data or 404 error if the file is missing.

2. **Chat Endpoint:**
   - Method: `POST`
   - URL: `http://localhost:5000/chat`
   - Headers: Include `GROQ_API_KEY`.
     ```json
     {
       "question": "What are the most popular hotels?"
     }
     ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Flask](https://flask.palletsprojects.com/) - Web framework.
- [LangChain](https://www.langchain.com/) - RAG capabilities.
- [Groq](https://groq.com/) - Language model provider.
- [Ollama](https://ollama.com/) - Embeddings.
- [Chroma](https://www.trychroma.com/) - Vector storage.

## Contributing

Contributions are welcome! Feel free to submit a pull request, ensuring adherence to best practices and providing a clear description of changes.
