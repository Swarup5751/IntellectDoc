# 📚 Document AI Chatbot

A powerful AI-powered chatbot that can process PDF and DOCX documents, provide intelligent summaries, extract topics, generate MCQs, and answer questions using LangChain and Google Gemini.

## ✨ Features

- **📄 Document Processing**: Upload and process PDF and DOCX files
- **🧠 AI-Powered Analysis**: Get intelligent summaries and topic extraction
- **❓ MCQ Generation**: Automatically generate multiple choice questions from documents
- **💬 Interactive Chat**: Ask questions about your documents and get contextual answers
- **🔍 Vector Search**: Advanced similarity search using embeddings
- **🌐 Web Interface**: Beautiful Streamlit web application
- **⚡ CLI Interface**: Command-line interface for automation and testing
- **📊 Analytics**: Comprehensive statistics and document insights

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key

### Getting Your API Key

1. **Visit Google AI Studio**
   - Go to [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account

2. **Create API Key**
   - Click "Create API Key"
   - Choose "Create API Key in new project" or select existing project
   - Copy the generated key (starts with `AIza...`)

3. **Save Your Key**
   - Keep it secure - you won't see it again
   - If lost, create a new one

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd document-ai-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API key** (Choose one method):

   **Method A: Interactive Setup**
   ```bash
   python setup_api_key.py
   ```

   **Method B: Manual Setup**
   ```bash
   # Copy the example environment file
   cp env_example.txt .env
   
   # Edit .env and add your Google Gemini API key
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   **Method C: Set in App**
   - Run the app and enter your API key in the sidebar

### Usage

#### Web Interface (Recommended)

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and go to `http://localhost:8501`

3. **Upload documents** and start chatting!

#### Command Line Interface

```bash
# Upload a document
python cli.py --api-key YOUR_API_KEY upload document.pdf

# Generate summary
python cli.py --api-key YOUR_API_KEY summary

# Extract topics
python cli.py --api-key YOUR_API_KEY topics

# Generate MCQs
python cli.py --api-key YOUR_API_KEY mcqs --count 5

# Interactive chat mode
python cli.py --api-key YOUR_API_KEY chat

# Full document analysis
python cli.py --api-key YOUR_API_KEY analyze

# View statistics
python cli.py --api-key YOUR_API_KEY stats
```

## 🏗️ Architecture

The application is built with a modular architecture:

### Core Components

1. **DocumentProcessor** (`document_processor.py`)
   - Handles PDF and DOCX file processing
   - Text extraction and chunking
   - Metadata management

2. **VectorStore** (`vector_store.py`)
   - ChromaDB integration for vector storage
   - Google Gemini embeddings for semantic search
   - Similarity search functionality

3. **AIAnalyzer** (`ai_analyzer.py`)
   - LangChain integration for AI operations
   - Summarization, topic extraction, MCQ generation
   - Question answering capabilities

4. **DocumentChatbot** (`chatbot.py`)
   - Main orchestrator class
   - Integrates all components
   - Provides high-level API

### Interfaces

- **Web Interface** (`app.py`): Streamlit-based web application
- **CLI Interface** (`cli.py`): Command-line tool for automation

## 📖 API Reference

### DocumentChatbot Class

```python
from chatbot import DocumentChatbot

# Initialize chatbot
chatbot = DocumentChatbot(
    db_path="./chroma_db",
    embedding_model="models/embedding-001",
    llm_model="gemini-1.5-pro"
)

# Upload document
result = chatbot.upload_document("document.pdf")

# Get summary
summary = chatbot.get_summary()

# Extract topics
topics = chatbot.get_topics()

# Generate MCQs
mcqs = chatbot.generate_mcqs(num_questions=5)

# Ask questions
answer = chatbot.ask_question("What is the main topic?")

# Get comprehensive analysis
analysis = chatbot.get_document_analysis()

# Get statistics
stats = chatbot.get_stats()
```

## 🔧 Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Your Google Gemini API key (required)
- `CHROMA_DB_PATH`: Vector database storage path
- `EMBEDDING_MODEL`: Embedding model name
- `LLM_MODEL`: Language model name

### Model Options

- **Embedding Models**: `models/embedding-001`
- **Language Models**: `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-pro`

## 📊 Features in Detail

### Document Processing
- Supports PDF and DOCX formats
- Automatic text extraction and cleaning
- Intelligent chunking with overlap
- Metadata preservation

### AI Analysis
- **Summarization**: Generate comprehensive summaries
- **Topic Extraction**: Identify main themes and topics
- **MCQ Generation**: Create multiple choice questions with explanations
- **Question Answering**: Context-aware responses

### Vector Search
- Semantic similarity search
- Context-aware document retrieval
- Configurable search parameters
- Persistent vector storage

### Web Interface
- Modern, responsive design
- Real-time chat interface
- Document upload and management
- Interactive analysis tools
- Statistics and monitoring

## 🛠️ Development

### Project Structure
```
document-ai-chatbot/
├── app.py                 # Streamlit web application
├── cli.py                 # Command-line interface
├── chatbot.py             # Main chatbot class
├── document_processor.py  # Document processing
├── vector_store.py        # Vector database operations
├── ai_analyzer.py         # AI analysis functions
├── requirements.txt       # Python dependencies
├── env_example.txt        # Environment variables template
└── README.md             # This file
```

### Adding New Features

1. **New Document Format**: Extend `DocumentProcessor` class
2. **New AI Model**: Modify `AIAnalyzer` class
3. **New Vector Store**: Implement new `VectorStore` class
4. **New Interface**: Create new interface file

### Testing

```bash
# Test document processing
python cli.py upload test_document.pdf

# Test AI features
python cli.py summary
python cli.py topics
python cli.py mcqs

# Test chat functionality
python cli.py chat
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the AI framework
- [Google Gemini](https://ai.google.dev/) for the language models
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the web interface

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed information
3. Include error messages and system information

## 🔮 Roadmap

- [ ] Support for more document formats (TXT, RTF, etc.)
- [ ] Multi-language support
- [ ] Advanced analytics and insights
- [ ] Export functionality (PDF reports, etc.)
- [ ] User authentication and multi-user support
- [ ] API endpoints for integration
- [ ] Mobile app support 