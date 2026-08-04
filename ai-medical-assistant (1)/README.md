# 🏥 AI Medical Assistant Backend

A comprehensive FastAPI backend for an AI-powered medical assistant with multilingual support, medical knowledge base, and PDF report generation.

## ✨ Features

### 🤖 Medical AI Services
- **Medical Question Validation**: Ensures only health-related questions are processed
- **Semantic Similarity Matching**: Uses sentence transformers for accurate medical advice
- **LangChain Integration**: Advanced conversation handling with Ollama LLM
- **Google Generative AI**: Enhanced Tamil-English medical translation

### 🌍 Multilingual Support
- **13 Supported Languages**: English, Hindi, Tamil, Telugu, Gujarati, Korean, Turkish, German, French, Arabic, Urdu, Chinese, Japanese
- **Medical Context Translation**: Specialized medical translation preserving context
- **Language Detection**: Automatic language detection with confidence scoring

### 📊 Advanced Features
- **Conversation Analysis**: AI-powered insights and pattern recognition
- **PDF Report Generation**: Comprehensive medical reports with ReportLab
- **Health Tips**: Personalized health recommendations
- **Medical Knowledge Search**: Searchable medical knowledge base

### 🔧 Technical Features
- **Modular Architecture**: Clean separation of concerns with service layers
- **Async/Await Support**: High-performance asynchronous operations
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Health Checks**: Built-in health monitoring endpoints
- **Docker Support**: Containerized deployment ready

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip or poetry
- Optional: Docker and Docker Compose

### Installation

1. **Clone the repository:**
\`\`\`bash
git clone <repository-url>
cd ai-medical-backend
\`\`\`

2. **Install dependencies:**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. **Set up environment variables:**
\`\`\`bash
cp .env.example .env
# Edit .env with your configuration
\`\`\`

4. **Run the application:**
\`\`\`bash
python main.py
\`\`\`

The API will be available at `http://localhost:8000`

### Docker Deployment

1. **Build and run with Docker Compose:**
\`\`\`bash
docker-compose up --build
\`\`\`

2. **Or build and run manually:**
\`\`\`bash
docker build -t medical-backend .
docker run -p 8000:8000 medical-backend
\`\`\`

## 📋 API Endpoints

### Health & Status
- `GET /` - API information
- `GET /health` - Health check with service status

### Language Services
- `POST /api/detect-language` - Detect text language
- `POST /api/translate` - Translate text with medical context

### Medical Services
- `POST /api/chat` - Main medical consultation endpoint
- `POST /api/validate-medical` - Validate if question is medical
- `POST /api/search-medical-knowledge` - Search medical knowledge base
- `GET /api/health-tips` - Get personalized health tips

### Analysis & Reports
- `POST /api/analyze-conversation` - Analyze conversation patterns
- `POST /api/generate-report` - Generate PDF medical report

## 🏗️ Architecture

\`\`\`
├── main.py                 # FastAPI application entry point
├── models/
│   └── schemas.py         # Pydantic models for requests/responses
├── services/
│   ├── medical_service.py # Core medical AI logic
│   ├── translation_service.py # Translation and language handling
│   ├── language_service.py # Language detection
│   └── report_service.py  # PDF report generation
├── config/
│   └── settings.py        # Application configuration
├── utils/
│   └── logger.py          # Logging utilities
└── requirements.txt       # Python dependencies
\`\`\`

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `DEBUG` | Debug mode | `true` |
| `GEMINI_API_KEY` | Google AI API key | `""` |
| `OLLAMA_MODEL` | Ollama model name | `mistral` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Medical Configuration
- **Similarity Threshold**: `0.5` for semantic matching
- **Max Chat History**: `100` messages
- **Supported Languages**: 13 languages with medical keyword support

## 🧪 Testing

### Manual Testing
\`\`\`bash
# Health check
curl http://localhost:8000/health

# Medical chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I have a headache", "language": "en"}'

# Language detection
curl -X POST http://localhost:8000/api/detect-language \
  -H "Content-Type: application/json" \
  -d '{"text": "I have a fever"}'
\`\`\`

### API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔒 Security Features

- **Medical Validation**: Only processes health-related questions
- **Input Sanitization**: Cleans and validates all inputs
- **CORS Protection**: Configurable CORS origins
- **Error Handling**: Comprehensive error handling and logging
- **Rate Limiting**: Built-in protection against abuse

## 📊 Monitoring & Logging

### Health Checks
- Service health monitoring
- Individual component status
- Dependency health verification

### Logging
- Structured logging with timestamps
- Configurable log levels
- Service-specific loggers
- Error tracking and debugging

## 🚀 Deployment

### Production Deployment
1. **Set environment variables for production**
2. **Use a production WSGI server** (included with uvicorn)
3. **Configure reverse proxy** (nginx recommended)
4. **Set up monitoring** and log aggregation
5. **Configure SSL/TLS** certificates

### Scaling
- **Horizontal scaling**: Multiple container instances
- **Load balancing**: Distribute requests across instances
- **Database**: Add persistent storage for chat history
- **Caching**: Redis for improved performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with proper tests
4. Follow the existing code style and patterns
5. Submit a pull request with clear description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Medical Disclaimer

This application is for educational and informational purposes only. The AI assistant provides general health information and should not be considered as medical advice, diagnosis, or treatment.

**Always consult qualified healthcare professionals for:**
- Medical diagnosis and treatment
- Emergency medical situations
- Prescription medications
- Serious health concerns

## 🆘 Support

For support and questions:
1. Check the API documentation at `/docs`
2. Review the logs for error messages
3. Ensure all dependencies are properly installed
4. Verify environment configuration
5. Check the health endpoint for service status

## 🔄 Updates

This project is actively maintained. Check for updates regularly and follow the changelog for new features and improvements.
