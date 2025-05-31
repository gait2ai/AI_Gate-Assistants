# AI Gate 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)

> **AI-powered conversational gateway for institutional knowledge bases**

AI Gate is a sophisticated, modular conversational interface that serves as an intelligent bridge between users and institutional knowledge systems. Leveraging advanced language models through OpenRouter, it provides contextually aware responses by seamlessly integrating with pre-processed institutional data.

## ✨ Key Features

- **🏗️ Modular Architecture** - Clean separation of concerns with testable, maintainable components
- **🤖 AI-Powered Intelligence** - Harnesses large language models via OpenRouter for natural language understanding
- **🏛️ Institution-Agnostic Design** - Easily configurable for different organizations through YAML configuration
- **📚 Knowledge Base Integration** - Seamlessly retrieves information from pre-processed institutional content
- **⚡ Intelligent Caching** - Optimized performance through strategic caching of queries and data
- **🌐 Multi-Language Support** - Built-in language detection and localized response capabilities
- **🚀 FastAPI Backend** - High-performance, modern API with automatic documentation
- **💻 Clean Web Interface** - Responsive, user-friendly chat interface
- **📊 Comprehensive Logging** - Detailed application monitoring and debugging capabilities
- **🔧 Environment Validation** - Automatic dependency and configuration checking

## 🏗️ Architecture Overview

The application follows a modular design pattern with distinct responsibilities:

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Question          │    │   Website            │    │   Prompt            │
│   Processor         │───▶│   Researcher         │───▶│   Builder           │
│                     │    │                      │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
          │                                                        │
          ▼                                                        ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Cache             │    │   AI Handler         │◀───│   Main Application  │
│   Manager           │    │   (OpenRouter)       │    │   (FastAPI)         │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### Core Components

| Component | Responsibility |
|-----------|----------------|
| **Question Processor** | Input validation, topic extraction, language detection |
| **Website Researcher** | Content search and retrieval from knowledge base |
| **Prompt Builder** | Dynamic system prompt construction |
| **AI Handler** | OpenRouter API communication with fallback logic |
| **Cache Manager** | Intelligent caching strategies |
| **Utils** | Shared utilities and configuration management |

## 🛠️ Technology Stack

- **Backend**: Python 3.8+, FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **AI Integration**: OpenRouter API
- **Configuration**: YAML, Environment Variables
- **Data Storage**: JSON, In-memory caching
- **Optional NLP**: NLTK, langdetect, scikit-learn
- **Testing**: pytest, pytest-asyncio

## 📁 Project Structure

```
ai-gate/
├── 📄 main.py                    # FastAPI application entry point
├── 📋 requirements.txt           # Python dependencies
├── 📖 README.md                  # Project documentation
├── 🔧 .env.example              # Environment template
├── 🧩 modules/                  # Core application logic
│   ├── question_processor.py    # Input processing & validation
│   ├── website_researcher.py    # Knowledge base search
│   ├── prompt_builder.py        # Dynamic prompt construction
│   ├── ai_handler.py            # AI model communication
│   ├── cache_manager.py         # Caching system
│   └── utils.py                 # Shared utilities
├── ⚙️ config/                   # Configuration files
│   ├── default.yaml             # Base configuration
│   ├── institution.yaml         # Institution-specific settings
│   └── system_prompt.txt        # AI system prompt template
├── 💾 data/                     # Data storage
│   ├── pages.json              # Processed knowledge base
│   └── cache/                  # Temporary cache files
├── 🌐 static/                   # Web interface
│   ├── index.html              # Main interface
│   ├── css/style.css           # Styling
│   └── js/app.js               # Frontend logic
├── 📊 logs/                     # Application logs
├── 🔧 scripts/                  # Utility scripts
│   └── website_scraper.py      # Content extraction tool
└── 🧪 tests/                    # Test suite
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed on your system
- **OpenRouter API key** ([Get one here](https://openrouter.ai/))
- **Git** for repository management

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-gate.git
   cd ai-gate
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your OpenRouter API key:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   OPENROUTER_API_URL=https://openrouter.ai/api/v1
   ```

5. **Prepare knowledge base**
   
   Generate your institution's knowledge base:
   ```bash
   python scripts/website_scraper.py --url https://your-institution.com --output data/pages.json
   ```

6. **Configure your institution**
   
   Edit `config/institution.yaml` with your organization's details:
   ```yaml
   institution:
     name: "Your Institution Name"
     description: "Brief description of your organization"
     # Add other institution-specific settings
   ```

7. **Launch the application**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

8. **Access the interface**
   
   Open your browser to: `http://localhost:8000`

## 📖 Usage

### Web Interface
- Navigate to the application URL in your browser
- Type questions about your institution in the chat interface
- Receive AI-powered responses based on your knowledge base

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send chat messages |
| `/api/institution` | GET | Get institution information |
| `/health` | GET | Application health status |
| `/api/stats` | GET | Usage statistics |
| `/api/docs` | GET | Interactive API documentation |

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | ✅ |
| `OPENROUTER_API_URL` | OpenRouter API endpoint | ❌ |
| `AIGATE_LOGGING_LEVEL` | Logging verbosity | ❌ |

### Configuration Files

- **`config/default.yaml`** - Base application settings
- **`config/institution.yaml`** - Institution-specific configuration
- **`config/system_prompt.txt`** - AI behavior and personality
- **`config/local.yaml`** - Local development overrides (gitignored)

## 🧪 Testing

Run the complete test suite:
```bash
pytest
```

Run specific test files:
```bash
pytest tests/test_question_processor.py
```

Run with coverage:
```bash
pytest --cov=modules --cov-report=html
```

## 🚀 Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations
- Use environment-specific configuration files
- Implement proper logging and monitoring
- Set up reverse proxy (nginx/Apache)
- Configure SSL certificates
- Use production ASGI server (gunicorn + uvicorn workers)

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenRouter](https://openrouter.ai/) - AI model access platform
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Uvicorn](https://www.uvicorn.org/) - Lightning-fast ASGI server

## 📞 Support

- 📖 [Documentation](https://github.com/yourusername/ai-gate/wiki)
- 🐛 [Issue Tracker](https://github.com/yourusername/ai-gate/issues)
- 💬 [Discussions](https://github.com/yourusername/ai-gate/discussions)

---

<div align="center">

**Built with ❤️ for institutional knowledge sharing**

[⭐ Star this repo](https://github.com/yourusername/ai-gate) • [🍴 Fork it](https://github.com/yourusername/ai-gate/fork) • [📝 Report Issues](https://github.com/yourusername/ai-gate/issues)

</div>