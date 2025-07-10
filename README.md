# ☁️ CloudyBot: AI-Powered DevOps Assistant

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io)

**CloudyBot is a sophisticated AI-powered assistant designed specifically for DevOps professionals and teams.**

</div>

## 🌟 Key Features

### 🤖 **Dual AI Backend**
- **OpenAI Integration**: Leverage GPT models for advanced reasoning and comprehensive responses
- **Hugging Face Support**: Run local models for privacy-conscious deployments
- **Intelligent Provider Switching**: Seamlessly switch between AI providers based on your needs

### 🛠️ **DevOps Expertise**
- **Cloud Infrastructure**: AWS, Azure, GCP best practices and troubleshooting
- **Container Orchestration**: Kubernetes, Docker, and container management
- **CI/CD Pipelines**: Jenkins, GitLab CI, GitHub Actions, and deployment strategies
- **Infrastructure as Code**: Terraform, CloudFormation, and Ansible guidance
- **Monitoring & Observability**: Prometheus, Grafana, ELK stack, and alerting

### 🎯 **Professional Features**
- **Async Architecture**: High-performance async/await implementation
- **Comprehensive Logging**: Structured logging with file rotation and level control
- **Type Safety**: Full type hints and mypy validation
- **Error Handling**: Robust exception handling with detailed error context
- **Configuration Management**: Environment-based settings with validation
- **Health Monitoring**: Built-in health checks and status monitoring

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (Python 3.11+ recommended)
- **Git** for version control
- **API Keys** (OpenAI API key or Hugging Face token)

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/akshaymittal143/CloudyBot-DevOps-AI.git
cd CloudyBot-DevOps-AI
```

#### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies

**For Production:**
```bash
pip install -e .
```

**For Development:**
```bash
pip install -e ".[dev,test,docs]"
make pre-commit-install  # Install pre-commit hooks
```

#### 4. Configure Environment

Create a `.env` file in the project root:

```env
# AI Provider Configuration
MODEL_PROVIDER=OPENAI  # Options: OPENAI, HUGGINGFACE

# OpenAI Configuration (if using OpenAI)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview

# Hugging Face Configuration (if using Hugging Face)
HUGGINGFACE_MODEL=google/flan-t5-base
HUGGINGFACE_API_TOKEN=your_hf_token_here  # Optional
HUGGINGFACE_DEVICE=auto  # Options: auto, cpu, cuda, mps

# Application Settings
MAX_CHAT_HISTORY=50
DEBUG=false

# Logging Configuration
LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_ROTATION=true
```

#### 5. Launch CloudyBot

```bash
streamlit run app.py
```

Or using the Makefile:
```bash
make run
```

The application will be available at `http://localhost:8501`

## 🏗️ Architecture

### Project Structure

```
cloudybot/
├── cloudybot/                    # Main package
│   ├── __init__.py              # Package initialization and version
│   ├── clients/                 # AI client implementations
│   │   ├── base.py             # Abstract base client and data models
│   │   ├── openai_client.py    # OpenAI integration
│   │   └── hf_client.py        # Hugging Face integration
│   ├── config/                  # Configuration management
│   │   ├── settings.py         # Settings classes and environment handling
│   │   └── logging.py          # Logging configuration and mixins
│   └── core/                    # Core functionality
│       ├── bot.py              # Main CloudyBot class
│       └── exceptions.py       # Custom exception hierarchy
├── tests/                       # Test suite
├── app.py                      # Streamlit web interface
├── pyproject.toml              # Modern Python packaging
├── Makefile                    # Development commands
└── README.md                   # This file
```

### Core Components

#### **CloudyBot Class** (`cloudybot.core.bot.CloudyBot`)
The main orchestrator that manages AI clients and provides the primary interface.

```python
from cloudybot import CloudyBot

# Initialize with auto-configuration
bot = CloudyBot()

# Ask a question
response = await bot.ask("How do I scale a Kubernetes deployment?")
print(response)

# Switch providers
bot.switch_provider("HUGGINGFACE")

# Get bot status
status = bot.get_status()
```

#### **AI Clients** (`cloudybot.clients`)
Abstract base class with concrete implementations for different AI providers.

#### **Configuration System** (`cloudybot.config`)
Environment-based configuration with type validation and Streamlit secrets support.

#### **Logging System** (`cloudybot.config.logging`)
Professional logging with file rotation, structured output, and configurable levels.

## 🛠️ Development

### Development Setup

```bash
# Complete development environment setup
make dev-setup

# Or manually:
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -e ".[dev,test,docs]"
pre-commit install
```

### Development Commands

```bash
# Code quality
make format          # Format code with Black and isort
make lint           # Run linting (flake8, bandit, safety)
make type-check     # Type checking with mypy
make pre-commit     # Run all pre-commit hooks

# Testing
make test           # Run tests
make test-cov       # Run tests with coverage
make test-quick     # Quick test run

# Development workflow
make dev-check      # Run all checks (format, lint, type, test)
make dev-fix        # Fix common issues

# Docker
make docker-build   # Build Docker image
make docker-run     # Run in container

# Documentation
make docs-serve     # Serve docs locally
make docs-build     # Build documentation

# Utilities
make clean          # Clean build artifacts
make version        # Show version info
make help           # Show all commands
```

### Testing

```bash
# Run the full test suite
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=cloudybot --cov-report=html

# Quick tests
pytest tests/ -x --no-cov
```

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting and style checks
- **mypy**: Static type checking
- **bandit**: Security scanning
- **pre-commit**: Git hooks for quality gates

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t cloudybot:latest .

# Run with environment file
docker run -p 8501:8501 --env-file .env cloudybot:latest

# Development mode with volume mount
docker run -p 8501:8501 -v $(pwd):/app --env-file .env cloudybot:latest
```

### Multi-stage Build

The Dockerfile uses multi-stage builds for optimized production images:
- **Builder stage**: Installs dependencies
- **Production stage**: Minimal runtime image
- **Development stage**: Additional development tools

## 📊 Usage Examples

### Basic DevOps Questions

- "How do I restart a Kubernetes pod?"
- "Explain blue-green deployment strategies"
- "How can I debug Docker container failures?"
- "What's the difference between Terraform and CloudFormation?"

### Advanced Scenarios

- "How do I set up a CI/CD pipeline for a microservices architecture?"
- "What are the best practices for monitoring containerized applications?"
- "How can I implement Infrastructure as Code for AWS?"
- "What's the recommended approach for secret management in Kubernetes?"

### Provider-Specific Usage

```python
# Using OpenAI for complex reasoning
response = await bot.ask(
    "Design a disaster recovery strategy for a multi-region AWS deployment",
    provider="OPENAI"
)

# Using Hugging Face for privacy-sensitive environments
response = await bot.ask(
    "How do I configure network policies in Kubernetes?",
    provider="HUGGINGFACE"
)
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MODEL_PROVIDER` | Primary AI provider | `OPENAI` | ✅ |
| `OPENAI_API_KEY` | OpenAI API key | - | If using OpenAI |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4-turbo-preview` | ❌ |
| `HUGGINGFACE_MODEL` | HF model path | `google/flan-t5-base` | ❌ |
| `HUGGINGFACE_API_TOKEN` | HF API token | - | ❌ |
| `HUGGINGFACE_DEVICE` | Compute device | `auto` | ❌ |
| `MAX_CHAT_HISTORY` | Chat history limit | `50` | ❌ |
| `LOG_LEVEL` | Logging level | `INFO` | ❌ |
| `DEBUG` | Debug mode | `false` | ❌ |

### Streamlit Secrets

For Streamlit Cloud deployment, add secrets in `.streamlit/secrets.toml`:

```toml
MODEL_PROVIDER = "OPENAI"
OPENAI_API_KEY = "your_key_here"
OPENAI_MODEL = "gpt-4-turbo-preview"
MAX_CHAT_HISTORY = 50
LOG_LEVEL = "INFO"
```

## 🚨 Troubleshooting

### Common Issues

#### **"Module not found" errors**
```bash
# Ensure package is installed in development mode
pip install -e .
```

#### **OpenAI API errors**
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Check API key permissions and billing
```

#### **Hugging Face model loading issues**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/flan-t5-base')"
```

#### **Memory issues with large models**
```bash
# Use CPU mode for Hugging Face
export HUGGINGFACE_DEVICE=cpu

# Or use a smaller model
export HUGGINGFACE_MODEL=google/flan-t5-small
```

### Performance Optimization

#### **For OpenAI:**
- Use `gpt-3.5-turbo` for faster responses
- Implement request caching for repeated queries
- Use streaming for long responses

#### **For Hugging Face:**
- Use GPU acceleration: `HUGGINGFACE_DEVICE=cuda`
- Choose smaller models for faster inference
- Enable model quantization for memory efficiency

### Health Monitoring

```python
# Check system health
health = await bot.health_check()
print(health)

# Get detailed status
status = bot.get_status()
print(status)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Install** development dependencies: `make install-dev`
4. **Make** your changes with proper tests
5. **Run** quality checks: `make dev-check`
6. **Commit** with conventional commits
7. **Push** and create a Pull Request

### Code Standards

- Follow **PEP 8** style guidelines
- Add **type hints** to all functions
- Write **comprehensive tests** for new features
- Update **documentation** for API changes
- Ensure **100% test coverage** for new code

## 📝 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for providing powerful language models
- **Hugging Face** for open-source model hosting and tools
- **Streamlit** for the excellent web framework
- **The DevOps Community** for inspiration and feedback

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/akshaymittal143/CloudyBot-DevOps-AI/issues)
- **Documentation**: [Project Wiki](https://github.com/akshaymittal143/CloudyBot-DevOps-AI/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/akshaymittal143/CloudyBot-DevOps-AI/discussions)

---

<div align="center">

**Made with ❤️ by [Akshay Mittal](https://github.com/akshaymittal143)**

*Empowering DevOps teams with AI-driven insights and automation*

</div>

