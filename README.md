
# CloudyBot: AI-Powered DevOps Assistant Chatbot

CloudyBot is an AI-powered DevOps assistant chatbot designed to help with cloud and DevOps-related queries and tasks. It provides a conversational interface using OpenAI's GPT models or Hugging Face's local models.

## Features

- **Dual AI Backend:** Supports OpenAI (requires API key) and local Hugging Face models (FLAN-T5).
- **User-Friendly UI:** Streamlit-based chat interface.
- **Easy Deployment:** Local, Streamlit Cloud, or Google Colab.

## Prerequisites

- Python 3.8+ - CloudyBot is a Python application. Ensure you have Python installed. You can check by running python --version in your terminal.
- API keys (for OpenAI usage): You can obtain an API key by creating an account on OpenAI’s platform and generating a key from their dashboard. Note: This is unrelated to Streamlit.
streamlit.io
. (If you only plan to use the local Hugging Face model, no external API key is required, although a Hugging Face Hub token could be used if you want to load models via their API or need to access gated models – for our default FLAN-T5, this is not necessary.)

## Quickstart

### Step 1: Clone Repository
```bash
git clone https://github.com/akshaymittal143/CloudyBot-DevOps-AI.git
cd CloudyBot-DevOps-AI

or 

# Create project files

mkdir cloudybot_project
touch bot.py openai_client.py hf_client.py app.py requirements.txt .env.example README.md
```

## Project Structure

```
cloudybot/
├── bot.py
├── openai_client.py
├── hf_client.py
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

### Step 2: Virtual Environment Setup (Optional)
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
```

#### Dependencies and Requirements
We list all required Python packages in requirements.txt. This ensures anyone setting up the project can install the correct libraries easily. Here’s what you should include in requirements.txt:
```bash
--index-url https://pypi.org/simple
streamlit>=1.28.0
openai>=0.27.8
transformers>=4.31.0
python-dotenv>=1.0.0
protobuf>=4.21.0
torch>=2.1.0
numpy<2.0.0
pillow>=9.5.0
```

- Streamlit: for the UI.
- OpenAI: official OpenAI Python SDK to communicate with OpenAI models.
- Transformers: Hugging Face library to run local language models.
- python-dotenv: to load environment variables from a .env file.

### Step 3: Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 4: Configuration (.env file)
Copy `.env.example` to `.env` and edit:
```ini
# .env.example - example configuration for CloudyBot
OPENAI_API_KEY=sk....
OPENAI_MODEL=gpt-3.5-turbo
HUGGINGFACE_MODEL=google/flan-t5-base
#HUGGINGFACE_API_TOKEN=your_optional_huggingface_token_here

# Default Model Provider (OPENAI or HUGGINGFACE)
MODEL_PROVIDER=OPENAI
TOKENIZERS_PARALLELISM=false
```
Note: Get your [OpenAI key from here](https://platform.openai.com/api-keys)

### Step 5: Run CloudyBot
```bash
streamlit run app.py
```

Visit [http://localhost:8501](http://localhost:8501).
- When you run this, Streamlit will start a local web server. It will print a URL in the terminal (usually http://localhost:8501).
- Hold Ctrl and click the URL, or open your web browser and navigate to it. You should see the CloudyBot interface come up.


### Sample Run Screenshot

Below is a screenshot of CloudyBot in action, showcasing its conversational interface and ability to answer DevOps-related queries:

![Sample Run of CloudyBot](resources/1.png)


## Deployment on Streamlit Cloud

- Deploy from GitHub via [Streamlit Cloud](https://streamlit.io).
- Configure secrets (`OPENAI_API_KEY`) through Streamlit secrets.


## Streamlit Cloud Deployment

1. Fork this repository
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Deploy from GitHub
4. Add these secrets in Streamlit Cloud dashboard:
   ```plaintext
   OPENAI_API_KEY = "your-api-key"
   MODEL_PROVIDER = "OPENAI"
   OPENAI_MODEL = "gpt-3.5-turbo"
   ```
   
## Running in Google Colab

In a Colab notebook:
```bash
!git clone https://github.com/akshaymittal143/CloudyBot-DevOps-AI.git
%cd CloudyBot-DevOps-AI
!pip install -r requirements.txt
```

Set keys in the notebook:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
```

Interactive Loop:
```python
from cloudybot.bot import ask_bot
while True:
    query = input("You: ")
    if query.lower() in ("exit", "quit"): break
    print("CloudyBot:", ask_bot(query))
```



## Example Queries

- "How do I restart a Kubernetes pod?"
- "Explain blue-green deployment."
- "How to debug Docker container failures?"

---

For detailed instructions and contributions, refer to the comprehensive guide above.

Future Improvements and Contributions
CloudyBot is a work in progress. There are many ways it could be enhanced, and we welcome ideas or contributions! If you're interested in contributing, please check out our [CONTRIBUTING.md](https://github.com/akshaymittal143/CloudyBot-DevOps-AI/blob/main/CONTRIBUTING.md) file for guidelines or visit our [GitHub Issues page](https://github.com/akshaymittal143/CloudyBot-DevOps-AI/issues) to see open tasks and feature requests.

Here are some potential future improvements (some of which we discussed in the slides):
- Live Data Integration: Connecting CloudyBot to live systems (APIs, databases, cloud services) so it can fetch real-time information. For instance, integrate with AWS/GCP SDKs to answer “What’s the CPU usage of EC2 instance X?” by actually calling CloudWatch metrics. This would make answers more dynamic and context-specific. It could be done securely by allowing certain read-only credentials configured in the environment.
- Executing Commands (Actionable Bot): Taking the above further, allow CloudyBot to execute certain actions when prompted. For example, “CloudyBot, restart the backend service” could trigger a predefined script or API call (maybe hitting a Kubernetes endpoint to restart a pod, etc.). This is essentially building ChatOps capabilities. We’d have to implement a permissions layer and confirmation (to avoid accidental destructive actions). Possibly maintain a list of allowed operations the bot can do.
- Enhanced Conversation Memory: Currently, the conversation memory might be limited (especially with local model). We can extend this by caching past interactions and summarizing them when they get too long, or by using a vector store to dynamically fetch relevant past bits. Also, if using OpenAI GPT-4, we’d get a larger context window enabling longer dialogues. Future models (like GPT-4 32k context or others) could vastly improve how much history CloudyBot can remember and reason over.
- Knowledge Base Integration: As mentioned, hooking up a documentation knowledge base. We could store documentation (Kubernetes docs, AWS docs, internal docs, etc.) and have CloudyBot retrieve and cite them. Perhaps using an approach with LangChain or similar frameworks: user question -> search docs -> give relevant snippets to LLM -> LLM answers using those. This would reduce hallucination and increase accuracy for domain-specific questions. It would also allow CloudyBot to answer questions like “According to our dev guide, how do we release a new version?” with exact steps from the guide.
- Support More Models/Providers: There’s interest in adding support for more open-source LLMs that are more powerful than FLAN-T5. For example: Alpaca/LLaMA derivatives, GPT-J/GPT-NeoX, Dolly 2.0, OpenAI’s newer models, Cohere API, etc. We could create a plugin system where if you have a model’s API or weights, you drop in a new client class and configure it. CloudyBot could even run a small model by default and escalate to a bigger model for harder questions (to optimize response time vs quality).
Multi-language Support: Maybe some users might ask questions in different languages. We could incorporate translation or use models that support multilingual queries so CloudyBot can assist non-English speakers on DevOps topics.
- UI Enhancements: The Streamlit UI can be made more robust: add an “Export conversation” feature to save the Q&A as a text or PDF. Add a “Clear chat” button to reset the session. Possibly allow the user to toggle between dark/light theme for better viewing. Another idea is to have a “persona” or “mode” switch: e.g., “Beginner mode” where CloudyBot gives more basic, step-by-step answers, vs “Expert mode” where it can be more concise and assume background knowledge. This could be a toggle that changes the system prompt to adjust the answer detail.
- Error Handling and Fail-safes: Currently, basic error handling is implemented to catch exceptions such as OpenAI request timeouts or rate limit errors, providing a friendly message like “Sorry, I’m having trouble reaching the AI service. Please try again.” Future improvements include detecting nonsensical outputs from the local model and responding appropriately, such as retrying or indicating uncertainty.
Contributing Guide: If opening the project to contributors, provide guidelines on how to add new features, coding style, etc. Perhaps write tests for the core logic (ensuring the OpenAI and HF clients work as expected given dummy inputs). This encourages community involvement to add the above features.
- Error Handling and Fail-safes: Improve how the bot handles cases where the AI model doesn’t know an answer or fails. We can catch exceptions (like OpenAI request timeouts or rate limit errors) and respond with a friendly message like “Sorry, I’m having trouble reaching the AI service. Please try again.” rather than just breaking. If the local model produces nonsense, maybe detect if output is empty or too off-topic and either retry or say it’s unsure.
- Security Considerations: If CloudyBot ever executes commands, ensure it cannot be tricked into doing something dangerous. Implement confirmations (“Are you sure? [Yes/No]”) for critical actions. Possibly maintain a whitelist of safe commands. Also, sanitize user input if it might go into any shell calls (to prevent injection). For now, as a read-only assistant, it’s mostly about not revealing secrets (which we handle via environment and not echoing them).

Feel free to raise issues or pull requests on the GitHub repo if you have ideas or improvements. CloudyBot can grow with community input, especially as new AI capabilities emerge.

