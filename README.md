LLM Application Sandbox Environment Setup Guide
Overview
A sandbox environment is an isolated development space where you can safely test, experiment, and develop LLM (Large Language Model) applications without affecting production systems. This guide will walk you through setting up a complete sandbox environment from scratch.
Why Do We Need a Sandbox Environment?

Safety: Test dangerous or experimental code without risk
Isolation: Keep development work separate from production systems
Experimentation: Try different models, parameters, and configurations freely
Learning: Practice and learn without consequences
Cost Control: Monitor and limit API usage during development

Prerequisites
Before starting, ensure you have:

A computer with at least 8GB RAM (16GB recommended)
Administrative privileges to install software
Stable internet connection
Basic command line knowledge

Step 1: Install Python and Environment Management
Install Python (3.9 or higher)
For Windows:

Download Python from python.org
Run installer and check "Add Python to PATH"
Verify installation: python --version

For macOS:
bash# Using Homebrew (recommended)
brew install python
For Linux (Ubuntu/Debian):
bashsudo apt update
sudo apt install python3 python3-pip python3-venv
Set Up Virtual Environment
Virtual environments keep your project dependencies isolated from other Python projects.
bash# Create a new directory for your LLM projects
mkdir llm-sandbox
cd llm-sandbox

# Create virtual environment
python -m venv llm-env

# Activate virtual environment
# Windows:
llm-env\Scripts\activate
# macOS/Linux:
source llm-env/bin/activate

# You should see (llm-env) in your terminal prompt
Step 2: Install Essential LLM Libraries
Install the core libraries you'll need for LLM development:
bash# Update pip first
pip install --upgrade pip

# Core LLM libraries
pip install openai anthropic langchain langchain-community

# Data handling and analysis
pip install pandas numpy matplotlib seaborn

# Web frameworks for building applications
pip install streamlit fastapi uvicorn flask

# Environment management
pip install python-dotenv

# Jupyter for interactive development
pip install jupyter notebook ipykernel

# Save installed packages to requirements file
pip freeze > requirements.txt
Step 3: Configure API Keys and Environment Variables
Create Environment Configuration
Create a .env file in your project root to store sensitive information securely:
bash# Create .env file
touch .env
Add your API keys to the .env file:
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORG_ID=your_org_id_here

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Other LLM Services
HUGGINGFACE_API_KEY=your_huggingface_key_here
COHERE_API_KEY=your_cohere_key_here

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
MAX_TOKENS=1000
TEMPERATURE=0.7
Create a Configuration Manager
Create a config.py file to manage your environment variables:
pythonimport os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
    
    # Model Settings
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gpt-3.5-turbo')
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 1000))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
    
    # Application Settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Validate required keys
def validate_config():
    required_keys = ['OPENAI_API_KEY']
    missing_keys = [key for key in required_keys if not getattr(Config, key)]
    
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {missing_keys}")

if __name__ == "__main__":
    validate_config()
    print("Configuration validated successfully!")
Step 4: Set Up Development Tools
Install and Configure Git
bash# Initialize git repository
git init

# Create .gitignore file
echo ".env
__pycache__/
*.pyc
.jupyter/
.vscode/
*.log
node_modules/
.DS_Store" > .gitignore

# Set up initial commit
git add .
git commit -m "Initial sandbox setup"
Set Up Jupyter Notebook
bash# Install Jupyter kernel for your virtual environment
python -m ipykernel install --user --name=llm-env --display-name="LLM Sandbox"

# Start Jupyter notebook
jupyter notebook
Configure VS Code (Optional but Recommended)
Install VS Code extensions:

Python
Jupyter
Python Docstring Generator
GitLens

Step 5: Create Project Structure
Organize your sandbox with a clear directory structure:
bashllm-sandbox/
├── .env                    # Environment variables (keep secret)
├── .gitignore             # Git ignore file
├── requirements.txt       # Python dependencies
├── config.py             # Configuration management
├── README.md             # Project documentation
├── notebooks/            # Jupyter notebooks for experiments
│   ├── experiments/      # Quick experiments
│   └── tutorials/        # Learning notebooks
├── src/                  # Source code
│   ├── __init__.py
│   ├── models/          # Model wrappers and utilities
│   ├── utils/           # Helper functions
│   └── apps/            # Complete applications
├── tests/               # Unit tests
├── data/                # Sample data files
│   ├── inputs/          # Input files
│   └── outputs/         # Generated outputs
├── logs/                # Application logs
└── docs/                # Documentation
Create this structure:
bashmkdir -p notebooks/{experiments,tutorials}
mkdir -p src/{models,utils,apps}
mkdir -p tests data/{inputs,outputs} logs docs
touch src/__init__.py src/models/__init__.py src/utils/__init__.py src/apps/__init__.py
Step 6: Create Your First LLM Application
Simple Chat Interface
Create src/apps/simple_chat.py:
pythonimport os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import Config, validate_config
import openai
from datetime import datetime

class SimpleLLMChat:
    def __init__(self):
        validate_config()
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.conversation_history = []
    
    def send_message(self, message):
        """Send a message to the LLM and get response"""
        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": message})
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=Config.DEFAULT_MODEL,
                messages=self.conversation_history,
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )
            
            # Extract response
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared.")
    
    def save_conversation(self, filename=None):
        """Save conversation to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.txt"
        
        filepath = os.path.join("logs", filename)
        with open(filepath, "w") as f:
            for message in self.conversation_history:
                f.write(f"{message['role'].upper()}: {message['content']}\n\n")
        
        print(f"Conversation saved to {filepath}")

def main():
    """Interactive chat session"""
    chat = SimpleLLMChat()
    print("LLM Sandbox Chat Interface")
    print("Commands: 'quit' to exit, 'clear' to clear history, 'save' to save conversation")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            chat.clear_history()
            continue
        elif user_input.lower() == 'save':
            chat.save_conversation()
            continue
        elif not user_input:
            continue
        
        response = chat.send_message(user_input)
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()
Test Your Setup
Run your first LLM application:
bashpython src/apps/simple_chat.py
Step 7: Create Utility Functions
Create src/utils/helpers.py:
pythonimport json
import logging
from datetime import datetime
from typing import Dict, List, Any

def setup_logging(log_level="INFO"):
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_json(data: Dict[str, Any], filename: str, directory: str = "data/outputs"):
    """Save data as JSON file"""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    return filepath

def load_json(filename: str, directory: str = "data/inputs") -> Dict[str, Any]:
    """Load data from JSON file"""
    filepath = os.path.join(directory, filename)
    
    with open(filepath, 'r') as f:
        return json.load(f)

def count_tokens_estimate(text: str) -> int:
    """Rough estimate of token count (1 token ≈ 4 characters)"""
    return len(text) // 4

def truncate_text(text: str, max_tokens: int = 1000) -> str:
    """Truncate text to approximate token limit"""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

def format_conversation_history(history: List[Dict[str, str]]) -> str:
    """Format conversation history for display"""
    formatted = []
    for message in history:
        role = message['role'].upper()
        content = message['content']
        formatted.append(f"{role}: {content}")
    
    return "\n\n".join(formatted)
Step 8: Safety and Best Practices
Rate Limiting and Cost Control
Create src/utils/rate_limiter.py:
pythonimport time
from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_calls_per_minute=60):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = defaultdict(list)
    
    def can_make_call(self, identifier="default"):
        """Check if a call can be made within rate limits"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Remove old calls
        self.calls[identifier] = [
            call_time for call_time in self.calls[identifier] 
            if call_time > minute_ago
        ]
        
        # Check if under limit
        if len(self.calls[identifier]) < self.max_calls_per_minute:
            self.calls[identifier].append(now)
            return True
        
        return False
    
    def wait_time(self, identifier="default"):
        """Calculate how long to wait before next call"""
        if not self.calls[identifier]:
            return 0
        
        oldest_call = min(self.calls[identifier])
        wait_until = oldest_call + timedelta(minutes=1)
        now = datetime.now()
        
        if wait_until > now:
            return (wait_until - now).total_seconds()
        
        return 0
Input Validation and Sanitization
Create src/utils/validators.py:
pythonimport re
from typing import List

class InputValidator:
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Basic input sanitization"""
        # Remove potential harmful patterns
        text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        # Limit length
        if len(text) > 10000:
            text = text[:10000] + "..."
        
        return text.strip()
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format"""
        if not api_key or len(api_key) < 10:
            return False
        
        # Basic format checks for different providers
        if api_key.startswith('sk-') and len(api_key) > 40:  # OpenAI
            return True
        elif api_key.startswith('claude-') or len(api_key) > 30:  # Anthropic
            return True
        
        return False
    
    @staticmethod
    def check_content_policy(text: str) -> List[str]:
        """Basic content policy checking"""
        warnings = []
        
        # Check for potentially harmful content patterns
        harmful_patterns = [
            (r'\b(hack|exploit|vulnerability)\b', "Security-related content detected"),
            (r'\b(bomb|weapon|violence)\b', "Violence-related content detected"),
            (r'personal\s+information', "Personal information handling detected")
        ]
        
        for pattern, warning in harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                warnings.append(warning)
        
        return warnings
Step 9: Testing Your Setup
Create Test Files
Create tests/test_basic_functionality.py:
pythonimport sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import unittest
from unittest.mock import patch, MagicMock
from src.utils.helpers import count_tokens_estimate, truncate_text
from src.utils.validators import InputValidator

class TestBasicFunctionality(unittest.TestCase):
    
    def test_token_counting(self):
        """Test token counting estimation"""
        text = "This is a test string"
        estimated_tokens = count_tokens_estimate(text)
        self.assertGreater(estimated_tokens, 0)
        self.assertLess(estimated_tokens, len(text))
    
    def test_text_truncation(self):
        """Test text truncation"""
        long_text = "x" * 10000
        truncated = truncate_text(long_text, max_tokens=100)
        self.assertLess(len(truncated), len(long_text))
    
    def test_input_sanitization(self):
        """Test input sanitization"""
        dangerous_input = "<script>alert('xss')</script>Hello"
        sanitized = InputValidator.sanitize_input(dangerous_input)
        self.assertNotIn("<script>", sanitized)
        self.assertIn("Hello", sanitized)
    
    def test_api_key_validation(self):
        """Test API key validation"""
        valid_key = "sk-1234567890abcdef1234567890abcdef12345678"
        invalid_key = "invalid"
        
        self.assertTrue(InputValidator.validate_api_key(valid_key))
        self.assertFalse(InputValidator.validate_api_key(invalid_key))

if __name__ == '__main__':
    unittest.main()
Run the tests:
bashpython -m pytest tests/ -v
Step 10: Documentation and Usage Examples
Create Usage Examples
Create notebooks/tutorials/getting_started.ipynb:
python# Cell 1: Setup and imports
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..', '..'))

from src.apps.simple_chat import SimpleLLMChat
from src.utils.helpers import setup_logging, count_tokens_estimate
from config import Config

# Setup logging
logger = setup_logging()
logger.info("Starting LLM sandbox tutorial")

# Cell 2: Basic chat example
chat = SimpleLLMChat()

# Send a simple message
response = chat.send_message("Hello! Can you explain what a Large Language Model is?")
print(f"Response: {response}")

# Cell 3: Token counting example
sample_text = "This is a sample text to demonstrate token counting."
token_count = count_tokens_estimate(sample_text)
print(f"Text: {sample_text}")
print(f"Estimated tokens: {token_count}")

# Cell 4: Conversation history
print("Conversation History:")
for i, message in enumerate(chat.conversation_history):
    print(f"{i+1}. {message['role'].upper()}: {message['content'][:100]}...")
Troubleshooting Guide
Common Issues and Solutions
Issue: "Module not found" errors

Solution: Ensure your virtual environment is activated and all packages are installed
Check your Python path and project structure

Issue: API key errors

Solution: Verify your .env file exists and contains valid API keys
Make sure .env is in your project root directory

Issue: Rate limiting errors

Solution: Implement the rate limiter utility or reduce API call frequency
Consider using cheaper models for testing

Issue: Memory issues with large models

Solution: Use cloud-based APIs instead of local models for development
Implement text truncation for long inputs

Next Steps
Now that your sandbox is set up, you can:

Experiment with different models: Try GPT-4, Claude, or open-source alternatives
Build applications: Create chatbots, content generators, or analysis tools
Learn prompt engineering: Practice crafting effective prompts
Explore advanced features: Function calling, embeddings, fine-tuning
Build web interfaces: Use Streamlit or FastAPI to create user-friendly apps

Resources for Learning

OpenAI Documentation: https://platform.openai.com/docs
LangChain Documentation: https://python.langchain.com/
Anthropic Claude Documentation: https://docs.anthropic.com/
Hugging Face Transformers: https://huggingface.co/docs/transformers

Security Reminders

Never commit .env files to version control
Regularly rotate your API keys
Monitor your API usage and costs
Validate and sanitize all user inputs
Use rate limiting in production applications

Your LLM sandbox environment is now ready for development and experimentation!
