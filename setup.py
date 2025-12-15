"""
Setup script for easy installation and execution.
Handles environment setup, dependency installation, and application launch.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"Python version: {sys.version.split()[0]}")
    return True

def check_pip():
    """Check if pip is available."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("pip is available")
        return True
    except subprocess.CalledProcessError:
        print("Error: pip is not available")
        return False

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print("Created .env file from template")
        print("Please edit .env file with your API keys before running the application")
        return True
    elif env_file.exists():
        print(".env file already exists")
        return True
    else:
        print("Error: .env.example file not found")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ["data", "uploads", "data/chroma_db"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def check_api_keys():
    """Check if required API keys are configured."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "your_openai_api_key_here":
            print("Warning: OPENAI_API_KEY not configured in .env file")
            return False
        
        print("OpenAI API key configured")
        
        # Check LangSmith (optional)
        langsmith_key = os.getenv("LANGCHAIN_API_KEY")
        if langsmith_key and langsmith_key != "your_langsmith_api_key_here":
            print("LangSmith API key configured")
        else:
            print("LangSmith API key not configured (optional)")
        
        return True
    
    except ImportError:
        print("Warning: python-dotenv not installed, skipping API key check")
        return True

def run_application():
    """Launch the Streamlit application."""
    print("Launching Study Notes Summarizer...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching application: {e}")
        return False
    except KeyboardInterrupt:
        print("Application stopped by user")
        return True

def main():
    """Main setup and launch function."""
    print("=" * 60)
    print("Study Notes Summarizer - Setup & Launch")
    print("=" * 60)
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Setup environment
    print("\n🔧 Setting up environment...")
    create_directories()
    
    if not create_env_file():
        sys.exit(1)
    
    # Install dependencies
    print("Installing dependencies...")
    if not install_dependencies():
        sys.exit(1)
    
    # Check configuration
    print("Checking configuration...")
    api_keys_ok = check_api_keys()
    
    if not api_keys_ok:
        print("\n" + "=" * 60)
        print("IMPORTANT: Please configure your API keys!")
        print("=" * 60)
        print("1. Open the .env file in a text editor")
        print("2. Replace 'your_openai_api_key_here' with your actual OpenAI API key")
        print("3. Optionally, add your LangSmith API key for monitoring")
        print("4. Save the file and run this script again")
        print("\nGet your OpenAI API key from: https://platform.openai.com/api-keys")
        print("Get your LangSmith API key from: https://smith.langchain.com/")
        return
    
    # Launch application
    print("Setup completed successfully!")
    print("=" * 60)
    
    choice = input("\nWould you like to launch the application now? (y/N): ").lower().strip()
    
    if choice in ['y', 'yes']:
        run_application()
    else:
        print("To launch the application later, run:")
        print("   streamlit run app.py")
        print("Setup complete!")

if __name__ == "__main__":
    main()