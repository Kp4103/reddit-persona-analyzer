#!/usr/bin/env python3
"""
One-Click Setup Script for Reddit Persona Analyzer
AI/LLM Engineer Intern Assignment - BeyondChats

This script automates the entire setup process for evaluators.
Just run: python setup.py
"""

import os
import subprocess
import sys
import platform

def check_python_version():
    """Ensure Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
    return True

def install_requirements():
    """Install required packages."""
    print("\n🔧 Installing required packages...")
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("💡 Try running manually: pip install -r requirements.txt")
        return False

def setup_env_file():
    """Set up environment file if it doesn't exist."""
    if not os.path.exists('.env'):
        print("\n📝 Setting up environment configuration...")
        import shutil
        shutil.copy('.env.example', '.env')
        print("✅ Created .env file from template")
        return False  # Needs manual configuration
    else:
        print("\n✅ Environment file already exists")
        return True

def check_gpu_availability():
    """Check if GPU is available for faster processing."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 GPU detected: {torch.cuda.get_device_name()} - Processing will be 5-10x faster!")
            return True
        else:
            print("💻 CPU processing mode - Will work fine, just slower")
            return False
    except ImportError:
        print("💻 PyTorch not installed yet - Will check GPU after installation")
        return False

def display_credentials_instructions():
    """Show clear instructions for getting Reddit API credentials."""
    print("\n" + "="*60)
    print("🔑 REDDIT API SETUP REQUIRED")
    print("="*60)
    print("To analyze Reddit profiles, you need free API credentials:")
    print("")
    print("1. Go to: https://www.reddit.com/prefs/apps")
    print("2. Click 'Create App' or 'Create Another App'")
    print("3. Fill out the form:")
    print("   - Name: Reddit Persona Analyzer")
    print("   - App type: Script")
    print("   - Redirect URI: http://localhost:8080")
    print("4. Copy the credentials to your .env file:")
    print("   - CLIENT_ID = (14-character string under app name)")
    print("   - CLIENT_SECRET = (longer string)")
    print("   - USER_AGENT = your_username_reddit_analyzer")
    print("")
    print("💡 This takes 2 minutes and is completely free!")
    print("="*60)

def run_test():
    """Test the setup with sample user."""
    print("\n🧪 Testing setup with sample user...")
    try:
        # Quick test to see if we can import main modules
        import praw
        from transformers import pipeline
        print("✅ All modules imported successfully!")
        
        print("\n🎯 Setup complete! Ready to analyze Reddit profiles.")
        print("\nTo test with sample user:")
        print("python main.py https://www.reddit.com/user/kojied/")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running setup again or install missing packages manually")
        return False

def main():
    """Main setup function with comprehensive checks."""
    print("🚀 Reddit Persona Analyzer - One-Click Setup")
    print("AI/LLM Engineer Intern Assignment - BeyondChats")
    print("="*60)
    
    # System checks
    if not check_python_version():
        return
    
    # Install dependencies
    print(f"\n🖥️  System: {platform.system()} {platform.release()}")
    deps_ok = install_requirements()
    
    if deps_ok:
        # Check GPU after installation
        check_gpu_availability()
    
    # Setup environment
    env_needs_config = not setup_env_file()
    
    # Final status
    print("\n" + "="*60)
    if deps_ok and not env_needs_config:
        print("🎉 SETUP COMPLETE! Ready to analyze Reddit profiles.")
        print("\n📋 Usage Examples:")
        print("python main.py https://www.reddit.com/user/kojied/")
        print("python main.py https://www.reddit.com/user/Hungry-Move-6603/")
        
        # Quick test
        run_test()
        
    elif deps_ok and env_needs_config:
        print("⚠️  SETUP PARTIALLY COMPLETE")
        display_credentials_instructions()
        print("\n📋 After configuring .env, run:")
        print("python main.py https://www.reddit.com/user/kojied/")
        
    else:
        print("❌ SETUP FAILED")
        print("\n🔧 Manual setup required:")
        print("1. pip install -r requirements.txt")
        print("2. cp .env.example .env")
        print("3. Configure Reddit API credentials in .env")
        print("4. python main.py <reddit_profile_url>")

if __name__ == "__main__":
    main()
