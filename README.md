# Reddit Persona Analyzer

**AI/LLM Engineer Intern Assignment - BeyondChats**

A sophisticated Reddit user persona generator that analyzes user profiles using free AI models and generates professional UX research-style personas with full citation tracking.

## 🚀 Features

- **Zero-Cost Implementation**: Uses only free APIs and models
- **GPU Acceleration**: Automatically detects and uses GPU when available for 5-10x faster processing
- **Professional Persona Output**: Industry-standard UX research format with rich behavioral insights
- **Complete Citation System**: Every insight linked to source content
- **AI-Powered Analysis**: Leverages multiple Hugging Face transformers for personality and interest analysis
- **Robust Error Handling**: Gracefully handles edge cases and API limits
- **Rich Behavioral Descriptions**: Detailed narratives like professional UX research personas

## 📋 Requirements

- Python 3.8 or higher
- Reddit API credentials (free)
- 4GB+ RAM (for AI models)
- Internet connection

## ⚡ Quick Start (One-Click Setup)

**For evaluators:** Run this single command for automated setup:

```bash
python setup.py
```

This will:
- ✅ Check Python version compatibility
- ✅ Install all dependencies automatically
- ✅ Create configuration template
- ✅ Provide clear Reddit API setup instructions
- ✅ Test the installation

## 🛠️ Manual Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Kp4103/reddit-persona-analyzer.git
cd reddit-persona-analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Reddit API Setup (Free)

1. Go to [Reddit App Preferences](https://www.reddit.com/prefs/apps/)
2. Click "Create App" or "Create Another App"
3. Fill in the form:
   - **Name**: PersonaAnalyzer (or any name)
   - **App Type**: Select "script"
   - **Description**: Reddit persona analysis tool
   - **About URL**: Leave blank
   - **Redirect URI**: http://localhost:8080
4. Click "Create app"
5. Note down your credentials:
   - **Client ID**: Found under the app name (short string)
   - **Client Secret**: The longer "secret" string

### 4. Configure Environment Variables

Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Edit `.env` with your Reddit credentials:
```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=PersonaAnalyzer/1.0 by YourRedditUsername
```

## 🎯 Usage

### Basic Usage
```bash
python main.py https://www.reddit.com/user/kojied/
```

### With Custom Limit
```bash
python main.py https://www.reddit.com/user/Hungry-Move-6603/ --limit 150
```

### Supported URL Formats
- `https://www.reddit.com/user/username/`
- `https://www.reddit.com/u/username/`
- `reddit.com/user/username`
- Just the username: `username`

## 📊 Output

The script generates a professional persona file named `{username}_persona.txt` containing:

- **Demographics & Overview**: Activity patterns, primary communities
- **Behavioral Insights**: AI-analyzed behavior patterns and habits
- **Personality Traits**: Characteristics with confidence levels and citations
- **Motivations**: Scored motivation analysis with visual progress bars
- **Interests & Activities**: Categorized interests from content analysis
- **Representative Quote**: Authentic statement from user's content
- **Activity Analysis**: Community engagement patterns
- **Data Sources & Citations**: Links to all analyzed content

## 🏗️ Architecture

### Core Components

1. **RedditScraper**: Handles API interactions and data collection
2. **PersonaAnalyzer**: AI-powered analysis using free Hugging Face models
3. **PersonaGenerator**: Formats professional persona output

### AI Models Used (All Free)

- **FLAN-T5-Large**: Text generation and analysis
- **RoBERTa**: Sentiment analysis for personality insights
- **BART-Large-MNLI**: Zero-shot classification for interests

### Rate Limiting

- Respects Reddit's API limits (60 requests/minute)
- Built-in delays between requests
- Graceful handling of API errors

## 📝 Sample Output Files

The repository includes sample outputs for the required test users:
- `kojied_persona.txt`
- `Hungry-Move-6603_persona.txt`

## 🔧 Technical Details

### Free Model Strategy
The analyzer uses a tiered approach for cost-free AI analysis:

1. **Primary**: Hugging Face Transformers (local inference)
2. **Fallback**: Smaller models if memory constraints
3. **Hybrid**: Combines AI insights with keyword-based analysis

### Error Handling
- Handles deleted/private content gracefully
- Manages API rate limits automatically
- Provides meaningful error messages
- Continues analysis even if some requests fail

### Citation System
Every persona characteristic includes:
- Direct links to supporting Reddit posts/comments
- Confidence levels for AI-generated insights
- Source post metadata (score, subreddit, type)

## 🚨 Troubleshooting

### Common Issues

**1. Reddit API Errors**
- Verify your credentials are correct
- Ensure the user profile is public
- Check your internet connection

**2. Model Loading Issues**
- Free up RAM (models require 2-4GB)
- Close other applications
- Consider using CPU if GPU memory is insufficient

**3. No Content Found**
- User might have deleted posts
- Account could be private/suspended
- Try a different user profile

**4. Slow Performance**
- AI models take time to load initially
- Subsequent analyses are faster
- Consider reducing the `--limit` parameter

### Debug Mode
For detailed logging, modify the logging level in `main.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Development Notes

### Code Quality
- Follows PEP-8 standards
- Comprehensive error handling
- Modular, extensible design
- Full type hints and documentation

### Performance Optimizations
- Batched AI inference where possible
- Efficient data structures
- Memory-conscious model loading
- Smart content filtering

## 🎯 Assignment Compliance

✅ **Required Features**:
- Takes Reddit profile URLs as input
- Scrapes posts and comments using free Reddit API
- Generates professional user personas
- Includes complete citation system
- Outputs to text files
- Works with both sample users

✅ **Technical Requirements**:
- Executable Python script
- PEP-8 compliant code
- Clear setup instructions
- Zero external costs
- Public GitHub repository

## 🔗 Repository

**GitHub Repository**: [https://github.com/Kp4103/reddit-persona-analyzer](https://github.com/Kp4103/reddit-persona-analyzer)

## 📄 License

This code is developed for the BeyondChats AI/LLM Engineer Intern assignment. 
The developer retains ownership unless selected for the paid internship position.

---

**Note**: This analyzer respects Reddit's terms of service and only accesses publicly available content through official APIs.
