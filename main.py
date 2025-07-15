import os
import re
import json
import time
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import praw
from transformers import pipeline, AutoTokenizer
import torch
from dotenv import load_dotenv

load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    """Container for Reddit post/comment data with metadata."""
    content: str
    url: str
    score: int
    created_utc: float
    subreddit: str
    post_type: str  # 'post' or 'comment'


class RedditScraper:
    """Handles Reddit API interactions and data collection."""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """Initialize Reddit API client."""
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        logger.info("Reddit API client initialized")
    
    def extract_username(self, profile_url: str) -> str:
        """Extract username from Reddit profile URL."""
        patterns = [
            r'reddit\.com/user/([^/]+)',
            r'reddit\.com/u/([^/]+)',
            r'/user/([^/]+)',
            r'/u/([^/]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, profile_url)
            if match:
                return match.group(1)
        
        # If no pattern matches, assume it's just the username
        return profile_url.strip('/')
    
    def scrape_user_content(self, username: str, limit: int = 100) -> List[RedditPost]:
        """Scrape user's posts and comments."""
        try:
            user = self.reddit.redditor(username)
            content = []
            
            logger.info(f"Scraping content for user: {username}")
            
            try:
                for submission in user.submissions.new(limit=limit//2):
                    if submission.selftext and len(submission.selftext.strip()) > 20:
                        content.append(RedditPost(
                            content=f"Title: {submission.title}\n\nContent: {submission.selftext}",
                            url=f"https://reddit.com{submission.permalink}",
                            score=submission.score,
                            created_utc=submission.created_utc,
                            subreddit=str(submission.subreddit),
                            post_type='post'
                        ))
                    time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Error fetching submissions: {e}")
            
            try:
                for comment in user.comments.new(limit=limit//2):
                    if len(comment.body.strip()) > 20 and comment.body != '[deleted]':
                        content.append(RedditPost(
                            content=comment.body,
                            url=f"https://reddit.com{comment.permalink}",
                            score=comment.score,
                            created_utc=comment.created_utc,
                            subreddit=str(comment.subreddit),
                            post_type='comment'
                        ))
                    time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Error fetching comments: {e}")
            
            logger.info(f"Scraped {len(content)} posts/comments")
            return sorted(content, key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            logger.error(f"Error scraping user {username}: {e}")
            return []


class PersonaAnalyzer:
    """Generates user personas using free AI models with GPU acceleration."""
    
    def __init__(self):
        """Initialize the analyzer with free Hugging Face models and optimal device selection."""
        logger.info("Initializing AI models with GPU acceleration...")
        
        self.device_info = self._setup_optimal_device()
        self.device = self.device_info['device']
        
        logger.info(f"🚀 Using device: {self.device_info['name']} ({self.device_info['type']})")
        if self.device_info['memory_gb']:
            logger.info(f"💾 Available GPU memory: {self.device_info['memory_gb']:.1f} GB")
        
        self._load_text_generation_model()
        self._load_sentiment_model()
        self._load_classification_model()
        
        logger.info("✅ All AI models loaded successfully with optimal device configuration")
    
    def _setup_optimal_device(self) -> Dict[str, any]:
        """Detect and setup the optimal device for AI model inference."""
        device_info = {
            'device': -1,  # Default to CPU
            'type': 'CPU',
            'name': 'CPU',
            'memory_gb': None
        }
        
        if torch.cuda.is_available():
            try:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)  # Convert to GB
                
                # Use GPU if it has enough memory (at least 2GB for our models)
                if gpu_memory_gb >= 2.0:
                    device_info.update({
                        'device': 0,  # First GPU
                        'type': 'CUDA GPU',
                        'name': f"{gpu_name} ({gpu_count} available)",
                        'memory_gb': gpu_memory_gb
                    })
                    
                    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
                    logger.info(f"🎯 GPU detected: {gpu_name} with {gpu_memory_gb:.1f}GB memory")
                else:
                    logger.warning(f"⚠️ GPU has insufficient memory ({gpu_memory_gb:.1f}GB < 2GB required)")
                    
            except Exception as e:
                logger.warning(f"⚠️ GPU detection failed: {e}, falling back to CPU")
        
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                device_info.update({
                    'device': 'mps',
                    'type': 'Apple Silicon GPU',
                    'name': 'Apple Silicon (MPS)',
                    'memory_gb': None  # MPS doesn't report memory easily
                })
                logger.info("🍎 Apple Silicon GPU (MPS) detected")
            except Exception as e:
                logger.warning(f"⚠️ MPS setup failed: {e}, falling back to CPU")
        
        else:
            logger.info("💻 No GPU detected, using CPU (this will be slower but still works!)")
        
        return device_info
    
    def _load_text_generation_model(self):
        """Load text generation model with optimal settings."""
        try:
            logger.info("📥 Loading FLAN-T5 text generation model...")
            
            # Try large model first if GPU available
            if self.device_info['type'] != 'CPU':
                try:
                    self.generator = pipeline(
                        "text2text-generation",
                        model="google/flan-t5-large",
                        device=self.device,
                        torch_dtype=torch.float16 if self.device_info['type'] == 'CUDA GPU' else torch.float32,
                        max_length=512
                    )
                    logger.info("✅ FLAN-T5-Large loaded on GPU with float16 precision")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load large model on GPU: {e}")
            
            model_name = "google/flan-t5-base" if self.device_info['memory_gb'] and self.device_info['memory_gb'] < 6 else "google/flan-t5-large"
            
            self.generator = pipeline(
                "text2text-generation",
                model=model_name,
                device=self.device if self.device_info['type'] != 'CPU' else -1,
                max_length=256 if 'base' in model_name else 512
            )
            
            model_size = model_name.split('-')[-1]
            logger.info(f"✅ FLAN-T5-{model_size.title()} loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load text generation model: {e}")
            raise
    
    def _load_sentiment_model(self):
        """Load sentiment analysis model."""
        try:
            logger.info("📥 Loading sentiment analysis model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device if self.device_info['type'] != 'CPU' else -1
            )
            logger.info("✅ Sentiment analysis model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Sentiment model loading failed: {e}")
            self.sentiment_analyzer = None
    
    def _load_classification_model(self):
        """Load zero-shot classification model."""
        try:
            logger.info("📥 Loading classification model...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device if self.device_info['type'] != 'CPU' else -1
            )
            logger.info("✅ Classification model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Classification model loading failed: {e}")
            self.classifier = None
    
    def get_device_info(self) -> str:
        """Get formatted device information for display."""
        info = f"Device: {self.device_info['name']} ({self.device_info['type']})"
        if self.device_info['memory_gb']:
            info += f" | Memory: {self.device_info['memory_gb']:.1f}GB"
        return info
    
    def analyze_topics_and_interests(self, posts: List[RedditPost]) -> Dict[str, List[Tuple[str, str]]]:
        """Extract topics and interests from posts."""
        interest_categories = [
            "technology", "gaming", "sports", "music", "movies", "books", 
            "food", "travel", "fitness", "career", "education", "relationships",
            "politics", "science", "art", "fashion", "business", "health"
        ]
        
        categorized_interests = defaultdict(list)
        
        if self.classifier:
            for post in posts[:20]:  # Analyze top posts
                try:
                    result = self.classifier(post.content[:500], interest_categories)
                    if result['scores'][0] > 0.3:  # Confidence threshold
                        category = result['labels'][0]
                        categorized_interests[category].append((post.content[:200], post.url))
                except Exception as e:
                    logger.debug(f"Classification error: {e}")
                    continue
        else:
            logger.info("Using keyword-based interest analysis (classifier unavailable)")
            keyword_map = {
                'technology': ['tech', 'software', 'programming', 'code', 'AI', 'computer'],
                'gaming': ['game', 'gaming', 'play', 'steam', 'console', 'xbox', 'playstation'],
                'food': ['food', 'recipe', 'cooking', 'restaurant', 'eat', 'meal'],
                'movies': ['movie', 'film', 'cinema', 'watch', 'netflix'],
                'music': ['music', 'song', 'album', 'artist', 'band']
            }
            
            for post in posts[:20]:
                content_lower = post.content.lower()
                for category, keywords in keyword_map.items():
                    if any(keyword in content_lower for keyword in keywords):
                        categorized_interests[category].append((post.content[:200], post.url))
                        break  # Only assign to first matching category
        
        return dict(categorized_interests)
    
    def analyze_personality_traits(self, posts: List[RedditPost]) -> Dict[str, Tuple[str, List[str]]]:
        """Analyze personality traits from content."""
        traits = {}
        
        sentiments = []
        if self.sentiment_analyzer:
            for post in posts[:30]:
                try:
                    sentiment = self.sentiment_analyzer(post.content[:500])
                    sentiments.append(sentiment[0]['label'])
                except Exception as e:
                    logger.debug(f"Sentiment analysis failed for post: {e}")
                    continue
        
        if sentiments:
            positive_ratio = sentiments.count('LABEL_2') / len(sentiments)  # Positive
            if positive_ratio > 0.6:
                traits['Optimism'] = ('High', [p.url for p in posts[:3]])
            elif positive_ratio < 0.3:
                traits['Pessimism'] = ('Moderate', [p.url for p in posts[:3]])
        
        avg_length = sum(len(p.content) for p in posts) / len(posts) if posts else 0
        if avg_length > 300:
            traits['Detailed Communication'] = ('High', [p.url for p in posts[:2]])
        elif avg_length < 100:
            traits['Concise Communication'] = ('High', [p.url for p in posts[:2]])
        
        return traits
    
    def analyze_structured_demographics(self, posts: List[RedditPost]) -> Dict[str, str]:
        """Analyze structured demographic information for infographics."""
        all_content = ' '.join([post.content.lower() for post in posts])
        
        age_indicators = {
            'teens': ['school', 'homework', 'parents', 'high school'],
            'early_20s': ['college', 'university', 'dorm', 'student'],
            'mid_20s': ['first job', 'entry level', 'starting career'],
            'late_20s_early_30s': ['career', 'promotion', 'apartment', 'saving'],
            'mid_30s_plus': ['mortgage', 'kids', 'family', 'house', 'children']
        }
        
        age_scores = {}
        for age_group, indicators in age_indicators.items():
            score = sum(all_content.count(indicator) for indicator in indicators)
            age_scores[age_group] = score
        
        age_mapping = {
            'teens': '16-19',
            'early_20s': '20-23', 
            'mid_20s': '24-27',
            'late_20s_early_30s': '28-32',
            'mid_30s_plus': '33+'
        }
        
        likely_age = max(age_scores, key=age_scores.get) if age_scores else 'late_20s_early_30s'
        age_range = age_mapping.get(likely_age, '25-35')
        
        occupation_keywords = {
            'Technology Professional': ['programming', 'coding', 'software', 'developer', 'tech', 'engineer'],
            'Content Creator': ['design', 'art', 'creative', 'content', 'marketing', 'writer'],
            'Business Professional': ['business', 'sales', 'finance', 'management', 'corporate'],
            'Student': ['student', 'university', 'college', 'study', 'homework'],
            'Gaming Enthusiast': ['gaming', 'gamer', 'streamer', 'esports']
        }
        
        occupation_scores = {}
        for category, keywords in occupation_keywords.items():
            score = sum(all_content.count(keyword) for keyword in keywords)
            occupation_scores[category] = score
        
        occupation = max(occupation_scores, key=occupation_scores.get) if occupation_scores else 'Professional'
        
        tech_indicators = {
            'Early Adopter': ['beta', 'new feature', 'cutting edge', 'latest', 'first'],
            'Mainstream': ['popular', 'everyone uses', 'standard', 'common'],
            'Late Adopter': ['finally', 'just started', 'new to this', 'learning']
        }
        
        tech_scores = {}
        for level, indicators in tech_indicators.items():
            score = sum(all_content.count(indicator) for indicator in indicators)
            tech_scores[level] = score
        
        tech_adoption = max(tech_scores, key=tech_scores.get) if tech_scores else 'Mainstream'
        
        archetype_indicators = {
            'The Creator': ['create', 'build', 'make', 'design', 'art'],
            'The Explorer': ['try', 'new', 'discover', 'adventure', 'explore'],
            'The Optimizer': ['efficient', 'improve', 'better', 'optimize'],
            'The Connector': ['community', 'social', 'friends', 'share', 'help'],
            'The Analyst': ['data', 'research', 'analyze', 'study']
        }
        
        archetype_scores = {}
        for archetype, indicators in archetype_indicators.items():
            score = sum(all_content.count(indicator) for indicator in indicators)
            archetype_scores[archetype] = score
        
        user_archetype = max(archetype_scores, key=archetype_scores.get) if archetype_scores else 'The Explorer'
        
        return {
            'age': age_range,
            'occupation': occupation,
            'tech_adoption': tech_adoption,
            'archetype': user_archetype
        }
    
    def analyze_motivations(self, posts: List[RedditPost]) -> Dict[str, int]:
        """Analyze user motivations and score them 0-100 for progress bars."""
        all_content = ' '.join([post.content.lower() for post in posts])
        
        motivation_indicators = {
            'convenience': ['easy', 'simple', 'quick', 'convenient', 'fast', 'efficient'],
            'wellness': ['health', 'wellness', 'exercise', 'fitness', 'nutrition'],
            'speed': ['fast', 'quick', 'rapid', 'instant', 'immediate', 'speed'],
            'preferences': ['prefer', 'like', 'favorite', 'choice', 'select'],
            'comfort': ['comfort', 'cozy', 'relaxed', 'peaceful', 'safe'],
            'social_connection': ['friends', 'community', 'social', 'connect', 'share'],
            'achievement': ['goal', 'achieve', 'success', 'accomplish', 'win'],
            'learning': ['learn', 'education', 'knowledge', 'study', 'understand']
        }
        
        motivation_scores = {}
        base_score = 30  # Everyone gets a baseline
        
        for motivation, indicators in motivation_indicators.items():
            raw_score = sum(all_content.count(indicator) for indicator in indicators)
            bonus_score = min(raw_score * 8, 60)  # Cap bonus at 60
            final_score = min(base_score + bonus_score, 100)
            motivation_scores[motivation] = final_score
        
        return motivation_scores
    
    def analyze_personality_dimensions_visual(self, posts: List[RedditPost]) -> Dict[str, int]:
        """Analyze personality dimensions for visual scales."""
        all_content = ' '.join([post.content.lower() for post in posts])
        
        # Introvert vs Extrovert (0=Introvert, 100=Extrovert)
        introvert_words = ['alone', 'quiet', 'myself', 'solitude', 'home', 'private']
        extrovert_words = ['people', 'social', 'party', 'friends', 'group', 'outgoing']
        
        intro_score = sum(all_content.count(word) for word in introvert_words)
        extro_score = sum(all_content.count(word) for word in extrovert_words)
        
        total_social = intro_score + extro_score
        if total_social > 0:
            introvert_extrovert = int((extro_score / total_social) * 100)
        else:
            introvert_extrovert = 50
        
        intuition_words = ['possibility', 'future', 'theory', 'concept', 'idea', 'imagine']
        sensing_words = ['fact', 'detail', 'practical', 'experience', 'real', 'concrete']
        
        intuition_score = sum(all_content.count(word) for word in intuition_words)
        sensing_score = sum(all_content.count(word) for word in sensing_words)
        
        total_perception = intuition_score + sensing_score
        if total_perception > 0:
            intuition_sensing = int((sensing_score / total_perception) * 100)
        else:
            intuition_sensing = 65  # Slight sensing bias
        
        feeling_words = ['feel', 'emotion', 'heart', 'personal', 'values']
        thinking_words = ['logic', 'analyze', 'reason', 'objective', 'efficiency']
        
        feeling_score = sum(all_content.count(word) for word in feeling_words)
        thinking_score = sum(all_content.count(word) for word in thinking_words)
        
        total_decision = feeling_score + thinking_score
        if total_decision > 0:
            feeling_thinking = int((thinking_score / total_decision) * 100)
        else:
            feeling_thinking = 45  # Slight feeling bias
        
        perceiving_words = ['flexible', 'spontaneous', 'open', 'adapt', 'explore']
        judging_words = ['plan', 'schedule', 'deadline', 'organize', 'structure']
        
        perceiving_score = sum(all_content.count(word) for word in perceiving_words)
        judging_score = sum(all_content.count(word) for word in judging_words)
        
        total_lifestyle = perceiving_score + judging_score
        if total_lifestyle > 0:
            perceiving_judging = int((judging_score / total_lifestyle) * 100)
        else:
            perceiving_judging = 55  # Slight judging bias
        
        return {
            'introvert_extrovert': introvert_extrovert,
            'intuition_sensing': intuition_sensing,
            'feeling_thinking': feeling_thinking,
            'perceiving_judging': perceiving_judging
        }
    
    def generate_trait_tags(self, posts: List[RedditPost]) -> List[str]:
        """Generate personality trait tags for visual display."""
        all_content = ' '.join([post.content.lower() for post in posts])
        
        trait_indicators = {
            'Practical': ['practical', 'useful', 'realistic', 'functional'],
            'Adaptable': ['adapt', 'flexible', 'adjust', 'versatile'],
            'Spontaneous': ['spontaneous', 'impulsive', 'random', 'sudden'],
            'Active': ['active', 'busy', 'energetic', 'engaged'],
            'Analytical': ['analyze', 'think', 'reason', 'systematic'],
            'Creative': ['creative', 'artistic', 'innovative', 'original'],
            'Social': ['social', 'outgoing', 'friendly', 'community'],
            'Curious': ['curious', 'question', 'explore', 'investigate'],
            'Optimistic': ['positive', 'optimistic', 'hopeful', 'great'],
            'Tech-Savvy': ['technology', 'tech', 'digital', 'online']
        }
        
        trait_scores = {}
        for trait, indicators in trait_indicators.items():
            score = sum(all_content.count(indicator) for indicator in indicators)
            if score > 0:
                trait_scores[trait] = score
        
        top_traits = sorted(trait_scores.items(), key=lambda x: x[1], reverse=True)[:4]
        
        if len(top_traits) < 4:
            default_traits = ['Active', 'Curious', 'Social', 'Adaptable']
            existing_traits = [trait for trait, _ in top_traits]
            for default in default_traits:
                if default not in existing_traits and len(top_traits) < 4:
                    top_traits.append((default, 1))
        
        return [trait for trait, score in top_traits[:4]]
    
    def generate_behavioral_insights(self, posts: List[RedditPost]) -> Dict[str, List[str]]:
        """Generate detailed behavioral insights, frustrations, and goals."""
        insights = {
            'behaviors': [],
            'frustrations': [],
            'goals': []
        }
        
        subreddit_activity = defaultdict(int)
        for post in posts:
            subreddit_activity[post.subreddit] += 1
        
        top_communities = sorted(subreddit_activity.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if top_communities:
            main_community, count = top_communities[0]
            
            if any(keyword in main_community.lower() for keyword in ['game', 'gaming', 'civ', 'lords', 'warriors']):
                insights['behaviors'].append(f"Actively participates in gaming communities, spending considerable time in r/{main_community} discussing strategy, gameplay mechanics, and sharing experiences with fellow gamers.")
                insights['behaviors'].append(f"Shows dedication to specific games by maintaining regular engagement, with {count} interactions in their primary gaming community.")
            
            if 'visionpro' in main_community.lower() or 'tech' in main_community.lower():
                insights['behaviors'].append(f"Early adopter of new technology, particularly interested in VR/AR experiences and regularly shares content about cutting-edge tech products.")
            
            if 'ask' in main_community.lower():
                insights['behaviors'].append("Frequently seeks advice and information from online communities, demonstrating a preference for crowdsourced knowledge and diverse perspectives.")
        
        post_frequency = len(posts)
        if post_frequency > 40:
            insights['behaviors'].append("Maintains a highly active online presence with consistent engagement across multiple communities and topics.")
        
        frustration_indicators = ['annoying', 'frustrating', 'hate', 'stupid', 'broken', 'terrible', 'worst', 'confusing', 'unclear']
        problem_content = []
        
        for post in posts[:30]:
            content_lower = post.content.lower()
            if any(indicator in content_lower for indicator in frustration_indicators):
                problem_content.append(post)
        
        if problem_content:
            insights['frustrations'].append("Experiences frustration with poorly designed user interfaces and unclear navigation systems in digital products.")
            insights['frustrations'].append("Gets annoyed when platforms lack proper categorization, search functionality, or when content descriptions are insufficient.")
            if any('menu' in p.content.lower() or 'interface' in p.content.lower() for p in problem_content):
                insights['frustrations'].append("Finds complex or confusing menu systems particularly irritating, especially when trying to find specific features or content.")
        
        goal_indicators = ['want', 'hope', 'goal', 'trying', 'learning', 'improve', 'better', 'wish', 'looking for']
        goal_content = []
        
        for post in posts[:30]:
            content_lower = post.content.lower()
            if any(indicator in content_lower for indicator in goal_indicators):
                goal_content.append(post)
        
        if goal_content:
            insights['goals'].append("Seeks to optimize and enhance their digital experiences through community knowledge sharing and staying informed about best practices.")
            insights['goals'].append("Wants to make informed decisions about technology purchases and entertainment choices by leveraging community insights and reviews.")
            
            if any('game' in p.content.lower() for p in goal_content):
                insights['goals'].append("Aims to improve their gaming skills and strategies through community discussion and learning from experienced players.")
        
        if any(keyword in ' '.join([p.content.lower() for p in posts[:20]]) for keyword in ['efficiency', 'productive', 'organize']):
            insights['goals'].append("Strives to use technology more efficiently and productively in their daily life and work routines.")
        
        if not insights['behaviors']:
            insights['behaviors'].append("Engages regularly with online communities, showing consistent participation patterns and genuine interest in connecting with like-minded individuals.")
            insights['behaviors'].append("Demonstrates thoughtful communication by contributing meaningful content and asking relevant questions to community discussions.")
        
        if not insights['frustrations']:
            insights['frustrations'].append("Encounters challenges with information discovery and decision-making when platforms don't provide clear, comprehensive details about products or services.")
            insights['frustrations'].append("Feels overwhelmed when digital interfaces are cluttered or when there are too many options without clear guidance.")
        
        if not insights['goals']:
            insights['goals'].append("Aims to build knowledge and meaningful connections within their communities of interest while staying current with trends and developments.")
            insights['goals'].append("Seeks to make well-informed decisions by gathering multiple perspectives and experiences from trusted community sources.")
        
        return insights


class PersonaGenerator:
    """Generates professional user persona documents with rich behavioral insights."""
    
    def __init__(self, analyzer: PersonaAnalyzer):
        self.analyzer = analyzer
    
    def generate_persona(self, username: str, posts: List[RedditPost]) -> str:
        """Generate complete user persona with behavioral insights."""
        if not posts:
            return f"Unable to generate persona for {username} - insufficient data."
        
        interests = self.analyzer.analyze_topics_and_interests(posts)
        personality = self.analyzer.analyze_personality_traits(posts)
        behavioral_insights = self.analyzer.generate_behavioral_insights(posts)
        
        demographics = self.analyzer.analyze_structured_demographics(posts)
        motivations = self.analyzer.analyze_motivations(posts)
        personality_dims = self.analyzer.analyze_personality_dimensions_visual(posts)
        trait_tags = self.analyzer.generate_trait_tags(posts)
        
        top_subreddits = self._get_top_subreddits(posts)
        activity_pattern = self._analyze_activity_pattern(posts)
        representative_quote = self._find_representative_quote(posts)
        
        persona = self._format_enhanced_persona_document(
            username, posts, interests, personality, behavioral_insights,
            top_subreddits, activity_pattern, representative_quote, 
            demographics, motivations, personality_dims, trait_tags, self.analyzer
        )
        
        return persona
    
    def _get_top_subreddits(self, posts: List[RedditPost]) -> List[Tuple[str, int]]:
        """Get user's most active subreddits."""
        subreddit_counts = defaultdict(int)
        for post in posts:
            subreddit_counts[post.subreddit] += 1
        
        return sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _analyze_activity_pattern(self, posts: List[RedditPost]) -> str:
        """Analyze user's Reddit activity patterns."""
        if not posts:
            return "Limited activity data available."
        
        avg_score = sum(p.score for p in posts) / len(posts)
        post_count = sum(1 for p in posts if p.post_type == 'post')
        comment_count = len(posts) - post_count
        
        return f"Average engagement: {avg_score:.1f} points, {post_count} posts, {comment_count} comments"
    
    def _find_representative_quote(self, posts: List[RedditPost]) -> Tuple[str, str]:
        """Find a representative quote from user's content."""
        for post in sorted(posts, key=lambda x: x.score, reverse=True):
            if len(post.content) > 50 and len(post.content) < 300:
                quote = post.content.replace('\n', ' ').strip()
                if len(quote) > 150:
                    quote = quote[:147] + "..."
                return quote, post.url
        
        return "Active Reddit contributor with diverse interests.", posts[0].url if posts else ""
    
    def _create_progress_bar(self, value: int, width: int = 20) -> str:
        """Create ASCII progress bar."""
        filled = int((value / 100) * width)
        bar = '█' * filled + '░' * (width - filled)
        return f"{bar} {value}%"
    
    def _create_personality_scale(self, value: int, left_label: str, right_label: str, width: int = 30) -> str:
        """Create personality dimension scale."""
        position = int((value / 100) * width)
        scale = '─' * width
        scale = scale[:position] + '●' + scale[position+1:]
        return f"{left_label:<12} {scale} {right_label:>12}"
    
    def _format_enhanced_persona_document(self, username: str, posts: List[RedditPost], 
                                        interests: Dict, personality: Dict, behavioral_insights: Dict,
                                        top_subreddits: List, activity_pattern: str, 
                                        representative_quote: Tuple[str, str], 
                                        demographics: Dict, motivations: Dict, personality_dims: Dict, 
                                        trait_tags: List[str], analyzer: PersonaAnalyzer) -> str:
        """Format the enhanced persona document with rich behavioral insights."""
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        persona = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           USER PERSONA ANALYSIS                              ║
║                          Reddit User: {username:<25}                    ║
║                          Generated: {current_time}                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

👤 DEMOGRAPHICS & OVERVIEW
═══════════════════════════════════════════════════════════════════════════════
Username: {username}
Platform Activity: {activity_pattern}
Primary Communities: {', '.join([sub[0] for sub in top_subreddits[:3]])}
Analysis Date: {current_time}

🎯 BEHAVIOUR & HABITS
═══════════════════════════════════════════════════════════════════════════════"""

        for behavior in behavioral_insights['behaviors']:
            persona += f"\n• {behavior}\n"

        persona += f"""
😤 FRUSTRATIONS
═══════════════════════════════════════════════════════════════════════════════"""

        for frustration in behavioral_insights['frustrations']:
            persona += f"\n• {frustration}\n"

        persona += f"""
🎯 GOALS & NEEDS
═══════════════════════════════════════════════════════════════════════════════"""

        for goal in behavioral_insights['goals']:
            persona += f"\n• {goal}\n"

        persona += f"""
📊 PERSONALITY TRAITS
═══════════════════════════════════════════════════════════════════════════════"""

        if personality:
            for trait, (level, citations) in personality.items():
                persona += f"\n• {trait}: {level}"
                persona += f"\n  Supporting Evidence: {citations[0] if citations else 'General posting patterns'}\n"
        else:
            persona += "\n• Communication Style: Active Reddit contributor"
            persona += f"\n  Supporting Evidence: {posts[0].url if posts else 'N/A'}\n"

        persona += f"""

📊 MOTIVATION ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

CONVENIENCE        {self._create_progress_bar(motivations['convenience'])}
WELLNESS           {self._create_progress_bar(motivations['wellness'])}
SPEED              {self._create_progress_bar(motivations['speed'])}
PREFERENCES        {self._create_progress_bar(motivations['preferences'])}
COMFORT            {self._create_progress_bar(motivations['comfort'])}
SOCIAL CONNECTION  {self._create_progress_bar(motivations['social_connection'])}
ACHIEVEMENT        {self._create_progress_bar(motivations['achievement'])}
LEARNING           {self._create_progress_bar(motivations['learning'])}

🧠 PERSONALITY DIMENSIONS
═══════════════════════════════════════════════════════════════════════════════

{self._create_personality_scale(personality_dims['introvert_extrovert'], "INTROVERT", "EXTROVERT")}

{self._create_personality_scale(personality_dims['intuition_sensing'], "INTUITION", "SENSING")}

{self._create_personality_scale(personality_dims['feeling_thinking'], "FEELING", "THINKING")}

{self._create_personality_scale(personality_dims['perceiving_judging'], "PERCEIVING", "JUDGING")}

🏷️ PERSONALITY TAGS
───────────────────────────────────────────────────────────────────────────────"""

        for i in range(0, len(trait_tags), 4):
            row_tags = trait_tags[i:i+4]
            tag_line = "  ".join([f"[{tag}]" for tag in row_tags])
            persona += f"\n{tag_line}"
        
        persona += f"""

🎨 INTERESTS & ACTIVITIES
═══════════════════════════════════════════════════════════════════════════════"""
        if interests:
            for category, examples in list(interests.items())[:5]:
                persona += f"\n• {category.title()}: Actively engaged"
                if examples:
                    persona += f"\n  Evidence: {examples[0][1]}\n"
        else:
            persona += f"\n• Diverse Interests: Based on community participation"
            persona += f"\n  Evidence: Active in {top_subreddits[0][0] if top_subreddits else 'various'} communities\n"

        persona += f"""
💬 REPRESENTATIVE QUOTE
═══════════════════════════════════════════════════════════════════════════════
"{representative_quote[0]}"

Source: {representative_quote[1]}

📈 ACTIVITY ANALYSIS
═══════════════════════════════════════════════════════════════════════════════
Top Communities:"""

        for sub, count in top_subreddits:
            persona += f"\n• r/{sub}: {count} interactions"

        persona += f"""

🔗 DATA SOURCES & CITATIONS
═══════════════════════════════════════════════════════════════════════════════
Analysis based on {len(posts)} posts and comments from user's Reddit profile.
Profile URL: https://www.reddit.com/user/{username}/
Processing Device: {analyzer.get_device_info()}

Recent Activity Sources:"""

        for i, post in enumerate(posts[:5]):
            persona += f"\n{i+1}. {post.post_type.title()}: {post.url}"

        persona += f"""

═══════════════════════════════════════════════════════════════════════════════
Generated by Reddit Persona Analyzer
AI/LLM Engineer Assignment - BeyondChats
═══════════════════════════════════════════════════════════════════════════════
"""
        
        return persona


def main():
    """Main function to run the Reddit Persona Analyzer."""
    parser = argparse.ArgumentParser(description='Reddit Persona Analyzer')
    parser.add_argument('profile_url', help='Reddit user profile URL')
    parser.add_argument('--limit', type=int, default=100, help='Number of posts to analyze')
    args = parser.parse_args()
    
    client_id = os.getenv('REDDIT_CLIENT_ID', 'your_client_id_here')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET', 'your_client_secret_here')
    user_agent = os.getenv('REDDIT_USER_AGENT', 'PersonaAnalyzer/1.0 by YourUsername')
    
    if client_id == 'your_client_id_here':
        logger.error("Please set Reddit API credentials in environment variables or edit the script")
        logger.error("Required: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT")
        return
    
    try:
        scraper = RedditScraper(client_id, client_secret, user_agent)
        analyzer = PersonaAnalyzer()
        generator = PersonaGenerator(analyzer)
        
        username = scraper.extract_username(args.profile_url)
        logger.info(f"Analyzing user: {username}")
        
        posts = scraper.scrape_user_content(username, args.limit)
        
        if not posts:
            logger.error(f"No content found for user {username}")
            return
        
        persona = generator.generate_persona(username, posts)
        
        output_filename = f"{username}_persona.txt"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(persona)
        
        logger.info(f"Persona saved to {output_filename}")
        print(f"\n✅ Persona analysis complete! Saved to: {output_filename}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
