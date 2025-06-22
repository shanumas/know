import requests
import time
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from content_extractor import ContentExtractor
from text_cleaner import TextCleaner

class HackerNewsDataManager:
    """Manages data fetching from HackerNews API"""
    
    def __init__(self, extract_url_content: bool = True):
        self.base_url = "https://hacker-news.firebaseio.com/v0"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HackerNews-RAG-App/1.0'
        })
        
        # Rate limiting
        self.requests_per_second = 10
        self.last_request_time = 0
        
        # Content extraction
        self.extract_url_content = extract_url_content
        self.content_extractor = ContentExtractor() if extract_url_content else None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _rate_limit(self):
        """Implement rate limiting to respect API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, max_retries: int = 3) -> Optional[Dict]:
        """Make HTTP request with retry logic"""
        self._rate_limit()
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Max retries exceeded for {url}")
                    return None
    
    def get_item(self, item_id: int) -> Optional[Dict]:
        """Fetch a single item by ID"""
        url = f"{self.base_url}/item/{item_id}.json"
        return self._make_request(url)
    
    def get_new_stories(self, limit: int = 500) -> List[int]:
        """Fetch new story IDs"""
        url = f"{self.base_url}/newstories.json"
        story_ids = self._make_request(url)
        
        if story_ids:
            return story_ids[:limit]
        return []
    
    def fetch_story_details(self, story_id: int) -> Optional[Dict]:
        """Fetch detailed story information including comments"""
        story = self.get_item(story_id)
        
        if not story or story.get('type') != 'story':
            return None
        
        # Filter out deleted or dead stories
        if story.get('deleted') or story.get('dead'):
            return None
        
        # Ensure required fields exist
        if not story.get('title'):
            return None
        
        # Fetch top-level comments if they exist
        comments = []
        if story.get('kids'):
            # Limit to first 5 comments to avoid too much data
            comment_ids = story['kids'][:5]
            comments = self._fetch_comments(comment_ids)
        
        story_data = {
            'id': story['id'],
            'title': story['title'],
            'text': story.get('text', ''),
            'url': story.get('url', ''),
            'score': story.get('score', 0),
            'by': story.get('by', ''),
            'time': story.get('time', 0),
            'descendants': story.get('descendants', 0),
            'comments': comments,
            'type': 'story'
        }
        
        # Extract content from URL if enabled and URL exists
        if self.extract_url_content and self.content_extractor and story_data.get('url'):
            try:
                story_data = self.content_extractor.enhance_document_with_url_content(story_data)
                if story_data.get('extracted_content'):
                    self.logger.info(f"Successfully extracted {story_data.get('content_type', 'content')} content for story {story['id']}")
            except Exception as e:
                self.logger.warning(f"Failed to extract URL content for story {story['id']}: {e}")
                story_data['extraction_error'] = str(e)

        story_data['text'] = TextCleaner.clean_extracted_text(story_data.get('text', '').strip())
        
        return story_data
    
    def _fetch_comments(self, comment_ids: List[int]) -> List[Dict]:
        """Fetch comment details for given IDs"""
        comments = []
        
        # Use threading for parallel comment fetching
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_id = {
                executor.submit(self.get_item, comment_id): comment_id 
                for comment_id in comment_ids
            }
            
            for future in as_completed(future_to_id):
                comment = future.result()
                if comment and not comment.get('deleted') and comment.get('text'):
                    comments.append({
                        'id': comment['id'],
                        'text': comment['text'],
                        'by': comment.get('by', ''),
                        'time': comment.get('time', 0),
                        'parent': comment.get('parent', 0)
                    })
        
        return comments
    
    def fetch_new_stories(self, limit: int = 100) -> List[Dict]:
        """Fetch new stories with full details"""
        story_ids = self.get_new_stories(limit * 2)  # Get more IDs in case some are filtered out
        stories = []
        
        self.logger.info(f"Fetching details for {len(story_ids)} stories...")
        
        # Use threading for parallel story fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_id = {
                executor.submit(self.fetch_story_details, story_id): story_id 
                for story_id in story_ids
            }
            
            for future in as_completed(future_to_id):
                story = future.result()
                if story:
                    stories.append(story)
                    if len(stories) >= limit:
                        break
        
        self.logger.info(f"Successfully fetched {len(stories)} stories")
        return stories
    
    def get_user_info(self, username: str) -> Optional[Dict]:
        """Fetch user information"""
        url = f"{self.base_url}/user/{username}.json"
        return self._make_request(url)
