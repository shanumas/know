from datetime import datetime, timezone, timedelta
from typing import Any
import time

def format_timestamp(dt: datetime) -> str:
    """Format datetime for display"""
    if dt.tzinfo is None:
        local_offset = -time.timezone if time.localtime().tm_isdst == 0 else -time.altzone
        dt = dt - timedelta(seconds=local_offset)
        dt = dt.replace(tzinfo=timezone.utc)
    
    now = datetime.now(timezone.utc)
    diff = now - dt
    
    if diff.days > 0:
        if diff.days == 1:
            return "1 day ago"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        elif diff.days < 30:
            weeks = diff.days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        else:
            months = diff.days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
    
    hours = diff.seconds // 3600
    if hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    
    minutes = diff.seconds // 60
    if minutes > 0:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    
    return "Just now"

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to specified length with ellipsis"""
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + "..."

def clean_html(text: str) -> str:
    """Basic HTML cleaning for HackerNews content"""
    if not text:
        return ""
    
    # Replace common HTML entities
    replacements = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#x27;': "'",
        '&#x2F;': '/',
        '<p>': '\n',
        '</p>': '',
        '<br>': '\n',
        '<br/>': '\n',
        '<br />': '\n'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.strip()

def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    if not url:
        return ""
    
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return url

def calculate_reading_time(text: str) -> int:
    """Calculate estimated reading time in minutes"""
    if not text:
        return 0
    
    # Average reading speed: 200 words per minute
    word_count = len(text.split())
    reading_time = max(1, round(word_count / 200))
    
    return reading_time

def format_score(score: int) -> str:
    """Format score for display"""
    if score >= 1000:
        return f"{score/1000:.1f}k"
    return str(score)

def validate_hackernews_id(item_id: Any) -> bool:
    """Validate HackerNews item ID"""
    try:
        id_int = int(item_id)
        return id_int > 0
    except (ValueError, TypeError):
        return False
