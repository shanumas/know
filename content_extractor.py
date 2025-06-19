import re
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs
import requests
import json

# Import optional dependencies
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

class ContentExtractor:
    """Extract text content from URLs and YouTube videos"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HackerNews-RAG-App/1.0 (Content Extractor)'
        })
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from URL - handles web pages and YouTube videos
        
        Returns:
        {
            'text': str,
            'title': str,
            'content_type': str,  # 'webpage', 'youtube', 'unknown'
            'success': bool,
            'error': str or None
        }
        """
        try:
            if self._is_youtube_url(url):
                return self._extract_youtube_content(url)
            else:
                return self._extract_webpage_content(url)
        except Exception as e:
            self.logger.error(f"Failed to extract content from {url}: {e}")
            return {
                'text': '',
                'title': '',
                'content_type': 'unknown',
                'success': False,
                'error': str(e)
            }
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube video"""
        youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
        parsed = urlparse(url)
        return parsed.netloc.lower() in youtube_domains
    
    def _extract_youtube_content(self, url: str) -> Dict[str, Any]:
        """Extract transcript and metadata from YouTube video"""
        if not YT_DLP_AVAILABLE:
            return {
                'text': '',
                'title': '',
                'content_type': 'youtube',
                'success': False,
                'error': 'yt-dlp not available for YouTube content extraction'
            }
        
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitlesformat': 'best',
                'skip_download': True,
                'subtitleslangs': ['en', 'en-US', 'en-GB'],
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video info
                info = ydl.extract_info(url, download=False)
                
                title = info.get('title', '')
                description = info.get('description', '')
                
                # Try to get subtitles/transcript
                transcript_text = self._extract_subtitles(info)
                
                # Combine all text content
                content_parts = []
                if title:
                    content_parts.append(f"Title: {title}")
                if description:
                    content_parts.append(f"Description: {description[:1000]}")  # Limit description
                if transcript_text:
                    content_parts.append(f"Transcript: {transcript_text}")
                
                combined_text = '\n\n'.join(content_parts)
                
                return {
                    'text': combined_text,
                    'title': title,
                    'content_type': 'youtube',
                    'success': True,
                    'error': None
                }
                
        except Exception as e:
            self.logger.error(f"Failed to extract YouTube content: {e}")
            return {
                'text': '',
                'title': '',
                'content_type': 'youtube',
                'success': False,
                'error': f"YouTube extraction failed: {str(e)}"
            }
    
    def _extract_subtitles(self, info: Dict) -> str:
        """Extract subtitle text from video info"""
        subtitles = info.get('subtitles', {})
        automatic_captions = info.get('automatic_captions', {})
        
        # Try manual subtitles first, then automatic captions
        for subs_dict in [subtitles, automatic_captions]:
            for lang in ['en', 'en-US', 'en-GB']:
                if lang in subs_dict:
                    sub_list = subs_dict[lang]
                    for sub in sub_list:
                        if sub.get('ext') in ['vtt', 'srt', 'ttml']:
                            try:
                                # Download subtitle content
                                response = requests.get(sub['url'], timeout=10)
                                if response.status_code == 200:
                                    return self._parse_subtitle_content(response.text, sub.get('ext', 'vtt'))
                            except Exception as e:
                                self.logger.warning(f"Failed to download subtitles: {e}")
                                continue
        
        return ""
    
    def _parse_subtitle_content(self, content: str, format_type: str) -> str:
        """Parse subtitle content and extract text"""
        lines = content.split('\n')
        text_lines = []
        
        if format_type == 'vtt':
            # WebVTT format
            for line in lines:
                line = line.strip()
                # Skip metadata lines, timestamps, and empty lines
                if (line and 
                    not line.startswith('WEBVTT') and 
                    not line.startswith('NOTE') and
                    not re.match(r'^\d+$', line) and
                    not re.match(r'^\d{2}:\d{2}:\d{2}', line) and
                    not '-->' in line):
                    # Remove HTML tags and add to text
                    clean_line = re.sub(r'<[^>]+>', '', line)
                    if clean_line:
                        text_lines.append(clean_line)
        
        elif format_type == 'srt':
            # SRT format
            for line in lines:
                line = line.strip()
                # Skip sequence numbers, timestamps, and empty lines
                if (line and 
                    not re.match(r'^\d+$', line) and
                    not re.match(r'^\d{2}:\d{2}:\d{2}', line) and
                    not '-->' in line):
                    text_lines.append(line)
        
        # Join lines and clean up
        text = ' '.join(text_lines)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_webpage_content(self, url: str) -> Dict[str, Any]:
        """Extract text content from web page"""
        if not TRAFILATURA_AVAILABLE:
            return self._fallback_webpage_extraction(url)
        
        try:
            # Use trafilatura for robust content extraction
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return {
                    'text': '',
                    'title': '',
                    'content_type': 'webpage',
                    'success': False,
                    'error': 'Failed to download webpage'
                }
            
            # Extract main content
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            
            # Extract title separately
            title = trafilatura.extract_metadata(downloaded)
            title_text = title.title if title and title.title else ''
            
            if not text:
                text = ''
            
            return {
                'text': text,
                'title': title_text,
                'content_type': 'webpage',
                'success': bool(text or title_text),
                'error': None if (text or title_text) else 'No content extracted'
            }
            
        except Exception as e:
            self.logger.error(f"Trafilatura extraction failed: {e}")
            return self._fallback_webpage_extraction(url)
    
    def _fallback_webpage_extraction(self, url: str) -> Dict[str, Any]:
        """Fallback method for webpage content extraction"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            content = response.text
            
            # Simple HTML parsing fallback
            import re
            
            # Extract title
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else ''
            
            # Remove script and style elements
            content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove HTML tags
            content = re.sub(r'<[^>]+>', ' ', content)
            
            # Clean up whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Limit content length
            if len(content) > 5000:
                content = content[:5000] + "..."
            
            return {
                'text': content,
                'title': title,
                'content_type': 'webpage',
                'success': bool(content or title),
                'error': None if (content or title) else 'No content extracted'
            }
            
        except Exception as e:
            self.logger.error(f"Fallback extraction failed: {e}")
            return {
                'text': '',
                'title': '',
                'content_type': 'webpage',
                'success': False,
                'error': f"Webpage extraction failed: {str(e)}"
            }
    
    def enhance_document_with_url_content(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a document by extracting content from its URL"""
        enhanced_doc = document.copy()
        
        url = document.get('url')
        if not url:
            return enhanced_doc
        
        # Extract content from URL
        extracted = self.extract_content(url)
        
        if extracted['success'] and extracted['text']:
            # Add extracted content to document
            enhanced_doc['extracted_content'] = extracted['text']
            enhanced_doc['extracted_title'] = extracted['title']
            enhanced_doc['content_type'] = extracted['content_type']
            
            # If document doesn't have text content, use extracted content
            if not document.get('text'):
                enhanced_doc['text'] = extracted['text']
            
            self.logger.info(f"Enhanced document {document.get('id', 'unknown')} with {extracted['content_type']} content")
        else:
            enhanced_doc['extraction_error'] = extracted.get('error', 'Unknown error')
            self.logger.warning(f"Failed to extract content for document {document.get('id', 'unknown')}: {extracted.get('error')}")
        
        return enhanced_doc