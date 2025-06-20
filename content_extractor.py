import re
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs
import requests
import json

MAX_DESCRIPTION_LENGTH = 100000  # 100,000 characters is the maximum length for a description

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
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Add common headers to appear more like a real browser
        self.session.headers.update({
            'Accept':
            'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
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
        self.logger.info(f"Extracting content from URL: {url}")
        try:
            if self._is_youtube_url(url):
                self.logger.info(f"Is youtube URL: {url}")
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
        youtube_domains = [
            'youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com'
        ]
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
            # Configure yt-dlp options for better success rate
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitlesformat': 'vtt',
                'skip_download': True,
                'subtitleslangs': ['en', 'en-US', 'en-GB', 'en-auto'],
                'ignoreerrors': True,
                'extract_flat': False,
                'format':
                'worst[height<=144]',  # We don't need video, just metadata
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video info
                self.logger.info(f"Extracting YouTube content from: {url}")
                info = ydl.extract_info(url, download=False)

                if not info:
                    raise Exception("Could not extract video information")

                title = info.get('title', '')
                description = info.get('description', '')
                duration = info.get('duration', 0)

                # Try multiple methods to get transcript
                transcript_text = ''

                self.logger.info(
                    f"Before collecting Transcript text:  {transcript_text}")
                # Method 1: Try to get subtitles from info
                if info.get('subtitles') or info.get('automatic_captions'):
                    self.logger.info(
                        f"Inside Transcript text collection:  {transcript_text}"
                    )
                    transcript_text = self._extract_subtitles(info)
                    self.logger.info(
                        f"After subtitles text collection:  {transcript_text}")
                    # Method 2: If no subtitles found, try alternative approach
                    #if not transcript_text:
                    transcript_text += self._extract_transcript_alternative(
                        url)
                    self.logger.info(f"Transcript text:  {transcript_text}")

                # Combine all text content
                content_parts = []
                if title:
                    content_parts.append(f"Video Title: {title}")
                if description and len(description.strip()) > 10:
                    # Clean and limit description
                    clean_desc = description.replace('\n', ' ').strip()
                    self.logger.info('Clean_desc length: ' +
                                     str(len(clean_desc)))
                    if len(clean_desc) > MAX_DESCRIPTION_LENGTH:
                        clean_desc = clean_desc[:MAX_DESCRIPTION_LENGTH] + "..."
                    content_parts.append(f"Description: {clean_desc}")
                if transcript_text:
                    content_parts.append(f"Transcript: {transcript_text}")

                combined_text = '\n\n'.join(content_parts)

                success = bool(title or description or transcript_text)

                self.logger.info(
                    f"YouTube extraction result - Title: {bool(title)}, Description: {bool(description)}, Transcript: {bool(transcript_text)}"
                )

                return {
                    'text': combined_text,
                    'title': title,
                    'content_type': 'youtube',
                    'success': success,
                    'error':
                    None if success else 'No content could be extracted'
                }

        except Exception as e:
            self.logger.error(f"Failed to extract YouTube content: {e}")
            # Try fallback method for YouTube metadata
            fallback_result = self._extract_youtube_fallback(url)
            if fallback_result['success']:
                return fallback_result

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
                                    return self._parse_subtitle_content(
                                        response.text, sub.get('ext', 'vtt'))
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to download subtitles: {e}")
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
                if (line and not line.startswith('WEBVTT')
                        and not line.startswith('NOTE')
                        and not re.match(r'^\d+$', line)
                        and not re.match(r'^\d{2}:\d{2}:\d{2}', line)
                        and not '-->' in line):
                    # Remove HTML tags and add to text
                    clean_line = re.sub(r'<[^>]+>', '', line)
                    if clean_line:
                        text_lines.append(clean_line)

        elif format_type == 'srt':
            # SRT format
            for line in lines:
                line = line.strip()
                # Skip sequence numbers, timestamps, and empty lines
                if (line and not re.match(r'^\d+$', line)
                        and not re.match(r'^\d{2}:\d{2}:\d{2}', line)
                        and not '-->' in line):
                    text_lines.append(line)

        # Join lines and clean up
        text = ' '.join(text_lines)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _extract_transcript_alternative(self, url: str) -> str:
        """Alternative method to extract transcript using simpler approach"""
        self.logger.info(f"Transcript alternative extraction:")
        try:
            # Try to get basic video info and any available transcript
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en'],
                'skip_download': True,
                'ignoreerrors': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # Try to get automatic captions
                auto_captions = info.get('automatic_captions', {})
                if 'en' in auto_captions:
                    for caption in auto_captions['en']:
                        if caption.get('ext') == 'vtt':
                            try:
                                response = requests.get(caption['url'],
                                                        timeout=15)
                                if response.status_code == 200:
                                    return self._parse_subtitle_content(
                                        response.text, 'vtt')
                            except:
                                continue

                return ""
        except Exception as e:
            self.logger.debug(f"Alternative transcript extraction failed: {e}")
            return ""

    def _extract_youtube_fallback(self, url: str) -> Dict[str, Any]:
        """Fallback method to extract basic YouTube metadata from page HTML"""
        try:
            # Parse video ID from URL
            video_id = None
            if 'youtube.com/watch?v=' in url:
                video_id = url.split('v=')[1].split('&')[0]
            elif 'youtu.be/' in url:
                video_id = url.split('youtu.be/')[1].split('?')[0]

            if not video_id:
                return {
                    'text': '',
                    'title': '',
                    'content_type': 'youtube',
                    'success': False,
                    'error': 'Could not parse video ID'
                }

            # Try to get basic page content
            watch_url = f"https://www.youtube.com/watch?v={video_id}"
            headers = {
                'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            }

            response = requests.get(watch_url, headers=headers, timeout=15)
            if response.status_code != 200:
                return {
                    'text': '',
                    'title': '',
                    'content_type': 'youtube',
                    'success': False,
                    'error': 'Could not access YouTube page'
                }

            html_content = response.text

            # Extract title from various sources
            title = self._extract_youtube_title(html_content)
            description = self._extract_youtube_description(html_content)

            content_parts = []
            if title:
                content_parts.append(f"Video Title: {title}")
            if description:
                content_parts.append(f"Description: {description}")

            combined_text = '\n\n'.join(content_parts)

            return {
                'text':
                combined_text,
                'title':
                title,
                'content_type':
                'youtube',
                'success':
                bool(title or description),
                'error':
                None if (title or description) else
                'Could not extract content from YouTube page'
            }

        except Exception as e:
            self.logger.debug(f"YouTube fallback extraction failed: {e}")
            return {
                'text': '',
                'title': '',
                'content_type': 'youtube',
                'success': False,
                'error': f"YouTube fallback failed: {str(e)}"
            }

    def _extract_youtube_title(self, html_content: str) -> str:
        """Extract YouTube video title from HTML"""
        title_patterns = [
            r'"videoDetails":\s*{[^}]*"title":\s*"([^"]+)"',
            r'<meta property="og:title" content="([^"]*)"',
            r'<meta name="title" content="([^"]*)"',
            r'"title":\s*"([^"]+)"[^}]*"videoId"',
            r'ytInitialPlayerResponse[^}]+title[^}]+text[^}]+([^"]+)',
            r'<title>([^<]+)</title>'
        ]

        for pattern in title_patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                title = match.group(1)
                # Clean title
                title = title.replace('\\u0026',
                                      '&').replace('\\/',
                                                   '/').replace('\\"', '"')
                title = re.sub(r'\\u[0-9a-fA-F]{4}', '',
                               title)  # Remove unicode escapes
                title = title.strip()

                # Filter out generic YouTube titles
                if (len(title) > 10 and not title.endswith(' - YouTube')
                        and title != 'YouTube' and
                        'Enjoy the videos and music you love' not in title):
                    return title.replace(' - YouTube', '').strip()

        return ""

    def _extract_youtube_description(self, html_content: str) -> str:
        """Extract YouTube video description from HTML"""
        desc_patterns = [
            r'"videoDetails":\s*{[^}]*"shortDescription":\s*"([^"]+)"',
            r'<meta property="og:description" content="([^"]*)"',
            r'<meta name="description" content="([^"]*)"',
            r'"shortDescription":\s*"([^"]+)"',
            r'"description":\s*{[^}]*"simpleText":\s*"([^"]+)"'
        ]

        for pattern in desc_patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                description = match.group(1)
                # Clean description
                description = description.replace('\\n', '\n').replace(
                    '\\u0026', '&').replace('\\/', '/')
                description = re.sub(r'\\u[0-9a-fA-F]{4}', '',
                                     description)  # Remove unicode escapes
                description = description.strip()

                # Filter out generic YouTube descriptions
                if (len(description) > 30
                        and 'Enjoy the videos and music you love'
                        not in description
                        and 'upload original content' not in description):
                    # Limit description length
                    if len(description) > 1000:
                        description = description[:1000] + "..."
                    return description

        return ""

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
            text = trafilatura.extract(downloaded,
                                       include_comments=False,
                                       include_tables=True)

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
                'error': None if
                (text or title_text) else 'No content extracted'
            }

        except Exception as e:
            self.logger.error(f"Trafilatura extraction failed: {e}")
            return self._fallback_webpage_extraction(url)

    def _fallback_webpage_extraction(self, url: str) -> Dict[str, Any]:
        """Enhanced fallback method for webpage content extraction"""
        try:
            # Try different request strategies
            for attempt in range(3):
                try:
                    if attempt == 0:
                        # First attempt: normal request
                        response = self.session.get(url,
                                                    timeout=15,
                                                    allow_redirects=True)
                    elif attempt == 1:
                        # Second attempt: with different headers
                        headers = self.session.headers.copy()
                        headers.update({
                            'User-Agent':
                            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                            'Accept': '*/*',
                        })
                        response = self.session.get(url,
                                                    timeout=15,
                                                    headers=headers,
                                                    allow_redirects=True)
                    else:
                        # Third attempt: minimal headers
                        response = requests.get(url,
                                                timeout=15,
                                                allow_redirects=True)

                    if response.status_code == 200:
                        break

                except requests.exceptions.SSLError:
                    # Try without SSL verification for the last attempt
                    if attempt == 2:
                        response = requests.get(url,
                                                timeout=15,
                                                verify=False,
                                                allow_redirects=True)
                        break
                except:
                    if attempt == 2:
                        raise
                    continue

            response.raise_for_status()

            # Handle different encodings
            if response.encoding is None:
                response.encoding = 'utf-8'

            content = response.text

            # Enhanced HTML parsing
            title = self._extract_title(content)
            text_content = self._extract_text_content(content)

            return {
                'text':
                text_content,
                'title':
                title,
                'content_type':
                'webpage',
                'success':
                bool(text_content or title),
                'error':
                None if (text_content or title) else 'No content extracted'
            }

        except Exception as e:
            self.logger.error(f"Fallback extraction failed for {url}: {e}")
            return {
                'text': '',
                'title': '',
                'content_type': 'webpage',
                'success': False,
                'error': f"Webpage extraction failed: {str(e)}"
            }

    def _extract_title(self, html_content: str) -> str:
        """Extract title from HTML content"""
        # Try different title extraction methods
        title_patterns = [
            r'<title[^>]*>([^<]+)</title>',
            r'<meta\s+property="og:title"\s+content="([^"]+)"',
            r'<meta\s+name="title"\s+content="([^"]+)"',
            r'<h1[^>]*>([^<]+)</h1>',
        ]

        for pattern in title_patterns:
            match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
            if match:
                title = match.group(1).strip()
                # Clean title
                title = re.sub(r'\s+', ' ', title)
                title = re.sub(r'[^\w\s\-\.\,\:\;\!\?]', '', title)
                if len(title) > 5:  # Reasonable title length
                    return title

        return ""

    def _extract_text_content(self, html_content: str) -> str:
        """Extract main text content from HTML"""
        # Remove unwanted elements
        content = html_content

        # Remove script, style, and other non-content elements
        unwanted_tags = [
            'script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript',
            'svg'
        ]
        for tag in unwanted_tags:
            content = re.sub(f'<{tag}[^>]*>.*?</{tag}>',
                             '',
                             content,
                             flags=re.DOTALL | re.IGNORECASE)

        # Remove comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

        # Extract text from common content containers
        content_patterns = [
            r'<main[^>]*>(.*?)</main>',
            r'<article[^>]*>(.*?)</article>',
            r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>',
            r'<div[^>]*class="[^"]*post[^"]*"[^>]*>(.*?)</div>',
            r'<div[^>]*id="[^"]*content[^"]*"[^>]*>(.*?)</div>',
        ]

        main_content = ""
        for pattern in content_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            if matches:
                main_content = max(matches, key=len)  # Get longest match
                break

        # If no specific content container found, use body
        if not main_content:
            body_match = re.search(r'<body[^>]*>(.*?)</body>', content,
                                   re.IGNORECASE | re.DOTALL)
            if body_match:
                main_content = body_match.group(1)
            else:
                main_content = content

        # Remove remaining HTML tags
        text = re.sub(r'<[^>]+>', ' ', main_content)

        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Multiple whitespace to single space
        text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines to single
        text = text.strip()

        # Remove very short lines that are likely navigation/menu items
        lines = text.split('\n')
        meaningful_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 20 or (len(line) > 5 and any(char.isalpha()
                                                        for char in line)):
                meaningful_lines.append(line)

        text = '\n'.join(meaningful_lines)

        # Limit content length but preserve complete sentences
        if len(text) > 8000:
            text = text[:8000]
            # Try to cut at sentence boundary
            last_period = text.rfind('.')
            if last_period > 6000:  # Only if we find a period in reasonable range
                text = text[:last_period + 1]
            text += "..."

        return text

    def enhance_document_with_url_content(
            self, document: Dict[str, Any]) -> Dict[str, Any]:
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

            self.logger.info(
                f"Enhanced document {document.get('id', 'unknown')} with {extracted['content_type']} content"
            )
        else:
            enhanced_doc['extraction_error'] = extracted.get(
                'error', 'Unknown error')
            self.logger.warning(
                f"Failed to extract content for document {document.get('id', 'unknown')}: {extracted.get('error')}"
            )

        return enhanced_doc
