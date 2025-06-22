"""
Automatic updater for HackerNews RAG system
Monitors for new posts and updates the knowledge base automatically
"""
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Set
import json
import os

from hn_data_manager import HackerNewsDataManager
from vector_store import VectorStore

class AutoUpdater:
    """Automatically updates the RAG system with new HackerNews content"""
    
    def __init__(self, 
                 vector_store: VectorStore, 
                 data_manager: HackerNewsDataManager,
                 update_interval_minutes: int = 30,
                 max_new_stories: int = 50):
        self.vector_store = vector_store
        self.data_manager = data_manager
        self.update_interval = update_interval_minutes * 60  # Convert to seconds
        self.max_new_stories = max_new_stories
        
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.update_thread = None
        
        # Track processed stories to avoid duplicates
        self.processed_ids_file = "processed_story_ids.json"
        self.processed_ids = self._load_processed_ids()
        
        # Last update timestamp
        self.last_update = None
    
    def _load_processed_ids(self) -> Set[int]:
        """Load previously processed story IDs from file"""
        try:
            if os.path.exists(self.processed_ids_file):
                with open(self.processed_ids_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('processed_ids', []))
        except Exception as e:
            self.logger.warning(f"Could not load processed IDs: {e}")
        
        # If file doesn't exist or loading fails, start with existing vector store IDs
        existing_ids = self.vector_store.get_existing_ids()
        self.logger.info(f"Initialized with {len(existing_ids)} existing story IDs")
        return existing_ids
    
    def _save_processed_ids(self):
        """Save processed story IDs to file"""
        try:
            data = {
                'processed_ids': list(self.processed_ids),
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
            with open(self.processed_ids_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.error(f"Could not save processed IDs: {e}")
    
    def check_for_new_stories(self) -> int:
        """Check for new stories and add them to the knowledge base"""
        try:
            self.logger.info("Checking for new HackerNews stories...")
            
            # Get latest story IDs (both top and new)
            new_story_ids = self.data_manager.get_new_stories(limit=100)
            
            if not new_story_ids:
                self.logger.info("No new stories found")
                return 0
            
            # Limit number of new stories to process
            if len(new_story_ids) > self.max_new_stories:
                new_story_ids = new_story_ids[:self.max_new_stories]
                self.logger.info(f"Limited to processing {self.max_new_stories} new stories")
            
            self.logger.info(f"Found {len(new_story_ids)} new stories to process")
            
            # Fetch story details
            new_stories = []
            for story_id in new_story_ids:
                try:
                    story = self.data_manager.fetch_story_details(story_id)
                    if story and story.get('type') == 'story':
                        new_stories.append(story)
                        self.processed_ids.add(story_id)
                except Exception as e:
                    self.logger.warning(f"Failed to fetch story {story_id}: {e}")
            
            if new_stories:
                # Add to vector store
                self.logger.info(f"Adding {len(new_stories)} new stories to vector store")
                self.vector_store.add_documents(new_stories)
                
                # Update last update time
                self.last_update = datetime.now()
                
                # Save processed IDs
                self._save_processed_ids()
                
                self.logger.info(f"Successfully added {len(new_stories)} new stories")
                return len(new_stories)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error checking for new stories: {e}")
            return 0
    
    def start_auto_update(self):
        """Start the automatic update process"""
        if self.is_running:
            self.logger.warning("Auto-updater is already running")
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        self.logger.info(f"Started auto-updater with {self.update_interval/60:.1f} minute intervals")
    
    def stop_auto_update(self):
        """Stop the automatic update process"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        self.logger.info("Stopped auto-updater")
    
    def _update_loop(self):
        """Main update loop that runs in background thread"""
        while self.is_running:
            try:
                new_count = self.check_for_new_stories()
                if new_count > 0:
                    self.logger.info(f"Auto-update: Added {new_count} new stories")
                
                # Wait for next update interval
                for _ in range(int(self.update_interval)):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error in auto-update loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def get_status(self) -> dict:
        """Get current status of the auto-updater"""
        return {
            'is_running': self.is_running,
            'last_update': self.last_update,
            'processed_count': len(self.processed_ids),
            'update_interval_minutes': self.update_interval / 60
        }
    
    def manual_update(self) -> int:
        """Perform a manual update and return number of new stories added"""
        return self.check_for_new_stories()