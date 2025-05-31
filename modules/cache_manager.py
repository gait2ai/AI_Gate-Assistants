"""
AI Gate for Artificial Intelligence Applications
Cache Manager Module

This module provides intelligent caching functionality for the AI Gate system,
including response caching, question analysis caching, website research caching,
and prompt template caching for optimal performance.
"""

import os
import json
import hashlib
import logging
import asyncio
import pickle
import gzip
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import OrderedDict
import threading
import time

# Import for LRU Cache implementation
from functools import lru_cache


@dataclass
class CacheEntry:
    """Cache entry data structure."""
    key: str
    value: Any
    timestamp: datetime
    ttl: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0
    category: str = "general"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
        if self.last_accessed is None:
            self.last_accessed = self.timestamp

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl <= 0:  # Permanent cache
            return False
        return datetime.now() > (self.timestamp + timedelta(seconds=self.ttl))

    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary for serialization."""
        return {
            'key': self.key,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'ttl': self.ttl,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'size_bytes': self.size_bytes,
            'category': self.category,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create cache entry from dictionary."""
        return cls(
            key=data['key'],
            value=data['value'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            ttl=data['ttl'],
            access_count=data.get('access_count', 0),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None,
            size_bytes=data.get('size_bytes', 0),
            category=data.get('category', 'general'),
            metadata=data.get('metadata', {})
        )


class CacheManager:
    """
    Intelligent cache manager for AI Gate application.
    
    Provides multi-level caching with different TTL policies for:
    - Chat responses
    - Question analysis results
    - Website research results
    - Prompt templates
    - AI model responses
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_cache_size: int = 1000,
        cache_ttl: int = 3600,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for persistent cache storage
            max_cache_size: Maximum number of entries in memory cache
            cache_ttl: Default TTL in seconds
            config: Additional configuration options (from cache section of YAML config)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size = max_cache_size
        self.default_ttl = cache_ttl
        self.config = config or {}
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # In-memory cache with LRU eviction
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Cache statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_bytes': 0,
            'last_cleanup': datetime.now(),
            'categories': {}
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Cache category configurations
        self._category_configs = {
            'chat_response': {'ttl': 3600, 'persistent': True, 'compress': True},
            'question_analysis': {'ttl': 1800, 'persistent': True, 'compress': False},
            'website_research': {'ttl': 7200, 'persistent': True, 'compress': True},
            'prompt_template': {'ttl': 0, 'persistent': True, 'compress': False},  # Permanent
            'ai_response': {'ttl': 3600, 'persistent': True, 'compress': True},
            'general': {'ttl': self.default_ttl, 'persistent': False, 'compress': False}
        }
        
        # Update category configs from provided config
        if 'categories' in self.config:
            self._category_configs.update(self.config['categories'])
        
        # Initialize cache files
        self._cache_files = {
            'persistent': self.cache_dir / 'persistent_cache.json.gz',
            'metadata': self.cache_dir / 'cache_metadata.json',
            'stats': self.cache_dir / 'cache_stats.json'
        }
        
        # Load existing caches
        self._load_persistent_cache()
        self._load_cache_metadata()
        
        # Start background cleanup task with configurable interval
        self._cleanup_task = None
        self._start_cleanup_task()
        
        self.logger.info(f"Cache manager initialized with {len(self._memory_cache)} entries")

    def _get_category_config(self, category: str) -> Dict[str, Any]:
        """Get configuration for cache category."""
        return self._category_configs.get(category, self._category_configs['general'])

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value, default=str).encode('utf-8'))
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 0

    def generate_cache_key(self, *args, category: str = "general", **kwargs) -> str:
        """
        Generate a cache key from arguments.
        
        Args:
            *args: Positional arguments to include in key
            category: Cache category for key prefix
            **kwargs: Keyword arguments to include in key
            
        Returns:
            str: Generated cache key
        """
        # Create a string representation of all arguments
        key_parts = []
        
        # Add category prefix
        key_parts.append(f"cat:{category}")
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (dict, list)):
                key_parts.append(json.dumps(arg, sort_keys=True, default=str))
            else:
                key_parts.append(str(arg))
        
        # Add keyword arguments (sorted for consistency)
        for key, value in sorted(kwargs.items()):
            if isinstance(value, (dict, list)):
                key_parts.append(f"{key}:{json.dumps(value, sort_keys=True, default=str)}")
            else:
                key_parts.append(f"{key}:{value}")
        
        # Create hash of the combined key
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Check if expired
                if entry.is_expired():
                    self._remove_entry(key)
                    self._stats['misses'] += 1
                    return default
                
                # Update access statistics
                entry.update_access()
                
                # Move to end (LRU)
                self._memory_cache.move_to_end(key)
                
                self._stats['hits'] += 1
                self._update_category_stats(entry.category, 'hits')
                
                self.logger.debug(f"Cache hit for key: {key[:16]}...")
                return entry.value
            
            self._stats['misses'] += 1
            self.logger.debug(f"Cache miss for key: {key[:16]}...")
            return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses category default if None)
            category: Cache category
            metadata: Additional metadata
            
        Returns:
            bool: True if successfully cached
        """
        try:
            with self._lock:
                # Get category configuration
                cat_config = self._get_category_config(category)
                
                # Use category TTL if not specified
                if ttl is None:
                    ttl = cat_config['ttl']
                
                # Calculate size
                size_bytes = self._calculate_size(value)
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=datetime.now(),
                    ttl=ttl,
                    size_bytes=size_bytes,
                    category=category,
                    metadata=metadata or {}
                )
                
                # Check if we need to evict entries
                self._ensure_cache_capacity()
                
                # Add to memory cache
                self._memory_cache[key] = entry
                
                # Update statistics
                self._stats['size_bytes'] += size_bytes
                self._update_category_stats(category, 'entries', 1)
                self._update_category_stats(category, 'size_bytes', size_bytes)
                
                # Save to persistent storage if configured for this category
                if cat_config.get('persistent', False):
                    await self._save_to_persistent(entry)
                
                self.logger.debug(f"Cached key: {key[:16]}... (category: {category}, size: {size_bytes} bytes)")
                return True
                
        except Exception as e:
            self.logger.error(f"Error caching key {key[:16]}...: {e}")
            return False

    async def get_cached_response(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached chat response.
        
        Args:
            key: Cache key for response
            
        Returns:
            Cached response dictionary or None
        """
        response = await self.get(key)
        if response and isinstance(response, dict):
            return response
        return None

    async def cache_response(self, key: str, response_data: Dict[str, Any]) -> bool:
        """
        Cache chat response data.
        
        Args:
            key: Cache key
            response_data: Response data to cache
            
        Returns:
            bool: True if successfully cached
        """
        return await self.set(
            key=key,
            value=response_data,
            category='chat_response',
            metadata={'type': 'chat_response', 'timestamp': datetime.now().isoformat()}
        )

    async def cache_question_analysis(self, question: str, analysis: Dict[str, Any]) -> bool:
        """
        Cache question analysis results.
        
        Args:
            question: Original question
            analysis: Analysis results
            
        Returns:
            bool: True if successfully cached
        """
        key = self.generate_cache_key(question, category='question_analysis')
        return await self.set(
            key=key,
            value=analysis,
            category='question_analysis',
            metadata={'question': question[:100], 'type': 'question_analysis'}
        )

    async def get_cached_question_analysis(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Get cached question analysis.
        
        Args:
            question: Original question
            
        Returns:
            Cached analysis or None
        """
        key = self.generate_cache_key(question, category='question_analysis')
        return await self.get(key)

    async def cache_website_research(self, topics: List[str], keywords: List[str], results: List[Dict[str, Any]]) -> bool:
        """
        Cache website research results.
        
        Args:
            topics: Research topics
            keywords: Research keywords
            results: Research results
            
        Returns:
            bool: True if successfully cached
        """
        key = self.generate_cache_key(topics, keywords, category='website_research')
        return await self.set(
            key=key,
            value=results,
            category='website_research',
            metadata={'topics': topics, 'keywords': keywords, 'type': 'website_research'}
        )

    async def get_cached_website_research(self, topics: List[str], keywords: List[str]) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached website research results.
        
        Args:
            topics: Research topics
            keywords: Research keywords
            
        Returns:
            Cached results or None
        """
        key = self.generate_cache_key(topics, keywords, category='website_research')
        return await self.get(key)

    # Prompt Template Caching Extension
    async def cache_prompt_template(self, template_name: str, template_data: Dict[str, Any]) -> bool:
        """
        Cache compiled prompt template.
        
        Args:
            template_name: Name/identifier of the template
            template_data: Template data including compiled template and metadata
            
        Returns:
            bool: True if successfully cached
        """
        key = self.generate_cache_key(template_name, category='prompt_template')
        return await self.set(
            key=key,
            value=template_data,
            category='prompt_template',
            ttl=0,  # Permanent cache for templates
            metadata={
                'template_name': template_name,
                'type': 'prompt_template',
                'compiled_at': datetime.now().isoformat()
            }
        )

    async def get_cached_prompt_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get cached prompt template.
        
        Args:
            template_name: Name/identifier of the template
            
        Returns:
            Cached template data or None
        """
        key = self.generate_cache_key(template_name, category='prompt_template')
        return await self.get(key)

    async def invalidate_prompt_templates(self) -> bool:
        """
        Invalidate all cached prompt templates (useful when templates are updated).
        
        Returns:
            bool: True if successfully invalidated
        """
        try:
            with self._lock:
                keys_to_remove = []
                for key, entry in self._memory_cache.items():
                    if entry.category == 'prompt_template':
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    self._remove_entry(key)
                
                self.logger.info(f"Invalidated {len(keys_to_remove)} prompt templates")
                return True
        except Exception as e:
            self.logger.error(f"Error invalidating prompt templates: {e}")
            return False

    def _ensure_cache_capacity(self):
        """Ensure cache doesn't exceed maximum capacity."""
        while len(self._memory_cache) >= self.max_cache_size:
            # Remove least recently used item
            key, entry = self._memory_cache.popitem(last=False)
            self._stats['size_bytes'] -= entry.size_bytes
            self._stats['evictions'] += 1
            self._update_category_stats(entry.category, 'entries', -1)
            self._update_category_stats(entry.category, 'size_bytes', -entry.size_bytes)

    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self._memory_cache:
            entry = self._memory_cache.pop(key)
            self._stats['size_bytes'] -= entry.size_bytes
            self._update_category_stats(entry.category, 'entries', -1)
            self._update_category_stats(entry.category, 'size_bytes', -entry.size_bytes)

    def _update_category_stats(self, category: str, stat: str, value: Union[int, float] = 1):
        """Update statistics for a category."""
        if category not in self._stats['categories']:
            self._stats['categories'][category] = {
                'hits': 0, 'entries': 0, 'size_bytes': 0
            }
        
        if stat in self._stats['categories'][category]:
            if isinstance(value, (int, float)) and stat in ['entries', 'size_bytes']:
                self._stats['categories'][category][stat] += value
            else:
                self._stats['categories'][category][stat] += 1

    def _load_persistent_cache(self):
        """Load persistent cache from disk."""
        try:
            if self._cache_files['persistent'].exists():
                with gzip.open(self._cache_files['persistent'], 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for entry_data in data.get('entries', []):
                        try:
                            entry = CacheEntry.from_dict(entry_data)
                            if not entry.is_expired():
                                self._memory_cache[entry.key] = entry
                                self._stats['size_bytes'] += entry.size_bytes
                        except Exception as e:
                            self.logger.warning(f"Failed to load cache entry: {e}")
                            
                self.logger.info(f"Loaded {len(self._memory_cache)} entries from persistent cache")
        except Exception as e:
            self.logger.warning(f"Failed to load persistent cache: {e}")

    async def _save_to_persistent(self, entry: CacheEntry):
        """Save entry to persistent storage."""
        # This is a simplified version - in production, you might want to batch saves
        pass

    def _load_cache_metadata(self):
        """Load cache metadata."""
        try:
            if self._cache_files['metadata'].exists():
                with open(self._cache_files['metadata'], 'r') as f:
                    metadata = json.load(f)
                    # Load any saved metadata
        except Exception as e:
            self.logger.warning(f"Failed to load cache metadata: {e}")

    def _start_cleanup_task(self):
        """Start background cleanup task with configurable interval."""
        # Get cleanup interval from config, default to 300 seconds (5 minutes)
        cleanup_interval = self.config.get('cleanup_interval', 300)
        
        def cleanup_worker():
            self.logger.info(f"Cache cleanup worker started with interval: {cleanup_interval} seconds")
            while True:
                try:
                    time.sleep(cleanup_interval)
                    asyncio.create_task(self._cleanup_expired())
                except Exception as e:
                    self.logger.error(f"Error in cleanup worker: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

    async def _cleanup_expired(self):
        """Clean up expired cache entries."""
        try:
            with self._lock:
                expired_keys = []
                current_time = datetime.now()
                
                for key, entry in self._memory_cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self._remove_entry(key)
                
                self._stats['last_cleanup'] = current_time
                
                if expired_keys:
                    self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")

    async def clear_cache(self, category: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            category: If specified, only clear entries from this category
        """
        with self._lock:
            if category:
                keys_to_remove = []
                for key, entry in self._memory_cache.items():
                    if entry.category == category:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    self._remove_entry(key)
                    
                self.logger.info(f"Cleared {len(keys_to_remove)} entries from category: {category}")
            else:
                self._memory_cache.clear()
                self._stats['size_bytes'] = 0
                self._stats['categories'] = {}
                self.logger.info("Cleared all cache entries")

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict containing cache statistics
        """
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'total_entries': len(self._memory_cache),
                'max_entries': self.max_cache_size,
                'size_bytes': self._stats['size_bytes'],
                'size_mb': round(self._stats['size_bytes'] / (1024 * 1024), 2),
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate_percent': round(hit_rate, 2),
                'evictions': self._stats['evictions'],
                'last_cleanup': self._stats['last_cleanup'].isoformat(),
                'categories': self._stats['categories'],
                'uptime_seconds': (datetime.now() - self._stats['last_cleanup']).total_seconds(),
                'cleanup_interval': self.config.get('cleanup_interval', 300)
            }

    def is_healthy(self) -> bool:
        """
        Check if cache manager is healthy.
        
        Returns:
            bool: True if healthy
        """
        try:
            # Basic health checks
            cache_usage = len(self._memory_cache) / self.max_cache_size
            memory_usage = self._stats['size_bytes'] / (1024 * 1024)  # MB
            
            # Consider unhealthy if cache is completely full or using excessive memory
            if cache_usage >= 1.0 or memory_usage > 500:  # 500MB limit
                return False
                
            return True
        except Exception:
            return False

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Save current cache state
            await self._save_cache_state()
            
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                
            self.logger.info("Cache manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")

    async def _save_cache_state(self):
        """Save current cache state to disk."""
        try:
            # Save statistics
            with open(self._cache_files['stats'], 'w') as f:
                stats = await self.get_statistics()
                json.dump(stats, f, indent=2, default=str)
            
            # Save persistent entries
            persistent_entries = []
            for entry in self._memory_cache.values():
                cat_config = self._get_category_config(entry.category)
                if cat_config.get('persistent', False) and not entry.is_expired():
                    persistent_entries.append(entry.to_dict())
            
            if persistent_entries:
                with gzip.open(self._cache_files['persistent'], 'wt', encoding='utf-8') as f:
                    json.dump({'entries': persistent_entries}, f, default=str)
                    
            self.logger.info(f"Saved {len(persistent_entries)} persistent cache entries")
            
        except Exception as e:
            self.logger.error(f"Error saving cache state: {e}")
