"""
AI Gate for Artificial Intelligence Applications
Main application entry point and FastAPI setup

This module serves as the central coordinator for the AI Gate system,
managing all components and handling the web interface and API endpoints.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, validator
from dotenv import load_dotenv
import uvicorn
from fastapi.responses import FileResponse

# Configure NLTK data path early - before any NLTK-dependent imports
import nltk
BASE_DIR = Path(__file__).parent
NLTK_DATA_LOCAL_DIR = BASE_DIR / "nltk_data_local"

def configure_nltk_data_path():
    """
    Configure NLTK to use bundled local data directory.
    This must be called early, before any NLTK-dependent components are initialized.
    """
    try:
        nltk_data_path = str(NLTK_DATA_LOCAL_DIR)
        
        # Check if the local NLTK data directory exists
        if NLTK_DATA_LOCAL_DIR.exists() and NLTK_DATA_LOCAL_DIR.is_dir():
            # Add to NLTK data path if not already present
            if nltk_data_path not in nltk.data.path:
                nltk.data.path.insert(0, nltk_data_path)
                print(f"NLTK data path configured: {nltk_data_path}")
            else:
                print(f"NLTK data path already configured: {nltk_data_path}")
        else:
            print(f"Warning: NLTK data directory not found at {nltk_data_path}")
            print("NLTK will attempt to use system-wide or downloaded data")
            
    except Exception as e:
        print(f"Error configuring NLTK data path: {e}")
        print("NLTK will attempt to use default data locations")

# Configure NLTK data path immediately
configure_nltk_data_path()

# Import custom modules (after NLTK configuration)
from modules.question_processor import QuestionProcessor
from modules.website_researcher import WebsiteResearcher
from modules.prompt_builder import PromptBuilder
from modules.ai_handler import AIHandler
from modules.cache_manager import CacheManager
from modules.utils import setup_logging, load_config, validate_environment

# Load environment variables
load_dotenv()

# Application metadata
APP_NAME = "AI Gate for Artificial Intelligence Applications"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Intelligent gateway for institutional AI assistance"

# Directory paths
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
LOGS_DIR = BASE_DIR / "logs"

# Ensure required directories exist
for directory in [CONFIG_DIR, DATA_DIR, STATIC_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Global logger (will be properly configured after config is loaded)
logger = None

# Pydantic models for API
class UserMessage(BaseModel):
    """User message model with validation."""
    message: str
    session_id: Optional[str] = None
    
    @validator('message')
    def validate_message(cls, v):
        """Validate that message is not empty and has reasonable length."""
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v.strip()) < 2:
            raise ValueError('Message too short')
        if len(v) > 2000:
            raise ValueError('Message too long (max 2000 characters)')
        return v.strip()

class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str
    sources: list = []
    processing_time: float = 0.0
    cached: bool = False
    session_id: Optional[str] = None

class InstitutionData(BaseModel):
    """Institution data response model."""
    name: str
    description: Optional[str] = None
    website: Optional[str] = None
    contact_email: Optional[str] = None
    timezone: Optional[str] = None
    logo_url: Optional[str] = None
    support_phone: Optional[str] = None
    address: Optional[str] = None

class HealthStatus(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]

# Global application components
app_components = {
    'question_processor': None,
    'website_researcher': None,
    'prompt_builder': None,
    'ai_handler': None,
    'cache_manager': None,
    'config': None
}

def setup_configured_logging() -> logging.Logger:
    """
    Setup logging using configuration values.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    global logger
    
    # Load basic configuration first to get logging settings
    try:
        temp_config = load_config(CONFIG_DIR)
        logging_config = temp_config.get('logging', {})
        
        # Extract logging parameters from config with fallbacks
        log_level = logging_config.get('level', 'INFO')
        max_file_size_mb = logging_config.get('max_file_size_mb', 10)
        backup_count = logging_config.get('backup_count', 5)
        console_output = logging_config.get('console_output', True)
        log_format = logging_config.get('log_format')
        
        # Convert MB to bytes
        max_file_size = max_file_size_mb * 1024 * 1024
        
        logger = setup_logging(
            logs_dir=LOGS_DIR,
            log_level=log_level,
            max_file_size=max_file_size,
            backup_count=backup_count,
            console_output=console_output,
            log_format=log_format
        )
        
        logger.info(f"Logging configured from config file - Level: {log_level}")
        
        # Log NLTK configuration status now that logging is available
        if NLTK_DATA_LOCAL_DIR.exists():
            logger.info(f"NLTK data path configured successfully: {NLTK_DATA_LOCAL_DIR}")
        else:
            logger.warning(f"NLTK local data directory not found: {NLTK_DATA_LOCAL_DIR}")
        
        return logger
        
    except Exception as e:
        # Fallback to basic logging if config loading fails
        logger = setup_logging(LOGS_DIR)
        logger.warning(f"Failed to load logging config, using defaults: {e}")
        
        # Log NLTK configuration status with basic logger
        if NLTK_DATA_LOCAL_DIR.exists():
            logger.info(f"NLTK data path configured successfully: {NLTK_DATA_LOCAL_DIR}")
        else:
            logger.warning(f"NLTK local data directory not found: {NLTK_DATA_LOCAL_DIR}")
        
        return logger

def load_initial_config() -> Dict[str, Any]:
    """
    Load initial configuration early in the application startup process.
    This is called at module level before FastAPI app creation.
    
    Returns:
        Dict[str, Any]: Loaded configuration dictionary
    """
    try:
        logger.info("Loading initial application configuration...")
        config = load_config(CONFIG_DIR)
        
        # Log configuration sources for debugging
        config_metadata = config.get('_metadata', {})
        config_sources = config_metadata.get('sources', [])
        logger.info(f"Configuration loaded from: {', '.join(config_sources)}")
        
        # Log institution configuration availability
        institution_config = config.get('institution', {})
        if institution_config:
            institution_name = institution_config.get('name', 'Not specified')
            logger.info(f"Institution configuration found: {institution_name}")
        else:
            logger.warning("No institution configuration section found in config")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to load initial configuration: {e}")
        # Return minimal fallback configuration
        logger.warning("Using minimal fallback configuration")
        return {
            'api': {
                'gzip_compression': True,
                'gzip_minimum_size': 1000,
                'cors_origins': ["*"],
                'cors_credentials': True,
                'cors_methods': ["*"],
                'cors_headers': ["*"]
            },
            'institution': {
                'name': 'المؤسسة',
                'description': 'مساعد ذكي للمؤسسة',
                'timezone': 'UTC'
            },
            'logging': {
                'level': 'INFO'
            }
        }

def create_basic_fastapi_app() -> FastAPI:
    """
    Create and configure basic FastAPI application without middleware.
    Middleware will be added immediately after this function returns.
    
    Returns:
        FastAPI: Basic FastAPI application instance
    """
    # Initialize FastAPI application with basic settings
    app = FastAPI(
        title=APP_NAME,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    logger.info("Basic FastAPI application created")
    return app

def configure_middleware(app: FastAPI, config: Dict[str, Any]) -> None:
    """
    Configure and add middleware to FastAPI application using loaded configuration.
    This function is called immediately after app creation, before any startup events.
    
    Args:
        app: FastAPI application instance to configure
        config: Loaded configuration dictionary
    """
    try:
        # Get API configuration with fallbacks
        api_config = config.get('api', {})
        
        # Configure GZip middleware
        gzip_compression = api_config.get('gzip_compression', True)
        if gzip_compression:
            gzip_minimum_size = api_config.get('gzip_minimum_size', 1000)
            app.add_middleware(GZipMiddleware, minimum_size=gzip_minimum_size)
            logger.info(f"GZip compression enabled with minimum size: {gzip_minimum_size}")
        else:
            logger.info("GZip compression disabled")
        
        # Configure CORS middleware
        cors_origins = api_config.get('cors_origins', ["*"])
        cors_credentials = api_config.get('cors_credentials', True)
        cors_methods = api_config.get('cors_methods', ["*"])
        cors_headers = api_config.get('cors_headers', ["*"])
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=cors_credentials,
            allow_methods=cors_methods,
            allow_headers=cors_headers,
        )
        
        logger.info(f"CORS configured - Origins: {cors_origins}")
        logger.info("Middleware configuration completed successfully")
        
    except Exception as e:
        logger.error(f"Error configuring middleware: {e}")
        # Configure safe fallback middleware
        logger.warning("Applying fallback middleware configuration")
        
        # Add fallback GZip middleware
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Add fallback CORS middleware with restrictive settings
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        logger.info("Fallback middleware configuration applied")

# Initialize logging early (before FastAPI app creation)
logger = setup_configured_logging()

# Load configuration early (before FastAPI app creation)
logger.info("Loading configuration for middleware setup...")
initial_config = load_initial_config()
app_components['config'] = initial_config

# Initialize basic FastAPI application (without middleware)
app = create_basic_fastapi_app()

# Configure middleware immediately after app creation
logger.info("Configuring middleware...")
configure_middleware(app, initial_config)

@app.get("/api/institution", response_model=InstitutionData)
async def get_institution_data():
    """
    Get institution data from centralized configuration.
    
    Returns:
        InstitutionData: Institution data from configuration with comprehensive fallbacks
        
    Raises:
        HTTPException: If configuration is completely unavailable
    """
    try:
        logger.debug("Processing institution data request")
        
        # Check if configuration is loaded
        if not app_components.get('config'):
            logger.warning("Configuration not loaded, returning default institution data")
            return InstitutionData(
                name="المؤسسة",
                description="مساعد ذكي للمؤسسة",
                timezone="UTC"
            )
        
        # Extract institution configuration
        institution_config = app_components['config'].get('institution', {})
        
        # Log configuration availability for debugging
        if not institution_config:
            logger.info("No institution configuration found, using default values")
        else:
            logger.debug(f"Institution configuration keys available: {list(institution_config.keys())}")
        
        # Build comprehensive institution data with fallbacks
        institution_data = InstitutionData(
            name=institution_config.get('name', 'المؤسسة'),
            description=institution_config.get('description', 'مساعد ذكي للمؤسسة'),
            website=institution_config.get('website'),
            contact_email=institution_config.get('contact_email'),
            timezone=institution_config.get('timezone', 'UTC'),
            logo_url=institution_config.get('logo_url'),
            support_phone=institution_config.get('support_phone'),
            address=institution_config.get('address')
        )
        
        logger.info(f"Successfully retrieved institution data for: {institution_data.name}")
        return institution_data
        
    except Exception as e:
        logger.error(f"Error retrieving institution data: {e}", exc_info=True)
        
        # Return minimal but functional fallback data
        logger.warning("Returning minimal fallback institution data due to error")
        return InstitutionData(
            name="المؤسسة",
            description="مساعد ذكي",
            timezone="UTC"
        )

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with enhanced session tracking."""
    start_time = datetime.now()
    
    # Extract session info if available
    session_info = ""
    if request.method == "POST" and "api/chat" in str(request.url.path):
        try:
            # This is for logging purposes only - don't interfere with actual request processing
            pass
        except:
            pass
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}{session_info}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response

async def initialize_remaining_components() -> bool:
    """
    Initialize the remaining application components that weren't needed for middleware setup.
    This excludes the config which was already loaded early for middleware configuration.
    
    Returns:
        bool: True if all remaining components initialized successfully
    """
    try:
        logger.info("Initializing remaining application components...")
        
        # Configuration is already loaded in app_components['config']
        config = app_components['config']
        
        # Get the current event loop for CacheManager integration
        try:
            loop = asyncio.get_running_loop()
            logger.debug("Retrieved current asyncio event loop for CacheManager")
        except RuntimeError:
            logger.warning("No running event loop found, CacheManager will create its own")
            loop = None
        
        # Initialize cache manager first (other components may use it)
        logger.info("Initializing cache manager...")
        cache_config = config.get('cache', {})
        
        # Pass the event loop instance to CacheManager for improved integration
        cache_manager_kwargs = {
            'cache_dir': DATA_DIR / "cache",
            'max_cache_size': cache_config.get('max_size', 1000),
            'cache_ttl': cache_config.get('ttl', 3600)
        }
        
        # Add event loop if available
        if loop is not None:
            cache_manager_kwargs['loop'] = loop
        
        app_components['cache_manager'] = CacheManager(**cache_manager_kwargs)
        logger.info(f"Cache manager initialized - Max size: {cache_config.get('max_size', 1000)}, "
                   f"TTL: {cache_config.get('ttl', 3600)}s, "
                   f"Event loop: {'provided' if loop is not None else 'will create own'}")
        
        # Initialize question processor
        logger.info("Initializing question processor...")
        question_config = config.get('question_processing', {})
        app_components['question_processor'] = QuestionProcessor(
            config=question_config,
            cache_manager=app_components['cache_manager']
        )
        logger.info(f"Question processor initialized - Min length: {question_config.get('min_length', 3)}, "
                   f"Max length: {question_config.get('max_length', 2000)}")
        
        # Initialize website researcher
        logger.info("Initializing website researcher...")
        pages_file = DATA_DIR / "pages.json"
        research_config = config.get('website_research', {})
        app_components['website_researcher'] = WebsiteResearcher(
            pages_file=pages_file,
            config=research_config,
            cache_manager=app_components['cache_manager']
        )
        logger.info(f"Website researcher initialized - Max results: {research_config.get('max_results', 10)}, "
                   f"Similarity threshold: {research_config.get('similarity_threshold', 0.6)}")
        
        # Initialize prompt builder
        logger.info("Initializing prompt builder...")
        institution_data = config.get('institution', {})
        prompt_templates = config.get('prompts', {})
        app_components['prompt_builder'] = PromptBuilder(
            config_dir=CONFIG_DIR,
            institution_data=institution_data,
            templates=prompt_templates
        )
        logger.info(f"Prompt builder initialized - Institution: {institution_data.get('name', 'Unknown')}")
        
        # Initialize AI handler
        logger.info("Initializing AI handler...")
        ai_models_config = config.get('ai_models', {})
        app_components['ai_handler'] = AIHandler(
            config=ai_models_config,
            cache_manager=app_components['cache_manager']
        )
        
        # Log AI model configuration
        primary_model = ai_models_config.get('primary_model')
        fallback_models = ai_models_config.get('fallback_models', [])
        if primary_model:
            logger.info(f"AI handler initialized - Primary model: {primary_model}, "
                       f"Fallback models: {len(fallback_models)}")
        else:
            models = ai_models_config.get('models', [])
            logger.info(f"AI handler initialized - Models: {len(models)}")
        
        logger.info("All remaining components initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize remaining components: {e}")
        return False

def get_component_status() -> Dict[str, str]:
    """
    Get the status of all application components.
    
    Returns:
        Dict[str, str]: Component status dictionary
    """
    status = {}
    for name, component in app_components.items():
        if component is None:
            status[name] = "not_initialized"
        elif hasattr(component, 'is_healthy') and callable(component.is_healthy):
            status[name] = "healthy" if component.is_healthy() else "unhealthy"
        else:
            status[name] = "initialized"
    return status

@app.on_event("startup")
async def startup_event():
    """Application startup event handler."""
    global startup_time
    startup_time = datetime.now()
    
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        raise Exception("Environment validation failed")
    
    # Initialize remaining components (config and middleware already configured)
    if not await initialize_remaining_components():
        logger.error("Remaining component initialization failed")
        raise Exception("Remaining component initialization failed")
    
    # Log final configuration summary
    config = app_components.get('config', {})
    institution_name = config.get('institution', {}).get('name', 'Unknown')
    cache_size = config.get('cache', {}).get('max_size', 1000)
    api_cors_origins = config.get('api', {}).get('cors_origins', ["*"])
    
    logger.info(f"Application startup completed successfully")
    logger.info(f"Institution: {institution_name}")
    logger.info(f"Cache max size: {cache_size}")
    logger.info(f"CORS origins: {len(api_cors_origins)} configured")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler with proper async cleanup."""
    logger.info("Shutting down application...")
    
    # Cleanup components with proper async handling
    for name, component in app_components.items():
        if component and hasattr(component, 'cleanup'):
            try:
                # Check if cleanup method is a coroutine and await it properly
                if asyncio.iscoroutinefunction(component.cleanup):
                    logger.debug(f"Awaiting async cleanup for {name}")
                    await component.cleanup()
                else:
                    logger.debug(f"Calling sync cleanup for {name}")
                    component.cleanup()
                logger.info(f"Cleaned up {name}")
            except Exception as e:
                logger.error(f"Error cleaning up {name}: {e}")
    
    logger.info("Application shutdown completed")

# API Endpoints

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    component_status = get_component_status()
    all_healthy = all(c is not None for c in app_components.values())
    
    return HealthStatus(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.now().isoformat(),
        version=APP_VERSION,
        components=component_status
    )

@app.post("/api/chat", response_model=ChatResponse)
async def process_chat_message(user_message: UserMessage, background_tasks: BackgroundTasks):
    """
    Process user chat message and return AI-generated response.
    
    Args:
        user_message: User message with validation (includes optional session_id)
        background_tasks: FastAPI background tasks for async operations
        
    Returns:
        ChatResponse: AI-generated response with metadata
    """
    start_time = datetime.now()
    
    try:
        # Check if components are initialized
        if not all(app_components.values()):
            raise HTTPException(
                status_code=503, 
                detail="Service temporarily unavailable - components not initialized"
            )
        
        # Log message processing with session info if available
        session_info = f" [Session: {user_message.session_id}]" if user_message.session_id else ""
        logger.info(f"Processing message: {user_message.message[:50]}...{session_info}")
        
        # Check cache first
        cache_key = app_components['cache_manager'].generate_cache_key(user_message.message)
        cached_response = await app_components['cache_manager'].get_cached_response(cache_key)
        
        if cached_response:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Cache hit - returning cached response in {processing_time:.3f}s{session_info}")
            return ChatResponse(
                answer=cached_response['answer'],
                sources=cached_response.get('sources', []),
                processing_time=processing_time,
                cached=True,
                session_id=user_message.session_id
            )
        
        # Step 1: Process and validate the question
        logger.info("Step 1: Processing question...")
        question_analysis = await app_components['question_processor'].process_question(
            user_message.message
        )
        
        if not question_analysis['is_valid']:
            return ChatResponse(
                answer=question_analysis.get('error_message', 'Invalid question format'),
                sources=[],
                processing_time=(datetime.now() - start_time).total_seconds(),
                cached=False,
                session_id=user_message.session_id
            )
        
        # Step 2: Research relevant website content
        logger.info("Step 2: Researching website content...")
        research_results = await app_components['website_researcher'].search_content(
            topics=question_analysis['topics'],
            keywords=question_analysis['keywords']
        )
        
        # Step 3: Build the system prompt
        logger.info("Step 3: Building system prompt...")
        system_prompt = await app_components['prompt_builder'].build_prompt(
            original_question=user_message.message,
            processed_question=question_analysis,
            research_results=research_results
        )
        
        # Step 4: Generate AI response
        logger.info("Step 4: Generating AI response...")
        ai_response = await app_components['ai_handler'].generate_response(
            user_message=user_message.message,
            system_prompt=system_prompt,
            context=research_results
        )
        
        # Prepare response
        sources = [result.get('source_url', '') for result in research_results if result.get('source_url')]
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = ChatResponse(
            answer=ai_response,
            sources=sources,
            processing_time=processing_time,
            cached=False,
            session_id=user_message.session_id
        )
        
        # Cache the response in background
        background_tasks.add_task(
            app_components['cache_manager'].cache_response,
            cache_key,
            {
                'answer': ai_response,
                'sources': sources,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        logger.info(f"Response generated successfully in {processing_time:.3f}s{session_info}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        session_info = f" [Session: {user_message.session_id}]" if user_message.session_id else ""
        logger.error(f"Error processing message: {e}{session_info}")
        
        return ChatResponse(
            answer="I apologize, but I'm experiencing technical difficulties. Please try again later or contact support.",
            sources=[],
            processing_time=processing_time,
            cached=False,
            session_id=user_message.session_id
        )

@app.post("/api/clear-cache")
async def clear_cache():
    """Clear application cache (admin endpoint)."""
    try:
        if app_components['cache_manager']:
            await app_components['cache_manager'].clear_cache()
            logger.info("Cache cleared successfully")
            return {"status": "success", "message": "Cache cleared successfully"}
        else:
            raise HTTPException(status_code=503, detail="Cache manager not available")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@app.get("/api/stats")
async def get_statistics():
    """Get application statistics."""
    try:
        stats = {}
        
        # Get cache statistics
        if app_components['cache_manager']:
            cache_stats = await app_components['cache_manager'].get_statistics()
            stats['cache'] = cache_stats
        
        # Get AI handler statistics
        if app_components['ai_handler']:
            ai_stats = await app_components['ai_handler'].get_statistics()
            stats['ai'] = ai_stats
        
        # Add configuration summary to stats
        if app_components['config']:
            config_summary = {
                'institution_name': app_components['config'].get('institution', {}).get('name', 'Unknown'),
                'cache_max_size': app_components['config'].get('cache', {}).get('max_size', 1000),
                'logging_level': app_components['config'].get('logging', {}).get('level', 'INFO'),
                'ai_primary_model': app_components['config'].get('ai_models', {}).get('primary_model', 'Not configured')
            }
            stats['configuration'] = config_summary
        
        stats['uptime'] = (datetime.now() - startup_time).total_seconds() if 'startup_time' in globals() else 0
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

# Static file serving
@app.get("/favicon.ico")
async def get_favicon():
    """Serve favicon."""
    favicon_path = STATIC_DIR / "assets" / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path))
    return JSONResponse(status_code=404, content={"detail": "Favicon not found"})

# Mount static files after API routes
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

@app.get("/")
async def read_index():
    """Serve the main index page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(
        status_code=404, 
        content={"detail": "Index page not found. Please ensure static files are properly configured."}
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

if __name__ == "__main__":
    # Store startup time for statistics
    startup_time = datetime.now()
    
    # Get port and host from environment with fallbacks
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Development vs Production settings
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port} (debug={debug})")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        access_log=True,
        log_level="info" if not debug else "debug"
    )