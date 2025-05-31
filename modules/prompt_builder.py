"""
AI Gate for Artificial Intelligence Applications
Prompt Builder Module

This module handles the construction of comprehensive system prompts by combining
default templates, institution-specific data, website research results, and user questions.
It manages prompt optimization for different AI models and handles template customization.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Handles construction and optimization of system prompts for AI models.
    
    This class combines various data sources to create contextually appropriate
    system prompts that guide AI model responses according to institutional
    requirements and available research data.
    """

    def __init__(self, config_dir: Path, institution_data: Dict[str, Any], templates: Dict[str, Any]):
        """
        Initialize the PromptBuilder with configuration and templates.
        
        Args:
            config_dir: Directory containing configuration files
            institution_data: Institution-specific configuration data
            templates: Template configuration for prompts
        """
        self.config_dir = Path(config_dir)
        self.institution_data = institution_data
        self.templates = templates
        
        # Default configuration
        self.default_config = {
            'max_context_length': 4000,
            'context_truncation_strategy': 'smart',  # 'smart', 'end', 'middle'
            'language_detection': True,
            'prompt_optimization': True,
            'template_variables': {}
        }
        
        # Language mapping for responses
        self.language_instructions = {
            'en': "Please respond in English.",
            'es': "Por favor responde en español.",
            'fr': "Veuillez répondre en français.",
            'de': "Bitte antworten Sie auf Deutsch.",
            'it': "Per favore rispondi in italiano.",
            'pt': "Por favor responda em português.",
            'ar': "يرجى الرد باللغة العربية.",
            'zh': "请用中文回答。",
            'ja': "日本語でお答えください。",
            'ko': "한국어로 답변해 주세요.",
            'ru': "Пожалуйста, отвечайте на русском языке.",
            'hi': "कृपया हिंदी में उत्तर दें।"
        }
        
        # Load system prompt template
        self.system_prompt_template = self._load_system_prompt_template()
        
        # Initialize prompt components
        self.base_prompt = None
        self.context_formatter = None
        
        logger.info("PromptBuilder initialized successfully")

    def _load_system_prompt_template(self) -> str:
        """
        Load the system prompt template from a configured file or use default.
        
        The method attempts to load the template from a filename specified in
        the configuration (prompts.system_template_file). If no valid filename
        is configured or if file loading fails, it falls back to the default
        embedded template.
        
        Returns:
            str: System prompt template
        """
        # Step 1: Retrieve filename from configuration
        configured_filename = self.templates.get('system_template_file')
        
        # Step 2: Validate the configured filename
        if not configured_filename:
            logger.info("No 'system_template_file' configured, using default embedded template")
            return self._get_default_system_prompt_template()
        
        if not isinstance(configured_filename, str) or not configured_filename.strip():
            logger.warning(f"Invalid 'system_template_file' value: {configured_filename!r}. Must be a non-empty string. Using default embedded template")
            return self._get_default_system_prompt_template()
        
        # Clean the filename (remove whitespace)
        configured_filename = configured_filename.strip()
        
        # Step 3: Attempt to load from configured file
        template_file = self.config_dir / configured_filename
        
        try:
            if not template_file.exists():
                logger.warning(f"Configured system prompt template file not found: '{configured_filename}' (full path: {template_file}). Using default embedded template")
                return self._get_default_system_prompt_template()
            
            if not template_file.is_file():
                logger.warning(f"Configured system prompt template path is not a file: '{configured_filename}' (full path: {template_file}). Using default embedded template")
                return self._get_default_system_prompt_template()
            
            # Attempt to read the file
            with open(template_file, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # Check if the file has meaningful content
            if not template_content or not template_content.strip():
                logger.warning(f"Configured system prompt template file is empty: '{configured_filename}' (full path: {template_file}). Using default embedded template")
                return self._get_default_system_prompt_template()
            
            # Successfully loaded template from configured file
            template_content = template_content.strip()
            logger.info(f"Successfully loaded system prompt template from configured file: '{configured_filename}'")
            return template_content
            
        except PermissionError:
            logger.warning(f"Permission denied reading configured system prompt template file: '{configured_filename}' (full path: {template_file}). Using default embedded template")
            return self._get_default_system_prompt_template()
        
        except UnicodeDecodeError as e:
            logger.warning(f"Unicode decode error reading configured system prompt template file: '{configured_filename}' (full path: {template_file}): {e}. Using default embedded template")
            return self._get_default_system_prompt_template()
        
        except OSError as e:
            logger.warning(f"OS error reading configured system prompt template file: '{configured_filename}' (full path: {template_file}): {e}. Using default embedded template")
            return self._get_default_system_prompt_template()
        
        except Exception as e:
            logger.warning(f"Unexpected error reading configured system prompt template file: '{configured_filename}' (full path: {template_file}): {e}. Using default embedded template")
            return self._get_default_system_prompt_template()

    def _get_default_system_prompt_template(self) -> str:
        """
        Get the default system prompt template.
        
        Returns:
            str: Default system prompt template
        """
        return """### Role
You are an intelligent assistant representing {institution_name}. Your primary function is to provide accurate and professional responses strictly based on the training data provided by the organization.

### Persona
- Identity: You are a virtual assistant trained on {institution_name} documents and internal sources. You must not impersonate a human or represent yourself as an official staff member.
- Purpose: To assist users in accessing relevant information about {institution_name} activities, projects, partners, sectors of work, mission, and vision.

### Constraints
1. Exclusive Data Reliance: You must rely exclusively on the provided internal data. You are not allowed to reference or fabricate external information.
2. Topic Focus: If the user asks unrelated or out-of-scope questions, politely decline and redirect the conversation back to the organization's context.
3. No Generalizations: Avoid making general claims or assumptions. Stick to what is explicitly documented.
4. Transparency: If the information is not found in your data, inform the user accordingly and suggest they contact official support.

### Knowledge Context
{context}

{language_instruction}"""

    async def build_prompt(self, 
                          original_question: str, 
                          processed_question: Dict[str, Any], 
                          research_results: List[Dict[str, Any]]) -> str:
        """
        Build a comprehensive system prompt for AI model interaction.
        
        Args:
            original_question: The original user question
            processed_question: Processed question analysis from QuestionProcessor
            research_results: Website research results from WebsiteResearcher
            
        Returns:
            str: Complete system prompt ready for AI model
        """
        try:
            logger.debug(f"Building prompt for question: {original_question[:50]}...")
            
            # Extract context from research results
            context = self._format_research_context(research_results)
            
            # Detect language and get instruction
            language_code = processed_question.get('language', 'en')
            language_instruction = self._get_language_instruction(language_code)
            
            # Get institution name
            institution_name = self.institution_data.get('name', 'the organization')
            
            # Format the system prompt
            formatted_prompt = self.system_prompt_template.format(
                institution_name=institution_name,
                context=context,
                language_instruction=language_instruction
            )
            
            # Apply prompt optimization if enabled
            if self.templates.get('prompt_optimization', True):
                formatted_prompt = self._optimize_prompt(formatted_prompt, processed_question)
            
            # Ensure prompt doesn't exceed length limits
            formatted_prompt = self._truncate_if_needed(formatted_prompt)
            
            logger.debug("System prompt built successfully")
            return formatted_prompt
            
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            # Return a basic fallback prompt
            return self._get_fallback_prompt(original_question)

    def _format_research_context(self, research_results: List[Dict[str, Any]]) -> str:
        """
        Format research results into context for the system prompt.
        
        Args:
            research_results: List of research results from website researcher
            
        Returns:
            str: Formatted context string
        """
        if not research_results:
            return "No specific institutional data was found for this query. Please provide general guidance based on your training."
        
        context_parts = []
        max_context_length = self.templates.get('max_context_length', 4000)
        current_length = 0
        
        for i, result in enumerate(research_results):
            # Extract relevant information
            content = result.get('content', '').strip()
            source_url = result.get('source_url', '')
            relevance_score = result.get('relevance_score', 0)
            
            if not content:
                continue
            
            # Format the context entry
            context_entry = f"--- Source {i + 1} ---\n"
            if source_url:
                context_entry += f"URL: {source_url}\n"
            if relevance_score > 0:
                context_entry += f"Relevance: {relevance_score:.2f}\n"
            context_entry += f"Content:\n{content}\n\n"
            
            # Check if adding this entry would exceed the limit
            if current_length + len(context_entry) > max_context_length:
                if self.templates.get('context_truncation_strategy', 'smart') == 'smart':
                    # Truncate the content to fit
                    remaining_space = max_context_length - current_length - 200  # Leave some buffer
                    if remaining_space > 100:  # Only add if there's meaningful space
                        truncated_content = content[:remaining_space] + "...[truncated]"
                        context_entry = f"--- Source {i + 1} ---\n"
                        if source_url:
                            context_entry += f"URL: {source_url}\n"
                        context_entry += f"Content:\n{truncated_content}\n\n"
                        context_parts.append(context_entry)
                break
            
            context_parts.append(context_entry)
            current_length += len(context_entry)
        
        if not context_parts:
            return "No specific institutional data was found for this query. Please provide general guidance based on your training."
        
        return "".join(context_parts)

    def _get_language_instruction(self, language_code: str) -> str:
        """
        Get language-specific instruction for the AI model.
        
        Args:
            language_code: ISO language code
            
        Returns:
            str: Language instruction
        """
        return self.language_instructions.get(language_code.lower(), self.language_instructions['en'])

    def _optimize_prompt(self, prompt: str, processed_question: Dict[str, Any]) -> str:
        """
        Optimize the prompt based on question analysis and model requirements.
        
        Args:
            prompt: The base prompt to optimize
            processed_question: Processed question analysis
            
        Returns:
            str: Optimized prompt
        """
        try:
            # Add question-specific guidance
            topics = processed_question.get('topics', [])
            keywords = processed_question.get('keywords', [])
            question_type = processed_question.get('question_type', 'general')
            
            optimization_notes = []
            
            # Add topic-specific guidance
            if topics:
                topic_list = ", ".join(topics[:3])  # Limit to top 3 topics
                optimization_notes.append(f"Focus on topics: {topic_list}")
            
            # Add question type specific guidance
            if question_type == 'factual':
                optimization_notes.append("Provide factual, specific information with sources when available.")
            elif question_type == 'procedural':
                optimization_notes.append("Provide step-by-step guidance based on organizational procedures.")
            elif question_type == 'comparative':
                optimization_notes.append("Compare options based on the provided institutional data.")
            
            # Add keyword emphasis
            if keywords and len(keywords) <= 5:
                keyword_list = ", ".join(keywords)
                optimization_notes.append(f"Pay special attention to: {keyword_list}")
            
            # Append optimization notes to prompt
            if optimization_notes:
                optimization_section = "\n### Additional Guidance\n" + "\n".join(f"- {note}" for note in optimization_notes)
                prompt += optimization_section
            
            return prompt
            
        except Exception as e:
            logger.warning(f"Error optimizing prompt: {e}")
            return prompt

    def _truncate_if_needed(self, prompt: str) -> str:
        """
        Truncate prompt if it exceeds maximum length limits.
        
        Args:
            prompt: The prompt to check and potentially truncate
            
        Returns:
            str: Truncated prompt if necessary
        """
        max_length = self.templates.get('max_prompt_length', 8000)
        
        if len(prompt) <= max_length:
            return prompt
        
        logger.warning(f"Prompt length ({len(prompt)}) exceeds maximum ({max_length}), truncating...")
        
        # Try to truncate intelligently by removing context first
        lines = prompt.split('\n')
        
        # Find context section
        context_start = -1
        context_end = -1
        
        for i, line in enumerate(lines):
            if '### Knowledge Context' in line:
                context_start = i + 1
            elif context_start > -1 and line.startswith('###') and 'Context' not in line:
                context_end = i
                break
        
        if context_start > -1:
            if context_end == -1:
                context_end = len(lines)
            
            # Calculate how much to remove from context
            context_lines = lines[context_start:context_end]
            other_lines = lines[:context_start] + lines[context_end:]
            other_length = len('\n'.join(other_lines))
            
            available_for_context = max_length - other_length - 100  # Buffer
            
            if available_for_context > 0:
                # Truncate context to fit
                truncated_context = []
                current_length = 0
                
                for line in context_lines:
                    if current_length + len(line) + 1 > available_for_context:
                        if truncated_context:  # Only add truncation notice if we have some content
                            truncated_context.append("...[Content truncated due to length limits]")
                        break
                    truncated_context.append(line)
                    current_length += len(line) + 1
                
                # Reconstruct prompt
                lines = lines[:context_start] + truncated_context + lines[context_end:]
        
        truncated_prompt = '\n'.join(lines)
        
        # Final length check and hard truncation if still too long
        if len(truncated_prompt) > max_length:
            truncated_prompt = truncated_prompt[:max_length - 50] + "\n...[Truncated]"
        
        return truncated_prompt

    def _get_fallback_prompt(self, original_question: str) -> str:
        """
        Get a basic fallback prompt when prompt building fails.
        
        Args:
            original_question: The original user question
            
        Returns:
            str: Basic fallback prompt
        """
        institution_name = self.institution_data.get('name', 'the organization')
        
        return f"""You are an assistant for {institution_name}. 
        
Please help the user with their question: {original_question}

Provide accurate information based on your training data. If you don't have specific information about {institution_name}, please let the user know and suggest they contact official support."""

    async def get_template_variables(self) -> Dict[str, Any]:
        """
        Get available template variables for prompt customization.
        
        Returns:
            Dict[str, Any]: Available template variables
        """
        return {
            'institution_name': self.institution_data.get('name', 'the organization'),
            'institution_website': self.institution_data.get('website', ''),
            'institution_description': self.institution_data.get('description', ''),
            'supported_languages': list(self.language_instructions.keys()),
            'template_file': str(self.config_dir / self.templates.get('system_template_file', 'system_prompt.txt')),
            'default_language': 'en'
        }

    async def validate_template(self, template_content: str) -> Dict[str, Any]:
        """
        Validate a prompt template for required placeholders and format.
        
        Args:
            template_content: The template content to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'required_placeholders': ['institution_name', 'context', 'language_instruction'],
            'found_placeholders': []
        }
        
        try:
            # Check for required placeholders
            for placeholder in validation_result['required_placeholders']:
                placeholder_pattern = f"{{{placeholder}}}"
                if placeholder_pattern in template_content:
                    validation_result['found_placeholders'].append(placeholder)
                else:
                    validation_result['errors'].append(f"Missing required placeholder: {placeholder_pattern}")
                    validation_result['is_valid'] = False
            
            # Check template length
            if len(template_content) > 2000:
                validation_result['warnings'].append("Template is quite long, consider condensing for better performance")
            
            # Check for common issues
            if '{' in template_content and '}' in template_content:
                # Check for unmatched braces
                open_braces = template_content.count('{')
                close_braces = template_content.count('}')
                if open_braces != close_braces:
                    validation_result['warnings'].append("Unmatched braces detected in template")
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Template validation error: {str(e)}")
        
        return validation_result

    def is_healthy(self) -> bool:
        """
        Check if the PromptBuilder is healthy and functioning properly.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Check if we have a valid template
            if not self.system_prompt_template:
                return False
            
            # Check if institution data is available
            if not self.institution_data:
                return False
            
            # Try to format a basic prompt
            test_context = "Test context"
            test_prompt = self.system_prompt_template.format(
                institution_name=self.institution_data.get('name', 'Test Org'),
                context=test_context,
                language_instruction=self.language_instructions['en']
            )
            
            return len(test_prompt) > 0
            
        except Exception as e:
            logger.error(f"PromptBuilder health check failed: {e}")
            return False

    async def cleanup(self):
        """Clean up resources and connections."""
        logger.info("PromptBuilder cleanup completed")
        pass