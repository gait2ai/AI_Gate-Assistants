# tests/test_prompt_builder.py

"""
AI Gate for Artificial Intelligence Applications
test_prompt_builder Module

Test Suite for the Prompt Builder Module (modules.prompt_builder)

This module provides comprehensive tests for the PromptBuilder class,
ensuring its ability to correctly construct system prompts based on
various inputs including templates, institutional data, research results,
and processed user questions. It covers aspects like:
- Initialization with different configurations and template sources.
- Formatting of research context.
- Language-specific instruction generation.
- Prompt optimization logic.
- Prompt truncation to meet length constraints.
- Template validation.
- Utility methods and health checks.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch # For specific method patching if needed

# Import the class to be tested
from modules.prompt_builder import PromptBuilder

# Pytest marker for all tests in this file to be treated as asyncio
pytestmark = pytest.mark.asyncio


# --- Sample Data Fixtures (Local to this test file) ---

@pytest.fixture
def sample_processed_question_data() -> dict:
    """Provides a sample dictionary for processed question data."""
    return {
        'is_valid': True,
        'original_question': "What are the admission requirements for Computer Science?",
        'cleaned_question': "what are the admission requirements for computer science",
        'language': 'en',
        'language_confidence': 0.95,
        'topics': ['admissions', 'computer science'],
        'keywords': ['admission', 'requirements', 'cs', 'program'],
        'entities': ['Computer Science'],
        'question_type': 'factual', # or 'what' depending on classification logic
        'complexity_score': 0.7,
        'confidence_score': 0.85,
        'error_message': None,
        'processing_time': 0.1,
        'word_count': 8,
        'sentence_count': 1,
        'metadata': {}
    }

@pytest.fixture
def sample_research_results_data() -> list:
    """Provides a sample list of research result dictionaries."""
    return [
        {
            'content_id': 'cs_admission_01',
            'url': 'https://institution.edu/cs/admission',
            'title': 'CS Admission Requirements',
            'summary': 'Details on CS program admission criteria.',
            # In PromptBuilder, 'content' is the key used, not 'relevant_content'
            'content': 'Computer Science admission requires a 3.0 GPA and relevant math courses.',
            'relevance_score': 0.9,
            'source_url': 'https://institution.edu/cs/admission', # Duplicate for clarity if needed
            'keywords': ['cs', 'admission', 'gpa'],
            'category': 'academics',
            'matched_terms': ['admission', 'requirements'],
            'word_count': 50
        },
        {
            'content_id': 'gen_admission_02',
            'url': 'https://institution.edu/admissions',
            'title': 'General Admission Info',
            'summary': 'General information about applying to the university.',
            'content': 'The general application deadline is January 15th. Apply online via our portal.',
            'relevance_score': 0.75,
            'source_url': 'https://institution.edu/admissions',
            'keywords': ['application', 'deadline', 'portal'],
            'category': 'admissions',
            'matched_terms': ['admission', 'deadline'],
            'word_count': 40
        }
    ]

@pytest.fixture
def temp_config_dir_with_custom_prompt(config_dir: Path, custom_system_prompt_text: str, prompt_builder_templates_config: dict) -> Path:
    """
    Creates a temporary config directory, writes a custom system prompt to a file
    within it (named as per prompt_builder_templates_config), and yields the directory path.
    Cleans up the directory afterwards.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="test_prompt_builder_custom_"))
    
    # Get the configured template filename, or use a default if not specified
    template_filename = prompt_builder_templates_config.get('system_template_file', "system_prompt.txt")
    if not template_filename: # Handle empty string case
        template_filename = "system_prompt.txt"
        
    custom_prompt_file = temp_dir / template_filename
    
    try:
        with open(custom_prompt_file, 'w', encoding='utf-8') as f:
            f.write(custom_system_prompt_text)
        yield temp_dir # Yield the temp directory containing the custom prompt
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


@pytest.fixture
def custom_system_prompt_text() -> str:
    """Provides a string for a custom system prompt template."""
    return """### Custom Test Prompt
Institution: {institution_name}
Context Provided:
{context}
Respond in: {language_instruction}
Guidance: Be brief.
"""

# --- Main Test Class for PromptBuilder ---

class TestPromptBuilder:
    """Groups tests for the PromptBuilder class."""

    @pytest.fixture
    def builder_instance(self, config_dir: Path, institution_data_config: dict, prompt_builder_templates_config: dict) -> PromptBuilder:
        """
        Provides a PromptBuilder instance initialized with configurations
        from conftest.py. This instance will use the default embedded prompt
        template if 'system_template_file' is not configured or the file is not found.
        """
        return PromptBuilder(
            config_dir=config_dir, # Real config_dir, but template might not exist there by default
            institution_data=institution_data_config,
            templates=prompt_builder_templates_config
        )

    @pytest.fixture
    def builder_with_custom_file_template(self, temp_config_dir_with_custom_prompt: Path, institution_data_config: dict, prompt_builder_templates_config: dict) -> PromptBuilder:
        """
        Provides a PromptBuilder instance that is configured to load its
        system prompt from a custom file created in a temporary directory.
        """
        # Ensure templates_config points to the file within temp_config_dir_with_custom_prompt
        # This assumes prompt_builder_templates_config['system_template_file'] is set correctly
        # (e.g., "system_prompt.txt") and the file exists in temp_config_dir_with_custom_prompt.
        return PromptBuilder(
            config_dir=temp_config_dir_with_custom_prompt, # Use the temp dir where custom prompt is
            institution_data=institution_data_config,
            templates=prompt_builder_templates_config # This should specify the filename
        )


    # --- Initialization and Configuration Tests ---
    class TestInitialization:
        """Tests related to PromptBuilder initialization and template loading."""

        def test_initialization_default_template(self, builder_instance: PromptBuilder, institution_data_config: dict):
            """
            Tests successful initialization and loading of the default embedded system prompt
            if no 'system_template_file' is found or configured.
            """
            assert builder_instance.institution_data == institution_data_config
            assert builder_instance.config_dir is not None # Passed from conftest
            assert "You are an intelligent assistant representing" in builder_instance.system_prompt_template # Part of default

        def test_initialization_with_custom_file_template(self, builder_with_custom_file_template: PromptBuilder, custom_system_prompt_text: str):
            """
            Tests successful initialization and loading of a custom system prompt
            from a file specified in the configuration.
            """
            assert custom_system_prompt_text.strip() == builder_with_custom_file_template.system_prompt_template.strip()
            assert "Custom Test Prompt" in builder_with_custom_file_template.system_prompt_template

        def test_initialization_template_file_not_configured(self, config_dir: Path, institution_data_config: dict):
            """
            Tests that default template is used if 'system_template_file' is not in templates config.
            """
            # templates_config without 'system_template_file'
            templates_config_no_file = {"max_context_length": 1000}
            builder = PromptBuilder(
                config_dir=config_dir,
                institution_data=institution_data_config,
                templates=templates_config_no_file
            )
            assert "You are an intelligent assistant representing" in builder.system_prompt_template

        def test_initialization_template_file_configured_but_not_found(self, config_dir: Path, institution_data_config: dict):
            """
            Tests that default template is used if 'system_template_file' is configured
            but the file does not exist in config_dir.
            """
            templates_config_missing_file = {"system_template_file": "non_existent_prompt.txt"}
            builder = PromptBuilder(
                config_dir=config_dir, # Real config_dir, where non_existent_prompt.txt isn't
                institution_data=institution_data_config,
                templates=templates_config_missing_file
            )
            assert "You are an intelligent assistant representing" in builder.system_prompt_template

        def test_language_instructions_loaded(self, builder_instance: PromptBuilder):
            """Ensures that language instructions are loaded."""
            assert isinstance(builder_instance.language_instructions, dict)
            assert 'en' in builder_instance.language_instructions
            assert 'es' in builder_instance.language_instructions


    # --- Prompt Building Core Logic Tests ---
    class TestPromptBuilding:
        """Tests the core build_prompt method."""

        @pytest.mark.asyncio
        async def test_build_prompt_successful_construction(self, builder_instance: PromptBuilder, sample_processed_question_data: dict, sample_research_results_data: list, institution_data_config: dict):
            """Tests successful prompt construction with all components."""
            prompt = await builder_instance.build_prompt(
                original_question=sample_processed_question_data['original_question'],
                processed_question=sample_processed_question_data,
                research_results=sample_research_results_data
            )
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            # Check for institution name placeholder filled
            assert institution_data_config['name'] in prompt
            # Check for context from research results
            assert sample_research_results_data[0]['content'] in prompt
            # Check for language instruction
            assert builder_instance.language_instructions[sample_processed_question_data['language']] in prompt

        @pytest.mark.asyncio
        async def test_build_prompt_no_research_results(self, builder_instance: PromptBuilder, sample_processed_question_data: dict):
            """Tests prompt construction when research_results list is empty."""
            prompt = await builder_instance.build_prompt(
                original_question=sample_processed_question_data['original_question'],
                processed_question=sample_processed_question_data,
                research_results=[] # Empty research results
            )
            assert "No specific institutional data was found" in prompt
            assert builder_instance.language_instructions['en'] in prompt # Assuming 'en' is default

        @pytest.mark.asyncio
        async def test_build_prompt_different_languages(self, builder_instance: PromptBuilder, sample_research_results_data: list):
            """Tests if language instructions are correctly inserted for different languages."""
            original_q = "Test"
            for lang_code, lang_instruction in builder_instance.language_instructions.items():
                processed_q_lang = {"language": lang_code, "topics": [], "keywords": [], "question_type": "general"}
                prompt = await builder_instance.build_prompt(original_q, processed_q_lang, sample_research_results_data)
                assert lang_instruction in prompt

        @pytest.mark.asyncio
        async def test_build_prompt_with_optimization_enabled(self, builder_instance: PromptBuilder, sample_processed_question_data: dict, sample_research_results_data: list):
            """Tests that prompt optimization section is added when enabled."""
            # Ensure optimization is enabled (default is True from PromptBuilder, or from fixture)
            builder_instance.templates['prompt_optimization'] = True # Explicitly set for test clarity
            
            prompt = await builder_instance.build_prompt(
                original_question=sample_processed_question_data['original_question'],
                processed_question=sample_processed_question_data,
                research_results=sample_research_results_data
            )
            assert "Additional Guidance" in prompt
            assert f"Focus on topics: {', '.join(sample_processed_question_data['topics'][:3])}" in prompt

        @pytest.mark.asyncio
        async def test_build_prompt_with_optimization_disabled(self, builder_instance: PromptBuilder, sample_processed_question_data: dict, sample_research_results_data: list):
            """Tests that prompt optimization is skipped if disabled in config."""
            builder_instance.templates['prompt_optimization'] = False # Disable optimization
            
            prompt = await builder_instance.build_prompt(
                original_question=sample_processed_question_data['original_question'],
                processed_question=sample_processed_question_data,
                research_results=sample_research_results_data
            )
            assert "Additional Guidance" not in prompt

        @pytest.mark.asyncio
        @patch('modules.prompt_builder.PromptBuilder._format_research_context', side_effect=Exception("Context formatting failed!"))
        async def test_build_prompt_handles_internal_error_gracefully(self, mock_format_context, builder_instance: PromptBuilder, sample_processed_question_data: dict):
            """
            Tests that build_prompt returns a fallback prompt if an internal error occurs.
            """
            fallback_prompt = await builder_instance.build_prompt(
                original_question=sample_processed_question_data['original_question'],
                processed_question=sample_processed_question_data,
                research_results=[] # Research results don't matter due to mocked error
            )
            assert mock_format_context.called
            # Check if the fallback prompt content is present
            assert "Please help the user with their question:" in fallback_prompt
            assert sample_processed_question_data['original_question'] in fallback_prompt


    # --- Context Formatting Tests ---
    class TestContextFormatting:
        """Tests the _format_research_context method."""

        def test_format_research_context_with_results(self, builder_instance: PromptBuilder, sample_research_results_data: list):
            """Tests formatting with valid research results."""
            context_str = builder_instance._format_research_context(sample_research_results_data)
            assert "--- Source 1 ---" in context_str
            assert f"URL: {sample_research_results_data[0]['source_url']}" in context_str
            assert f"Content:\n{sample_research_results_data[0]['content']}" in context_str
            assert f"Relevance: {sample_research_results_data[0]['relevance_score']:.2f}" in context_str
            assert "--- Source 2 ---" in context_str

        def test_format_research_context_empty(self, builder_instance: PromptBuilder):
            """Tests formatting with an empty list of research results."""
            context_str = builder_instance._format_research_context([])
            assert "No specific institutional data was found" in context_str

        def test_format_research_context_truncation(self, builder_instance: PromptBuilder):
            """Tests context truncation if total length exceeds max_context_length."""
            long_content = "This is very long content. " * 200 # Approx 4000 chars
            research_results = [
                {"content": long_content, "source_url": "url1", "relevance_score": 0.9},
                {"content": "Short content.", "source_url": "url2", "relevance_score": 0.8}
            ]
            # Set max_context_length from templates config for the builder instance
            max_len = builder_instance.templates.get('max_context_length', 4000)
            
            context_str = builder_instance._format_research_context(research_results)
            
            assert len(context_str) <= max_len + 500 # Allow some overhead for formatting
            assert "Short content." not in context_str or "...[truncated]" in context_str # Second item might be truncated or omitted
            if "Short content." in context_str:
                assert "...[truncated]" in research_results[0]['content'] or "...[truncated]" in context_str

        def test_format_research_context_result_without_content(self, builder_instance: PromptBuilder):
            """Tests handling of research results where a result item might lack 'content'."""
            results_mixed = [
                {"content": "Valid content.", "source_url": "url1", "relevance_score": 0.9},
                {"source_url": "url2", "relevance_score": 0.8} # No 'content' key
            ]
            context_str = builder_instance._format_research_context(results_mixed)
            assert "Valid content." in context_str
            # The entry without 'content' should be skipped, not cause an error.
            # Check that the context doesn't contain an error or an empty "Content:" section for source 2.
            assert "--- Source 2 ---" not in context_str or "Content:\n\n" not in context_str.split("--- Source 2 ---")[-1]


    # --- Prompt Truncation Tests ---
    class TestPromptTruncation:
        """Tests the _truncate_if_needed method."""

        def test_truncate_prompt_below_limit(self, builder_instance: PromptBuilder):
            """Tests that a prompt shorter than max_prompt_length is not truncated."""
            prompt = "Short prompt."
            max_prompt_len = builder_instance.templates.get('max_prompt_length', 8000)
            truncated_prompt = builder_instance._truncate_if_needed(prompt)
            assert truncated_prompt == prompt
            assert len(truncated_prompt) <= max_prompt_len

        def test_truncate_prompt_exceeding_limit(self, builder_instance: PromptBuilder):
            """Tests that a prompt longer than max_prompt_length is truncated."""
            max_prompt_len = builder_instance.templates.get('max_prompt_length', 8000)
            long_prompt_text = "a" * (max_prompt_len + 100)
            
            truncated_prompt = builder_instance._truncate_if_needed(long_prompt_text)
            
            assert len(truncated_prompt) <= max_prompt_len
            assert "...[Truncated]" in truncated_prompt or "...[Content truncated due to length limits]" in truncated_prompt


    # --- Template Validation Tests ---
    class TestTemplateValidation:
        """Tests the validate_template method."""

        @pytest.mark.asyncio
        async def test_validate_template_valid(self, builder_instance: PromptBuilder, custom_system_prompt_text: str):
            """Tests validation of a correctly formatted template."""
            validation = await builder_instance.validate_template(custom_system_prompt_text)
            assert validation['is_valid'] is True
            assert not validation['errors']
            assert all(p in validation['found_placeholders'] for p in ['institution_name', 'context', 'language_instruction'])

        @pytest.mark.asyncio
        @pytest.mark.parametrize("invalid_template, expected_error_part", [
            ("Missing all placeholders.", "Missing required placeholder: {institution_name}"),
            ("Only {institution_name} and {context}.", "Missing required placeholder: {language_instruction}"),
        ])
        async def test_validate_template_missing_placeholders(self, builder_instance: PromptBuilder, invalid_template: str, expected_error_part: str):
            """Tests validation when required placeholders are missing."""
            validation = await builder_instance.validate_template(invalid_template)
            assert validation['is_valid'] is False
            assert any(expected_error_part in error for error in validation['errors'])

        @pytest.mark.asyncio
        async def test_validate_template_with_warnings(self, builder_instance: PromptBuilder):
            """Tests validation for issues that generate warnings (e.g., long template, unmatched braces)."""
            long_template_content = "{institution_name}{context}{language_instruction}" + ("_very_long_template_" * 300)
            validation_long = await builder_instance.validate_template(long_template_content)
            assert validation_long['is_valid'] is True # Still valid structure-wise
            assert any("quite long" in warning for warning in validation_long['warnings'])

            template_unmatched_braces = "{institution_name {context} {language_instruction}"
            validation_braces = await builder_instance.validate_template(template_unmatched_braces)
            assert any("Unmatched braces" in warning for warning in validation_braces['warnings'])


    # --- Utility Method Tests ---
    class TestUtilityMethods:
        """Tests for helper/utility methods like get_template_variables, is_healthy."""

        @pytest.mark.asyncio
        async def test_get_template_variables(self, builder_instance: PromptBuilder, institution_data_config: dict, prompt_builder_templates_config: dict, config_dir: Path):
            """Tests retrieval of available template variables."""
            variables = await builder_instance.get_template_variables()
            assert isinstance(variables, dict)
            assert variables['institution_name'] == institution_data_config['name']
            assert variables['default_language'] == 'en'
            expected_template_file = config_dir / prompt_builder_templates_config.get('system_template_file', 'system_prompt.txt')
            assert variables['template_file'] == str(expected_template_file)


        @pytest.mark.asyncio
        async def test_get_fallback_prompt(self, builder_instance: PromptBuilder, institution_data_config: dict):
            """Tests the generation of a fallback prompt."""
            original_question = "A complex user query"
            fallback = builder_instance._get_fallback_prompt(original_question)
            assert institution_data_config['name'] in fallback
            assert original_question in fallback

        def test_is_healthy_true(self, builder_instance: PromptBuilder):
            """Tests is_healthy when the builder is properly configured."""
            assert builder_instance.is_healthy() is True

        def test_is_healthy_false_no_template(self, config_dir: Path, institution_data_config: dict, prompt_builder_templates_config: dict):
            """Tests is_healthy when the system_prompt_template is None (e.g. failed load and no default)."""
            builder = PromptBuilder(config_dir, institution_data_config, prompt_builder_templates_config)
            builder.system_prompt_template = None # Force unhealthy state
            assert builder.is_healthy() is False
        
        def test_is_healthy_false_no_institution_data(self, config_dir: Path, prompt_builder_templates_config: dict):
            """Tests is_healthy when institution_data is empty."""
            builder = PromptBuilder(config_dir, {}, prompt_builder_templates_config) # Empty institution data
            assert builder.is_healthy() is False

        @pytest.mark.asyncio
        async def test_cleanup(self, builder_instance: PromptBuilder):
            """Ensures the cleanup method runs without error."""
            try:
                await builder_instance.cleanup() # PromptBuilder's cleanup is currently a pass
            except Exception as e:
                pytest.fail(f"cleanup() raised an exception: {e}")