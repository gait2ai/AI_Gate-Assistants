#!/usr/bin/env python3
"""
Knowledge Base Builder GUI
A web-based administrative control panel for building and managing the AI Gate knowledge base.
"""

import gradio as gr
import subprocess
import sys
import threading
import os
import logging
from typing import Iterator, Tuple, Any, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time
import yaml


class ProcessType(Enum):
    """Enumeration of available process types."""
    WEBSITE_SCRAPER = "website_scraper.py"
    DOCUMENT_PROCESSOR = "document_processor.py"
    MERGE_KNOWLEDGE = "merge_knowledge.py"
    INSTITUTIONAL_DICT_BUILDER = "institutional_dict_builder.py"


class StatusType(Enum):
    """Enumeration of status types."""
    READY = "ready"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class ProcessConfig:
    """Configuration for a process type."""
    script_name: str
    display_name: str
    icon: str
    description: str


@dataclass
class StatusMessage:
    """Status message configuration."""
    text: str
    css_class: str


class KnowledgeBaseManager:
    """Main manager class for knowledge base operations."""
    
    # Process configurations
    PROCESS_CONFIGS = {
        ProcessType.WEBSITE_SCRAPER: ProcessConfig(
            script_name=ProcessType.WEBSITE_SCRAPER.value,
            display_name="Build from Website",
            icon="🌐",
            description="Scrapes and processes content from specified websites"
        ),
        ProcessType.DOCUMENT_PROCESSOR: ProcessConfig(
            script_name=ProcessType.DOCUMENT_PROCESSOR.value,
            display_name="Build from Documents", 
            icon="📄",
            description="Processes uploaded documents and files"
        ),
        ProcessType.MERGE_KNOWLEDGE: ProcessConfig(
            script_name=ProcessType.MERGE_KNOWLEDGE.value,
            display_name="Merge Knowledge Base",
            icon="🔄",
            description="Combines all processed data into the final knowledge base"
        ),
        ProcessType.INSTITUTIONAL_DICT_BUILDER: ProcessConfig(
            script_name=ProcessType.INSTITUTIONAL_DICT_BUILDER.value,
            display_name="Build Institutional Dictionary",
            icon="📖",
            description="Generates categorized keyword dictionary from processed knowledge base data"
        )
    }
    
    # Status messages
    STATUS_MESSAGES = {
        StatusType.READY: StatusMessage("Status: Ready", "status-ready"),
        StatusType.IN_PROGRESS: StatusMessage("Status: In Progress - {}", "status-in-progress"),
        StatusType.SUCCESS: StatusMessage("Status: Complete - {}", "status-success"),
        StatusType.ERROR: StatusMessage("Status: Error - {}", "status-error")
    }
    
    def __init__(self):
        """Initialize the knowledge base manager."""
        self.current_process: Optional[subprocess.Popen] = None
        self.script_dir = Path(__file__).parent
        self.config_dir = self.script_dir / "config"
        self.default_config_path = self.config_dir / "default.yaml"
        self.institution_config_path = self.config_dir / "institution.yaml"
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('knowledge_base_gui.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_status_message(self, status_type: StatusType, process_name: str = "") -> StatusMessage:
        """Get formatted status message."""
        message = self.STATUS_MESSAGES[status_type]
        if "{}" in message.text and process_name:
            formatted_text = message.text.format(process_name)
        else:
            formatted_text = message.text
        return StatusMessage(formatted_text, message.css_class)
        
    def validate_script_exists(self, script_name: str) -> bool:
        """Check if a script file exists."""
        script_path = self.script_dir / script_name
        return script_path.exists()
        
    def get_missing_scripts(self) -> list[str]:
        """Get list of missing required scripts."""
        missing = []
        for config in self.PROCESS_CONFIGS.values():
            if not self.validate_script_exists(config.script_name):
                missing.append(config.script_name)
        return missing
        
    def load_institution_config(self) -> Tuple[str, str, str, str, str]:
        """
        Load institution configuration from YAML files.
        
        Returns:
            Tuple of (name, description, website, contact_email, timezone)
        """
        try:
            # Default values
            default_values = {
                'name': '',
                'description': '',
                'website': '',
                'contact_email': '',
                'timezone': ''
            }
            
            # First, try to load defaults
            if self.default_config_path.exists():
                try:
                    with open(self.default_config_path, 'r', encoding='utf-8') as f:
                        default_config = yaml.safe_load(f) or {}
                        institution_defaults = default_config.get('institution', {})
                        for key in default_values:
                            if key in institution_defaults:
                                default_values[key] = institution_defaults[key]
                except Exception as e:
                    self.logger.warning(f"Could not load default config: {e}")
            
            # Then, override with institution-specific values
            if self.institution_config_path.exists():
                try:
                    with open(self.institution_config_path, 'r', encoding='utf-8') as f:
                        institution_config = yaml.safe_load(f) or {}
                        institution_data = institution_config.get('institution', {})
                        for key in default_values:
                            if key in institution_data:
                                default_values[key] = institution_data[key]
                except Exception as e:
                    self.logger.warning(f"Could not load institution config: {e}")
            
            return (
                default_values['name'],
                default_values['description'],
                default_values['website'],
                default_values['contact_email'],
                default_values['timezone']
            )
            
        except Exception as e:
            self.logger.error(f"Error loading institution configuration: {e}")
            return "", "", "", "", ""
    
    def save_institution_config(self, name: str, description: str, website: str, 
                              contact_email: str, timezone: str) -> str:
        """
        Save institution configuration to institution.yaml file.
        
        Args:
            name: Institution name
            description: Institution description
            website: Institution website URL
            contact_email: Institution contact email
            timezone: Institution timezone
            
        Returns:
            Status message indicating success or failure
        """
        try:
            # Ensure config directory exists
            self.config_dir.mkdir(exist_ok=True)
            
            # Prepare the new institution data
            new_institution_data = {
                'name': name.strip(),
                'description': description.strip(),
                'website': website.strip(),
                'contact_email': contact_email.strip(),
                'timezone': timezone.strip()
            }
            
            # Load existing institution config or create new
            existing_config = {}
            if self.institution_config_path.exists():
                try:
                    with open(self.institution_config_path, 'r', encoding='utf-8') as f:
                        existing_config = yaml.safe_load(f) or {}
                except Exception as e:
                    self.logger.warning(f"Could not load existing institution config: {e}")
            
            # Update institution section
            existing_config['institution'] = new_institution_data
            
            # Write the updated config
            with open(self.institution_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(existing_config, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
            
            self.logger.info("Institution configuration saved successfully")
            return "✅ Institution settings saved successfully!"
            
        except Exception as e:
            error_msg = f"❌ Error saving institution configuration: {str(e)}"
            self.logger.error(f"Error saving institution configuration: {e}")
            return error_msg
        
    def run_process(self, process_type: ProcessType, status_box, console_box, *buttons) -> Iterator[Tuple[Any, ...]]:
        """
        Execute a process and stream its output in real-time.
        
        Args:
            process_type: Type of process to execute
            status_box: Gradio component for status display
            console_box: Gradio component for console output
            *buttons: Button components to disable/enable
            
        Yields:
            Tuple of updated component states
        """
        config = self.PROCESS_CONFIGS[process_type]
        script_path = self.script_dir / config.script_name
        
        self.logger.info(f"Starting process: {config.display_name}")
        
        # Initial state: disable all buttons, set in-progress status, clear console
        status_msg = self.get_status_message(StatusType.IN_PROGRESS, config.display_name)
        yield self._create_update_tuple(
            status_msg, "", False, *buttons
        )
        
        try:
            # Validate script exists
            if not script_path.exists():
                error_msg = f"Error: Script '{config.script_name}' not found at {script_path}"
                self.logger.error(error_msg)
                status_msg = self.get_status_message(StatusType.ERROR, f"Script not found")
                yield self._create_update_tuple(
                    status_msg, error_msg, True, *buttons
                )
                return
            
            # Start the subprocess
            self.current_process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=str(self.script_dir)
            )
            
            output_lines = []
            
            # Stream output in real-time
            while True:
                line = self.current_process.stdout.readline()
                if line:
                    output_lines.append(line.rstrip())
                    current_output = "\n".join(output_lines)
                    yield self._create_update_tuple(
                        None, current_output, None, *buttons
                    )
                elif self.current_process.poll() is not None:
                    break
            
            # Get the return code
            return_code = self.current_process.poll()
            
            # Final state based on success or failure
            if return_code == 0:
                success_msg = f"✅ Process completed successfully (exit code: {return_code})"
                final_output = "\n".join(output_lines) + f"\n\n{success_msg}"
                status_msg = self.get_status_message(StatusType.SUCCESS, f"{config.display_name} finished successfully")
                self.logger.info(f"Process {config.display_name} completed successfully")
                
                yield self._create_update_tuple(
                    status_msg, final_output, True, *buttons
                )
            else:
                error_msg = f"❌ Process failed (exit code: {return_code})"
                final_output = "\n".join(output_lines) + f"\n\n{error_msg}"
                status_msg = self.get_status_message(StatusType.ERROR, f"Failed to {config.display_name.lower()}")
                self.logger.error(f"Process {config.display_name} failed with exit code {return_code}")
                
                yield self._create_update_tuple(
                    status_msg, final_output, True, *buttons
                )
                
        except Exception as e:
            # Handle any unexpected errors
            error_msg = f"Unexpected error occurred: {str(e)}"
            self.logger.exception(f"Unexpected error in process {config.display_name}")
            final_output = "\n".join(output_lines) + f"\n\n{error_msg}" if 'output_lines' in locals() else error_msg
            
            status_msg = self.get_status_message(StatusType.ERROR, f"Failed to {config.display_name.lower()}")
            yield self._create_update_tuple(
                status_msg, final_output, True, *buttons
            )
        finally:
            self.current_process = None
            
    def _create_update_tuple(self, status_msg: Optional[StatusMessage], console_output: Optional[str], 
                           buttons_interactive: Optional[bool], *buttons) -> tuple:
        """Create update tuple for Gradio components."""
        updates = []
        
        # Status update
        if status_msg:
            updates.append(gr.update(value=status_msg.text, elem_classes=[status_msg.css_class]))
        else:
            updates.append(gr.update())
            
        # Console output update
        if console_output is not None:
            updates.append(gr.update(value=console_output))
        else:
            updates.append(gr.update())
            
        # Button updates
        for _ in buttons:
            if buttons_interactive is not None:
                updates.append(gr.update(interactive=buttons_interactive))
            else:
                updates.append(gr.update())
                
        return tuple(updates)


class GradioInterface:
    """Gradio interface builder and manager."""
    
    # Custom CSS for styling
    CUSTOM_CSS = """
    .status-ready {
        background-color: #f0f0f0 !important;
        color: #333 !important;
        padding: 12px !important;
        border-radius: 8px !important;
        text-align: center !important;
        font-weight: bold !important;
        border: 2px solid #ddd !important;
    }
    .status-in-progress {
        background-color: #fff3cd !important;
        color: #856404 !important;
        padding: 12px !important;
        border-radius: 8px !important;
        text-align: center !important;
        font-weight: bold !important;
        border: 2px solid #ffeaa7 !important;
        animation: pulse 2s infinite;
    }
    .status-success {
        background-color: #d4edda !important;
        color: #155724 !important;
        padding: 12px !important;
        border-radius: 8px !important;
        text-align: center !important;
        font-weight: bold !important;
        border: 2px solid #00b894 !important;
    }
    .status-error {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        padding: 12px !important;
        border-radius: 8px !important;
        text-align: center !important;
        font-weight: bold !important;
        border: 2px solid #e17055 !important;
    }
    .console-output {
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
        background-color: #1e1e1e !important;
        color: #f8f8f2 !important;
        border-radius: 8px !important;
    }
    .institution-config-status {
        margin-top: 10px !important;
        padding: 8px !important;
        border-radius: 6px !important;
        text-align: center !important;
        font-weight: bold !important;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .main-header {
        text-align: center !important;
        margin-bottom: 2rem !important;
    }
    .button-row {
        margin: 1.5rem 0 !important;
    }
    """
    
    def __init__(self, manager: KnowledgeBaseManager):
        """Initialize the Gradio interface."""
        self.manager = manager
        
    def create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface."""
        
        with gr.Blocks(title="AI Gate Knowledge Base Builder", css=self.CUSTOM_CSS) as interface:
            
            # Header
            with gr.Column(elem_classes=["main-header"]):
                gr.Markdown("# 🔧 AI Gate Knowledge Base Builder")
                gr.Markdown("*Administrative control panel for building and managing the AI Gate knowledge base*")
            
            # Create tabs
            with gr.Tabs():
                
                # Knowledge Base Builder Tab (original functionality)
                with gr.Tab("Knowledge Base Builder", id="kb_builder"):
                    
                    # Status indicator
                    status_box = gr.Textbox(
                        value=self.manager.get_status_message(StatusType.READY).text,
                        label="System Status",
                        interactive=False,
                        elem_classes=["status-ready"],
                        show_label=True
                    )
                    
                    # Action buttons
                    with gr.Row(elem_classes=["button-row"]):
                        buttons = []
                        for process_type in ProcessType:
                            config = self.manager.PROCESS_CONFIGS[process_type]
                            btn = gr.Button(
                                f"{config.icon} {config.display_name}",
                                variant="primary",
                                size="lg",
                                interactive=True
                            )
                            buttons.append((btn, process_type))
                    
                    # Live output console
                    console_box = gr.Textbox(
                        label="Live Console Output",
                        lines=22,
                        max_lines=35,
                        interactive=False,
                        elem_classes=["console-output"],
                        show_label=True,
                        placeholder="Console output will appear here when a process is running..."
                    )
                    
                    # Instructions and information
                    self._create_instructions_section()
                
                # Institution Configuration Tab (new functionality)
                with gr.Tab("Institution Configuration", id="institution_config"):
                    
                    gr.Markdown("## 🏛️ Institution Configuration")
                    gr.Markdown("Configure core institutional details that will be used throughout the AI Gate application.")
                    
                    with gr.Column():
                        # Institution configuration inputs
                        inst_name = gr.Textbox(
                            label="Institution Name",
                            placeholder="Enter your institution's name",
                            info="The official name of your organization"
                        )
                        
                        inst_description = gr.Textbox(
                            label="Institution Description",
                            lines=3,
                            placeholder="Enter a description of your institution",
                            info="A brief description of your organization's purpose and activities"
                        )
                        
                        inst_website = gr.Textbox(
                            label="Website URL",
                            placeholder="https://www.example.com",
                            info="Your institution's primary website URL"
                        )
                        
                        inst_contact_email = gr.Textbox(
                            label="Contact Email",
                            placeholder="contact@example.com",
                            info="Primary contact email for your institution"
                        )
                        
                        inst_timezone = gr.Textbox(
                            label="Timezone",
                            placeholder="UTC, America/New_York, Europe/London, etc.",
                            info="Institution's primary timezone (e.g., UTC, America/New_York)"
                        )
                        
                        # Save button and status
                        with gr.Row():
                            save_institution_btn = gr.Button(
                                "💾 Save Institution Details",
                                variant="primary",
                                size="lg"
                            )
                        
                        # Status message for institution config
                        inst_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            elem_classes=["institution-config-status"],
                            show_label=False,
                            visible=False
                        )
                    
                    # Information section
                    with gr.Accordion("ℹ️ Configuration Information", open=False):
                        gr.Markdown("""
                        ### About Institution Configuration
                        
                        This tab allows you to configure core institutional details that will be used throughout the AI Gate application. The settings are saved to `config/institution.yaml` and will override any default values.
                        
                        **Configuration Files:**
                        - `config/default.yaml` - Contains default template values
                        - `config/institution.yaml` - Contains your institution-specific overrides
                        
                        **Field Descriptions:**
                        - **Institution Name**: The official name displayed in the application
                        - **Description**: A brief description shown to users
                        - **Website URL**: Your institution's primary website
                        - **Contact Email**: Primary contact for support or inquiries
                        - **Timezone**: Used for displaying times and scheduling
                        
                        After saving, restart the AI Gate application for changes to take effect.
                        """)
            
            # Institution configuration components for reference
            institution_components = [inst_name, inst_description, inst_website, inst_contact_email, inst_timezone]
            
            # Set up event handlers
            
            # Load institution config on interface startup
            interface.load(
                fn=self.manager.load_institution_config,
                inputs=[],
                outputs=institution_components
            )
            
            # Save institution config button click
            save_institution_btn.click(
                fn=self._save_institution_with_status,
                inputs=institution_components,
                outputs=[inst_status]
            )
            
            # Set up knowledge base builder button click events
            button_components = [btn for btn, _ in buttons]
            for btn, process_type in buttons:
                btn.click(
                    fn=lambda pt=process_type: self.manager.run_process(
                        pt, status_box, console_box, *button_components
                    ),
                    inputs=[],
                    outputs=[status_box, console_box] + button_components,
                    show_progress=False
                )
        
        return interface
    
    def _save_institution_with_status(self, name: str, description: str, website: str, 
                                    contact_email: str, timezone: str) -> str:
        """
        Save institution config and handle status display.
        
        Returns:
            Status message with visibility update
        """
        try:
            # Save the configuration
            status_msg = self.manager.save_institution_config(
                name, description, website, contact_email, timezone
            )
            
            # Return update that makes status visible and sets the message
            return gr.update(value=status_msg, visible=True)
            
        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            self.manager.logger.error(f"Unexpected error saving institution config: {e}")
            return gr.update(value=error_msg, visible=True)
        
    def _create_instructions_section(self):
        """Create the instructions and information section."""
        with gr.Accordion("📋 Instructions & Information", open=False):
            gr.Markdown(f"""
            ## Process Descriptions
            
            {self._format_process_descriptions()}
            
            ## Usage Guidelines
            
            - **Sequential Processing**: Only one process can run at a time for system stability
            - **Real-time Feedback**: Console output streams live during execution
            - **Status Monitoring**: Color-coded status indicator shows current system state
            - **Error Handling**: Detailed error messages help troubleshoot issues
            
            ## Recommended Workflow
            
            1. **🌐 Build from Website**: Scrape and process content from target websites
            2. **📄 Build from Documents**: Process any uploaded documents or files
            3. **🔄 Merge Knowledge Base**: Combine all processed data into unified knowledge base
            4. **📖 Build Institutional Dictionary**: Generate categorized keyword dictionary for analysis
            
            ## Status Indicators
            
            | Color | Status | Description |
            |-------|--------|-------------|
            | 🔘 Gray | Ready | System ready for new operations |
            | 🟡 Yellow | In Progress | Process currently running |
            | 🟢 Green | Success | Process completed successfully |
            | 🔴 Red | Error | Process failed or encountered error |
            
            ## Troubleshooting
            
            - **Script Not Found**: Ensure all required Python scripts are in the same directory
            - **Permission Errors**: Check file permissions and Python environment
            - **Process Failures**: Review console output for detailed error information
            """)
            
    def _format_process_descriptions(self) -> str:
        """Format process descriptions for display."""
        descriptions = []
        for process_type in ProcessType:
            config = self.manager.PROCESS_CONFIGS[process_type]
            descriptions.append(f"- **{config.icon} {config.display_name}**: {config.description}")
        return "\n".join(descriptions)


def main():
    """Main function to launch the GUI application."""
    print("🚀 Starting AI Gate Knowledge Base Builder GUI...")
    
    # Initialize manager
    manager = KnowledgeBaseManager()
    
    # Log startup information
    manager.logger.info("GUI application starting...")
    manager.logger.info(f"Working directory: {os.getcwd()}")
    manager.logger.info(f"Script location: {__file__}")
    
    # Change to script directory if needed
    script_dir = Path(__file__).parent
    if script_dir != Path.cwd():
        os.chdir(script_dir)
        manager.logger.info(f"Changed working directory to: {os.getcwd()}")
    
    # Check for missing scripts
    missing_scripts = manager.get_missing_scripts()
    if missing_scripts:
        manager.logger.warning("Missing scripts detected:")
        for script in missing_scripts:
            manager.logger.warning(f"  - {script}")
        print("⚠️  Warning: Some required scripts are missing. Check logs for details.")
    
    # Create and launch interface
    interface_builder = GradioInterface(manager)
    interface = interface_builder.create_interface()
    
    print("🌐 Launching web interface...")
    print("📱 The GUI will open in your default web browser")
    print("🔗 If it doesn't open automatically, check the console for the local URL")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    
    try:
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True,
            quiet=False,
            favicon_path=None
        )
    except KeyboardInterrupt:
        manager.logger.info("Application stopped by user")
        print("\n👋 GUI application stopped")
    except Exception as e:
        manager.logger.error(f"Failed to launch interface: {e}")
        print(f"❌ Failed to launch GUI: {e}")


if __name__ == "__main__":
    main()