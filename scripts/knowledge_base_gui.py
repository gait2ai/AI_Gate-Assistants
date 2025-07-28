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
from typing import Iterator, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time


class ProcessType(Enum):
    """Enumeration of available process types."""
    WEBSITE_SCRAPER = "website_scraper.py"
    DOCUMENT_PROCESSOR = "document_processor.py"
    MERGE_KNOWLEDGE = "merge_knowledge.py"


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
            
            # Set up button click events
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
