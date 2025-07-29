#!/usr/bin/env python3
"""
Merge Knowledge Script - AI Gate Ingestion Pipeline

This script performs the final data consolidation step in the AI Gate ingestion pipeline.
It merges the contents of two high-quality, consistently structured JSON files:
- data/pages.json (from the web scraper)
- data/ins_info.json (from the document processor)

The output is a single, unified knowledge base file: data/knowledge_base.json

Author: AI Gate Pipeline
Version: 1.0
"""

import json
import os
import sys

# --- Path Correction ---
# Add the project's root directory to the Python path.
# This allows this script to be run from anywhere and still import modules 
# from the 'modules' directory correctly, just as if it were run from the root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from datetime import datetime


def main():
    """
    Main function to execute the knowledge base merging workflow.
    """
    # Define file paths
    WEB_DATA_PATH = "data/pages.json"
    DOCUMENT_DATA_PATH = "data/ins_info.json"
    OUTPUT_PATH = "data/knowledge_base.json"
    
    print("Starting knowledge base merge process...")
    print(f"Web data source: {WEB_DATA_PATH}")
    print(f"Document data source: {DOCUMENT_DATA_PATH}")
    print(f"Output destination: {OUTPUT_PATH}")
    print("-" * 50)
    
    # Load web scraper data
    try:
        with open(WEB_DATA_PATH, 'r', encoding='utf-8') as f:
            web_data = json.load(f)
        print(f"✓ Successfully loaded web data from {WEB_DATA_PATH}")
    except FileNotFoundError:
        print(f"✗ Error: Could not find web data file at {WEB_DATA_PATH}")
        print("Please ensure the website scraper has completed successfully.")
        return
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON format in {WEB_DATA_PATH}: {e}")
        return
    
    # Load document processor data
    try:
        with open(DOCUMENT_DATA_PATH, 'r', encoding='utf-8') as f:
            document_data = json.load(f)
        print(f"✓ Successfully loaded document data from {DOCUMENT_DATA_PATH}")
    except FileNotFoundError:
        print(f"✗ Error: Could not find document data file at {DOCUMENT_DATA_PATH}")
        print("Please ensure the document processor has completed successfully.")
        return
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON format in {DOCUMENT_DATA_PATH}: {e}")
        return
    
    # Extract pages lists from both sources
    web_data_pages = web_data.get('pages', [])
    document_data_pages = document_data.get('pages', [])
    
    print(f"✓ Web data contains {len(web_data_pages)} pages")
    print(f"✓ Document data contains {len(document_data_pages)} pages")
    
    # Combine the lists
    combined_pages = web_data_pages + document_data_pages
    total_items = len(combined_pages)
    
    print(f"✓ Combined total: {total_items} pages")
    
    # Extract component versions from source metadata
    component_versions = {}
    
    # Get website scraper version
    web_metadata = web_data.get('metadata', {})
    if 'version' in web_metadata:
        component_versions['website_scraper'] = web_metadata['version']
    
    # Get document processor version
    doc_metadata = document_data.get('metadata', {})
    if 'version' in doc_metadata:
        component_versions['document_processor'] = doc_metadata['version']
    
    # Generate unified metadata
    unified_metadata = {
        "processed_at": datetime.now().isoformat(),
        "total_items": total_items,
        "source_files": [WEB_DATA_PATH, DOCUMENT_DATA_PATH],
        "component_versions": component_versions,
        "merge_script_version": "1.0",
        "source_breakdown": {
            "web_pages": len(web_data_pages),
            "document_pages": len(document_data_pages)
        }
    }
    
    # Construct final knowledge base structure
    knowledge_base = {
        "pages": combined_pages,
        "metadata": unified_metadata
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Save the unified knowledge base
    try:
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        print(f"✓ Successfully saved unified knowledge base to {OUTPUT_PATH}")
    except Exception as e:
        print(f"✗ Error saving knowledge base: {e}")
        return
    
    # Print completion summary
    print("-" * 50)
    print("🎉 Knowledge base merge completed successfully!")
    print(f"📊 Total items in unified knowledge base: {total_items}")
    print(f"   • Web pages: {len(web_data_pages)}")
    print(f"   • Document pages: {len(document_data_pages)}")
    print(f"📁 Output file: {OUTPUT_PATH}")
    print("📋 Component versions:")
    for component, version in component_versions.items():
        print(f"   • {component}: {version}")
    print("\n✅ The unified knowledge base is now ready for the WebsiteResearcher module.")


if __name__ == "__main__":
    main()
