import argparse
import sys
import os
import shutil
import json
import logging
from typing import List

from dotenv import load_dotenv

from webis.core.pipeline import Pipeline
from webis.core.schema import WebisDocument, DocumentType, DocumentMetadata, PipelineContext
from webis.core.plugin import get_default_registry
from webis.core.agent.crawler_agent import CrawlerAgent

# Import plugins to register them
import webis.plugins.sources
import webis.plugins.processors
import webis.plugins.extractors
import webis.plugins.outputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("webis.cli")

def main():
    # 0. Auto-Init Configuration (Onboarding)
    # If .env is missing but .env.example exists, copy it and warn user.
    env_path = ".env"
    example_path = ".env.example"
    
    if not os.path.exists(env_path):
        if os.path.exists(example_path):
            print("🚀 Welcome to Webis! Initializing configuration...")
            try:
                shutil.copy(example_path, env_path)
                print(f"✅ Created {env_path} from template.")
                print("⚠️  IMPORTANT: Please edit .env and add your API keys (WENDALOG_API_KEY recommended) to proceed.")
                print("   Opening .env for you..." if sys.platform == "darwin" else "")
                
                # Optional: Open file automatically on Mac
                # if sys.platform == "darwin":
                #     os.system(f"open {env_path}")
                    
            except Exception as e:
                logger.warning(f"Failed to auto-create .env: {e}")
        else:
            logger.warning("No .env found and no .env.example template available.")

    # Load environment variables from .env
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Webis CLI v2")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run command (End-to-End)
    run_parser = subparsers.add_parser("run", help="Run end-to-end pipeline")
    run_parser.add_argument("task", help="Natural language task description")
    run_parser.add_argument("--limit", type=int, default=5, help="Max results")
    run_parser.add_argument("--output", "-o", help="Output directory")

    # crawl command
    crawl_parser = subparsers.add_parser("crawl", help="Crawl data for a task")
    crawl_parser.add_argument("task", help="Task or Query")
    crawl_parser.add_argument("--limit", type=int, default=5)
    crawl_parser.add_argument("--output", "-o", help="Output file (JSON)")

    # extract command
    extract_parser = subparsers.add_parser("extract", help="Extract structure from files")
    extract_parser.add_argument("files", nargs="+", help="Files to extract from")
    extract_parser.add_argument("--task", help="Extraction goal/task", default="Extract main information")
    extract_parser.add_argument("--schema", help="Path to JSON schema")
    extract_parser.add_argument("--output", "-o", help="Output file")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args.task, args.limit, args.output)
    elif args.command == "crawl":
        cmd_crawl(args.task, args.limit, args.output)
    elif args.command == "extract":
        cmd_extract(args.files, args.task, args.schema, args.output)
    else:
        parser.print_help()

def cmd_run(task: str, limit: int, output_dir: str = None):
    logger.info(f"Starting Webis Pipeline for: {task}")
    
    # 1. Source (Crawler Agent)
    logger.info("Phase 1: Sourcing...")
    agent = CrawlerAgent()
    docs = agent.run(task, limit=limit)
    
    if not docs:
        logger.error("No documents found.")
        return

    logger.info(f"Fetched {len(docs)} documents.")

    # 2. Pipeline (Processing + Extraction)
    logger.info("Phase 2: Processing & Extraction...")
    
    pipeline = Pipeline()
    
    # Add processors to clean/parse content
    # Note: Pipeline.run() typically runs source stages too. 
    # But here we inject docs directly. 
    # We will manually construct the pipeline flow or modify Pipeline to accept docs.
    # For CLI simplicity, we can use Pipeline's processor runner if we expose it, 
    # or just use plugins directly.
    # Let's use Pipeline proper by configuring it without sources, 
    # but Pipeline.run() doesn't take docs.
    # Refactoring: Let's use the registry directly for flexibility in CLI script.
    
    registry = get_default_registry()
    
    # Auto-detect processors needed?
    # For v2 default pipeline:
    # - HTML Fetcher (if content empty)
    # - HTML Cleaner
    # - PDF/Doc Parser
    
    processed_docs = []
    
    # Helper to run processors
    html_fetcher = registry.get_processor("html_fetcher")
    html_cleaner = registry.get_processor("html_cleaner")
    pdf_parser = registry.get_processor("pdf_extractor")
    doc_parser = registry.get_processor("document_parser")

    context = PipelineContext(task=task, output_dir=output_dir)

    for doc in docs:
        # Fetch if needed
        if html_fetcher:
            doc = html_fetcher.process(doc, context) or doc

        # Parse/Clean based on type
        if doc.doc_type == DocumentType.PDF and pdf_parser:
             doc = pdf_parser.process(doc, context) or doc
        elif doc.doc_type in (DocumentType.HTML, DocumentType.UNKNOWN) and html_cleaner:
             doc = html_cleaner.process(doc, context) or doc
        
        # Try doc parser for others
        if doc_parser:
             doc = doc_parser.process(doc, context) or doc
             
        processed_docs.append(doc)

    # 3. Extraction
    logger.info("Phase 3: Extraction...")
    extractor = registry.get_extractor("llm_extractor")
    if extractor:
        result = extractor.extract(processed_docs, context)
        
        print("\n=== Extraction Result ===")
        print(json.dumps(result.data, indent=2, ensure_ascii=False))
        
    # 4. Reporting (Phase 4)
    logger.info("Phase 4: Reporting...")
    
    # Determine output directory (Auto-save)
    if not output_dir:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", timestamp)
        
    os.makedirs(output_dir, exist_ok=True)

    if extractor and 'result' in locals():
        # Save JSON
        json_path = os.path.join(output_dir, "result.json")
        with open(json_path, "w") as f:
            f.write(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
        
        # Generate HTML Report
        html_plugin = registry.get_output("html_report")
        if html_plugin:
            html_plugin.save(result, context=context, output_dir=output_dir, documents=processed_docs)
            print(f"\n✨ Report generated: {os.path.join(output_dir, 'report.html')}")
            print(f"📁 JSON saved to:   {json_path}")
        else:
             logger.warning("HtmlReportPlugin not found.")
    else:
        logger.error("No result to save.")

def cmd_crawl(task: str, limit: int, output_file: str = None):
    agent = CrawlerAgent()
    docs = agent.run(task, limit=limit)
    
    data = [doc.model_dump(mode="json") for doc in docs]
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
    if output_file:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(docs)} docs to {output_file}")

def cmd_extract(files: List[str], task: str, schema_path: str = None, output_file: str = None):
    registry = get_default_registry()
    extractor = registry.get_extractor("llm_extractor")
    
    config = {}
    if schema_path:
        with open(schema_path) as f:
            config["schema"] = json.load(f)
            
    # Re-init extractor with config if needed, or pass in kwargs
    # Plugins act as singletons in registry usually, but `extract` takes kwargs.
    # LLMExtractorPlugin uses `self.schema` from init. 
    # We should probably instantiate a new one or pass schema in kwargs (if plugin supports it).
    # Current implementation supports `self.schema`. Let's assume kwargs support or re-register.
    # For now, simplistic approach:
    
    if schema_path:
         # Create a temporary instance with config
         from webis.plugins.extractors.llm_extractor_plugin import LLMExtractorPlugin
         extractor = LLMExtractorPlugin(config={"schema": config["schema"]})

    if not extractor:
        logger.error("LLM Extractor not found.")
        return

    docs = []
    # Load files as docs
    doc_parser = registry.get_processor("document_parser")
    pdf_parser = registry.get_processor("pdf_extractor")
    context = PipelineContext(task=task)
    
    for fp in files:
        doc = WebisDocument(content=fp, doc_type=DocumentType.UNKNOWN, meta=DocumentMetadata(title=os.path.basename(fp), custom={"file_path": fp}))
        
        # Parse
        if fp.lower().endswith(".pdf") and pdf_parser:
             doc.doc_type = DocumentType.PDF
             doc = pdf_parser.process(doc, context)
        elif doc_parser:
             doc = doc_parser.process(doc, context)
             
        if doc:
            docs.append(doc)
            
    result = extractor.extract(docs, context=context)
    print(json.dumps(result.data, indent=2, ensure_ascii=False))
    
    if output_file:
        with open(output_file, "w") as f:
            f.write(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
