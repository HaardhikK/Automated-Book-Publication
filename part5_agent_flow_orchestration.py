import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentFlowOrchestrator:
    def __init__(self, base_url: str = None, chapter_numbers: List[int] = None):
        """
        Initialize the orchestrator
        
        Args:
            base_url: Base URL template for chapters (e.g., "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_{}")
            chapter_numbers: List of chapter numbers to process
        """
        self.base_url = base_url or "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_{}"
        self.chapter_numbers = chapter_numbers or [1]  # Default to chapter 1
        
        # Directories
        self.src_dir = Path("src") if Path("src").exists() else Path(".")
        self.logs_dir = self.src_dir / "logs"
        self.chapters_dir = self.src_dir / "chapters"
        
        # Create directories
        for directory in [self.logs_dir, self.chapters_dir]:
            directory.mkdir(exist_ok=True)
        
        # Flow tracking
        self.flow_id = str(uuid.uuid4())[:8]
        self.flow_log_file = self.logs_dir / f"flow_log_{self.flow_id}.json"
        self.flow_log = {
            "flow_id": self.flow_id,
            "started_at": datetime.now().isoformat(),
            "base_url": self.base_url,
            "chapters_to_process": self.chapter_numbers,
            "steps": [],
            "status": "running",
            "completed_chapters": [],
            "failed_chapters": []
        }
        
        # Module paths
        self.part1_path = self.src_dir / "part1_scraper.py"
        self.part2_path = self.src_dir / "part2_ai_writer.py"
        self.part3_path = self.src_dir / "part3_ai_reviewer.py"
        self.part4_path = self.src_dir / "part4_human_interface.py"
    
    def log_step(self, step_name: str, status: str, chapter_num: int = None, 
                 input_file: str = None, output_file: str = None, 
                 metadata: Dict = None, error: str = None):
        """Log a step in the flow"""
        step_entry = {
            "step_id": len(self.flow_log["steps"]) + 1,
            "step_name": step_name,
            "chapter_number": chapter_num,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "input_file": input_file,
            "output_file": output_file,
            "metadata": metadata or {},
            "error": error
        }
        
        self.flow_log["steps"].append(step_entry)
        self.save_flow_log()
        
        # Log to console
        if status == "success":
            logger.info(f"✅ {step_name} completed for chapter {chapter_num}")
        elif status == "failed":
            logger.error(f"❌ {step_name} failed for chapter {chapter_num}: {error}")
        else:
            logger.info(f"🔄 {step_name} {status} for chapter {chapter_num}")
    
    def save_flow_log(self):
        """Save the flow log to file"""
        with open(self.flow_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.flow_log, f, indent=2)
    
    async def run_scraper(self, chapter_num: int) -> Dict[str, Any]:
        """Run Part 1: Scraper"""
        step_name = "Web Scraping"
        self.log_step(step_name, "started", chapter_num)
        
        try:
            # Import and run scraper
            sys.path.append(str(self.src_dir))
            from part1_scraper import WikisourceScraper
            
            # Create scraper with chapter-specific URL
            chapter_url = self.base_url.format(chapter_num)
            scraper = WikisourceScraper(
                output_dir=str(self.chapters_dir),
                chapter_url=chapter_url,
                chapter_num=chapter_num
            )
            
            result = await scraper.scrape_chapter()
            
            if result["success"]:
                self.log_step(
                    step_name, "success", chapter_num,
                    output_file=result.get("text_file"),
                    metadata={
                        "content_length": result.get("content_length"),
                        "screenshot_file": result.get("screenshot_file"),
                        "url_scraped": chapter_url
                    }
                )
            else:
                self.log_step(step_name, "failed", chapter_num, error=result.get("error"))
            
            return result
        
        except Exception as e:
            error_msg = str(e)
            self.log_step(step_name, "failed", chapter_num, error=error_msg)
            return {"success": False, "error": error_msg}
    
    def run_ai_writer(self, chapter_num: int) -> Dict[str, Any]:
        """Run Part 2: AI Writer"""
        step_name = "AI Writing"
        self.log_step(step_name, "started", chapter_num)
        
        try:
            # Import and run AI writer
            sys.path.append(str(self.src_dir))
            from part2_ai_writer import AIWriter
            
            writer = AIWriter(chapter_num=chapter_num, chapters_dir=str(self.chapters_dir))
            result = writer.rewrite_chapter()
            
            if result["success"]:
                self.log_step(
                    step_name, "success", chapter_num,
                    input_file=result.get("input_file"),
                    output_file=result.get("output_file"),
                    metadata={
                        "original_length": result.get("original_length"),
                        "ai_length": result.get("ai_length"),
                        "tokens_used": result.get("tokens_used")
                    }
                )
            else:
                self.log_step(step_name, "failed", chapter_num, error=result.get("error"))
            
            return result
        
        except Exception as e:
            error_msg = str(e)
            self.log_step(step_name, "failed", chapter_num, error=error_msg)
            return {"success": False, "error": error_msg}
    
    def run_ai_reviewer(self, chapter_num: int) -> Dict[str, Any]:
        """Run Part 3: AI Reviewer"""
        step_name = "AI Review"
        self.log_step(step_name, "started", chapter_num)
        
        try:
            # Import and run AI reviewer
            sys.path.append(str(self.src_dir))
            from part3_ai_reviewer import AIReviewer
            
            reviewer = AIReviewer(chapter_num=chapter_num, chapters_dir=str(self.chapters_dir))
            result = reviewer.review_chapter()
            
            if result["success"]:
                self.log_step(
                    step_name, "success", chapter_num,
                    input_file=result.get("input_file"),
                    output_file=result.get("output_file"),
                    metadata={
                        "ai_length": result.get("ai_length"),
                        "reviewed_length": result.get("reviewed_length"),
                        "tokens_used": result.get("tokens_used")
                    }
                )
            else:
                self.log_step(step_name, "failed", chapter_num, error=result.get("error"))
            
            return result
        
        except Exception as e:
            error_msg = str(e)
            self.log_step(step_name, "failed", chapter_num, error=error_msg)
            return {"success": False, "error": error_msg}
    
    async def process_chapter(self, chapter_num: int) -> Dict[str, Any]:
        """Process a single chapter through the entire pipeline"""
        logger.info(f"📖 Starting processing for Chapter {chapter_num}")
        
        chapter_results = {
            "chapter_number": chapter_num,
            "scraper": None,
            "ai_writer": None,
            "ai_reviewer": None,
            "overall_success": False
        }
        
        try:
            # Step 1: Web Scraping
            scraper_result = await self.run_scraper(chapter_num)
            chapter_results["scraper"] = scraper_result
            
            if not scraper_result["success"]:
                raise Exception(f"Scraping failed: {scraper_result.get('error')}")
            
            # Step 2: AI Writing
            writer_result = self.run_ai_writer(chapter_num)
            chapter_results["ai_writer"] = writer_result
            
            if not writer_result["success"]:
                raise Exception(f"AI Writing failed: {writer_result.get('error')}")
            
            # Step 3: AI Review
            reviewer_result = self.run_ai_reviewer(chapter_num)
            chapter_results["ai_reviewer"] = reviewer_result
            
            if not reviewer_result["success"]:
                raise Exception(f"AI Review failed: {reviewer_result.get('error')}")
            
            chapter_results["overall_success"] = True
            self.flow_log["completed_chapters"].append(chapter_num)
            logger.info(f"✅ Chapter {chapter_num} processing completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Chapter {chapter_num} processing failed: {e}")
            self.flow_log["failed_chapters"].append({"chapter": chapter_num, "error": str(e)})
            chapter_results["error"] = str(e)
        
        return chapter_results
    
    async def run_flow(self) -> Dict[str, Any]:
        """Run the complete flow for all chapters"""
        logger.info(f"🚀 Starting Agent Flow Orchestration (ID: {self.flow_id})")
        logger.info(f"📚 Processing chapters: {self.chapter_numbers}")
        logger.info(f"🔗 Base URL: {self.base_url}")
        
        all_results = []
        
        # Process each chapter
        for chapter_num in self.chapter_numbers:
            chapter_result = await self.process_chapter(chapter_num)
            all_results.append(chapter_result)
        
        # Final flow status
        successful_chapters = len(self.flow_log["completed_chapters"])
        failed_chapters = len(self.flow_log["failed_chapters"])
        
        self.flow_log["completed_at"] = datetime.now().isoformat()
        self.flow_log["status"] = "completed"
        self.flow_log["summary"] = {
            "total_chapters": len(self.chapter_numbers),
            "successful_chapters": successful_chapters,
            "failed_chapters": failed_chapters,
            "success_rate": f"{(successful_chapters/len(self.chapter_numbers)*100):.1f}%"
        }
        
        self.save_flow_log()
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 AGENT FLOW ORCHESTRATION COMPLETED")
        print("="*60)
        print(f"Flow ID: {self.flow_id}")
        print(f"✅ Successful: {successful_chapters}/{len(self.chapter_numbers)} chapters")
        print(f"❌ Failed: {failed_chapters}/{len(self.chapter_numbers)} chapters")
        print(f"📊 Success Rate: {self.flow_log['summary']['success_rate']}")
        print(f"📋 Flow Log: {self.flow_log_file}")
        
        if successful_chapters > 0:
            print(f"\n🎉 Ready for human review! Run Part 4:")
            print(f"   cd {self.src_dir}")
            print(f"   streamlit run part4_human_interface.py")
        
        return {
            "flow_id": self.flow_id,
            "results": all_results,
            "flow_log_file": str(self.flow_log_file),
            "summary": self.flow_log["summary"]
        }

def main():
    """Main execution function with configuration options"""
    
    # Configuration - Modify these as needed
    CONFIG = {
        "base_url": "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_{}",
        "chapters": [3],  # List of chapters to process
        # "chapters": [1],  # For single chapter
    }
    
    print("📚 Agent Flow Orchestration - Book Processing Pipeline")
    print(f"🔗 Base URL: {CONFIG['base_url']}")
    print(f"📖 Chapters to process: {CONFIG['chapters']}")
    print("\nStarting automated pipeline: Scraping → AI Writing → AI Review")
    print("Note: Part 4 (Human Interface) requires manual Streamlit execution")
    
    try:
        # Create orchestrator
        orchestrator = AgentFlowOrchestrator(
            base_url=CONFIG["base_url"],
            chapter_numbers=CONFIG["chapters"]
        )
        
        # Run the flow
        result = asyncio.run(orchestrator.run_flow())
        
        return result
    
    except KeyboardInterrupt:
        print("\n⚠️ Flow interrupted by user")
        return {"success": False, "error": "User interruption"}
    except Exception as e:
        print(f"\n❌ Flow failed: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = main()