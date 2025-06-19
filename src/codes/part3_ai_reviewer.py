import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIReviewer:
    def __init__(self, chapter_num: int = 1, chapters_dir: str = "chapters"):
        self.chapter_num = chapter_num
        
        # Folder paths
        self.chapters_dir = Path(chapters_dir)
        self.logs_dir = Path("logs")
        self.prompts_dir = Path("prompts")
        
        # Create directories
        for directory in [self.chapters_dir, self.logs_dir, self.prompts_dir]:
            directory.mkdir(exist_ok=True)
        
        # Groq API configuration
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "deepseek-r1-distill-llama-70b"
        
        # File paths - now dynamic based on chapter number
        self.input_file = self.chapters_dir / f"chapter_{self.chapter_num}_ai_v1.txt"
        self.output_file = self.chapters_dir / f"chapter_{self.chapter_num}_reviewed_v1.txt"
        self.prompt_file = self.prompts_dir / "reviewer_prompt.txt"
        self.meta_file = self.logs_dir / f"chapter_{self.chapter_num}_reviewed_v1_meta.json"
    
    def create_reviewer_prompt(self) -> str:
        """Create and save the reviewer prompt template"""
        prompt_template = """You are an expert book editor with decades of experience in refining literary works. Your task is to polish and improve the following chapter text while maintaining its essence and style.

FOCUS AREAS:
- Enhance grammar and punctuation accuracy
- Improve sentence flow and rhythm
- Ensure paragraph transitions are smooth
- Maintain consistent tone and voice
- Preserve all plot points, characters, and dialogue
- Keep the original meaning and story structure intact
- Ensure professional publishing quality

IMPORTANT: Return ONLY the polished chapter content. Do not include commentary, explanations, or editing notes.

TEXT TO REVIEW:
{ai_text}

POLISHED VERSION:"""
        
        # Save prompt template
        with open(self.prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt_template)
        
        logger.info(f"Reviewer prompt saved to {self.prompt_file}")
        return prompt_template
    
    def read_ai_chapter(self) -> str:
        """Read the AI-written chapter"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                raise ValueError(f"AI chapter {self.chapter_num} file is empty")
            
            logger.info(f"Read AI chapter {self.chapter_num} from {self.input_file} ({len(content)} characters)")
            return content
        
        except FileNotFoundError:
            raise FileNotFoundError(f"AI chapter {self.chapter_num} file not found: {self.input_file}")
        except Exception as e:
            raise Exception(f"Error reading AI chapter {self.chapter_num}: {e}")
    
    def call_groq_api(self, prompt: str) -> Dict[str, Any]:
        """Make API call to Groq with DeepSeek model"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": self.model,
            "temperature": 0.3,  # Lower temp for editing consistency
            "max_tokens": 4000,
            "top_p": 0.95,
            "stream": False
        }
        
        try:
            logger.info(f"Sending chapter {self.chapter_num} to Groq API (DeepSeek model)...")
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Received response from DeepSeek model for chapter {self.chapter_num}")
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for chapter {self.chapter_num}: {e}")
            raise Exception(f"Groq API error: {e}")
    
    def extract_reviewed_content(self, api_response: Dict[str, Any]) -> str:
        """Extract the reviewed content from API response"""
        try:
            content = api_response["choices"][0]["message"]["content"]
            
            # Remove DeepSeek thinking blocks
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            
            # Clean up any reviewer commentary
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip reviewer commentary lines
                if any(phrase in line.lower() for phrase in [
                    'here is the polished', 'i have improved', 
                    'the following changes', 'polished version',
                    'here\'s the refined', 'i have refined'
                ]):
                    continue
                
                if line:
                    cleaned_lines.append(line)
            
            return '\n'.join(cleaned_lines).strip()
        except (KeyError, IndexError) as e:
            raise Exception(f"Failed to extract content from API response: {e}")
    
    def save_reviewed_version(self, content: str) -> None:
        """Save the reviewed version"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Reviewed chapter {self.chapter_num} saved to {self.output_file}")
    
    def save_metadata(self, ai_text: str, reviewed_text: str, api_response: Dict[str, Any]) -> None:
        """Save metadata about the review process"""
        metadata = {
            "chapter_number": self.chapter_num,
            "model": self.model,
            "api_provider": "Groq",
            "source_file": str(self.input_file),
            "output_file": str(self.output_file),
            "prompt_file": str(self.prompt_file),
            "timestamp": datetime.now().isoformat(),
            "prompt_summary": "Improve flow, grammar, tone without changing meaning",
            "ai_length": len(ai_text),
            "reviewed_length": len(reviewed_text),
            "length_change": len(reviewed_text) - len(ai_text),
            "api_usage": {
                "prompt_tokens": api_response.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": api_response.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": api_response.get("usage", {}).get("total_tokens", 0)
            },
            "model_settings": {
                "temperature": 0.3,
                "max_tokens": 4000,
                "top_p": 0.95
            }
        }
        
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata for chapter {self.chapter_num} saved to {self.meta_file}")
    
    def review_chapter(self) -> Dict[str, Any]:
        """Main method to review the AI-written chapter"""
        try:
            # Read AI chapter
            ai_text = self.read_ai_chapter()
            
            # Create reviewer prompt
            prompt_template = self.create_reviewer_prompt()
            
            # Format prompt with AI text
            formatted_prompt = prompt_template.format(ai_text=ai_text)
            
            # Call Groq API with DeepSeek
            api_response = self.call_groq_api(formatted_prompt)
            
            # Extract reviewed content
            reviewed_content = self.extract_reviewed_content(api_response)
            
            # Save reviewed version
            self.save_reviewed_version(reviewed_content)
            
            # Save metadata
            self.save_metadata(ai_text, reviewed_content, api_response)
            
            return {
                "success": True,
                "chapter_number": self.chapter_num,
                "input_file": str(self.input_file),
                "output_file": str(self.output_file),
                "ai_length": len(ai_text),
                "reviewed_length": len(reviewed_content),
                "tokens_used": api_response.get("usage", {}).get("total_tokens", 0)
            }
        
        except Exception as e:
            logger.error(f"Error during chapter {self.chapter_num} review: {e}")
            return {
                "success": False,
                "chapter_number": self.chapter_num,
                "error": str(e)
            }

def main(chapter_num: int = 1, chapters_dir: str = "chapters"):
    """Main execution function"""
    logger.info(f"Starting AI Reviewer (Part 3) for Chapter {chapter_num}...")
    
    try:
        # Initialize AI Reviewer
        reviewer = AIReviewer(chapter_num=chapter_num, chapters_dir=chapters_dir)
        
        # Review chapter
        result = reviewer.review_chapter()
        
        if result["success"]:
            logger.info(f"✅ Chapter {chapter_num} review completed successfully!")
            logger.info(f"📄 Input: {result['input_file']}")
            logger.info(f"📄 Output: {result['output_file']}")
            logger.info(f"📊 AI Version: {result['ai_length']} chars")
            logger.info(f"📊 Reviewed: {result['reviewed_length']} chars")
            logger.info(f"🎯 Tokens used: {result['tokens_used']}")
        else:
            logger.error(f"❌ Chapter {chapter_num} review failed: {result['error']}")
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI Reviewer for chapter {chapter_num}: {e}")
        return {"success": False, "chapter_number": chapter_num, "error": str(e)}

if __name__ == "__main__":
    import argparse
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="AI Reviewer - Review and polish AI-written chapters")
    parser.add_argument("--chapter", type=int, default=1, help="Chapter number to process")
    parser.add_argument("--chapters_dir", default="chapters", help="Directory containing chapters")
    
    args = parser.parse_args()
    
    # Install dependencies if needed
    try:
        import requests
        from dotenv import load_dotenv
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install requests python-dotenv")
        exit(1)
    
    # Run the AI reviewer
    result = main(chapter_num=args.chapter, chapters_dir=args.chapters_dir)
    
    if result["success"]:
        print("\n" + "="*50)
        print(f"PART 3 COMPLETED SUCCESSFULLY - CHAPTER {result['chapter_number']}")
        print("="*50)
        print(f"✅ Reviewed chapter saved!")
        print(f"📊 {result['ai_length']} → {result['reviewed_length']} characters")
        print(f"🎯 API tokens used: {result['tokens_used']}")
    else:
        print(f"\n❌ Failed: {result['error']}")