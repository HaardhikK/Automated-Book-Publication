#!/usr/bin/env python3
"""
Part 2: AI Writer
Reads chapter text and rewrites it using Groq LLM API
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIWriter:
    def __init__(self, chapter_num: int = 1, chapters_dir: str = "chapters"):
        self.chapter_num = chapter_num
        
        # Folder structure paths
        self.chapters_dir = Path(chapters_dir)
        self.logs_dir = Path("logs")
        self.prompts_dir = Path("prompts")
        
        # Create directories if they don't exist
        for directory in [self.chapters_dir, self.logs_dir, self.prompts_dir]:
            directory.mkdir(exist_ok=True)
        
        # Groq API configuration
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-70b-8192"  # Free tier model
        
        # File paths - now dynamic based on chapter number
        self.input_file = self.chapters_dir / f"chapter_{self.chapter_num}.txt"
        self.output_file = self.chapters_dir / f"chapter_{self.chapter_num}_ai_v1.txt"
        self.prompt_file = self.prompts_dir / "writer_prompt.txt"
        self.meta_file = self.logs_dir / f"chapter_{self.chapter_num}_ai_v1_meta.json"
    
    def create_prompt_template(self) -> str:
        """Create and save the writer prompt template"""
        prompt_template = """You are a professional book editor and rewriter. Your task is to rewrite the following chapter text to make it more engaging, modern, and vivid while preserving the original story, characters, and plot.

INSTRUCTIONS:
- Maintain the original story structure and plot points
- Keep all character names and relationships intact
- Improve sentence flow and readability
- Use more vivid and descriptive language
- Make dialogue more natural and engaging
- Preserve the historical/literary context
- Ensure smooth transitions between paragraphs
- Maintain the original chapter length (don't drastically expand or reduce)

IMPORTANT: Return ONLY the rewritten chapter content. Do not include any commentary, explanations, or notes about your changes. Start directly with the title and story content.

ORIGINAL TEXT:
{original_text}

REWRITTEN VERSION:"""
        
        # Save prompt template
        with open(self.prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt_template)
        
        logger.info(f"Prompt template saved to {self.prompt_file}")
        return prompt_template
    
    def read_chapter(self) -> str:
        """Read the original chapter text"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                raise ValueError(f"Chapter {self.chapter_num} file is empty")
            
            logger.info(f"Read chapter {self.chapter_num} from {self.input_file} ({len(content)} characters)")
            return content
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Chapter {self.chapter_num} file not found: {self.input_file}")
        except Exception as e:
            raise Exception(f"Error reading chapter {self.chapter_num}: {e}")
    
    def call_groq_api(self, prompt: str) -> Dict[str, Any]:
        """Make API call to Groq"""
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
            "temperature": 0.7,
            "max_tokens": 4000,
            "top_p": 1,
            "stream": False
        }
        
        try:
            logger.info(f"Sending chapter {self.chapter_num} to Groq API...")
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Received response from Groq API for chapter {self.chapter_num}")
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for chapter {self.chapter_num}: {e}")
            raise Exception(f"Groq API error: {e}")
    
    def extract_ai_content(self, api_response: Dict[str, Any]) -> str:
        """Extract the rewritten content from API response"""
        try:
            content = api_response["choices"][0]["message"]["content"]
            
            # Clean up the content to remove AI commentary
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip lines that are AI commentary/explanations
                if any(phrase in line.lower() for phrase in [
                    'here is the rewritten', 'i made significant changes', 
                    'i used more engaging', 'i preserved the original',
                    'rewritten version', 'here\'s the rewritten'
                ]):
                    continue
                
                # Keep the line if it's not empty
                if line:
                    cleaned_lines.append(line)
            
            return '\n'.join(cleaned_lines).strip()
        except (KeyError, IndexError) as e:
            raise Exception(f"Failed to extract content from API response: {e}")
    
    def save_ai_version(self, content: str) -> None:
        """Save the AI-rewritten version"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"AI-rewritten chapter {self.chapter_num} saved to {self.output_file}")
    
    def save_metadata(self, original_text: str, ai_text: str, api_response: Dict[str, Any]) -> None:
        """Save metadata about the AI generation process"""
        metadata = {
            "chapter_number": self.chapter_num,
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "api_provider": "Groq",
            "input_file": str(self.input_file),
            "output_file": str(self.output_file),
            "prompt_file": str(self.prompt_file),
            "original_length": len(original_text),
            "ai_length": len(ai_text),
            "compression_ratio": len(ai_text) / len(original_text) if original_text else 0,
            "api_usage": {
                "prompt_tokens": api_response.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": api_response.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": api_response.get("usage", {}).get("total_tokens", 0)
            },
            "model_settings": {
                "temperature": 0.7,
                "max_tokens": 4000,
                "top_p": 1
            }
        }
        
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata for chapter {self.chapter_num} saved to {self.meta_file}")
    
    def rewrite_chapter(self) -> Dict[str, Any]:
        """Main method to rewrite the chapter"""
        try:
            # Read original chapter
            original_text = self.read_chapter()
            
            # Create/load prompt template
            prompt_template = self.create_prompt_template()
            
            # Format prompt with original text
            formatted_prompt = prompt_template.format(original_text=original_text)
            
            # Call Groq API
            api_response = self.call_groq_api(formatted_prompt)
            
            # Extract AI-generated content
            ai_content = self.extract_ai_content(api_response)
            
            # Save AI version
            self.save_ai_version(ai_content)
            
            # Save metadata
            self.save_metadata(original_text, ai_content, api_response)
            
            return {
                "success": True,
                "chapter_number": self.chapter_num,
                "input_file": str(self.input_file),
                "output_file": str(self.output_file),
                "original_length": len(original_text),
                "ai_length": len(ai_content),
                "tokens_used": api_response.get("usage", {}).get("total_tokens", 0)
            }
        
        except Exception as e:
            logger.error(f"Error during chapter {self.chapter_num} rewriting: {e}")
            return {
                "success": False,
                "chapter_number": self.chapter_num,
                "error": str(e)
            }


def main(chapter_num: int = 1, chapters_dir: str = "chapters"):
    """Main execution function"""
    logger.info(f"Starting AI Writer (Part 2) for Chapter {chapter_num}...")
    
    
    
    try:
        # Initialize AI Writer
        writer = AIWriter(chapter_num=chapter_num, chapters_dir=chapters_dir)
        
        # Rewrite chapter
        result = writer.rewrite_chapter()
        
        if result["success"]:
            logger.info(f"✅ Chapter {chapter_num} rewriting completed successfully!")
            logger.info(f"📄 Input: {result['input_file']}")
            logger.info(f"📄 Output: {result['output_file']}")
            logger.info(f"📊 Original: {result['original_length']} chars")
            logger.info(f"📊 AI Version: {result['ai_length']} chars")
            logger.info(f"🎯 Tokens used: {result['tokens_used']}")
        else:
            logger.error(f"❌ Chapter {chapter_num} rewriting failed: {result['error']}")
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI Writer for chapter {chapter_num}: {e}")
        return {"success": False, "chapter_number": chapter_num, "error": str(e)}

if __name__ == "__main__":
    import argparse
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="AI Writer - Rewrite chapters using LLM")
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
    
    # Run the AI writer
    result = main(chapter_num=args.chapter, chapters_dir=args.chapters_dir)
    
    if result["success"]:
        print("\n" + "="*50)
        print(f"PART 2 COMPLETED SUCCESSFULLY - CHAPTER {result['chapter_number']}")
        print("="*50)
        print(f"✅ AI-rewritten chapter saved!")
        print(f"📊 {result['original_length']} → {result['ai_length']} characters")
        print(f"🎯 API tokens used: {result['tokens_used']}")
    else:
        print(f"\n❌ Failed: {result['error']}")