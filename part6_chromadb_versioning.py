import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBVersionManager:
    def __init__(self, persist_directory: str = "chromadb_data", 
                 model_name: str = "all-MiniLM-L6-v2",
                 collection_name: str = "chapter_versions"):
        
        # Base path to this file's directory
        base_path = Path(__file__).resolve().parent
        self.persist_directory = base_path / persist_directory
        self.persist_directory.mkdir(exist_ok=True)
        
        self.feedback_file = base_path / "feedback.json"
        self.chapters_dir = base_path / "chapters"
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Chapter versions with metadata"}
        )
        
        # Load feedback data
        self.feedback_data = self.load_feedback()
        
        print(f"ChromaDB initialized. Collection: {collection_name}")
        print(f"Current documents in collection: {self.collection.count()}")
    
    def load_feedback(self) -> Dict[str, float]:
        """Load feedback scores from file"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load feedback file: {e}")
        return {}
    
    def save_feedback(self):
        """Save feedback scores to file"""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save feedback file: {e}")
    
    def parse_filename(self, filename: str) -> Dict[str, Any]:
        """Parse filename to extract chapter info"""
        # Pattern: chapter_{num}_{type}_v{version}.txt
        pattern = r'chapter_(\d+)_([^_]+)_v(\d+)\.txt'
        match = re.match(pattern, filename)
        
        if match:
            return {
                "chapter": int(match.group(1)),
                "type": match.group(2),
                "version": int(match.group(3))
            }
        
        # Fallback for chapter_1.txt format
        pattern2 = r'chapter_(\d+)\.txt'
        match2 = re.match(pattern2, filename)
        if match2:
            return {
                "chapter": int(match2.group(1)),
                "type": "original",
                "version": 1
            }
        
        return None
    
    def index_single_file(self, file_path: Path) -> bool:
        """Index a single file"""
        try:
            # Parse filename
            parsed = self.parse_filename(file_path.name)
            if not parsed:
                print(f"Could not parse filename: {file_path.name}")
                return False
            
            # Read content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print(f"Empty file: {file_path.name}")
                return False
            
            # Create document ID
            doc_id = f"chapter_{parsed['chapter']}_{parsed['type']}_v{parsed['version']}"
            
            # Check if already exists
            try:
                existing = self.collection.get(ids=[doc_id])
                if existing['ids']:
                    print(f"Already indexed: {doc_id}")
                    return False
            except:
                pass  # Collection might be empty
            
            # Prepare metadata
            metadata = {
                "chapter": parsed['chapter'],
                "version": parsed['version'],
                "type": parsed['type'],
                "filename": file_path.name,
                "content_length": len(content),
                "word_count": len(content.split()),
                "indexed_at": datetime.now().isoformat(),
                "feedback_score": 0.0
            }
            
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()
            
            # Add to ChromaDB
            self.collection.add(
                ids=[doc_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            
            print(f"✅ Indexed: {doc_id}")
            return True
            
        except Exception as e:
            print(f"Error indexing {file_path.name}: {e}")
            return False
    
    def index_all_files(self) -> Dict[str, Any]:
        """Index all files in chapters directory"""
        if not self.chapters_dir.exists():
            print(f"❌ Chapters directory not found: {self.chapters_dir}")
            return {"success": False, "error": "Chapters directory not found", "indexed": 0}
        
        txt_files = list(self.chapters_dir.glob("*.txt"))
        if not txt_files:
            print(f"❌ No .txt files found in: {self.chapters_dir}")
            return {"success": False, "error": "No .txt files found in chapters directory", "indexed": 0}
        
        print(f"Found {len(txt_files)} files to index:")
        for f in txt_files:
            print(f"  - {f.name}")
        
        indexed_count = 0
        for txt_file in txt_files:
            if self.index_single_file(txt_file):
                indexed_count += 1
        
        return {
            "success": True,
            "indexed": indexed_count,
            "total_files": len(txt_files),
            "message": f"Indexed {indexed_count}/{len(txt_files)} files"
        }
    
    def search_versions(self, query: str, n_results: int = 10, 
                       chapter_filter: Optional[int] = None,
                       type_filter: Optional[str] = None) -> Dict[str, Any]:
        """Search for relevant versions"""
        try:
            # Check if collection has documents
            count = self.collection.count()
            if count == 0:
                return {"success": False, "error": "No documents in collection. Please index files first."}
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Prepare filters
            where_clause = {}
            if chapter_filter is not None:
                where_clause["chapter"] = chapter_filter
            if type_filter is not None:
                where_clause["type"] = type_filter
            
            # Search ChromaDB
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, count),
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            processed_results = []
            
            for i, doc_id in enumerate(search_results['ids'][0]):
                document = search_results['documents'][0][i]
                metadata = search_results['metadatas'][0][i]
                similarity_score = 1 - search_results['distances'][0][i]
                feedback_score = self.feedback_data.get(doc_id, 0.0)
                combined_score = (0.7 * similarity_score) + (0.3 * feedback_score)
                
                processed_results.append({
                    "doc_id": doc_id,
                    "document": document,
                    "metadata": metadata,
                    "similarity_score": similarity_score,
                    "feedback_score": feedback_score,
                    "combined_score": combined_score,
                    "preview": document[:300] + "..." if len(document) > 300 else document
                })
            
            # Sort by combined score
            processed_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            return {
                "success": True,
                "query": query,
                "total_results": len(processed_results),
                "results": processed_results
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def add_feedback(self, doc_id: str, feedback: float) -> bool:
        """Add feedback score for a document"""
        try:
            feedback = max(-1.0, min(1.0, feedback))
            self.feedback_data[doc_id] = feedback
            self.save_feedback()
            
            # Update metadata in ChromaDB
            try:
                existing = self.collection.get(ids=[doc_id])
                if existing['ids']:
                    metadata = existing['metadatas'][0]
                    metadata['feedback_score'] = feedback
                    metadata['last_feedback'] = datetime.now().isoformat()
                    
                    self.collection.update(
                        ids=[doc_id],
                        metadatas=[metadata]
                    )
                    return True
            except:
                pass
            
            return True
                
        except Exception as e:
            print(f"Error adding feedback: {e}")
            return False
    
    def get_all_documents(self) -> Dict[str, Any]:
        """Get all documents in collection"""
        try:
            all_docs = self.collection.get(include=["metadatas"])
            return {
                "success": True,
                "count": len(all_docs['ids']),
                "documents": all_docs
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_version_stats(self) -> Dict[str, Any]:
        """Get basic statistics"""
        try:
            all_docs = self.collection.get(include=["metadatas"])
            
            if not all_docs['ids']:
                return {"total_versions": 0, "success": True}
            
            chapters = set()
            types = {}
            
            for metadata in all_docs['metadatas']:
                chapters.add(metadata['chapter'])
                version_type = metadata['type']
                types[version_type] = types.get(version_type, 0) + 1
            
            feedback_scores = list(self.feedback_data.values())
            
            return {
                "success": True,
                "total_versions": len(all_docs['ids']),
                "total_chapters": len(chapters),
                "chapters": sorted(list(chapters)),
                "version_types": types,
                "feedback_stats": {
                    "total_feedbacks": len(feedback_scores),
                    "average_score": np.mean(feedback_scores) if feedback_scores else 0.0
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

def main():
    """CLI interface for testing"""
    print("🗂️ ChromaDB Version Manager")
    print("=" * 40)
    
    try:
        manager = ChromaDBVersionManager()
        
        while True:
            print("\nCommands:")
            print("1. Index all files")
            print("2. Search versions")
            print("3. Add feedback")
            print("4. View stats")
            print("5. Exit")
            
            choice = input("\nChoice (1-5): ").strip()
            
            if choice == "1":
                print("\n📂 Indexing files...")
                result = manager.index_all_files()
                if result["success"]:
                    print(f"✅ {result['message']}")
                else:
                    print(f"❌ Error: {result['error']}")
                print(f"Collection now has {manager.collection.count()} documents")
            
            elif choice == "2":
                count = manager.collection.count()
                if count == 0:
                    print("❌ No documents indexed yet. Use option 1 first.")
                    continue
                    
                query = input(f"\n🔍 Search query (collection has {count} docs): ").strip()
                if query:
                    print("Searching...")
                    results = manager.search_versions(query)
                    if results["success"]:
                        print(f"\nFound {results['total_results']} results:")
                        for i, result in enumerate(results["results"][:3], 1):
                            print(f"\n{i}. {result['doc_id']}")
                            print(f"   📊 Score: {result['combined_score']:.3f}")
                            print(f"   📝 Preview: {result['preview'][:150]}...")
                    else:
                        print(f"❌ Error: {results['error']}")
                else:
                    print("❌ Empty query")
            
            elif choice == "3":
                doc_id = input("\n📝 Document ID: ").strip()
                if doc_id:
                    try:
                        feedback = float(input("Feedback (-1.0 to 1.0): "))
                        if manager.add_feedback(doc_id, feedback):
                            print("✅ Feedback added")
                        else:
                            print("❌ Failed to add feedback")
                    except ValueError:
                        print("❌ Invalid feedback score")
                else:
                    print("❌ Empty document ID")
            
            elif choice == "4":
                print("\n📊 Collection Stats:")
                stats = manager.get_version_stats()
                if stats["success"]:
                    print(f"- Total versions: {stats['total_versions']}")
                    print(f"- Chapters: {stats['chapters']}")
                    print(f"- Types: {stats['version_types']}")
                    print(f"- Feedback entries: {stats['feedback_stats']['total_feedbacks']}")
                else:
                    print(f"❌ Error: {stats['error']}")
            
            elif choice == "5":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice (1-5)")
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()