import os
import json
import glob
import pickle
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RLSearchOptimizer:
    """Query-specific RL optimizer for search ranking."""
    
    def __init__(self):
        self.query_doc_scores = defaultdict(lambda: defaultdict(list))  # query -> doc_id -> [scores]
        self.query_patterns = defaultdict(list)  # Similar query clustering
        
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better matching."""
        return ' '.join(query.lower().split())
    
    def _find_similar_queries(self, query: str, threshold: float = 0.7) -> List[str]:
        """Find similar queries using simple word overlap."""
        normalized_query = self._normalize_query(query)
        query_words = set(normalized_query.split())
        
        similar_queries = []
        for stored_query in self.query_doc_scores.keys():
            stored_words = set(stored_query.split())
            if len(query_words) == 0 or len(stored_words) == 0:
                continue
                
            overlap = len(query_words.intersection(stored_words))
            similarity = overlap / max(len(query_words), len(stored_words))
            
            if similarity >= threshold:
                similar_queries.append(stored_query)
        
        return similar_queries
    
    def update_score(self, doc_id: str, query: str, feedback_score: float):
        """Update query-specific document relevance."""
        normalized_query = self._normalize_query(query)
        
        # Store feedback for this exact query-document pair
        self.query_doc_scores[normalized_query][doc_id].append(feedback_score)
        
        # Keep only recent feedback (last 10 scores)
        if len(self.query_doc_scores[normalized_query][doc_id]) > 10:
            self.query_doc_scores[normalized_query][doc_id] = \
                self.query_doc_scores[normalized_query][doc_id][-10:]
    
    def get_relevance_boost(self, doc_id: str, query: str) -> float:
        """Get query-specific relevance boost for a document."""
        normalized_query = self._normalize_query(query)
        
        # Direct query-document score
        direct_scores = self.query_doc_scores[normalized_query].get(doc_id, [])
        direct_score = np.mean(direct_scores) if direct_scores else 0.0
        
        # Score from similar queries
        similar_queries = self._find_similar_queries(normalized_query)
        similar_scores = []
        
        for sim_query in similar_queries:
            if sim_query != normalized_query:
                scores = self.query_doc_scores[sim_query].get(doc_id, [])
                if scores:
                    similar_scores.extend(scores)
        
        similar_score = np.mean(similar_scores) if similar_scores else 0.0
        
        # Combine direct and similar query scores
        # Direct feedback is more important than similar query feedback
        final_boost = direct_score * 0.8 + similar_score * 0.2
        
        # Scale the boost to reasonable range (-0.3 to +0.3)
        return np.clip(final_boost * 0.3, -0.3, 0.3)

class FAISSVersionManager:
    """FAISS-based version manager with RL-optimized search."""
    
    def __init__(self, db_path: str = "faiss_db", chapters_dir: str = "chapters"):
        self.db_path = db_path
        self.chapters_dir = chapters_dir
        self.index_file = os.path.join(db_path, "faiss_index.bin")
        self.metadata_file = os.path.join(db_path, "metadata.json")
        self.rl_state_file = os.path.join(db_path, "rl_state.pkl")
        
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.documents = []
        self.metadata = []
        
        # Initialize RL optimizer
        self.rl_optimizer = RLSearchOptimizer()
        
        # Load existing data
        self._load_data()
        
        logger.info(f"Initialized with {len(self.documents)} documents")
    
    def _parse_filename(self, filepath: str) -> Dict[str, str]:
        """Extract metadata from filename following your naming convention."""
        filename = os.path.basename(filepath)
        base_name = filename.replace('.txt', '')
        parts = base_name.split('_')
        
        # Default metadata
        metadata = {
            'chapter': 'unknown',
            'type': 'final',
            'version': 'v1',
            'editor': 'unknown'
        }
        
        if len(parts) >= 2:
            metadata['chapter'] = f"{parts[0]}_{parts[1]}"  # e.g., "chapter_1"
            
            # Find type and version in remaining parts
            for part in parts[2:]:
                if part in ['ai', 'human', 'reviewed', 'draft', 'final']:
                    metadata['type'] = part
                    if part == 'ai':
                        metadata['editor'] = 'ai_writer'
                    elif part == 'human':
                        metadata['editor'] = 'human_editor'
                    elif part == 'reviewed':
                        metadata['editor'] = 'ai_reviewer'
                elif part.startswith('v') and part[1:].isdigit():
                    metadata['version'] = part
        
        return metadata
    
    def store_versions(self):
        """Index all chapter versions from the chapters directory."""
        pattern = os.path.join(self.chapters_dir, "*.txt")
        txt_files = glob.glob(pattern)
        
        if not txt_files:
            logger.warning(f"No .txt files found in {self.chapters_dir}")
            return
        
        logger.info(f"Found {len(txt_files)} files to process")
        
        new_embeddings = []
        new_documents = []
        new_metadata = []
        
        # Get existing document IDs to avoid duplicates
        existing_ids = {meta.get('id') for meta in self.metadata}
        
        for filepath in txt_files:
            try:
                # Skip files without version tags
                filename = os.path.basename(filepath)
                if not any(tag in filename for tag in ['ai', 'human', 'reviewed', 'draft', 'final']):
                    continue
                
                # Parse metadata
                file_metadata = self._parse_filename(filepath)
                doc_id = f"{file_metadata['chapter']}_{file_metadata['type']}_{file_metadata['version']}"
                
                # Skip if already indexed
                if doc_id in existing_ids:
                    logger.info(f"Skipping existing: {doc_id}")
                    continue
                
                # Read content
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    logger.warning(f"Empty file: {filepath}")
                    continue
                
                # Generate embedding
                embedding = self.model.encode(content, normalize_embeddings=True)
                
                # Create metadata
                metadata = {
                    'id': doc_id,
                    'filepath': filepath,
                    'timestamp': datetime.now().isoformat(),
                    'content_length': len(content),
                    **file_metadata
                }
                
                new_embeddings.append(embedding)
                new_documents.append(content)
                new_metadata.append(metadata)
                
                logger.info(f"✅ Indexed: {doc_id}")
                
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
        
        if new_embeddings:
            # Add to FAISS index
            embeddings_array = np.array(new_embeddings).astype('float32')
            self.index.add(embeddings_array)
            
            # Update document store
            self.documents.extend(new_documents)
            self.metadata.extend(new_metadata)
            
            # Save to disk
            self._save_data()
            
            logger.info(f"Successfully indexed {len(new_embeddings)} new documents")
        else:
            logger.info("No new documents to index")
    
    def search_versions(self, query: str, n_results: int = 5, 
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for relevant versions with RL optimization."""
        if self.index.ntotal == 0:
            logger.warning("No documents in index")
            return []
        
        # Encode query
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search FAISS (get more results for filtering)
        k = min(n_results * 3, self.index.ntotal)
        similarities, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                break
            
            similarity = float(similarities[0][i])
            metadata = self.metadata[idx]
            content = self.documents[idx]
            
            # Apply filters if provided
            if filters and not self._matches_filters(metadata, filters):
                continue
            
            # Get RL boost
            rl_boost = self.rl_optimizer.get_relevance_boost(metadata['id'], query)
            
            # Combined score (similarity + RL boost)
            final_score = similarity + rl_boost
            
            results.append({
                'id': metadata['id'],
                'content': content,
                'metadata': metadata,
                'similarity': similarity,
                'rl_boost': rl_boost,
                'final_score': final_score
            })
        
        # Sort by final score and return top results
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:n_results]
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
    
    def add_feedback(self, doc_id: str, query: str, score: float):
        """Add user feedback for RL learning."""
        # Clamp score to valid range
        score = max(-1.0, min(1.0, score))
        
        # Update RL optimizer
        self.rl_optimizer.update_score(doc_id, query, score)
        
        # Save RL state
        self._save_rl_state()
        
        logger.info(f"Added feedback for {doc_id}: {score}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed documents."""
        if not self.metadata:
            return {'total_documents': 0}
        
        stats = {
            'total_documents': len(self.metadata),
            'by_type': defaultdict(int),
            'by_chapter': defaultdict(int),
            'by_version': defaultdict(int)
        }
        
        for meta in self.metadata:
            stats['by_type'][meta.get('type', 'unknown')] += 1
            stats['by_chapter'][meta.get('chapter', 'unknown')] += 1
            stats['by_version'][meta.get('version', 'unknown')] += 1
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats['by_type'] = dict(stats['by_type'])
        stats['by_chapter'] = dict(stats['by_chapter'])
        stats['by_version'] = dict(stats['by_version'])
        
        return stats
    
    def _save_data(self):
        """Save FAISS index and metadata."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            
            # Save metadata and documents
            data = {
                'documents': self.documents,
                'metadata': self.metadata
            }
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Save RL state
            self._save_rl_state()
            
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def _load_data(self):
        """Load existing FAISS index and metadata."""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                # Load FAISS index
                self.index = faiss.read_index(self.index_file)
                
                # Load metadata and documents
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', [])
                    self.metadata = data.get('metadata', [])
                
                # Load RL state
                self._load_rl_state()
                
                logger.info(f"Loaded {len(self.documents)} existing documents")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def _save_rl_state(self):
        """Save RL optimizer state."""
        try:
            rl_state = {
                'query_doc_scores': {
                    query: {doc_id: scores for doc_id, scores in docs.items()}
                    for query, docs in self.rl_optimizer.query_doc_scores.items()
                }
            }
            with open(self.rl_state_file, 'wb') as f:
                pickle.dump(rl_state, f)
        except Exception as e:
            logger.error(f"Error saving RL state: {e}")
    
    def _load_rl_state(self):
        """Load RL optimizer state."""
        try:
            if os.path.exists(self.rl_state_file):
                with open(self.rl_state_file, 'rb') as f:
                    rl_state = pickle.load(f)
                    query_doc_scores = rl_state.get('query_doc_scores', {})
                    
                    # Reconstruct the nested defaultdict structure
                    self.rl_optimizer.query_doc_scores = defaultdict(lambda: defaultdict(list))
                    for query, docs in query_doc_scores.items():
                        for doc_id, scores in docs.items():
                            self.rl_optimizer.query_doc_scores[query][doc_id] = scores
        except Exception as e:
            logger.error(f"Error loading RL state: {e}")

def main():
    """Interactive CLI for the version manager."""
    print("🚀 FAISS Version Manager with RL Search")
    print("=" * 50)
    
    # Initialize with your folder structure
    vm = FAISSVersionManager(chapters_dir="chapters")
    
    while True:
        print("\n📋 Available Commands:")
        print("1. Index versions")
        print("2. Search versions")  
        print("3. Search with filters")
        print("4. Add feedback")
        print("5. View statistics")
        print("6. Exit")
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                print("📂 Indexing versions...")
                vm.store_versions()
                
            elif choice == '2':
                query = input("🔍 Enter search query: ").strip()
                if query:
                    results = vm.search_versions(query)
                    print(f"\n📊 Search Results for: '{query}'")
                    print("-" * 50)
                    
                    for i, result in enumerate(results, 1):
                        meta = result['metadata']
                        print(f"\n{i}. ID: {result['id']}")
                        print(f"   Type: {meta['type']} | Version: {meta['version']}")
                        print(f"   Similarity: {result['similarity']:.3f} | RL Boost: {result['rl_boost']:.3f}")
                        print(f"   Preview: {result['content'][:100]}...")
                        
            elif choice == '3':
                query = input("🔍 Enter search query: ").strip()
                filter_type = input("Filter by type (ai/human/reviewed/final) or press Enter: ").strip()
                
                filters = {}
                if filter_type:
                    filters['type'] = filter_type
                    
                if query:
                    results = vm.search_versions(query, filters=filters)
                    print(f"\n📊 Filtered Results: {len(results)} found")
                    for i, result in enumerate(results, 1):
                        meta = result['metadata']
                        print(f"{i}. {result['id']} ({meta['type']}) - Score: {result['final_score']:.3f}")
                        
            elif choice == '4':
                doc_id = input("📝 Document ID: ").strip()
                query = input("🔍 Related query: ").strip()
                try:
                    score = float(input("👍 Feedback score (-1 to 1): "))
                    vm.add_feedback(doc_id, query, score)
                    print("✅ Feedback recorded")
                except ValueError:
                    print("❌ Invalid score")
                    
            elif choice == '5':
                stats = vm.get_stats()
                print("\n📈 Statistics:")
                print(f"Total Documents: {stats['total_documents']}")
                
                if stats['total_documents'] > 0:
                    print("\nBy Type:")
                    for doc_type, count in stats['by_type'].items():
                        print(f"  {doc_type}: {count}")
                    
                    print("\nBy Chapter:")
                    for chapter, count in stats['by_chapter'].items():
                        print(f"  {chapter}: {count}")
                        
            elif choice == '6':
                print("👋 Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()