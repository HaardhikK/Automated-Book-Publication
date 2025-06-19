import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path

# Import our ChromaDB manager
from part6_chromadb_versioning import ChromaDBVersionManager

# Page config
st.set_page_config(
    page_title="ChromaDB Version Manager",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'manager' not in st.session_state:
    st.session_state.manager = ChromaDBVersionManager()
if 'last_search_results' not in st.session_state:
    st.session_state.last_search_results = None

def main():
    st.title("🗂️ ChromaDB Version Manager & RL-based Retrieval")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Statistics
        with st.expander("📊 Statistics", expanded=True):
            stats = st.session_state.manager.get_version_stats()
            if "error" not in stats:
                st.metric("Total Versions", stats.get('total_versions', 0))
                st.metric("Total Chapters", stats.get('total_chapters', 0))
                st.metric("Feedback Entries", stats.get('feedback_stats', {}).get('total_feedbacks', 0))
                
                avg_feedback = stats.get('feedback_stats', {}).get('average_score', 0)
                st.metric("Avg Feedback Score", f"{avg_feedback:.3f}")
        
        # Quick actions
        st.header("⚡ Quick Actions")
        
        if st.button("🔍 Index All Versions", use_container_width=True):
            with st.spinner("Indexing versions..."):
                result = st.session_state.manager.scan_and_index_versions()
                if result["success"]:
                    st.success(f"✅ Indexed {result['indexed_count']} versions")
                else:
                    st.error(f"❌ Error: {result['error']}")
                st.rerun()
        
        if st.button("💾 Export Versions", use_container_width=True):
            filename = f"versions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            if st.session_state.manager.export_versions(filename):
                st.success(f"✅ Exported to {filename}")
            else:
                st.error("❌ Export failed")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📊 Analytics", "👍 Feedback", "🗂️ Browse"])
    
    with tab1:
        search_tab()
    
    with tab2:
        analytics_tab()
    
    with tab3:
        feedback_tab()
    
    with tab4:
        browse_tab()

def search_tab():
    st.header("🔍 Semantic Search")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'character development', 'battle scene', 'emotional dialogue'",
            key="search_query"
        )
    
    with col2:
        n_results = st.selectbox("Results", [5, 10, 15, 20], index=1)
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        chapter_filter = st.selectbox(
            "Filter by Chapter",
            ["All Chapters"] + [f"Chapter {i}" for i in range(1, 11)],
            key="chapter_filter"
        )
        chapter_num = None if chapter_filter == "All Chapters" else int(chapter_filter.split()[-1])
    
    with col2:
        type_filter = st.selectbox(
            "Filter by Type",
            ["All Types", "original", "ai", "human", "reviewed"],
            key="type_filter"
        )
        type_val = None if type_filter == "All Types" else type_filter
    
    # Search button
    if st.button("🔍 Search", use_container_width=True) or st.session_state.get('auto_search', False):
        if query.strip():
            with st.spinner("Searching..."):
                results = st.session_state.manager.search_versions(
                    query=query,
                    n_results=n_results,
                    chapter_filter=chapter_num,
                    type_filter=type_val
                )
                
                st.session_state.last_search_results = results
        else:
            st.warning("Please enter a search query")
    
    # Display results
    if st.session_state.last_search_results:
        results = st.session_state.last_search_results
        
        if results["success"]:
            st.subheader(f"📋 Search Results ({results['total_results']} found)")
            
            for i, result in enumerate(results["results"]):
                with st.expander(f"📄 {result['doc_id']} (Score: {result['combined_score']:.3f})", expanded=i<3):
                    # Metadata
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Similarity", f"{result['similarity_score']:.3f}")
                    with col2:
                        st.metric("Feedback", f"{result['feedback_score']:.3f}")
                    with col3:
                        st.metric("Combined", f"{result['combined_score']:.3f}")
                    with col4:
                        word_count = result['metadata'].get('word_count', 0)
                        st.metric("Words", word_count)
                    
                    # Metadata details
                    metadata = result['metadata']
                    st.write("**Metadata:**")
                    meta_col1, meta_col2 = st.columns(2)
                    
                    with meta_col1:
                        st.write(f"- **Chapter:** {metadata.get('chapter', 'N/A')}")
                        st.write(f"- **Type:** {metadata.get('type', 'N/A')}")
                        st.write(f"- **Version:** {metadata.get('version', 'N/A')}")
                    
                    with meta_col2:
                        st.write(f"- **Editor:** {metadata.get('editor', 'N/A')}")
                        st.write(f"- **Timestamp:** {metadata.get('timestamp', 'N/A')}")
                        st.write(f"- **Length:** {metadata.get('content_length', 'N/A')} chars")
                    
                    # Content preview
                    st.write("**Content Preview:**")
                    st.text_area(
                        "Preview",
                        result['preview'],
                        height=100,
                        key=f"preview_{result['doc_id']}",
                        label_visibility="collapsed"
                    )
                    
                    # Quick feedback buttons
                    feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
                    
                    with feedback_col1:
                        if st.button("👍 Good", key=f"good_{result['doc_id']}"):
                            st.session_state.manager.add_feedback(result['doc_id'], 0.5)
                            st.success("Positive feedback added!")
                            st.rerun()
                    
                    with feedback_col2:
                        if st.button("👎 Bad", key=f"bad_{result['doc_id']}"):
                            st.session_state.manager.add_feedback(result['doc_id'], -0.5)
                            st.success("Negative feedback added!")
                            st.rerun()
                    
                    with feedback_col3:
                        if st.button("⭐ Excellent", key=f"excellent_{result['doc_id']}"):
                            st.session_state.manager.add_feedback(result['doc_id'], 1.0)
                            st.success("Excellent feedback added!")
                            st.rerun()
        
        else:
            st.error(f"Search failed: {results['error']}")

def analytics_tab():
    st.header("📊 Analytics Dashboard")
    
    # Get stats
    stats = st.session_state.manager.get_version_stats()
    
    if "error" in stats:
        st.error(f"Error loading stats: {stats['error']}")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Versions", stats.get('total_versions', 0))
    with col2:
        st.metric("Total Chapters", stats.get('total_chapters', 0))
    with col3:
        feedback_stats = stats.get('feedback_stats', {})
        st.metric("Total Feedback", feedback_stats.get('total_feedbacks', 0))
    with col4:
        avg_score = feedback_stats.get('average_score', 0)
        st.metric("Avg Feedback", f"{avg_score:.3f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Version types distribution
        if stats.get('version_types'):
            fig_types = px.pie(
                values=list(stats['version_types'].values()),
                names=list(stats['version_types'].keys()),
                title="Version Types Distribution"
            )
            st.plotly_chart(fig_types, use_container_width=True)
    
    with col2:
        # Chapters distribution
        if stats.get('chapters'):
            chapter_counts = {}
            # This would need to be implemented in the manager to get actual chapter distribution
            for chapter in stats['chapters']:
                chapter_counts[f"Chapter {chapter}"] = 1  # Placeholder
            
            fig_chapters = px.bar(
                x=list(chapter_counts.keys()),
                y=list(chapter_counts.values()),
                title="Chapters Available"
            )
            st.plotly_chart(fig_chapters, use_container_width=True)
    
    # Feedback distribution
    if feedback_stats.get('total_feedbacks', 0) > 0:
        st.subheader("📈 Feedback Score Distribution")
        
        feedback_data = st.session_state.manager.feedback_data
        scores = list(feedback_data.values())
        
        if scores:
            fig_feedback = px.histogram(
                x=scores,
                nbins=20,
                title="Feedback Score Distribution",
                labels={'x': 'Feedback Score', 'y': 'Count'}
            )
            st.plotly_chart(fig_feedback, use_container_width=True)

def feedback_tab():
    st.header("👍 Feedback Management")
    
    # Manual feedback entry
    st.subheader("📝 Add Manual Feedback")
    
    col1, col2 = st.columns(2)
    
    with col1:
        doc_id = st.text_input(
            "Document ID",
            placeholder="e.g., chapter_1_ai_v1",
            key="feedback_doc_id"
        )
    
    with col2:
        feedback_score = st.slider(
            "Feedback Score",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Rate from -1.0 (worst) to 1.0 (best)"
        )
    
    if st.button("💾 Submit Feedback"):
        if doc_id.strip():
            if st.session_state.manager.add_feedback(doc_id.strip(), feedback_score):
                st.success(f"✅ Feedback {feedback_score:.1f} added for {doc_id}")
            else:
                st.error("❌ Failed to add feedback. Check if document ID exists.")
        else:
            st.warning("Please enter a document ID")
    
    # Current feedback data
    st.subheader("📊 Current Feedback Data")
    
    feedback_data = st.session_state.manager.feedback_data
    
    if feedback_data:
        # Convert to DataFrame for display
        feedback_df = pd.DataFrame([
            {"Document ID": doc_id, "Feedback Score": score}
            for doc_id, score in feedback_data.items()
        ])
        
        feedback_df = feedback_df.sort_values("Feedback Score", ascending=False)
        
        # Display with editing capabilities
        edited_df = st.data_editor(
            feedback_df,
            use_container_width=True,
            num_rows="dynamic",
            key="feedback_editor"
        )
        
        # Save changes button
        if st.button("💾 Save Changes"):
            try:
                # Update feedback data from edited DataFrame
                new_feedback = {}
                for _, row in edited_df.iterrows():
                    new_feedback[row["Document ID"]] = row["Feedback Score"]
                
                st.session_state.manager.feedback_data = new_feedback
                st.session_state.manager.save_feedback()
                st.success("✅ Feedback data updated!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error saving changes: {e}")
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Entries", len(feedback_data))
        with col2:
            avg_score = sum(feedback_data.values()) / len(feedback_data)
            st.metric("Average Score", f"{avg_score:.3f}")
        with col3:
            best_doc = max(feedback_data.items(), key=lambda x: x[1])
            st.metric("Best Rated", f"{best_doc[0][:20]}... ({best_doc[1]:.2f})")
    
    else:
        st.info("No feedback data available yet. Start by adding some feedback!")

def browse_tab():
    st.header("🗂️ Browse All Versions")
    
    # Get all documents
    try:
        all_docs = st.session_state.manager.collection.get(include=["metadatas"])
        
        if not all_docs['ids']:
            st.info("No versions found. Please index some versions first.")
            return
        
        # Convert to DataFrame
        rows = []
        for i, doc_id in enumerate(all_docs['ids']):
            metadata = all_docs['metadatas'][i]
            feedback_score = st.session_state.manager.feedback_data.get(doc_id, 0.0)
            
            rows.append({
                "Document ID": doc_id,
                "Chapter": metadata.get('chapter', 'N/A'),
                "Type": metadata.get('type', 'N/A'),
                "Version": metadata.get('version', 'N/A'),
                "Word Count": metadata.get('word_count', 0),
                "Editor": metadata.get('editor', 'N/A'),
                "Feedback Score": feedback_score,
                "Timestamp": metadata.get('timestamp', 'N/A')
            })
        
        df = pd.DataFrame(rows)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chapter_filter = st.selectbox(
                "Filter by Chapter",
                ["All"] + sorted(df['Chapter'].unique().tolist()),
                key="browse_chapter_filter"
            )
        
        with col2:
            type_filter = st.selectbox(
                "Filter by Type",
                ["All"] + sorted(df['Type'].unique().tolist()),
                key="browse_type_filter"
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Document ID", "Chapter", "Type", "Version", "Word Count", "Feedback Score", "Timestamp"],
                index=5  # Default to Feedback Score
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if chapter_filter != "All":
            filtered_df = filtered_df[filtered_df['Chapter'] == chapter_filter]
        
        if type_filter != "All":
            filtered_df = filtered_df[filtered_df['Type'] == type_filter]
        
        # Sort
        filtered_df = filtered_df.sort_values(sort_by, ascending=False)
        
        # Display
        st.subheader(f"📋 Versions ({len(filtered_df)} found)")
        
        # Interactive table
        selected_rows = st.dataframe(
            filtered_df,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # Show selected document details
        if selected_rows and selected_rows.selection and selected_rows.selection.rows:
            selected_idx = selected_rows.selection.rows[0]
            selected_doc_id = filtered_df.iloc[selected_idx]['Document ID']
            
            st.subheader(f"📄 Document Details: {selected_doc_id}")
            
            # Get full document
            doc_data = st.session_state.manager.collection.get(
                ids=[selected_doc_id],
                include=["documents", "metadatas"]
            )
            
            if doc_data['ids']:
                document = doc_data['documents'][0]
                metadata = doc_data['metadatas'][0]
                
                # Metadata display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📊 Metadata:**")
                    for key, value in metadata.items():
                        st.write(f"- **{key.title()}:** {value}")
                
                with col2:
                    st.write("**📈 Stats:**")
                    st.write(f"- **Character Count:** {len(document):,}")
                    st.write(f"- **Word Count:** {len(document.split()):,}")
                    st.write(f"- **Lines:** {document.count(chr(10)) + 1:,}")
                    
                    feedback_score = st.session_state.manager.feedback_data.get(selected_doc_id, 0.0)
                    st.write(f"- **Feedback Score:** {feedback_score:.3f}")
                
                # Full document content
                st.write("**📝 Full Content:**")
                st.text_area(
                    "Document Content",
                    document,
                    height=400,
                    key=f"full_content_{selected_doc_id}",
                    label_visibility="collapsed"
                )
                
                # Quick actions for selected document
                st.write("**⚡ Quick Actions:**")
                action_col1, action_col2, action_col3, action_col4 = st.columns(4)
                
                with action_col1:
                    if st.button("👍 Rate Good", key=f"rate_good_{selected_doc_id}"):
                        st.session_state.manager.add_feedback(selected_doc_id, 0.5)
                        st.success("Positive feedback added!")
                        st.rerun()
                
                with action_col2:
                    if st.button("👎 Rate Bad", key=f"rate_bad_{selected_doc_id}"):
                        st.session_state.manager.add_feedback(selected_doc_id, -0.5)
                        st.success("Negative feedback added!")
                        st.rerun()
                
                with action_col3:
                    if st.button("⭐ Rate Excellent", key=f"rate_excellent_{selected_doc_id}"):
                        st.session_state.manager.add_feedback(selected_doc_id, 1.0)
                        st.success("Excellent feedback added!")
                        st.rerun()
                
                with action_col4:
                    # Copy to clipboard (simulated)
                    if st.button("📋 Copy Text", key=f"copy_{selected_doc_id}"):
                        st.code(document, language="text")
                        st.info("Text displayed above for copying")
    
    except Exception as e:
        st.error(f"Error browsing versions: {e}")

# Add some custom CSS for better styling
st.markdown("""
<style>
    .stMetric > div > div > div > div {
        font-size: 1.2rem;
    }
    
    .stExpander > div > div > div > div {
        font-size: 0.9rem;
    }
    
    .stDataFrame {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()