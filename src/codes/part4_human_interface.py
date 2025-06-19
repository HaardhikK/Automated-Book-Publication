import streamlit as st
import json
import difflib
from datetime import datetime
from pathlib import Path
import glob

# Page config
st.set_page_config(
    page_title="Chapter Editor",
    page_icon="📚",
    layout="wide"
)

class ChapterEditor:
    def __init__(self):
        self.chapters_dir = Path("chapters")
        self.logs_dir = Path("logs")
        
        # Create directories
        for directory in [self.chapters_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
    
    def get_chapter_files(self):
        """Get all chapter text files"""
        pattern = str(self.chapters_dir / "chapter_*.txt")
        files = glob.glob(pattern)
        return sorted([Path(f).name for f in files])
    
    def load_chapter(self, filename):
        """Load chapter content"""
        try:
            with open(self.chapters_dir / filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None
    
    def save_chapter(self, content, filename):
        """Save chapter content"""
        filepath = self.chapters_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath
    
    def save_metadata(self, filename, source_version, editor_name="Anonymous"):
        """Save edit metadata"""
        metadata = {
            "filename": filename,
            "source_version": source_version,
            "editor_name": editor_name,
            "timestamp": datetime.now().isoformat(),
            "edit_type": "human_review"
        }
        
        meta_filename = filename.replace('.txt', '_meta.json')
        with open(self.logs_dir / meta_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
    def generate_diff_html(self, text1, text2, label1="Version 1", label2="Version 2"):
        """Generate improved HTML diff between two texts with better styling"""
        differ = difflib.HtmlDiff(wrapcolumn=120)
        lines1 = text1.splitlines(keepends=True)
        lines2 = text2.splitlines(keepends=True)
        
        # Generate the diff table
        diff_table = differ.make_table(lines1, lines2, "", "", context=True, numlines=3)
        
        # Remove navigation links by parsing and cleaning the HTML
        import re
        # Remove the first column (navigation links) by removing first td/th in each row
        diff_table = re.sub(r'<th[^>]*>[^<]*</th>', '', diff_table, count=1)  # Remove first header
        diff_table = re.sub(r'<td[^>]*class="diff_next"[^>]*>.*?</td>', '', diff_table, flags=re.DOTALL)  # Remove navigation cells
        
        # Custom CSS
        custom_css = """
        <style>
        body {
            margin: 0;
            padding: 0;
            background: transparent;
        }
        table.diff {
            width: 100%;
            border-collapse: collapse;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            background-color: #ffffff;
            border: 1px solid #ddd;
        }
        .diff td {
            padding: 3px 6px;
            vertical-align: top;
            white-space: pre-wrap;
            word-break: break-word;
            border: 1px solid #ddd;
        }
        .diff th {
            background-color: #f5f5f5;
            padding: 6px;
            text-align: center;
            font-weight: bold;
            color: #333;
            border: 1px solid #ddd;
        }
        /* Line numbers styling */
        .diff_header {
            background-color: #e9ecef !important;
            color: #495057 !important;
            text-align: center;
            font-weight: bold;
            width: 50px;
        }
        /* Added lines (green background) */
        .diff_add {
            background-color: #d4edda !important;
            color: #155724 !important;
        }
        /* Deleted lines (red background) */
        .diff_chg {
            background-color: #f8d7da !important;
            color: #721c24 !important;
        }
        /* Modified content (yellow background) */
        .diff_sub {
            background-color: #fff3cd !important;
            color: #856404 !important;
        }
        /* Unchanged lines */
        .diff td:not(.diff_add):not(.diff_chg):not(.diff_sub):not(.diff_header) {
            color: #212529 !important;
            background-color: #ffffff !important;
        }
        /* Remove any remaining navigation elements */
        .diff_next {
            display: none !important;
        }
        </style>
        """
        
        return f"{custom_css}{diff_table}"
        """Generate a simple, readable diff display"""
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()
        
        diff = list(difflib.unified_diff(lines1, lines2, fromfile=label1, tofile=label2, lineterm=''))
        
        if not diff:
            return "No differences found."
        
        diff_text = ""
        for line in diff:
            if line.startswith('+++') or line.startswith('---'):
                diff_text += f"**{line}**\n"
            elif line.startswith('@@'):
                diff_text += f"\n`{line}`\n\n"
            elif line.startswith('+'):
                diff_text += f"🟢 **Added:** {line[1:]}\n"
            elif line.startswith('-'):
                diff_text += f"🔴 **Removed:** {line[1:]}\n"
            else:
                diff_text += f"{line}\n"
        
        return diff_text

def main():
    st.title("📚 Chapter Editor")
    st.markdown("Compare, edit and save chapter versions")
    
    editor = ChapterEditor()
    
    # Sidebar for file selection
    st.sidebar.header("📁 File Selection")
    
    available_files = editor.get_chapter_files()
    if not available_files:
        st.error("No chapter files found in chapters/ directory")
        return
    
    version1 = st.sidebar.selectbox("Sidebar Default Version 1:", available_files, key="v1_sidebar")
    version2 = st.sidebar.selectbox("Sidebar Default Version 2:", available_files, key="v2_sidebar")
    
    tab1, tab2, tab3 = st.tabs(["📄 Side-by-Side", "🔍 Diff View", "✏️ Editor"])
    
    with tab1:
        st.subheader("Side-by-Side Comparison")
        col1, col2 = st.columns(2)

        with col1:
            left_file = st.selectbox("Left Version", available_files, index=available_files.index(version1), key="left_select")
            left_content = editor.load_chapter(left_file)
            st.markdown(f"**{left_file}**")
            st.text_area("Left Version Content", left_content, height=500, disabled=True, key="left_content_display")
            st.caption(f"Length: {len(left_content)} characters")

        with col2:
            right_file = st.selectbox("Right Version", available_files, index=available_files.index(version2), key="right_select")
            right_content = editor.load_chapter(right_file)
            st.markdown(f"**{right_file}**")
            st.text_area("Right Version Content", right_content, height=500, disabled=True, key="right_content_display")
            st.caption(f"Length: {len(right_content)} characters")
    
    with tab2:
        st.subheader("🔍 Diff Viewer")
        
        # Add legend for diff colors
        st.markdown("""
        🟢 **Green**: Added lines 🔴 **Red**: Deleted lines  🟡 **Yellow**: Modified lines 
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            diff_file_1 = st.selectbox("Diff Version 1", available_files, index=0, key="diff_v1")
            diff_content_1 = editor.load_chapter(diff_file_1)
        with col2:
            diff_file_2 = st.selectbox("Diff Version 2", available_files, index=1 if len(available_files) > 1 else 0, key="diff_v2")
            diff_content_2 = editor.load_chapter(diff_file_2)

        if diff_content_1 and diff_content_2 and diff_content_1 != diff_content_2:
            diff_html = editor.generate_diff_html(diff_content_1, diff_content_2, diff_file_1, diff_file_2)
            st.components.v1.html(diff_html, height=600, scrolling=True)
        else:
            if diff_content_1 == diff_content_2:
                st.info("✅ Both versions are identical - no differences found")
            else:
                st.warning("⚠️ Select two different versions to see differences")
    
    with tab3:
        st.subheader("✏️ Edit Chapter")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            source_version = st.selectbox("Base version for editing:", available_files)
        with col2:
            editor_name = st.text_input("Editor name:", value="Anonymous")
        
        base_content = editor.load_chapter(source_version) if source_version else ""
        edited_content = st.text_area(f"Edit {source_version}:", base_content, height=400, key="editor_textarea")
        st.caption(f"Length: {len(edited_content)} characters")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            new_filename = st.text_input("Save as:", value="chapter_1_human_v2.txt")
        
        with col2:
            if st.button("💾 Save Version", type="primary"):
                if edited_content and new_filename:
                    try:
                        saved_path = editor.save_chapter(edited_content, new_filename)
                        editor.save_metadata(new_filename, source_version, editor_name)
                        st.success(f"✅ Saved to {saved_path}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error saving: {e}")
                else:
                    st.warning("⚠️ Please enter content and filename")
        
        with col3:
            if st.button("📝 Save Final"):
                final_filename = "chapter_1_final.txt"
                if edited_content:
                    try:
                        saved_path = editor.save_chapter(edited_content, final_filename)
                        editor.save_metadata(final_filename, source_version, editor_name)
                        st.success("✅ Final version saved!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error saving final: {e}")
    
    # Footer: available files
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Available Files")
    for file in available_files:
        file_path = editor.chapters_dir / file
        if file_path.exists():
            size = file_path.stat().st_size
            st.sidebar.text(f"{file} ({size} bytes)")

if __name__ == "__main__":
    main()