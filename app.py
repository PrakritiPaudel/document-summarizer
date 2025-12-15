"""
Notes/Document Summarizer with Basic NLP
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import re
from collections import Counter

# Page config
st.set_page_config(
    page_title="Document Summarizer",
    page_icon="📚",
    layout="wide"
)

# Basic imports
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import nltk
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab')
        except:
            nltk.download('punkt')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    
    DEPENDENCIES_OK = True
except ImportError as e:
    st.error(f"Install missing dependencies: pip install nltk")
    DEPENDENCIES_OK = False

class PDOPrompts:
    """PDO (Purpose-Details-Output) Prompt Engineering"""
    
    @staticmethod
    def get_summarization_prompt():
        return """
PURPOSE: You are an expert academic research assistant analyzing study materials and research papers.

DETAILS: 
- Extract key concepts, methodologies, and findings
- Focus on academic rigor and factual accuracy  
- Maintain original context and meaning
- Use clear, structured academic language

Document Context: {context}

Question: {question}

OUTPUT FORMAT:
Provide a comprehensive response that:
1. Directly answers the question using document evidence
2. Cites specific sections when relevant
3. Maintains academic tone and precision
4. Identifies key concepts and relationships

Answer:"""

class RAGSummarizer:
    """Document analysis using basic NLP - No API required"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.documents = []
        self.processed_text = ""
        self.stop_words = set(stopwords.words('english'))
        
    def process_documents(self, uploaded_files):
        """Process uploaded documents using basic text processing"""
        all_text = []
        
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.endswith('.pdf'):
                    # Save temporary file for PDF processing
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        for doc in docs:
                            all_text.append(doc.page_content)
                    finally:
                        os.unlink(tmp_path)
                        
                else:
                    # Handle text files directly
                    text_content = uploaded_file.read().decode('utf-8')
                    all_text.append(text_content)
                    
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
                continue
        
        if not all_text:
            return False
            
        # Combine all text and clean it
        combined_text = "\n\n".join(all_text)
        # Clean up excessive whitespace and normalize text
        self.processed_text = ' '.join(combined_text.split())
        
        # Split into sentences for basic "retrieval"
        raw_sentences = sent_tokenize(self.processed_text)
        # Clean each sentence to remove extra whitespace
        self.documents = [' '.join(sentence.split()) for sentence in raw_sentences if sentence.strip()]
        
        return True
    
    def generate_summary(self, summary_type: str):
        """Generate summary using rule-based approach"""
        if not self.documents:
            return "No documents processed yet."
        
        try:
            if summary_type == "key_points":
                return self._extract_key_points()
            elif summary_type == "methodology":
                return self._extract_methodology()
            elif summary_type == "findings":
                return self._extract_findings()
            else:  # comprehensive
                return self._extract_comprehensive()
                
        except Exception as e:
            return f"Error generating summary: {e}"
    
    def _extract_key_points(self):
        """Extract key points using keyword frequency"""
        # Get most common important words
        words = word_tokenize(self.processed_text.lower())
        words = [w for w in words if w.isalpha() and w not in self.stop_words and len(w) > 3]
        
        word_freq = Counter(words)
        top_words = [word for word, _ in word_freq.most_common(10)]
        
        # Find sentences containing top keywords
        key_sentences = []
        for sentence in self.documents:
            score = sum(1 for word in top_words if word.lower() in sentence.lower())
            if score >= 2:  # Sentence contains at least 2 key terms
                key_sentences.append((score, sentence))
        
        # Sort by relevance and return top sentences
        key_sentences.sort(reverse=True, key=lambda x: x[0])
        
        summary = "**KEY POINTS:**\n\n"
        for i, (_, sentence) in enumerate(key_sentences[:5], 1):
            # Clean and normalize the sentence
            clean_sentence = ' '.join(sentence.strip().split())
            summary += f"{i}. {clean_sentence}\n\n"
        
        return summary
    
    def _extract_methodology(self):
        """Extract methodology-related content"""
        method_keywords = ['method', 'approach', 'technique', 'procedure', 'analysis', 
                          'experiment', 'study', 'research', 'data', 'sample']
        
        method_sentences = []
        for sentence in self.documents:
            score = sum(1 for keyword in method_keywords if keyword in sentence.lower())
            if score >= 1:
                method_sentences.append(sentence)
        
        if not method_sentences:
            return "No clear methodology sections found in the documents."
        
        summary = "**METHODOLOGY:**\n\n"
        for i, sentence in enumerate(method_sentences[:7], 1):
            # Clean and normalize the sentence
            clean_sentence = ' '.join(sentence.strip().split())
            summary += f"• {clean_sentence}\n\n"
        
        return summary
    
    def _extract_findings(self):
        """Extract findings and results"""
        finding_keywords = ['result', 'finding', 'conclusion', 'outcome', 'evidence',
                           'showed', 'demonstrated', 'found', 'discovered', 'revealed']
        
        finding_sentences = []
        for sentence in self.documents:
            score = sum(1 for keyword in finding_keywords if keyword in sentence.lower())
            if score >= 1:
                finding_sentences.append(sentence)
        
        if not finding_sentences:
            return "No clear findings sections found in the documents."
        
        summary = "**FINDINGS & RESULTS:**\n\n"
        for i, sentence in enumerate(finding_sentences[:7], 1):
            # Clean and normalize the sentence
            clean_sentence = ' '.join(sentence.strip().split())
            summary += f"• {clean_sentence}\n\n"
        
        return summary
    
    def _extract_comprehensive(self):
        """Generate comprehensive summary"""
        # Get document statistics
        total_sentences = len(self.documents)
        total_words = len(word_tokenize(self.processed_text))
        
        # Extract most important sentences (combination of position and keywords)
        important_sentences = []
        
        # First and last few sentences are often important
        if total_sentences > 0:
            important_sentences.extend(self.documents[:2])  # First 2
            if total_sentences > 4:
                important_sentences.extend(self.documents[-2:])  # Last 2
        
        # Add key point sentences
        key_points = self._extract_key_points()
        
        summary = f"**COMPREHENSIVE SUMMARY**\n\n"
        summary += f"*Document Statistics: {total_sentences} sentences, {total_words} words*\n\n"
        summary += "**Overview:**\n"
        for sentence in important_sentences[:3]:
            # Clean and normalize the sentence
            clean_sentence = ' '.join(sentence.strip().split())
            summary += f"• {clean_sentence}\n\n"
        
        summary += key_points
        
        return summary
    
    def ask_question(self, question: str):
        """Answer questions using keyword matching"""
        if not self.documents:
            return "No documents processed yet."
        
        try:
            # Extract keywords from question
            question_words = word_tokenize(question.lower())
            question_words = [w for w in question_words if w.isalpha() and w not in self.stop_words]
            
            if not question_words:
                return "Please ask a more specific question with keywords."
            
            # Find sentences that match question keywords
            relevant_sentences = []
            for sentence in self.documents:
                score = sum(1 for word in question_words if word.lower() in sentence.lower())
                if score > 0:
                    relevant_sentences.append((score, sentence))
            
            # Sort by relevance
            relevant_sentences.sort(reverse=True, key=lambda x: x[0])
            
            if not relevant_sentences:
                return "No relevant information found for your question."
            
            # Build answer from most relevant sentences
            answer = f"**Answer to: {question}**\n\n"
            answer += "Based on the document content:\n\n"
            
            for i, (_, sentence) in enumerate(relevant_sentences[:5], 1):
                # Clean and normalize the sentence
                clean_sentence = ' '.join(sentence.strip().split())
                answer += f"{i}. {clean_sentence}\n\n"
            
            return answer
            
        except Exception as e:
            return f"Error answering question: {e}"

def main():
    """Main application"""
    st.title("📚 Document Summarizer")
    
    if not DEPENDENCIES_OK:
        st.error("Please install: `pip install nltk langchain-community`")
        return
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSummarizer()
    
    # File upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files", 
        accept_multiple_files=True,
        type=['pdf', 'txt'],
        help="Upload PDF or TXT files for analysis"
    )
    
    if uploaded_files:
        if st.button("🔄 Process Documents"):
            with st.spinner("Processing documents with local RAG..."):
                success = st.session_state.rag_system.process_documents(uploaded_files)
                if success:
                    st.success(f"Processed {len(uploaded_files)} documents!")
                    st.session_state.documents_processed = True
                else:
                    st.error("Error processing documents")
    
    # Summary generation
    if st.session_state.get('documents_processed', False):
        st.subheader("Generate Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            summary_type = st.selectbox(
                "Summary Type",
                ["comprehensive", "key_points", "methodology", "findings"]
            )
            
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = st.session_state.rag_system.generate_summary(summary_type)
                    st.subheader(f"📋 {summary_type.title()} Summary")
                    st.markdown(summary)
        
        with col2:
            st.subheader("Ask Questions?")
            question = st.text_input("Ask about your documents:")
            
            if question and st.button("🔍 Get Answer"):
                with st.spinner("Searching documents..."):
                    answer = st.session_state.rag_system.ask_question(question)
                    st.subheader("💡 Answer")
                    st.markdown(answer)

if __name__ == "__main__":
    main()