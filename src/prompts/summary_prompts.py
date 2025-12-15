"""
PDO (Purpose-Details-Output) Prompt Templates for Research Paper Summarization.
Specifically designed for sentiment analysis and research paper processing.
"""

from typing import Optional
from langchain.prompts import PromptTemplate

class SummaryPromptTemplates:
    """Collection of PDO-structured prompts for different summary types."""
    
    def __init__(self):
        """Initialize prompt templates following PDO methodology."""
        self._qa_prompt = None
        self._summary_prompts = {}
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize all prompt templates."""
        
        # QA Prompt Template (PDO Structure)
        qa_template = """
        PURPOSE: You are an expert research assistant analyzing academic papers, particularly in sentiment analysis and related fields. Your goal is to provide accurate, comprehensive answers based on the provided context.

        DETAILS: 
        - Use only the information provided in the context below
        - If the answer isn't in the context, state that clearly
        - For sentiment analysis papers, pay special attention to methodologies, datasets, evaluation metrics, and findings
        - Maintain academic tone and precision
        - Cite specific sections when possible

        CONTEXT: {context}

        QUESTION: {question}

        OUTPUT FORMAT:
        Provide a clear, structured answer that:
        1. Directly addresses the question
        2. References specific parts of the paper when relevant
        3. Uses technical terminology appropriately
        4. Maintains objectivity and accuracy

        Answer:"""
        
        self._qa_prompt = PromptTemplate(
            template=qa_template,
            input_variables=["context", "question"]
        )
        
        # Comprehensive Summary Prompt
        comprehensive_template = """
        PURPOSE: Create a comprehensive summary of this research paper, focusing on all major aspects including methodology, findings, and implications.

        DETAILS:
        - Summarize the paper's main contribution to the field
        - Include research objectives, methodology, key findings, and conclusions
        - Highlight novel approaches or significant results
        - Mention limitations and future work suggestions
        - For sentiment analysis papers, emphasize datasets, models, and performance metrics

        CONTEXT: {context}

        OUTPUT FORMAT:
        ## Research Paper Summary

        **Title & Objective:**
        [Main research question and objectives]

        **Methodology:**
        [Approach, datasets, models, experimental setup]

        **Key Findings:**
        [Main results and discoveries]

        **Significance:**
        [Contribution to the field and implications]

        **Limitations & Future Work:**
        [Acknowledged limitations and suggested directions]

        Summary:"""
        
        # Key Points Summary Prompt
        key_points_template = """
        PURPOSE: Extract and present the most important key points from this research paper in a concise, bullet-point format.

        DETAILS:
        - Focus on the most critical insights and contributions
        - Include quantitative results where available
        - Highlight methodological innovations
        - Keep each point concise but informative
        - Prioritize findings that advance the field

        CONTEXT: {context}

        OUTPUT FORMAT:
        ## Key Research Points

        **Main Contributions:**
        • [Key contribution 1]
        • [Key contribution 2]
        • [Key contribution 3]

        **Methodology Highlights:**
        • [Important methodological aspect 1]
        • [Important methodological aspect 2]

        **Significant Results:**
        • [Key finding 1 with metrics if available]
        • [Key finding 2 with metrics if available]

        **Novel Insights:**
        • [New insight or approach 1]
        • [New insight or approach 2]

        Key Points:"""
        
        # Methodology Summary Prompt
        methodology_template = """
        PURPOSE: Provide a detailed summary of the research methodology, experimental design, and technical approach used in this paper.

        DETAILS:
        - Focus on the technical implementation details
        - Include information about datasets, models, and evaluation metrics
        - Explain the experimental setup and validation approach
        - Highlight any novel methodological contributions
        - For sentiment analysis, emphasize data preprocessing, feature extraction, and model architecture

        CONTEXT: {context}

        OUTPUT FORMAT:
        ## Methodology Summary

        **Research Approach:**
        [Overall methodological framework]

        **Data & Preprocessing:**
        [Dataset details, size, preprocessing steps]

        **Model Architecture:**
        [Model design, algorithms used, technical specifications]

        **Experimental Setup:**
        [Training procedure, hyperparameters, validation approach]

        **Evaluation Metrics:**
        [Performance measures and evaluation methodology]

        **Technical Innovations:**
        [Novel methodological contributions]

        Methodology:"""
        
        # Findings & Results Summary Prompt
        findings_template = """
        PURPOSE: Summarize the key findings, results, and conclusions from this research paper with emphasis on quantitative outcomes and their significance.

        DETAILS:
        - Focus on empirical results and statistical findings
        - Include performance metrics, comparisons, and benchmarks
        - Highlight significant improvements or novel discoveries
        - Explain the practical implications of the results
        - Compare with baseline or state-of-the-art methods when mentioned

        CONTEXT: {context}

        OUTPUT FORMAT:
        ## Research Findings & Results

        **Primary Results:**
        [Main experimental outcomes with specific metrics]

        **Performance Metrics:**
        [Quantitative results: accuracy, precision, recall, F1-score, etc.]

        **Comparative Analysis:**
        [How results compare to existing methods/baselines]

        **Statistical Significance:**
        [Significance of improvements and confidence levels]

        **Practical Implications:**
        [Real-world applications and impact]

        **Limitations Identified:**
        [Constraints and areas for improvement]

        Findings:"""
        
        # Store all summary prompts
        self._summary_prompts = {
            'comprehensive': PromptTemplate(
                template=comprehensive_template,
                input_variables=["context"]
            ),
            'key_points': PromptTemplate(
                template=key_points_template,
                input_variables=["context"]
            ),
            'methodology': PromptTemplate(
                template=methodology_template,
                input_variables=["context"]
            ),
            'findings': PromptTemplate(
                template=findings_template,
                input_variables=["context"]
            )
        }
    
    def get_qa_prompt(self) -> PromptTemplate:
        """Get the QA prompt template."""
        return self._qa_prompt
    
    def get_summary_prompt(self, summary_type: str) -> PromptTemplate:
        """
        Get a specific summary prompt template.
        
        Args:
            summary_type: Type of summary ('comprehensive', 'key_points', 'methodology', 'findings')
            
        Returns:
            PromptTemplate for the requested summary type
        """
        if summary_type not in self._summary_prompts:
            raise ValueError(f"Unknown summary type: {summary_type}. Available types: {list(self._summary_prompts.keys())}")
        
        return self._summary_prompts[summary_type]
    
    def get_summary_query(self, summary_type: str, focus_area: Optional[str] = None) -> str:
        """
        Generate a query for document summarization.
        
        Args:
            summary_type: Type of summary to generate
            focus_area: Optional specific focus area
            
        Returns:
            Formatted query string
        """
        base_queries = {
            'comprehensive': "Provide a comprehensive summary of this research paper including methodology, findings, and implications.",
            'key_points': "Extract and list the key points and main contributions of this research paper.",
            'methodology': "Describe the methodology, experimental design, and technical approach used in this research.",
            'findings': "Summarize the key findings, results, and conclusions of this research with specific metrics and outcomes."
        }
        
        query = base_queries.get(summary_type, base_queries['comprehensive'])
        
        if focus_area:
            query += f" Pay special attention to aspects related to {focus_area}."
        
        # Add sentiment analysis specific context if relevant
        query += " If this is a sentiment analysis paper, emphasize the datasets used, model architectures, evaluation metrics (accuracy, precision, recall, F1-score), and comparative performance results."
        
        return query
    
    def get_available_summary_types(self) -> list:
        """Get list of available summary types."""
        return list(self._summary_prompts.keys())
    
    def create_custom_prompt(self, purpose: str, details: str, output_format: str) -> PromptTemplate:
        """
        Create a custom prompt following PDO structure.
        
        Args:
            purpose: The purpose/objective of the prompt
            details: Detailed instructions and context
            output_format: Desired output structure and format
            
        Returns:
            Custom PromptTemplate
        """
        template = f"""
        PURPOSE: {purpose}

        DETAILS: {details}

        CONTEXT: {{context}}

        OUTPUT FORMAT: {output_format}

        Response:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context"]
        )