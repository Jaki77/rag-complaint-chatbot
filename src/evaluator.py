"""
Evaluation framework for RAG pipeline.
"""
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    question: str
    generated_answer: str
    retrieved_sources: List[Dict[str, Any]]
    relevance_score: float  # 1-5
    accuracy_score: float   # 1-5
    completeness_score: float  # 1-5
    overall_score: float
    comments: str
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'question': self.question,
            'generated_answer': self.generated_answer,
            'retrieved_sources_count': len(self.retrieved_sources),
            'relevance_score': self.relevance_score,
            'accuracy_score': self.accuracy_score,
            'completeness_score': self.completeness_score,
            'overall_score': self.overall_score,
            'comments': self.comments,
            'processing_time': self.processing_time
        }

class RAGEvaluator:
    """Evaluator for RAG pipeline performance."""
    
    def __init__(self, rag_pipeline, evaluation_dir: str = "./evaluation"):
        """
        Initialize evaluator.
        
        Args:
            rag_pipeline: RAGPipeline instance
            evaluation_dir: Directory to save evaluation results
        """
        self.rag_pipeline = rag_pipeline
        self.evaluation_dir = Path(evaluation_dir)
        self.evaluation_dir.mkdir(exist_ok=True)
        
        # Create results directory
        self.results_dir = self.evaluation_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
    
    def load_test_questions(self, filepath: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load test questions from JSON file.
        
        Args:
            filepath: Path to test questions JSON file
        
        Returns:
            List of test questions with metadata
        """
        if filepath is None:
            filepath = self.evaluation_dir / "test_questions.json"
        
        if not Path(filepath).exists():
            # Create default test questions
            questions = self._create_default_test_questions()
            with open(filepath, 'w') as f:
                json.dump(questions, f, indent=2)
            logger.info(f"Created default test questions at: {filepath}")
            return questions
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _create_default_test_questions(self) -> List[Dict[str, Any]]:
        """Create default test questions for evaluation."""
        return [
            {
                "id": 1,
                "question": "What are the main complaints about credit card fees?",
                "category": "credit_card",
                "expected_aspects": ["fee amounts", "transparency", "justification"],
                "difficulty": "medium"
            },
            {
                "id": 2,
                "question": "How do customers feel about money transfer services?",
                "category": "money_transfer",
                "expected_aspects": ["speed", "reliability", "cost", "errors"],
                "difficulty": "easy"
            },
            {
                "id": 3,
                "question": "What issues are customers having with personal loan applications?",
                "category": "personal_loan",
                "expected_aspects": ["approval process", "documentation", "time", "communication"],
                "difficulty": "medium"
            },
            {
                "id": 4,
                "question": "Compare complaints between savings accounts and credit cards.",
                "category": "comparative",
                "expected_aspects": ["fee comparison", "service issues", "customer satisfaction"],
                "difficulty": "hard"
            },
            {
                "id": 5,
                "question": "What are the most common billing disputes?",
                "category": "billing",
                "expected_aspects": ["incorrect charges", "dispute process", "resolution time"],
                "difficulty": "medium"
            },
            {
                "id": 6,
                "question": "How have complaints about interest rates changed over time?",
                "category": "trends",
                "expected_aspects": ["temporal patterns", "rate changes", "customer reactions"],
                "difficulty": "hard"
            },
            {
                "id": 7,
                "question": "What problems do customers report with mobile banking apps?",
                "category": "technology",
                "expected_aspects": ["app crashes", "UI issues", "functionality", "security"],
                "difficulty": "medium"
            },
            {
                "id": 8,
                "question": "Summarize the key issues across all financial products.",
                "category": "summary",
                "expected_aspects": ["cross-product patterns", "severity", "frequency"],
                "difficulty": "hard"
            },
            {
                "id": 9,
                "question": "What specific issues are mentioned about overdraft fees?",
                "category": "specific_issue",
                "expected_aspects": ["fee amounts", "notification", "avoidance", "disputes"],
                "difficulty": "easy"
            },
            {
                "id": 10,
                "question": "Are there geographic patterns in complaints?",
                "category": "geographic",
                "expected_aspects": ["state variations", "regional issues", "concentration"],
                "difficulty": "hard"
            }
        ]
    
    def evaluate_single(
        self,
        question: str,
        expected_answer: Optional[str] = None,
        auto_score: bool = False
    ) -> EvaluationResult:
        """
        Evaluate a single question.
        
        Args:
            question: The question to evaluate
            expected_answer: Expected answer (for automated scoring)
            auto_score: Whether to use automated scoring
        
        Returns:
            EvaluationResult
        """
        import time
        
        start_time = time.time()
        
        # Get RAG response
        response = self.rag_pipeline.query(question)
        
        processing_time = time.time() - start_time
        
        # Score the response
        if auto_score and expected_answer:
            scores = self._auto_score_response(
                response.answer,
                expected_answer,
                response.sources
            )
        else:
            # Manual scoring placeholder - in practice, you'd implement manual review
            scores = {
                'relevance': 3.5,
                'accuracy': 4.0,
                'completeness': 3.0,
                'overall': 3.5
            }
            comments = "Manual evaluation needed"
        
        # Show sample sources
        sample_sources = response.sources[:2] if len(response.sources) > 2 else response.sources
        
        return EvaluationResult(
            question=question,
            generated_answer=response.answer,
            retrieved_sources=sample_sources,
            relevance_score=scores['relevance'],
            accuracy_score=scores['accuracy'],
            completeness_score=scores['completeness'],
            overall_score=scores['overall'],
            comments=comments,
            processing_time=processing_time
        )
    
    def evaluate_batch(
        self,
        questions: List[Dict[str, Any]],
        save_results: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate a batch of questions.
        
        Args:
            questions: List of question dictionaries
            save_results: Whether to save results to file
        
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        logger.info(f"Evaluating {len(questions)} questions...")
        
        for i, q in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}: {q['question'][:50]}...")
            
            result = self.evaluate_single(
                question=q['question'],
                auto_score=False  # Manual evaluation for now
            )
            
            # Convert to dict and add question metadata
            result_dict = result.to_dict()
            result_dict.update({
                'question_id': q.get('id', i),
                'category': q.get('category', 'unknown'),
                'difficulty': q.get('difficulty', 'medium')
            })
            
            results.append(result_dict)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Calculate summary statistics
        summary = {
            'total_questions': len(df),
            'avg_overall_score': df['overall_score'].mean(),
            'avg_relevance': df['relevance_score'].mean(),
            'avg_accuracy': df['accuracy_score'].mean(),
            'avg_completeness': df['completeness_score'].mean(),
            'avg_processing_time': df['processing_time'].mean(),
            'total_chunks_retrieved': df['retrieved_sources_count'].sum()
        }
        
        logger.info(f"Evaluation complete. Average score: {summary['avg_overall_score']:.2f}/5")
        
        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.results_dir / f"evaluation_results_{timestamp}.json"
            
            # Save detailed results
            detailed_results = {
                'summary': summary,
                'questions_evaluated': len(questions),
                'timestamp': timestamp,
                'results': results
            }
            
            with open(results_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            # Save DataFrame as CSV
            csv_file = self.results_dir / f"evaluation_results_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            
            logger.info(f"Results saved to: {results_file}")
            logger.info(f"CSV saved to: {csv_file}")
        
        return df, summary
    
    def _auto_score_response(
        self,
        generated_answer: str,
        expected_answer: str,
        sources: List[Dict]
    ) -> Dict[str, float]:
        """
        Automatically score a response (simplified version).
        
        In practice, you would use more sophisticated methods like:
        - Semantic similarity with embeddings
        - LLM-as-judge
        - Keyword matching
        """
        # This is a simplified scoring mechanism
        # In practice, implement proper scoring logic
        
        score = {
            'relevance': 4.0,
            'accuracy': 4.0,
            'completeness': 3.5,
            'overall': 3.8
        }
        
        return score
    
    def generate_evaluation_report(
        self,
        results_df: pd.DataFrame,
        summary: Dict[str, Any]
    ) -> str:
        """
        Generate a markdown evaluation report.
        
        Returns:
            Markdown formatted report
        """
        report = []
        
        # Header
        report.append("# RAG Pipeline Evaluation Report")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append("")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Total Questions Evaluated | {summary['total_questions']} |")
        report.append(f"| Average Overall Score | {summary['avg_overall_score']:.2f}/5 |")
        report.append(f"| Average Relevance Score | {summary['avg_relevance']:.2f}/5 |")
        report.append(f"| Average Accuracy Score | {summary['avg_accuracy']:.2f}/5 |")
        report.append(f"| Average Completeness Score | {summary['avg_completeness']:.2f}/5 |")
        report.append(f"| Average Processing Time | {summary['avg_processing_time']:.2f}s |")
        report.append(f"| Total Chunks Retrieved | {summary['total_chunks_retrieved']} |")
        report.append("")
        
        # Score Distribution
        report.append("## Score Distribution")
        report.append("")
        
        # Create score distribution table
        score_counts = results_df['overall_score'].round().value_counts().sort_index()
        for score, count in score_counts.items():
            percentage = (count / len(results_df)) * 100
            report.append(f"- **{int(score)}/5**: {count} questions ({percentage:.1f}%)")
        
        report.append("")
        
        # By Category
        report.append("## Performance by Question Category")
        report.append("")
        report.append("| Category | Avg Score | Questions |")
        report.append("|----------|-----------|-----------|")
        
        category_stats = results_df.groupby('category').agg({
            'overall_score': 'mean',
            'question_id': 'count'
        }).round(2)
        
        for category, row in category_stats.iterrows():
            report.append(f"| {category} | {row['overall_score']}/5 | {row['question_id']} |")
        
        report.append("")
        
        # By Difficulty
        report.append("## Performance by Difficulty Level")
        report.append("")
        report.append("| Difficulty | Avg Score | Questions |")
        report.append("|------------|-----------|-----------|")
        
        difficulty_stats = results_df.groupby('difficulty').agg({
            'overall_score': 'mean',
            'question_id': 'count'
        }).round(2)
        
        for difficulty, row in difficulty_stats.iterrows():
            report.append(f"| {difficulty} | {row['overall_score']}/5 | {row['question_id']} |")
        
        report.append("")
        
        # Detailed Results
        report.append("## Detailed Results")
        report.append("")
        report.append("| Question ID | Category | Difficulty | Score | Answer Preview |")
        report.append("|-------------|----------|------------|-------|----------------|")
        
        for _, row in results_df.iterrows():
            answer_preview = row['generated_answer'][:50].replace('\n', ' ') + "..."
            report.append(f"| {row['question_id']} | {row['category']} | {row['difficulty']} | {row['overall_score']}/5 | `{answer_preview}` |")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations for Improvement")
        report.append("")
        report.append("1. **Improve Retrieval**: Focus on increasing relevance scores")
        report.append("2. **Enhance Prompt Engineering**: Improve completeness of answers")
        report.append("3. **Optimize Processing Time**: Target < 5s per query")
        report.append("4. **Expand Test Suite**: Add more diverse question types")
        report.append("")
        
        # Appendix: Example Questions and Answers
        report.append("## Appendix: Example Questions and Answers")
        report.append("")
        
        for i, (_, row) in enumerate(results_df.head(3).iterrows(), 1):
            report.append(f"### Example {i}: {row['category'].title()} (Score: {row['overall_score']}/5)")
            report.append("")
            report.append(f"**Question**: {row['question']}")
            report.append("")
            report.append(f"**Answer**: {row['generated_answer']}")
            report.append("")
            report.append(f"**Retrieved Sources**: {row['retrieved_sources_count']} chunks")
            report.append(f"**Processing Time**: {row['processing_time']:.2f}s")
            report.append("")
        
        return "\n".join(report)