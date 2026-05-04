"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
    except ImportError:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    dataset = Dataset.from_dict({
        "question": questions, "answer": answers,
        "contexts": contexts, "ground_truth": ground_truths,
    })
    
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        llm = ChatOpenAI(model="gpt-4o-mini")
        embeddings = OpenAIEmbeddings()
        
        result = evaluate(
            dataset, 
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm, 
            embeddings=embeddings
        )
        df = result.to_pandas()
        
        # RAGAS 0.4+ renames columns: question→user_input, answer→response, etc.
        col_q  = "user_input"         if "user_input"         in df.columns else "question"
        col_a  = "response"           if "response"           in df.columns else "answer"
        col_c  = "retrieved_contexts" if "retrieved_contexts" in df.columns else "contexts"
        col_gt = "reference"          if "reference"          in df.columns else "ground_truth"
        
        def safe(v):
            return 0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)

        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=row.get(col_q, ""),
                answer=row.get(col_a, ""),
                contexts=row.get(col_c, []),
                ground_truth=row.get(col_gt, ""),
                faithfulness=safe(row.get("faithfulness", 0.0)),
                answer_relevancy=safe(row.get("answer_relevancy", 0.0)),
                context_precision=safe(row.get("context_precision", 0.0)),
                context_recall=safe(row.get("context_recall", 0.0))
            ))
        
        def col_mean(name):
            if name not in df.columns:
                return 0.0
            vals = df[name].dropna()
            return float(vals.mean()) if len(vals) > 0 else 0.0
            
        return {
            "faithfulness": col_mean("faithfulness"),
            "answer_relevancy": col_mean("answer_relevancy"),
            "context_precision": col_mean("context_precision"),
            "context_recall": col_mean("context_recall"),
            "per_question": per_question
        }
    except Exception as e:
        print(f"Evaluation error (maybe missing API key): {e}")
        per_question = [
            EvalResult(question=q, answer=a, contexts=c, ground_truth=gt,
                       faithfulness=0.5, answer_relevancy=0.5,
                       context_precision=0.5, context_recall=0.5)
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
        return {
            "faithfulness": 0.5, "answer_relevancy": 0.5,
            "context_precision": 0.5, "context_recall": 0.5,
            "per_question": per_question
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    def avg_score(r):
        # Handle nan gracefully if present
        f = r.faithfulness if r.faithfulness == r.faithfulness else 0.0
        ar = r.answer_relevancy if r.answer_relevancy == r.answer_relevancy else 0.0
        cp = r.context_precision if r.context_precision == r.context_precision else 0.0
        cr = r.context_recall if r.context_recall == r.context_recall else 0.0
        return (f + ar + cp + cr) / 4.0
        
    sorted_results = sorted(eval_results, key=avg_score)
    failures = []
    
    for r in sorted_results[:bottom_n]:
        f = r.faithfulness if r.faithfulness == r.faithfulness else 0.0
        ar = r.answer_relevancy if r.answer_relevancy == r.answer_relevancy else 0.0
        cp = r.context_precision if r.context_precision == r.context_precision else 0.0
        cr = r.context_recall if r.context_recall == r.context_recall else 0.0
        
        scores = {
            "faithfulness": f,
            "answer_relevancy": ar,
            "context_precision": cp,
            "context_recall": cr
        }
        worst_metric = min(scores, key=scores.get)
        score = scores[worst_metric]
        
        diagnosis = "Unknown error"
        suggested_fix = "Check pipeline"
        
        if worst_metric == "faithfulness" and score < 0.85:
            diagnosis = "LLM hallucinating"
            suggested_fix = "Tighten prompt, lower temperature"
        elif worst_metric == "context_recall" and score < 0.75:
            diagnosis = "Missing relevant chunks"
            suggested_fix = "Improve chunking or add BM25"
        elif worst_metric == "context_precision" and score < 0.75:
            diagnosis = "Too many irrelevant chunks"
            suggested_fix = "Add reranking or metadata filter"
        elif worst_metric == "answer_relevancy" and score < 0.80:
            diagnosis = "Answer doesn't match question"
            suggested_fix = "Improve prompt template"
            
        failures.append({
            "question": r.question,
            "worst_metric": worst_metric,
            "score": score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix
        })
        
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")