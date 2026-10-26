"""
main.py — Entry point for the Simple RAG Pipeline.

Usage examples:
  python main.py reset
  python main.py add --docs sample_data/source/doc1.pdf sample_data/source/doc2.pdf
  python main.py query --question "Who founded Swan Lagoon?"
  python main.py evaluate
  python main.py interactive
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Add src/ to path so imports resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.impl.datastore import Datastore
from src.impl.evaluator import Evaluator
from src.impl.indexer import Indexer
from src.impl.response_generator import ResponseGenerator
from src.impl.retriever import Retriever
from src.rag_pipeline import RAGPipeline


def build_pipeline(top_k: int = 3) -> RAGPipeline:
    datastore = Datastore()
    return RAGPipeline(
        datastore=datastore,
        indexer=Indexer(),
        retriever=Retriever(datastore),
        response_generator=ResponseGenerator(),
        top_k=top_k,
    )


def cmd_reset(args, pipeline: RAGPipeline):
    pipeline.reset()


def cmd_add(args, pipeline: RAGPipeline):
    if not args.docs:
        print("❌ No documents provided. Use --docs path/to/file.pdf ...")
        sys.exit(1)
    pipeline.add_documents(args.docs)


def cmd_query(args, pipeline: RAGPipeline):
    if not args.question:
        print("[ERROR] No question provided. Use --question \"your question\"")
        sys.exit(1)
    answer = pipeline.process_query(args.question)
    print(f"\nAnswer:\n{answer}")


def cmd_evaluate(args, pipeline: RAGPipeline):
    evaluator = Evaluator()
    results = evaluator.run_evaluation(pipeline.process_query)

    if args.output:
        import json
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in results], f, indent=2)
        print(f"\n📁 Results saved to {args.output}")


def cmd_interactive(args, pipeline: RAGPipeline):
    print("🤖 RAG Pipeline — Interactive Mode  (type 'exit' to quit)\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question or question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        answer = pipeline.process_query(question)
        print(f"\nBot: {answer}\n")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="rag",
        description="Simple RAG Pipeline CLI",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per query (default: 3)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # reset
    subparsers.add_parser("reset", help="Drop and recreate the vector datastore.")

    # add
    add_parser = subparsers.add_parser("add", help="Index and store documents.")
    add_parser.add_argument(
        "--docs",
        nargs="+",
        metavar="FILE",
        help="Paths to documents to index (PDF).",
    )

    # query
    query_parser = subparsers.add_parser("query", help="Ask a single question.")
    query_parser.add_argument(
        "--question",
        type=str,
        help="The question to answer.",
    )

    # evaluate
    eval_parser = subparsers.add_parser(
        "evaluate", help="Run evaluation suite against sample_questions.json."
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Optional path to save evaluation results as JSON.",
    )

    # interactive
    subparsers.add_parser("interactive", help="Start an interactive Q&A session.")

    args = parser.parse_args()
    pipeline = build_pipeline(top_k=args.top_k)

    commands = {
        "reset": cmd_reset,
        "add": cmd_add,
        "query": cmd_query,
        "evaluate": cmd_evaluate,
        "interactive": cmd_interactive,
    }

    commands[args.command](args, pipeline)


if __name__ == "__main__":
    main()
