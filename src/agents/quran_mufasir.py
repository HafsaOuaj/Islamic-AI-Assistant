import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag import (
    retrieval,
    VectorDatabase,
)

from ollama import Client
from rag import init_reranker

def build_prompt(question: str, context: str) -> str:
    """
    Build a grounded prompt for Quran + Tafsir question answering.
    """
    return f"""
        You are an Islamic AI assistant specialized in the Quran and classical tafsir.

        STRICT RULES:
        1. Answer ONLY using the provided context.
        2. Do not invent interpretations not present in the tafsir.
        3. If the answer is not clearly supported by the context, say:
        "The retrieved tafsir does not provide enough information to answer this question."
        4. Always cite Surah and Ayah references.
        5. Be respectful, precise, and concise.

        OUTPUT FORMAT:
        - Direct Answer
        - Quran References
        - Tafsir Insights

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """.strip()


class QuranRAG:
    def __init__(
        self,
        docs_path: str = "data/gold/quran_docs.pkl",
        collection_name: str = "quran_vdb",
        vector_db_path: str = "data/gold",
        model_name: str = 'mistral:latest',
        ollama_host: str = "http://localhost:11434",
    ):
        # Vector database
        self.vector_db = VectorDatabase(
            collection_name=collection_name,
            path=vector_db_path,
        )

        # Path to serialized QuranDoc objects
        self.docs_path = docs_path

        # Local LLM client
        self.client = Client(host=ollama_host)

        # Model served by Ollama
        self.model_name = model_name

        # Retrieval reranker
        self.reranker =init_reranker()

    def ask(self, question: str, top_k: int = 1, verbose: bool = True) -> str:
        # 1. Retrieve relevant context
        if verbose:
            print("=" * 80)
            print("STEP 1: RETRIEVING CONTEXT")
            print("=" * 80)

        context = retrieval(
            docs_path=self.docs_path,
            query=question,
            vector_db=self.vector_db,
            top_k=top_k,
            reranker=self.reranker
        )

        if not context.strip():
            return "No relevant Quran or tafsir passages were found."

        if verbose:
            print(f"Retrieved context length: {len(context)} characters")

        # 2. Build prompt
        if verbose:
            print("=" * 80)
            print("STEP 2: BUILDING PROMPT")
            print("=" * 80)

        prompt = build_prompt(question, context)

        if verbose:
            print(f"Prompt length: {len(prompt)} characters")

        # 3. Generate answer
        if verbose:
            print("=" * 80)
            print("STEP 3: GENERATING ANSWER")
            print("=" * 80)
            print(f"Model: {self.model_name}")

        stream = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],stream=True
        )
        full_answer =""
        for chunk in stream:
            token = chunk['message']['content']
            full_answer += token

        print("\n" + "=" * 80)
        print("DONE")
        print("=" * 80)

        if verbose:
            print("=" * 80)
            print("GENERATION COMPLETED")
            print("=" * 80)

        return full_answer


if __name__ == "__main__":
    # Initialize the Quran RAG system
    rag = QuranRAG(
        model_name='mistral:latest',
    )

    # Example question
    question = "What does the Quran say about Allah's mercy?"

    # Ask the system
    answer = rag.ask(
        question=question,
        top_k=1,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(answer)