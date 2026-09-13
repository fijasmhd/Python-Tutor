import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import chromadb
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph,END
import re

load_dotenv()

class AgentState(TypedDict):
    question: str
    rewritten_question: str
    documents: list[str]
    answer: str
    grade: str
    retry_count: int


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.environ.get("API_KEY")
)

generation_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.environ.get("API_KEY")
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

def check_relevance(state: AgentState) -> AgentState:
    """
    Determine whether the question belongs to the Python/ML domain.
    """

    question = state["question"]

    prompt = f"""
You are a classifier.

Determine whether the following question is related to:

- Python programming
- Machine Learning
- Artificial Intelligence
- Data Science
- Software Development
- Programming concepts

Question:
{question}

Reply ONLY with:

YES
or
NO
"""

    try:
        response = llm.invoke([
            HumanMessage(content=prompt)
        ])

        result = response.content.strip().upper()

    except Exception:
        result = "NO"

    if result == "NO":
        return {
            **state,
            "answer": "Sorry! I am a Python and ML tutor.",
            "grade": "FAIL"
        }

    return state

def retrieve(state: AgentState)-> AgentState:
    """Search ChromaDB for documents relevant to the current question."""
    question=state.get("rewritten_question") or state["question"]
    print(f"\n[RETRIEVE] Searching for: {question}")
    try:
        results=collection.query(
            query_texts=[question],
            n_results=3
        )
        documents=results["documents"][0]
        distances = results["distances"][0]
        if distances[0] > 1.2:
            documents = []
        print(f"[RETRIEVE] Found {len(documents)} chunks.")
    except Exception as error:
        print(f"[RETRIEVE] Chromadb query failed: {error}")
        documents=[]
    
    return {**state, "documents": documents}


def generate(state: AgentState) -> AgentState:
    question = state.get("rewritten_question") or state["question"]
    documents = state.get("documents", [])
    print(f"\n[GENERATE] Generating answer for:{question}")

    if not documents:
        return{**state,"answer":"No relavant documents were found."}
    
    context = "\n\n".join(documents)

    system_prompt = """
    You are a helpful assistant that answers questions based strictly on the provided documents.
    Do NOT mention document numbers, document names, sources, chunks, or references in your answer.
    Answer naturally as if you already know the information.
    If the documents do not contain enough information, say:
    'I don't know based on the available documents.'
    """

    user_prompt = f"""Documents:
    {context}
    Question:{question}
    Answer based only the documents above:"""

    try:
        response=generation_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        answer = re.sub(
        r"<think>.*?</think>",
        "",
        response.content,
        flags=re.DOTALL
        ).strip()
        print(f"[GENERATE] Answer Generated ({len(answer)} chars).")
    except Exception as error:
        print(f"[GENERATE] LLM Call failed: {error}")
        answer = "An error occured while generating the answer."
    return{**state, "answer":answer}

def grade_answer(state: AgentState) -> AgentState:
    """Grade whether the answer is grounded in the retrieved documents."""
    question = state["question"]
    documents = state.get("documents", [])
    answer = state.get("answer", "")
    if "i don't know" in answer.lower():
        print("[GRADE] Auto FAIL")
        return {**state, "grade": "FAIL"}    
    print(f"\n[GRADE] Evaluating answer quality...")

    context = "\n\n".join(documents) if documents else "No documents."

    grading_prompt = f"""
You are a strict evaluator.

Question:
{question}

Retrieved Context:
{context}

Answer:
{answer}

Rules:

PASS only if:
- The answer directly answers the question.
- The answer is supported by the context.
- The answer is relevant.

FAIL if:
- The answer says "I don't know".
- The answer avoids the question.
- The answer is unrelated.
- The answer introduces unsupported facts.

Reply ONLY:

PASS

or

FAIL
"""

    try:
        response = llm.invoke([
            HumanMessage(content=grading_prompt)
        ])

        grade = response.content.strip().upper()

        print(f"[GRADE] Result: {grade}")

    except Exception as error:
        print(f"[GRADE] Error: {error}")
        grade = "FAIL"

    return {**state, "grade": grade}


def rewrite_question(state: AgentState) -> AgentState:
    """Rewrite the question to improve retrieval."""

    question = state["question"]

    print("\n[REWRITE] Rewriting question...")

    prompt = f"""
Rewrite the following question to improve document retrieval.

Question:
{question}

Return only the rewritten question.
"""

    try:
        response = llm.invoke([
            HumanMessage(content=prompt)
        ])

        rewritten = response.content.strip()

    except Exception as error:
        print(f"[REWRITE] Error: {error}")
        rewritten = question

    return {
        **state,
        "rewritten_question": rewritten,
        "retry_count": state.get("retry_count", 0) + 1
    }


def decide_next_step(state: AgentState):
    if state["grade"] == "PASS":
        return "end"

    if state.get("retry_count", 0) >= 2:
        return "end"
    return "rewrite"


graph = StateGraph(AgentState)
graph.add_node("relevance", check_relevance)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("grade", grade_answer)
graph.add_node("rewrite", rewrite_question)
graph.set_entry_point("relevance")
graph.add_edge("relevance", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "grade")
graph.add_conditional_edges(
    "grade",
    decide_next_step,
    {
        "end": END,
        "rewrite": "rewrite"
    }
)
graph.add_edge("rewrite", "retrieve")
app = graph.compile()


if __name__ == "__main__":
    question = input("Ask a question: ")
    result = app.invoke(
        {
            "question": question,
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "grade": "",
            "retry_count": 0,
        }
    )
    print("\nFINAL ANSWER")
    print("-" * 50)

    if result["grade"] == "PASS":
        print(result["answer"])
    else:
        print("Sorry! I am your Python assistant, this doesn't look like what I am trained in.")



def ask_question(question: str):

    result = app.invoke(
        {
            "question": question,
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "grade": "",
            "retry_count": 0,
        }
    )

    answer = result.get("answer", "")

    if (
        result.get("grade") == "PASS"
        and "i don't know" not in answer.lower()
    ):
        return answer

    return "Sorry! I am your Python assistant, this doesn't look like what I am trained in."