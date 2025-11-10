"""
Agentic RAG Module - Crew.AI with Ollama via ChatOpenAI
Two agents: Researcher (validates facts) and Finisher (formats response)
"""

import logging

from crewai import LLM, Agent, Crew, Task

from app.core.config import GENERATION_MODEL, OLLAMA_HOST

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_llm(model):
    """Create LLM for CrewAI using CrewAI LLM pointing to Ollama"""
    return LLM(model=f"ollama/{model or GENERATION_MODEL}", base_url=OLLAMA_HOST)


def create_researcher_agent(llm):
    """Agent that validates facts from retrieved documents"""
    return Agent(
        role="Fact Researcher",
        goal="Validate information from retrieved documents",
        backstory="You analyze documents to verify facts and ensure accuracy. You cross-check claims against sources.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def create_finisher_agent(llm):
    """Agent that formats response with citations"""
    return Agent(
        role="Response Finisher",
        goal="Create well-formatted answer with citations",
        backstory="You format validated facts into comprehensive answers with proper citations [1][2][3].",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def format_retrieved_docs(docs):
    """Format documents for agent context with metadata"""
    if not docs:
        return "No documents available"

    formatted = []
    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "")
        score = doc.get("score", 0.0)
        doc_id = doc.get("document_id", doc.get("metadata", {}).get("id", i))
        source = doc.get(
            "source", doc.get("metadata", {}).get("source_file", "unknown")
        )

        formatted.append(
            f"Document {i} [ID: {doc_id}, Source: {source}, Relevance: {score:.2f}]:\n{text}"
        )

    return "\n\n" + "=" * 50 + "\n\n".join(formatted)


def process_with_agents(model, user_query, retrieved_docs):
    """Process query with CrewAI agents"""
    logger.info(f"[AGENT] Starting agent processing for query: {user_query[:50]}...")

    if not retrieved_docs:
        logger.warning("[AGENT] No documents retrieved")
        return {
            "response": "No relevant information found.",
            "query": user_query,
            "sources": [],
            "count": 0,
        }

    try:
        logger.info(f"[AGENT] Retrieved {len(retrieved_docs)} documents")
        logger.info("[AGENT] Step 1: Initializing LLM...")
        llm = create_llm(model)

        logger.info("[AGENT] Step 2: Formatting context from retrieved documents...")
        context = format_retrieved_docs(retrieved_docs)

        logger.info("[AGENT] Step 3: Creating agents...")
        researcher = create_researcher_agent(llm)
        finisher = create_finisher_agent(llm)

        logger.info("[AGENT] Step 4: Creating tasks...")
        research_task = Task(
            description=f"Validate facts for: '{user_query}'\n\nDocuments:\n{context}",
            agent=researcher,
            expected_output="Validated facts summary",
        )

        # Extract unique source file names for finisher context
        source_files = []
        seen_sources = set()
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.get(
                "source", doc.get("metadata", {}).get("source_file", "unknown")
            )
            if source != "unknown" and source not in seen_sources:
                source_files.append(f"[{i}] Source: {source}")
                seen_sources.add(source)
        sources_info = (
            "\n".join(source_files) if source_files else "No sources available"
        )

        finish_task = Task(
            description=f"""Create final answer for: '{user_query}' with citations.

Source files used:
{sources_info}

At the end of your answer, mention which source files you referenced.""",
            agent=finisher,
            expected_output="Complete answer with citations [1][2][3] and mention source file names at the end",
            context=[research_task],
        )

        logger.info(
            "[AGENT] Step 5: Running Crew with Researcher and Finisher agents..."
        )
        crew = Crew(
            agents=[researcher, finisher],
            tasks=[research_task, finish_task],
            verbose=True,
        )

        logger.info("[AGENT] Researcher agent: Validating facts...")
        result = crew.kickoff()
        logger.info("[AGENT] Finisher agent: Formatting response with citations...")
        logger.info("[AGENT] Agent processing completed successfully")

        # Format response with metadata and source references
        sources = [
            {
                "id": str(d.get("document_id", i)),
                "text": d["text"][:200],
                "score": d["score"],
                "source": d.get("source", "unknown"),
                "metadata": d.get("metadata", {}),
            }
            for i, d in enumerate(retrieved_docs[:3], 1)
        ]

        return {
            "response": str(result),
            "query": user_query,
            "sources": sources,
            "count": len(sources),
        }

    except Exception as e:
        logger.error(f"[AGENT] Error during processing: {e}", exc_info=True)

        # Fallback response
        response_text = (
            f"Based on retrieved information: {retrieved_docs[0]['text'][:300]}..."
        )
        sources = [
            {"id": str(i), "text": d["text"][:200], "score": d["score"]}
            for i, d in enumerate(retrieved_docs[:3], 1)
        ]

        return {
            "response": response_text,
            "query": user_query,
            "sources": sources,
            "count": len(sources),
            "error": str(e),
        }
