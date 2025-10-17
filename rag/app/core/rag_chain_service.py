import logging
from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LcDocument
from app.core.llm import AIService
from app.data_models import QueryRequest, RAGResponse, ContextDocument

logger = logging.getLogger("RAGChainService")

def _build_rag_prompt(question: str, context_docs: List[LcDocument]) -> str:
    """
    Constructs the final, engineered prompt combining system instructions, 
    context, and the user's question.
    """
    # 1. Format the retrieved context
    formatted_context = []
    for i, doc in enumerate(context_docs):
        # Extract filename from metadata
        source = doc.metadata.get('filename', 'Unknown Source')
        content = doc.page_content.strip()
        formatted_context.append(f"--- Document {i+1} (Source: {source}) ---\n{content}\n")

    context_str = "\n".join(formatted_context)

    # 2. Define the system prompt
    system_instruction = (
        "You are an expert HR Policy Assistant. Use ONLY the provided context "
        "documents below to answer the user's question. "
        "If the context does not contain the answer, state clearly that you cannot "
        "find the answer in the provided policies. Do not use external knowledge or invent facts. "
        "Cite the document source(s) (e.g., Source: [filename]) for the information you use."
    )

    # 3. Combine everything into the final prompt
    final_prompt = (
        f"{system_instruction}\n\n"
        f"----------------------------------------\n"
        f"CONTEXT DOCUMENTS:\n\n{context_str}\n"
        f"----------------------------------------\n"
        f"USER QUESTION: {question}"
    )
    
    return final_prompt

async def process_rag_query(
    request: QueryRequest,
    retriever: BaseRetriever,
    ai_service: AIService
) -> RAGResponse:
    """
    Executes the full RAG pipeline: retrieve, prompt, generate.
    """
    try:
        # 1. Retrieve relevant documents (embed and retrieve)
        logger.info(f"Retrieving {request.k} documents for query: {request.query[:50]}...")
        # Note: LangChain's BaseRetriever's get_relevant_documents is synchronous.
        context_docs: List[LcDocument] = retriever.invoke(
            request.query, 
            k=request.k
        )
        logger.info(f"Retrieved {len(context_docs)} documents.")

        # 2. Prompt Engineering
        final_prompt = _build_rag_prompt(request.query, context_docs)
        
        # 3. LLM Generation
        logger.info("Sending prompt to LLM for final answer generation.")
        llm_result = ai_service.query(final_prompt)

        # 4. Format Output (RAGResponse)
        if llm_result.get("error"):
            raise RuntimeError(f"LLM generation failed: {llm_result['error']}")

        formatted_context = [
            ContextDocument(
                page_content=doc.page_content,
                source=doc.metadata.get('filename', 'Unknown'),
                score=doc.metadata.get('relevance_score') # Score may not be exposed by PDR/Chroma
            )
            for doc in context_docs
        ]
        
        # Determine model name (safely access _client attribute)
        model_name = getattr(ai_service._client, 'model', 'Unknown LLM')

        return RAGResponse(
            answer=llm_result["answer"],
            context=formatted_context,
            model_name=model_name,
            response_time_seconds=llm_result["response_time"],
            tokens_estimated=llm_result["tokens"],
            error=None
        )

    except Exception as e:
        logger.error(f"RAG pipeline failure: {e}")
        # Return an error response if anything fails
        return RAGResponse(
            answer="I am currently unable to process your request due to a system error.",
            context=[],
            model_name=getattr(ai_service._client, 'model', 'Unknown LLM'),
            response_time_seconds=0.0,
            tokens_estimated=0,
            error=str(e)
        )
