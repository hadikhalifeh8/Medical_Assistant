import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_llm_chain(retriever):
    """Create and return a RetrievalQA chain using ChatGroq.

    Langchain-related imports are performed lazily so the app can start
    even when incompatible langchain versions are installed. If the
    environment lacks the expected langchain layout, this raises a
    helpful ModuleNotFoundError with instructions to install the
    project's compatible versions.
    """
    # Try multiple import paths to support different langchain versions.
    prompt_exc = chains_exc = groq_exc = None
    PromptTemplate = None
    RetrievalQA = None
    ChatGroq = None

    try:
        from langchain.prompts.prompt import PromptTemplate
    except Exception as e:
        prompt_exc = e
        try:
            from langchain.prompts import PromptTemplate
        except Exception as e2:
            prompt_exc = e2
            try:
                from langchain import PromptTemplate
            except Exception as e3:
                prompt_exc = e3

    try:
        from langchain.chains import RetrievalQA
    except Exception as e:
        chains_exc = e
        try:
            # Older/newer variants may expose QA via question_answering
            from langchain.chains.question_answering import RetrievalQA
        except Exception as e2:
            chains_exc = e2

    try:
        from langchain.groq import ChatGroq
    except Exception as e:
        groq_exc = e
        try:
            from langchain_groq import ChatGroq
        except Exception as e2:
            groq_exc = e2

    if PromptTemplate is None or RetrievalQA is None or ChatGroq is None:
        raise ModuleNotFoundError(
            "Missing or incompatible langchain components. \n"
            "PromptTemplate error: %s\n" % repr(prompt_exc)
            + "RetrievalQA error: %s\n" % repr(chains_exc)
            + "ChatGroq error: %s\n" % repr(groq_exc)
            + "\nTry installing compatible versions, for example:\n"
            "pip install 'langchain==0.3.27' 'langchain-groq==0.3.8' 'groq==0.36.0'"
        )

    # ChatGroq should not be passed unknown kwargs that are forwarded to
    # the underlying client (which raises errors). The GROQ API key is
    # read from the environment by the Groq client, so pass only the
    # supported parameters here.
    model_name = os.getenv("GROQ_MODEL_NAME")
    try:
        llm = ChatGroq(model_name=model_name)
    except Exception as e:
        msg = str(e)
        if "model_not_found" in msg or "does not exist" in msg or "not found" in msg:
            raise RuntimeError(
                f"Groq model '{model_name}' not found or not accessible.\n"
                "Confirm the model name and that your Groq account has access to it.\n"
                "Set a different model name with the env var `GROQ_MODEL_NAME` or check your Groq dashboard for available models.\n"
                f"Original error: {msg}"
            ) from e
        raise

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        you are a **Medicalbot**, an AI Powered Assistant to help users to understand medical documents
        and health-related questions.
        
        your job is to provide clear, accurate, and helpful responses based **only on the provided
        context**.
        
        ---
        
            🔍 **Context**:
            {context}

            🙋‍♂️ **User Question**:
            {question}

            ---

            💬 **Answer**:
            - Respond in a calm, factual, and respectful tone.
            - Use simple explanations when needed.
            - If the context does not contain the answer, say: "I'm sorry, but I couldn't find relevant information in the provided documents."
            - Do NOT make up facts.
            - Do NOT give medical advice or diagnoses.
        """
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
