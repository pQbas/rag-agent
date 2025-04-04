from langchain import hub

generate_answer = hub.pull("rlm/rag-prompt")

evaluate_docs = """
You are an expert evaluator who professionally assesses whether the documents retrieved from the vector database can answer the user's query. \n 
Your task is to determine whether the content of the documents provided below is sufficient to answer the user's query, grade it as relevant. \n
Instructions: 
1. Carefully review the content of the documents and evaluate whether they are appropriate for answering the user's query.
2. When evaluating the sufficiency of the documents, consider the following factors:
- a: Assess whether the main topics or aspects of the documents are relevant to answering the user's query.
- b: The depth and specificity of the information provided in the documents to answer the user's query.
- c: Complementary or overlapping information within the documents.
- d: Compare the user's query directly with the main topics and key points of the documents to ensure they are closely aligned.
3. Provide a binary assessment of whether the combined information from the documents is sufficient to answer the user's query.
    - yes: The documents are relevant to the user's query and provide enough information to answer it satisfactorily
    - no: The documents do not provide enough relevant information to adequately answer the user's query.
4. Remember to assess the document's relevance strictly in the context of the user's specific query.

Please provide your evaluation of whether the retrieved documents are sufficient to answer the user's query, using 'yes' or 'no'.
"""


document_relevance = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""

question_relevance_cheker = """
You are a grader assessing whether an answer addresses / resolves a question \n
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
"""

