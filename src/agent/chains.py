from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from src.agent.classes.chain import Chain

load_dotenv()

import src.agent.llm as llm
import src.agent.prompts as prompts


class EvaluateDocs(BaseModel):
    score : str = Field(
        description="Indicates whether the documents are relevant to the question, respond with 'yes' or 'no'"
    )

evaluate_docs = Chain(llm = llm.model,
                      system = prompts.evaluate_docs,
                      human = "Document retrieved: \n\n {document} \n\n Question from user: {question}", 
                      output = EvaluateDocs)



class QuestionRelevance(BaseModel):
    binary_score: bool = Field(
        description="Indicates whether the answer sufficiently addresses the user's question, respond with 'yes' or 'no'"
    )

question_relevance_cheker = Chain(llm     = llm.model,
                                   system = prompts.question_relevance_cheker,
                                   human  = "User's query: \n\n {question} \n\n Generated response: {solution}",
                                   output = QuestionRelevance)



class DocumentRelevance(BaseModel):
    binary_score: bool = Field(
        description="Indicates whether the answer is based on the provided documents, respond with 'yes' or 'no'"
    )

document_relevance = Chain(llm = llm.model,
                           system = prompts.document_relevance,
                           human = "Collection of facts: \n\n {documents} \n\n LLM output: {solution}",
                           output = DocumentRelevance)



