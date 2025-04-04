__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver

from typing import List, TypedDict
from langgraph.graph import END, StateGraph
from src.classes.chain import Chain
import src.agent.llm as llm
import src.prompts as prompts

class GraphState(TypedDict):
    question: str
    solution: str
    online_search: bool
    documents: List[str]


class Evaluate:
    def __init__(self):
        self.evaluate_docs = Chain(
                llm = llm.model,
                system = prompts.evaluate_docs,
                human = "Document retrieved: \n\n {document} \n\n Question from user: {question}", 
                output = (str, "Indicates whether the documents are relevant to the question, respond with 'yes' or 'no'")
                )

    def invoke(self, state: GraphState):
        """
        Filters documents based on their relevance to the question.
        """
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        online_search = False
        for document in documents:
            response = self.evaluate_docs.invoke({"question": question, "document": document.page_content})
            result = response.score
            print(result)
            if result.lower() == "yes":
                filtered_docs.append(document)
            else:
                online_search = True
                
        return {"documents": filtered_docs, "question": question, "online_search": online_search}


class GenerateAnswer:

    def __init__(self):

        self.generate = Chain(
                llm    = llm.model,
                system = prompts.generate_answer, 
            )

    def invoke(self, state: GraphState):
        """
        Generates an answer based on the retrieved documents.
        """    
        question = state["question"]
        documents = state["documents"]

        solution = self.generate.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "solution": solution}



class Hallucinations:

    def __init__(self):

        self.question_relevance = Chain(
                llm    = llm.model,
                system = prompts.question_relevance_cheker,
                human  = "User's query: \n\n {question} \n\n Generated response: {solution}",
                output = (bool, "Indicates whether the answer sufficiently addresses the user's question, respond with 'yes' or 'no'"))

        self.document_relevance = Chain(
                llm = llm.model,
                system = prompts.document_relevance,
                human = "Collection of facts: \n\n {documents} \n\n LLM output: {solution}",
                output = (bool, "Indicates whether the answer is based on the provided documents, respond with 'yes' or 'no'"))


    def invoke(self, state: GraphState):
        """
        Checks for hallucinations in the generated answers.
        """    
        question = state["question"]
        documents = state["documents"]
        solution = state["solution"]

        score = self.document_relevance.invoke(
            {"documents": documents, "solution": solution}
        )

        if score.score:
            score = self.question_relevance.invoke({"question": question, "solution": solution})
            if score.score:
                return "Answers Question"
            else:
                return "Question not addressed"
        else:
            return "Hallucinations detected"

 
class Retriever:
    def __init__(self, content):

        if content is None:
            return

        documents = [content]
        splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=30)
        doc_splits = splitter.create_documents(documents)

        embedding_function = OllamaEmbeddings(model="nomic-embed-text")

        chroma_db = Chroma.from_documents(
            documents=doc_splits, 
            collection_name= 'rag-chroma', 
            embedding = embedding_function,
            persist_directory="./.chroma"
        ) 

        self.retriever = chroma_db.as_retriever() 


    def invoke(self, state: GraphState):
        """
        Retrieves documents relevant to the user's question.
        """    
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}


class RagGraph:
    def __init__(self, content): 
        self.retriever = Retriever(content)
        self.evaluate = Evaluate()
        self.generate_answer = GenerateAnswer()
        self.hallucinations = Hallucinations()
        self.graph = self.build()

    def build(self): 
        workflow = StateGraph(GraphState)
         
        workflow.add_node("Retrieve Documents", self.retriever.invoke)
        workflow.add_node("Grade Documents", self.evaluate.invoke)
        workflow.add_node("Generate Answer", self.generate_answer.invoke)

        workflow.set_entry_point("Retrieve Documents")
        workflow.add_edge("Retrieve Documents", "Grade Documents")
        workflow.add_edge("Grade Documents", "Generate Answer")

        workflow.add_conditional_edges(
            "Generate Answer",
            self.hallucinations.invoke,
            {
                "Hallucinations detected": "Generate Answer",
                "Answers Question": END, 
                "Question not addressed": END,
            },
        )

        workflow.add_edge("Generate Answer", END)

        memory = MemorySaver()

        graph = workflow.compile(checkpointer=memory) 
        graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
        
        return graph 


    def invoke(self, question, config):
        return self.graph.invoke(input={"question": question}, config = config)
