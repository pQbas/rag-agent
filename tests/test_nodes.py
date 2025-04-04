from src.agent.graph import Evaluate, GenerateAnswer, Hallucinations, Retriever, GraphState
from langchain_core.documents import Document


DOCUMENTS = [
        Document(page_content="AI is revolutionizing various fields, especially Coding with tools like Cursor."),
        Document(page_content="AI is also advancing in autonomous vehicles"),
        Document(page_content="AI is also advancing in healthcare innovations")
        ]

QUESTION = "What are the latest trends in AI?"

SOLUTION = "The latest trends in AI include advancements in coding tools like Cursor, autonomous vehicles, and healthcare innovations."


def test_generate_answer_node():

    generate_node = GenerateAnswer()
    graph_state_instance = GraphState(question = QUESTION, documents = DOCUMENTS)

    response = generate_node.invoke(graph_state_instance)

    assert type(response) is dict 



def test_evaluate_node():

    evaluate_node = Evaluate()
    graph_state_instance = GraphState(question = QUESTION, documents = DOCUMENTS)

    response = evaluate_node.invoke(graph_state_instance)

    assert type(response) is dict 


def test_hallucinations_node():
    any_doc_node = Hallucinations()
    graph_state_instance = GraphState(question = QUESTION, documents = DOCUMENTS, solution = SOLUTION)

    response = any_doc_node.invoke(graph_state_instance)

    assert response in ["Answers Question", "Question not adressed", "Hallucinations detected"]

