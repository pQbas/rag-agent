from src.graph import RagGraph 
from langchain_core.documents import Document

CONTENT = (
    "The latest trend of AI is revolutionizing various fields, especially Coding with tools like Cursor.\n"
    "The latest trend in AI is also advancing in autonomous vehicles.\n"
    "The latest trend of AI is also advancing in healthcare innovations."
)

DOCUMENTS = [
        Document(page_content="The latest trend of AI is revolutionizing various fields, especially Coding with tools like Cursor."),
        Document(page_content="The latest trend in AI is also advancing in autonomous vehicles"),
        Document(page_content="The latest trend of AI is also advancing in healthcare innovations")
        ]

QUESTION = "What are the latest trends in AI?"


SOLUTION = "The latest trends in AI include advancements in coding tools like Cursor, autonomous vehicles, and healthcare innovations."


CONFIG = {"configurable": {"thread_id": "1"}}


def test_rag_graph():

    rag_graph = RagGraph(CONTENT)

    response = rag_graph.invoke(question = QUESTION, config = CONFIG)

    print(response["solution"])

    assert type(response["solution"]) is str 

