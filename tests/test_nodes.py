from src.agent.graph import Evaluate, GenerateAnswer, AnyDoc, Retriever, GraphState


def test_generate_answer():

    generate_node = GenerateAnswer() 

    graph_state_instance = GraphState(
        question="What are the latest trends in AI?",
        solution="AI is evolving in areas like natural language processing and reinforcement learning.",
        online_search=False,
        documents=["AI is a revolution in dfiferent areas of technology, one of them is Coding, Cursor is the best tool",
                   "Monkeys are like humans",
                   "A type of Pokemon is Charizard"]
    )

    response = generate_node.invoke(graph_state_instance) 

    assert type(response) is dict 


