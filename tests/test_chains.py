from src.agent.chains import evaluate_docs, question_relevance_cheker, document_relevance


def test_evaluate_docs():

    question = 'what are you?'
    document = 'Im a person who born in the year 2000, my name is Percy'

    response = evaluate_docs.invoke({"question": question, "document": document}) 

    assert response.score in ['yes', 'no']


def test_question_relevance_cheker():

    question = 'what are you'
    solution = 'Im a person who born in the year 2000'

    response = question_relevance_cheker.invoke({'question' : question, 'solution' : solution})

    assert response.score in [True, False]


def test_document_relevance():

    documents = 'Im a Person who born the year of 2000'
    solution = 'This is a person and he born in 2000'

    response = document_relevance.invoke({'documents' : documents, 'solution' : solution})

    assert response.score in [True, False]

