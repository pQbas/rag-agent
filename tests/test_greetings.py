from src.greetings import say_hello

def test_say_hello():
    # Check if the function returns 'hello world'
    assert say_hello() == 'hello world'

