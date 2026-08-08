from app.chat_engine import ChatEngine


def test_chat_engine_initializes():
    chat = ChatEngine()

    assert chat is not None
    assert chat.search_engine is not None