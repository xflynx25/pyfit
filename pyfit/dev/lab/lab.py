

__all__ = ['hello_worldy']

from .helper import hello_worldy_helper


def hello_worldy():
    print('hello worldy')
    hello_worldy_helper()
    return 'hello worldy'