"""Example pytests adapted from pytest docs

https://docs.pytest.org/en/latest/getting-started.html
"""

# content of test_sysexit.py
import pytest


def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()


# content of test_sample.py
def func(x):
    return x + 1


def test_answer():
    assert func(3) != 5
    assert func(3) == 4


# content of test_class.py
class TestClass:
    """
    pytest discovers all tests following its Conventions for Python test
    discovery, so it finds both test_ prefixed functions. There is no need to
    subclass anything, but make sure to prefix your class with Test otherwise
    the class will be skipped.
    """

    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        x = "hello"
        assert not hasattr(x, "check")
        assert hasattr(x, "capitalize")


# content of test_tmp_path.py
def test_needsfiles(tmp_path):
    print(f"tmp_path: {tmp_path}")
    assert 1
