from hoopy.utils import Pipe, context, FunctionContext


def glob():
    pass


class TestContext:
    def cls(self):
        pass

    def test_global_context(self):
        assert context(glob) == FunctionContext.Global

    def test_class_context(self):
        assert context(self.cls) == FunctionContext.Class

    def test_local_context(self):
        def loc():
            pass

        assert context(loc) == FunctionContext.Local

    def test_weird_nest(self):
        def inner():
            assert context(inner) == FunctionContext.Local

            class Inner:
                @staticmethod
                def some():
                    assert context(Inner.some) == FunctionContext.Class

                    def another():
                        assert context(another) == FunctionContext.Local

                    another()

            Inner.some()

        inner()


def test_pipe_application_order():
    def add_one(x: int):
        return x + 1

    def neg(x: int):
        return -x

    assert (Pipe(0) | add_one | add_one | neg | add_one)() == -1
