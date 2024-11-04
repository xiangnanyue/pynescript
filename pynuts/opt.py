
class Operator:
    @classmethod
    def add(cls, left, right):
        return left[0] + right[0]

    @classmethod
    def sub(cls, left, right):
        return left[0] - right[0]

    @classmethod
    def mul(cls, left, right):
        return left[0] * right[0]

    @classmethod
    def truediv(cls, left, right):
        return left[0] / right[0]

    @classmethod
    def mod(cls, left, right):
        return left[0] % right[0]

    @classmethod
    def gt(cls, left, right):
        return left[0] > right[0]

    @classmethod
    def lt(cls, left, right):
        return left[0] < right[0]

    @classmethod
    def eq(cls, left, right):
        return left[0] == right[0]

    @classmethod
    def ne(cls, left, right):
        return left[0] != right[0]

    @classmethod
    def le(cls, left, right):
        return left[0] <= right[0]

    @classmethod
    def ge(cls, left, right):
        return left[0] >= right[0]
