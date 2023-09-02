# coding: hoopy

def ($)(f, x):
    return f x

# The backtick disambiguates this from attribute access
def (.`)(f, g):
    return lambda x: f(g(x))

class (<:):
    def __init__(self, rest, last):
        self.rest = rest
        self.last = last

    def __repr__(self):
        return f"{self.rest} <: {self.last}"

    def (++)(self, other):
        if other.rest == []:
            return self <: other.last
        else:
            return (self ++ other.rest) <: other.last

    # Notice the flipped order of arguments
    def (<$>)(fn, self):
        if self.rest == []:
            return [] <: fn self.last
        else:
            return (fn <$> self.rest) <: fn self.last

    # The backtick disambiguates this operator from the built-in >>=
    def (>>=`)(fn, self):
        def foldl(fn, self):
            if self.rest == []:
                return self.last
            else:
                return foldl fn self.rest `fn` self.last
        return foldl((++), fn <$> self)


xs = [] <: 1 <: 2 <: 3
print xs
# prints "[] <: 1 <: 2 <: 3"
square = lambda x: x * x
print (square <$> xs)
# prints "[] <: 1 <: 4 <: 9"
square_roots = lambda x: [] <: x ** 0.5 <: -x ** 0.5
print (round .` square <$> (square_roots >>=` xs))
# prints "[] <: 1 <: 1 <: 2 <: 2 <: 3 <: 3"
