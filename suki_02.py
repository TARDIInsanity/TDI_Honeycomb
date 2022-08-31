# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 20:11:40 2022

@author: github.com/TARDIInsanity
"""

import suki_02_parser_03
for m in [suki_02_parser_03]:
    Block = m.Block
    ParserBase = m.Parser
    ConstComb = m.ConstComb
    DataComb = m.DataComb
    Exprs = m.Exprs
    DataHalfComb = m.DataHalfComb
del m

def unfold_human(pattern:"tuple[(int, ...)]"):
    result = []
    for i in pattern:
        while hasattr(i, "__len__") and len(i) == 1:
            i = i[0]
        if isinstance(i, int):
            result.append(i)
        else:
            result.append(-1)
            result.extend(unfold_human(i))
            result.append(-2)
    return result

class Context(dict):
    def __init__(self, prior=None):
        super().__init__()
        self.prior = prior
    def __getitem__(self, key):
        node = self
        while node is not None:
            if key in node:
                return dict.__getitem__(node, key)
            node = node.prior
        raise NameError("name "+repr(key)+" is not defined in "+repr(self))
    def newsub(self):
        return self.__class__(self)

# turing complete subsets:
#    S K ~ S K I
#    B C K W (I = W K)

S = ConstComb["S", ("f", "g", "x"), [0, 2, -1, 1, 2, -2]] # = B (B (B W) B) C
K = ConstComb["K", ("x", "y"), [0]]
# note: SK behaves like KI but SKx stores x while KIx returns I immediately
I = ConstComb["I", ("x",), [0]] # = S K x = C K x
B = ConstComb["B", ("f", "g", "x"), [0, -1, 1, 2, -2]] # = S (K S) K
BBB = ConstComb["BBB", ("f", "g", "x", "y"), [0, -1, 1, 2, 3, -2]] # = B B B
C = ConstComb["C", ("f", "x", "y"), [0, 2, 1]] # = B (S B (K K)) S = S (B B S) (K K)
V = ConstComb["V", ("x", "y", "f"), [2, 0, 1]] # = B C (C I)
O = ConstComb["O", ("a", "b"), [1, -1, 0, 1, -2]] # = S I; M = O I
W = ConstComb["W", ("a", "b"), [0, 1, 1]] # = C S I = S S (K I)
Yi = ConstComb["Yi", ("a", "b"), [0, 1, 0]] # = W C = S C I
Yb = ConstComb["Yb", ("y", "f"), [1, -1, 0, 0, 1, -2]] # = B (S I) (S I I) = S (K (S I)) (S I I)
# Y = Yb Yb # theoretically it could unfold to O (O (O (...))) if not lazily evaluated

CONT = Context()
CONT.update(S.val.__class__.INST)
CONT["KI"] = DataHalfComb.build((K.val, I.val)) # best possible definition
CONT["Y"] = DataHalfComb.build((Yb.val,)*2)

###########################
###########################
###########################
#                         #
#                         #
# !!! END PREDECLARATION  #
# !!! BEGIN PARSER        #
#                         #
#                         #
###########################
###########################
###########################

class Parser(ParserBase):
    @classmethod
    def parse(cls, code:str) -> (bool, Block):
        return super().parse("\n"+code)
    @classmethod
    def parse_to_expr(cls, code:str) -> (bool, Exprs):
        return super().parse_to_expr(code)

nj = lambda *i: "\n".join(i)

TESTS = {
    1:nj(
        "# initial comment",
        "a = B 2 3",
        "def test[i, j, k=3] -> 99:",
        "  # comment",
        "    if i j: # comment if",
        "        return k",
        "      # comment end",
        "### multiline",
        "comment",
        "def ignore[a=3, b]: # remove",
        "  nonsense = b a a",
        "    while K:",
        "   nonsense = M nonsense",
        "       return 9",
        "###",
        "def other[a, b, c]: # func comment",
        "    return b",
        "def loop[a]:",
        "    while a:",
        "        print a",
        "        a = a I",
        "    return a",
        "def wrap[a]:",
        "    def test[b]:",
        "        return V a b",
        "    return test",
        "c = wrap[2]",
        "d = c[5]",
        "print test[K I, I]",
        "print test[K, I]",
        "print loop[K]",
        "print d",
        ),
    "bad":nj(
        "def test[]:",
        "    if K:",
        "\t\treturn 3",
        "  return 5"
        ), # mismatched indentation, even if it might look correct to the user
    2:nj(
        "print 3",
        "print K",
        "print K I",
        "print K 4 2",
        "print C K",
        "print C K 4 2",
        ),
    3:nj(
        "def test[i]:",
        "  print i",
        "# comment",
        "x = test[4]",
        "x = test[2]",
        "def probe[i, j]:",
        "  print i j S",
        "  return j",
        "x = probe[K, C]",
        "x = probe[K I, C]",
        "x = probe[C K, C]",
        "x = probe[S K, C]",
        "x = probe[C (K I), C]",
        "x = probe[C (C K), C]",
        "x = probe[C (S K), C]",
        "x = probe[V, 3]",
        "x = probe[V I, K]",
        ),
    4:nj(
        "def wrapper[i]:",
        "  def printer[j]:",
        "    print i j",
        "    return j i",
        "  print i",
        "  return printer",
        "x = wrapper[K]",
        "print x[C]",
        ),
    5:nj(
        "print S(K(S(K(S(K S)K))(S(K S)K)))",
        "print (comb a b c: a b b)",
        "print (comb a b c: a b b) V",
        "print (comb a b c: a b b) V 2",
        "print (comb a b c: a b b) V 2 4",
        "print (comb a b c: a b b) V 2 4 K",
        ),
    6:nj(
        "print O B O (O I)",
        "print Yi (C (B O Yi))",
        ),
    7:nj(
        "inc = (comb n f x: n f (f x))",
        "zero = K I",
        "one = inc zero",
        "two = inc one",
        "three = inc two",
        "print three (V 1) 0",
        ),
    8:nj(
        "def test[i]:",
        "    return ADD i i",
        "print test[4]",
        "print test[2]",
        "print test[0u111]",
        "print ADD[2,3]"
        ),
    9:nj(
        "def wrapper[i, j]:",
        "  def wrapped[k]:",
        "    def utility[l]:",
        "      return V l (k i j)",
        "    return utility",
        "  return wrapped",
        "x = wrapper[2, 3]",
        "y = x[V]",
        "z = wrapper[S, K][V]",
        "print y[99]",
        "print z[69]",
        "print wrapper[0aleet, 0s33][V][0x420]",
        ),
    10:nj(
        "OR = W",
        "AND = W C",
        "NOT = V (K I) K",
        "XOR = V NOT I",
        "def bool[i]:",
        "    if i:",
        "        return K",
        "    return K I",
        )
    }

def parse(code:str) -> object:
    success, result = Parser.parse(code)
    if success:
        print("success")
    return result
def interpret(tree, context:dict=None):
    if context is None:
        context = CONT
    iterator = tree.process(context)
    flag = None
    try:
        while True:
            flag = next(iterator)
            if flag is not None:
                return flag
    except StopIteration as e:
        return e.value

def read(path:str) -> str:
    with open(path, mode="r") as f:
        code = f.read()
    return code

def ipr(path:str, context:dict=None):
    return interpret(parse(read(path)), context)
