# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 20:11:40 2022

@author: github.com/TARDIInsanity

Honeycomb (Suki) syntax guide

-pythonic indentation layers
-pythonic string declaration
-expanded integer declaration

NAME - any pythonic identifier
EXPR - expression; evaluates to some result object
STMT - statement; may transmit a Flag to the governing statement, and
        may evoke expressions, functions, and code blocks when run.
        each statement appears on its own line with some amount of
        whitespace preceding it, known as its indentation level.
        this indentation may be nothing, at which time it is referred
        to as the "base" level, which all statements ultimately will
        fall under.
BLOCK - a series of statements of the same or deeper indentation level
        an indentation level is defined as being "same or deeper" if
        the indentation whitespace begins with the group's overall
        indentation whitespace string. The first statement which
        does not begin with this string marks the end of the group.

expressions:
    (comb NAMES: PATTERN)
    (comb f g x: f x (g x)) # S
        NAMES is a space-separated group of 0 or more names
        PATTERN is a space-separated group of 0 or more names and ()-enclosed patterns
        within PATTERN, all names must appear in NAMES.
        this defines a proper combinator.
        Avoid defining combinators that are identical to the builtins

S = Combnode["S", [0, 2, -1, 1, 2, -2], 3]
K = Combnode["K", [0], 2]
I = Combnode["I", [0], 1]
B = Combnode["B", [0, -1, 1, 2, -2], 3]
C = Combnode["C", [0, 2, 1], 3]
V = Combnode["V", [2, 0, 1], 3]
O = Combnode["O", [1, -1, 0, 1, -2], 2] # = S I
Yi = Combnode["Yi", [0, 1, 0], 2]
KI = K I
        
    CONST - any integer constant
    COMB - any combinator constant name or user definition
    NAME - any variable name which does not refer to a function
    EXPR[ARGS] - when NAME refers to a function and ARGS is 0 or more ,-separated expressions
        *EXPR may not be of the form "EXPR EXPR"
        EXPR EXPR[ARGS] -> EXPR (EXPR[ARGS]), not (EXPR EXPR)[ARGS]
        to achieve (EXPR EXPR)[ARGS], enter it in that form.
    (EXPR) - the result of evaluating EXPR
    EXPR EXPR - juxtaposition is calling of combinators
        COMB left expressions are evaluated with the given arg
        integers are treated like church numerals
            W = (comb x y. x y y)
            INC = (comb n f x. n f (f x)) = B W (B B)
            (comb n f x. B W (B B) n f x)
            0 = K I                       = 0 INC 0
            1 = I     = INC 0             = 1 INC 0
            2 = INC 1 = INC (INC 0)       = 2 INC 0
            3 = INC 2 = INC (INC (INC 0)) = 3 INC 0
            N = N INC 0
        except that they ONLY interface directly with the
            integer-interfacing functions, provided as
            constants to make integers useful.

ADD x y:
    if x and y are integers, return x+y
    if x is 0, return y
    if x is 1, return W (B B y)
    return x (ADD 1) y
MUL x y:
    if x and y are integers, return x * y
    if x is 0, return 0
    if x is 1, return y
    return x (ADD y) 0
POW x y:
    if x and y are integers, return y ** x
    if y is 0, return 0
    if y is 1, return 1
    if x is 0, return I
    if x is 1, return y
    return x y
DIF x y:
    if x and y are integers, return max(x-y, 0)
    otherwise error
DIV x y:
    if x and y are integers, return x//y
    otherwise error
MOD x y:
    if x and y are integers, return x%y
    otherwise error

all integers are >= 0

ISINT x:
    K if x is an integer else K I
IS x y:
    K if x and y refer to the same object in memory else K I
ISCOM:
    K if x is a combinator (builtin or user defined) else K I

statement types not classified below:
    NAME = EXPR

function definition:
    def NAME[ ARGS ]:
        BLOCK
    
    def NAME[ ARGS ] -> DEFAULT:
        BLOCK
    
    ARGS, if present, is zero or more NAMEs followed by zero or more
        NAME=EXPR pairs, all comma separated. any EXPRs in ARGS are
        evaluated on function definition.
    DEFAULT, if present, denotes a value to return if the function doesn't
        return anywhere else. Its value is stored with the function, and
        evaluated on definition.
        if absent, the default is the number 0

flow control:
    anywhere an "else:..." appears, it could also not appear.
        an absent "else" functions identically to a present "else: pass"
    
    if EXPR:
        BLOCK
    else:
        BLOCK
    
    while EXPR:
        BLOCK
    else:
        BLOCK
    
    dowhile EXPR:
        BLOCK
    else:
        BLOCK
    
    do:
        BLOCK
    else:
        BLOCK
    
    "while": evaluate EXPR.
        if True: do the first block and loop, else do the else block
    "dowhile": like "while", but instead of evaluating EXPR, it assumes True
        for the very first iteration.
    "do": like "dowhile", but instead of specifying an EXPR, its expr is just False.
        how is this useful? consider the logic of "continue" and "break".

flow interrupt:
    pass
        pass
    continue
        proceed to the end of a block in the nearest
            while, dowhile, or do first-block.
    break
        like "continue", but force the loop to exit, without
        evaluating the else block.
    
    return EXPR
        evaluate EXPR and then exit out of the nearest function
        the result of calling that function is the result of this EXPR
        a function which exits without a return will use its default
    
    return
        exits out of the nearest function
        the result of calling that function is its default return value
        this is only useful if you want to quit in the middle of a function
        such as with "if stop_now: return"

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
Yb = ConstComb["Yb", ("y", "f"), [1, -1, 0, 0, 1, -2]]
# Y = Yb Yb

CONT = Context()
CONT.update(S.val.__class__.INST)
CONT["KI"] = DataHalfComb.build((K.val, I.val)) # best possible definition
CONT["Y"] = DataHalfComb.build((Yb.val,)*2)

'''
# Y = f. (x. f (x x)) (x. f (x x))
# Y = f. (g x. g (x g x)) f (g x. g (x g x))
# Y = f. Yi (g x. g (x g x)) f
# Y = Yi (g x. g (x g x))
# Y = Yi (g x. g (Yi x g))
# Y = Yi (g x. O (Yi x) g)
# Y = Yi (g x. B O Yi x g)
# Y = Yi (g x. C (B O Yi) g x)
# Y = Yi (C (B O Yi))
# Y = Yi (B C (B O) Yi)
# Y = O (B C (B O)) Yi
# Y = B O (B C) (B O) Yi
# Y = Yi (B O) (B C) Yi

Yi (C (B O Yi)) f
B O Yi (C (B O Yi)) f
O (Yi (C (B O Yi))) f
f (Yi (C (B O Yi)) f)

O (M (B O M)) f
f (M (B O M) f)

# Y = f. (x. f (x x)) (x. f (x x))
# Y = f. (x g. g (x x g)) (x g. g (x x g)) f
# Y = (x g. g (x x g)) (x g. g (x x g))
# Y = M (x g. g (x x g))
# Y = M (x g. O (x x) g)
# Y = M (x. O (x x))
# Y = M (x. O (M x))
# Y = M (x. B O M x)
# Y = M (B O M)
# Y = O (B O) M
# Y = O B O M
# Y = S I ((S (K S) K) (S I)) (S I I)

S a b c = a c (b c)
S M b c = c c (b c)
S O b c = b c (c (b c))
S O I c = c (c c)
S M I c = c c c
S B b c = B c (b c)
S B I c d = c (c d)
S (S B I) b c = c (c (b c))

(comb a b c d: b d (c d))
X a b c d = S b c d
X I = S
S X b c d e = S (b c) d e

(comb a b c d: b c (a c))
X a b c d = S b a c
X I I I I = S I I I = I
X I b c . = b c c
X I (X I b) c . = X I b c c = b c c
X I (X I (X I b)) c . = X I (X I b) c c = b c c
X I (X I) c . . = X I c c . = c (c c)
X I (X I (X I)) c . . = X I c c . = c c c
X I (X I (X I) I) = K Ki
X I (X I (X I I I) I) c d = c
X I (X I (X I I I) I) c = K c
X I (X I (X I I I) I) = K
X . K = K
X (K a) b c = K (b c a)
X K I c d = S I K c = c (K c)
X K I K = K (K (K K))
X . (X K I K) = X K I K
X (X K I K) I c = K (c (K (K K)))
X (X K I K) I (X K I K) = X K I K
X a X c d e f = S X a c = a c e (c e)
X I X c d e f = c e (c e)
X I X I = K (B K M)
X I X (K (B K M)) = K (K (K (...)))
X K I (K (B K M)) = K (B K M)
X I I I = Ki
X I Ki Ki = K Ki
(every other combination of X followed by three choices of I or Ki) = Ki
X (K Ki) I I = K Ki
X (K Ki) Ki Ki = K Ki
X (K Ki) Ki I = K Ki
X (K Ki) I Ki = Ki

(comb a b c d: a (b c (a c)))
X a b c d = a (S b a c)
X I b c d = b c c
X I I = B K M
X I I I = Ki
X I (X I I) = X I I
X Ki = K (K Ki)
X I Ki = K
X (K a) = K (K (K a))
X K b c = K (K (b c (K c)))
X K K c = K (K c)
X K Ki c = K (K (K c))
X K b c d e = b c (K c)
X K (X K) c d e f = K (S c K (K c))
X K (X K) X d e f g h i = X
X K (X Ki) = K (X Ki)
X I (X K) c = K(K(K(c c (K c))))

(comb a b c d: b (c d (b d)))
X a b c d = b (S c b d)
X I a b c = a (S b a c)
X I I = W
W I = M
'''

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











