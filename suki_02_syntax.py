# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 21:28:20 2022

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

