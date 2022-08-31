# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 20:43:12 2022 - v3
Created on Wed Aug 24 23:27:09 2022 - v2
Created on Mon Aug 22 18:05:33 2022 - v1

@author: github.com/TARDIInsanity
"""

####################
####################
####################
#                  #
#                  #
# !!! BEGIN LEXER  #
#                  #
#                  #
####################
####################
####################

from weakref import WeakValueDictionary

import suki_02_lexer
for m in [suki_02_lexer]:
    Token = m.Token
    Indent = m.Indent
    Name = m.Name
    Keyword = m.Keyword
    Paren = m.Paren
    Punct = m.Punct
    Special = m.Special
    Comment = m.Comment
    Comment = m.Comment
    String = m.String
    Integer = m.Integer
    ZERO = m.Integer(0)
    
    ignore_iterator = m.lex_integer_2.ignore_iterator
del m
LexerBase = suki_02_lexer.Lexer
class Lexer(LexerBase):
    '''push, pop, init: redefined so that spyder shows the hints'''
    def __init__(self, code:str):
        super().__init__(code)
        self.layers = []
    def pop(self, record:bool=True):
        result = super().pop()
        if record and result is not None and self.layers:
            self.layers[-1].append(result)
        return result
    def push(self, val, record:bool=True):
        if record and self.layers and self.layers[-1] and self.layers[-1][-1] is val:
            self.layers[-1].pop()
        super().push(val)
    def peek(self, comparison:Token) -> bool:
        token = self.pop(record=False)
        if token is None:
            return False
        self.push(token, record=False)
        if isinstance(comparison, type):
            return token.be(comparison, None)
        else:
            return token.be(comparison.__class__, comparison.val)
    def feed(self, source:iter):
        for i in reversed(source):
            self.push(i)
    # new in v3: Lexer is fully receptive of backtracking parse style
    def enter(self):
        self.layers.append([])
    def accept(self):
        if self.layers:
            last = self.layers.pop()
            if self.layers:
                self.layers[-1].extend(last)
    def reject(self, exception=None):
        if self.layers:
            self.feed(self.layers.pop())
        return exception
    def conclude(self, accepted:bool):
        if accepted:
            self.accept()
        else:
            self.reject()
        return accepted

##################
##################
##################
#                #
#                #
# !!! END LEXER  #
# !!! BEGIN AST  #
#                #
#                #
##################
##################
##################

class AST:
    '''not as abstract as you would think'''
    @classmethod
    def pull(cls, parser, lexer:Lexer, indent:Indent) -> (bool, object):
        lexer.enter()
        raise lexer.reject(NotImplementedError(cls))
    def __repr__(self):
        raise NotImplementedError(self.__class__)
#
class Data(AST):
    CALLABLE = False
    def to_bool(self) -> bool:
        raise NotImplementedError
class Flag(AST):
    INST = {}
    def __init__(self, name:str, value:Data=None):
        self.name = name
        self.value = value
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.name)}, {repr(self.value)})"
    def be(self, name, value=None):
        return name==self.name and value in (None, self.value)
    def __class_getitem__(cls, name:str, value:Data=None):
        if value is not None:
            return cls(name, value)
        if name not in cls.INST:
            cls.INST[name] = cls(name)
        return cls.INST[name]
class Expression(AST):
    def evaluate(self, context:dict) -> Data:
        raise NotImplementedError
class Statement(AST):
    def state(self, context:dict) -> Flag:
        raise NotImplementedError
class Block(AST):
    def __init__(self, *statements:Statement):
        self.statements = statements
    def __repr__(self):
        return f"{self.__class__.__name__}{repr(self.statements)}"
    @classmethod
    def pull(cls, parser, lexer:Lexer, indent:Indent) -> (bool, object):
        '''if you hit the recursion depth here, then that's because your
        program is so deeply layered that it would do so at runtime'''
        lexer.enter()
        statements = []
        while True:
            # !!! basically the only reference to Parser.pop_stmt
            success, stmt = parser.pop_stmt(lexer, indent)
            if not success:
                break
            statements.append(stmt)
        return (lexer.conclude(bool(statements)), cls(*statements))
    def process(self, context:dict) -> Flag:
        for stmt in self.statements:
            result = yield from stmt.state(context)
            if result is not None:
                return result
Block.PASS = Block()

######################
######################
######################
#                    #
#                    #
# !!! END AST BASES  #
# !!! BEGIN DATA     #
#                    #
#                    #
######################
######################
######################

class DataFunction(Data):
    CALLABLE = True
    def __init__(self, name:Name, args:"tuple[Name]", defaults:"tuple[Data]", block:Block, defreturn:Data, context:dict):
        self.name = name
        self.args = args
        self.defaults = defaults
        self.block = block
        self.defreturn = defreturn
        self.context = context
    def __str__(self):
        return "<function '{self.name.val}'>"
    def __repr__(self):
        body = (self.name, self.args, self.defaults, self.block, self.defreturn)
        return f"{self.__class__.__name__}{body}"
    def call(self, args, context:dict):
        # completely ignores context!
        sub = self.context.newsub() # expects a dict-like with this method to make a subcontext
        ls, la, ld = len(self.args), len(args), len(self.defaults)
        assert 0 <= ls-la <= ld, "invalid number of args passed to function"
        gap = ls-la
        if gap:
            args = tuple(args)+tuple(self.defaults[-gap:])
        for name, arg in zip(self.args, args):
            sub[name.val] = arg
        result = yield from self.block.process(sub)
        if result is None:
            return Flag("return", self.defreturn)
        if not result.be("return"):
            print("FUNCTION ERROR")
            print(result)
            return Flag("return", self.defreturn)
        return result
    def to_bool(self) -> bool:
        return True
    def step(self, context:dict) -> (bool, Data):
        return (False, self)
class DataObjectBase(Data):
    pass
class DataImmutableBase(DataObjectBase):
    def __str__(self):
        return repr(self.val)
    def str_arg(self):
        return str(self)
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.val)})"
    def step(self, context:dict) -> (bool, Data):
        return (False, self)
class DataBool(DataImmutableBase):
    INST = {}
    def __init__(self, val:bool):
        self.val = val
    def to_bool(self) -> bool:
        return self.val
    def __class_getitem__(cls, val):
        if not isinstance(val, bool):
            raise ValueError(val)
        if val not in cls.INST:
            cls.INST[val] = cls(val)
        return cls.INST[val]
class DataInt(DataImmutableBase):
    arity = 2
    INST = WeakValueDictionary()
    def __init__(self, val:int):
        self.val = val
    def substitute(self, args, context=None):
        args = tuple(args)
        assert len(args) == 2, str((self, args))
        f, x = args
        if self.val <= 0:
            return x
        if self.val == 1:
            return DataHalfComb(f, x)
        new = self.__class__(self.val-1)
        return DataHalfComb.build((new, args[0], (args[0], args[1])))
    def to_bool(self) -> bool:
        return bool(self.val)
    def __class_getitem__(cls, val:int):
        if val in cls.INST:
            return cls.INST[val]
        result = cls(val)
        cls.INST[val] = result
        return result
class DataStr(DataImmutableBase):
    arity = None
    def __init__(self, val:str):
        self.val = val
    def to_bool(self) -> bool:
        return bool(self.val)
class DataBuiltin(DataImmutableBase):
    CALLABLE = True
    INST = {}
    def __init__(self, name:str, func:callable, arity:int):
        self.name = name
        self.func = func
        self.arity = arity
    def __str__(self):
        return self.name
    def __repr__(self):
        body = (self.name, self.func, self.arity)
        return f"{self.__class__.__name__}{repr(body)}"
    def call(self, args, context:dict):
        if False:
            yield
        return Flag("return", self._call(args, context))
    def _call(self, args, context:dict):
        return self.func(*args, context=context)
    @classmethod
    def wrap(cls, arity:int):
        def wrapper(function):
            cls.INST[function.__name__] = cls(function.__name__, function, arity)
            return cls.INST[function.__name__]
        return wrapper
    def __class_getitem__(cls, key):
        return cls.INST[key]
    @classmethod
    def pull(cls, parser, lexer:Lexer) -> (bool, object):
        lexer.enter()
        tok = lexer.pop()
        if tok is None or not tok.be(Name) or tok.val not in cls.INST:
            return (lexer.conclude(False), None)
        return (lexer.conclude(True), Constant(cls.INST[tok.val]))
    def substitute(self, args, context=None):
        return self._call(args, context=context)

@DataBuiltin.wrap(2)
def ADD(x, y, context):
    if isinstance(x, DataInt):
        if isinstance(y, DataInt):
            return DataInt[x.val+y.val]
        if x.val == 0:
            return y
        if x.val == 1:
            return DataHalfComb.build((context["W"], (context["B"], context["B"]), y))
    return DataHalfComb.build((x, (context["ADD"], DataInt[1]), y))
@DataBuiltin.wrap(2)
def MUL(x, y, context):
    if isinstance(x, DataInt):
        if isinstance(y, DataInt):
            return DataInt[x.val*y.val]
        if x.val == 0:
            return DataInt[0]
        if x.val == 1:
            return y
    return DataHalfComb.build((x, (context["ADD"], y), DataInt[0]))
@DataBuiltin.wrap(2)
def POW(x, y, context):
    if isinstance(x, DataInt):
        if isinstance(y, DataInt):
            return DataInt[y.val**x.val]
        if y.val in (0, 1):
            return y
        if x.val == 0:
            return context["I"]
        if x.val == 1:
            return y
    return DataHalfComb.build((x, y))
@DataBuiltin.wrap(2)
def DIF(x, y, context=None):
    if isinstance(x, DataInt):
        if isinstance(y, DataInt):
            return DataInt[max(x.val-y.val, 0)]
    raise TypeError(f"DIF(x:{type(x)}, y:{type(y)})")
@DataBuiltin.wrap(2)
def DIV(x, y, context=None):
    if isinstance(x, DataInt):
        if isinstance(y, DataInt):
            return DataInt[x.val//y.val]
    raise TypeError(f"DIV(x:{type(x)}, y:{type(y)})")
@DataBuiltin.wrap(2)
def MOD(x, y, context=None):
    if isinstance(x, DataInt):
        if isinstance(y, DataInt):
            return DataInt[x.val%y.val]
    raise TypeError(f"MOD(x:{type(x)}, y:{type(y)})")

@DataBuiltin.wrap(1)
def ISINT(x, context=None):
    if isinstance(x, DataInt):
        return context["K"]
    return context["KI"]
@DataBuiltin.wrap(2)
def IS(x, y, context=None):
    if x is y:
        return context["K"]
    return context["KI"]
@DataBuiltin.wrap(1)
def ISCOM(x, context=None):
    if isinstance(x, DataComb):
        return context["K"]
    return context["KI"]

class DataComb(DataImmutableBase):
    INST = {}
    def __init__(self, args:"tuple[str]", pattern:"tuple[int]"):
        self.args = args
        self.pattern = pattern
        self._string = None
    def __str__(self):
        if self in self.INST.values():
            for key, value in self.INST.items():
                if self is value:
                    return key
        if self._string is None:
            pattern = [self.args[i] if i >= 0 else "()"[~i] for i in self.pattern]
            body = " ".join(pattern).replace("( ", "(").replace(" )", ")")
            self._string = f"(comb {' '.join(self.args)}: {body})"
        return self._string
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.args)}, {repr(self.pattern)})"
    @property
    def arity(self):
        return len(self.args)
    def substitute(self, args, context=None):
        args = tuple(args)
        assert len(args) == len(self.args)
        return self._substitute(iter(self.pattern), args)
    @classmethod
    def _substitute(cls, pattern:iter, args:"tuple[Data]") -> Data:
        results = []
        for i in pattern:
            if i == -1:
                results.append(cls._substitute(pattern, args))
                continue
            if i == -2:
                break
            results.append(args[i])
        output = results.pop(0)
        while results:
            output = DataHalfComb(output, results.pop(0))
        return output
    def to_bool(self) -> bool:
        return True
    def __class_getitem__(cls, key):
        if isinstance(key, tuple):
            key, args, pattern = key
            if key not in cls.INST:
                cls.INST[key] = cls(args, pattern)
        return cls.INST[key]

class DataIntermediateBase(DataObjectBase):
    '''only under very very specific circumstances
    can this be stored in a variable.'''
    def step(self, context:dict) -> (bool, Data):
        '''return true when the result is changed AND implements .step'''
        raise NotImplementedError
    def to_bool(self) -> bool:
        return False
class DataHalfComb(DataIntermediateBase):
    def __init__(self, head:Data, arg:Data, _=None):
        #self.arity = None # set on setting .head
        self.head = head
        self.arg = arg
    def str_arg(self):
        return f"({str(self)})"
    def __str__(self):
        return f"{str(self.head)} {self.arg.str_arg()}"
    def __repr__(self):
        body = (self.head, self.arg, self.arity)
        return f"{self.__class__.__name__}{body}"
    @property
    def head(self):
        return self._head
    @head.setter
    def head(self, value:Data):
        self._head = value
        #self.arity = value.arity-1
    @property
    def arity(self):
        return self.head.arity-1
    def step(self, context:dict) -> (bool, Data):
        '''return true when the result is changed AND implements .step'''
        if self.arity > 0:
            return (False, self)
        prior = None
        node = self
        while node.arity < 0:
            prior, node = node, node.head
        ref = node
        found = []
        while isinstance(ref, DataHalfComb):
            found.append(ref)
            ref = ref.head
        found.reverse()
        for i in found:
            val = i.arg
            if not isinstance(val, DataHalfComb):
                continue
            changed, i.arg = val.step(context)
            if changed:
                return (True, self)
        #input(str((self, node, found)))
        result = ref.substitute((i.arg for i in found), context=context)
        if prior is None:
            return (hasattr(result, "step"), result)
        prior.head = result
        return (True, self)
    @classmethod
    def build(cls, args):
        if not isinstance(args, tuple):
            return args
        result = cls.build(args[0])
        for i in args[1:]:
            result = cls(result, cls.build(i))
        return result

##########################
##########################
##########################
#                        #
#                        #
# !!! END DATA           #
# !!! BEGIN EXPRESSIONS  #
#                        #
#                        #
##########################
##########################
##########################

class Constant(Expression):
    DATA = DataImmutableBase
    SPECIFIC = None
    def __init__(self, val:DataImmutableBase):
        self.val = val
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.val)})"
    def evaluate(self, context:dict) -> Data:
        if False:
            yield
        return self.val
    @classmethod
    def pull(cls, parser, lexer:Lexer) -> (bool, object):
        lexer.enter()
        const = lexer.pop()
        if const is None or not const.be(cls.SPECIFIC):
            return (lexer.conclude(False), None)
        return (lexer.conclude(True), cls(cls.DATA(const.val)))
class ConstInt(Constant):
    DATA = DataInt
    SPECIFIC = Integer
class ConstStr(Constant):
    DATA = DataStr
    SPECIFIC = String
class ConstComb(Constant):
    DATA = DataComb
    INST = {}
    def __class_getitem__(cls, key):
        return cls(DataComb[key])
    @classmethod
    def pull(cls, parser, lexer:Lexer) -> (bool, object):
        if not lexer.peek(Paren["("]):
            return (False, None)
        lexer.enter()
        lexer.pop() # "("
        name = lexer.pop()
        if not name.be(Keyword, "comb"):
            return (lexer.conclude(False), None)
        args = cls.pull_comb_names(lexer)
        pattern = cls.pull_comb_pattern(lexer, args)
        if not lexer.peek(Paren[")"]):
            raise lexer.reject(SyntaxError("Unexpected lack of ) when pulling combinator expression"))
        lexer.pop() # ")"
        return (lexer.conclude(True), cls(DataComb(args, pattern)))
    @classmethod
    def pull_comb_names(cls, lexer:Lexer) -> "tuple[str]":
        lexer.enter()
        names = []
        while True:
            token = lexer.pop()
            if token is None:
                raise lexer.reject(SyntaxError("comb name sequence must be terminated with ':', but got EOF"))
            if token.be(Punct, ":"):
                break
            if token.be(Comment):
                continue
            if token.be(Indent):
                continue
            if token.be(Name):
                names.append(token.val)
                continue
            raise lexer.reject(SyntaxError("Unexpected token while parsing (comb ...) names: "+repr(token)))
        lexer.accept()
        return names
    @classmethod
    def pull_comb_pattern(cls, lexer:Lexer, names:"tuple[str]") -> "tuple[int]":
        lexer.enter()
        pattern = []
        depth = 0
        while True:
            token = lexer.pop()
            if token is None:
                raise lexer.reject(SyntaxError("comb pattern sequence must be terminated with ')', but got EOF"))
            if token.be(Comment):
                continue
            if token.be(Indent):
                continue
            if token.be(Name):
                if token.val not in names:
                    raise lexer.reject(SyntaxError("(comb ...) pattern may only refer to names present in the comb's ARGS. "+
                        "found name "+token.val+" under names "+repr(names)))
                pattern.append(names.index(token.val))
                continue
            if token.be(Paren, "("):
                pattern.append(-1)
                depth += 1
                continue
            if token.be(Paren, ")"):
                if depth == 0:
                    lexer.push(token)
                    break
                pattern.append(-2)
                depth -= 1
                continue
            raise lexer.reject(SyntaxError("Unexpected token while parsing (comb ...) pattern: "+repr(token)))
        lexer.accept()
        return pattern

class Variable(Expression):
    def __init__(self, name:str):
        self.name = name
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.name)})"
    def evaluate(self, context:dict) -> Data:
        if False:
            yield
        return context[self.name]
    @classmethod
    def pull(cls, parser, lexer:Lexer) -> (bool, object):
        lexer.enter()
        name = lexer.pop()
        if name is None or not name.be(Name):
            return (lexer.conclude(False), None)
        return (lexer.conclude(True), cls(name.val))
class Call(Expression):
    def __init__(self, func:Expression, args:"tuple[Expression]"):
        self.func = func
        self.args = args
    def __repr__(self):
        body = (self.func, self.args)
        return f"{self.__class__.__name__}{body}"
    def evaluate(self, context:dict) -> Data:
        func = yield from self.func.evaluate(context)
        exprs = []
        for expr in self.args:
            val = yield from expr.evaluate(context)
            exprs.append(val)
        if not func.CALLABLE:
            raise TypeError("tried to call object of type "+func.__class__.__name__)
        result = yield from func.call(exprs, context=context)
        if result is None:
            raise RuntimeError("Defunct function did not return a value from "+func.__class__.__name__)
        if not isinstance(result, Flag) or not result.be("return"):
            raise RuntimeError("Defunct function did not return a return-flag from "+func.__class__.__name__+"; got: "+repr(result))
        if result.value is None:
            return func.DEFAULT
        return result.value
    @classmethod
    def pull(cls, parser, lexer:Lexer) -> (bool, object):
        success, function = parser.pop_expr(lexer, [cls])
        if not success:
            return (False, None)
        while lexer.peek(Paren["["]):
            lexer.enter()
            lexer.pop()
            args = []
            CLOSER = Paren["]"]
            SEPP = Punct[","]
            if lexer.peek(CLOSER):
                lexer.pop()
            else:
                while True:
                    success, expr = Exprs.pull(parser, lexer)
                    if not success:
                        raise lexer.reject(SyntaxError("failed to parse ARG in function call"))
                    args.append(expr)
                    if lexer.peek(CLOSER):
                        lexer.pop()
                        break
                    if lexer.peek(SEPP):
                        lexer.pop()
                        continue
                    raise lexer.reject(SyntaxError("expected comma in function call"))
            lexer.accept()
            function = cls(function, args)
        return (True, function)
class Exprs(Expression):
    def __init__(self, *args:Expression):
        self.args = args
    def __repr__(self):
        return f"{self.__class__.__name__}{self.args}"
    def evaluate(self, context:dict) -> Data:
        exprs = []
        for expr in self.args:
            val = yield from expr.evaluate(context)
            exprs.append(val)
        head = DataHalfComb.build(tuple(exprs))
        changed = True
        while changed:
            changed, head = head.step(context)
        return head
    @classmethod
    def pull(cls, parser, lexer:Lexer) -> (bool, object):
        lexer.enter()
        exprs = []
        while True:
            # !!! basically the only reference to Parser.pop_expr
            success, expr = parser.pop_expr(lexer)
            if not success:
                break
            exprs.append(expr)
        return (lexer.conclude(bool(exprs)), cls(*exprs))
class Parenthetical(Exprs):
    # parses with () around Exprs
    @classmethod
    def pull(cls, parser, lexer:Lexer) -> (bool, object):
        if not lexer.peek(Paren["("]):
            return (False, None)
        lexer.enter()
        lexer.pop() # "("
        success, instance = super().pull(parser, lexer)
        if not success:
            raise lexer.reject(SyntaxError("could not find an expression after '('"))
        if not lexer.peek(Paren[")"]):
            raise lexer.reject(SyntaxError("could not find closing paren ')'"))
        lexer.pop() # ")"
        return (lexer.conclude(True), instance)

#########################
#########################
#########################
#                       #
#                       #
# !!! END EXPRESSIONS   #
# !!! BEGIN STATEMENTS  #
#                       #
#                       #
#########################
#########################
#########################

class Assign(Statement):
    # ref:Expression, expr:Expression
    def __init__(self, ref:Expression, expr:Expression):
        self.ref = ref
        self.expr = expr
    def __repr__(self):
        body = (self.ref, self.expr)
        return f"{self.__class__.__name__}{body}"
    def state(self, context:dict) -> Flag:
        expr = yield from self.expr.evaluate(context)
        if isinstance(self.ref, Variable):
            context[self.ref.name] = expr
            return
        raise NotImplementedError(type(self.ref))
    @classmethod
    def pull(cls, parser, lexer:Lexer, indent:Indent) -> (bool, object):
        lexer.enter()
        name = lexer.pop()
        if name is None or not name.be(Name) or not lexer.peek(Special["="]):
            return (lexer.conclude(False), None)
        lexer.pop() # "="
        success, expr = Exprs.pull(parser, lexer)
        if not success:
            raise lexer.reject(SyntaxError("could not parse expr to the right of '='"))
        return (lexer.conclude(True), cls(Variable(name.val), expr))
class Def(Statement):
    SUPERDEFAULT = ZERO
    def __init__(self, name:Name, args:"tuple[Name]", defaults:"tuple[Expression]", block:Block, defreturn:Expression):
        self.name = name
        self.args = args
        self.defaults = defaults
        self.block = block
        self.defreturn = defreturn
    def __repr__(self):
        body = (self.name, self.args, self.defaults, self.block, self.defreturn)
        return f"{self.__class__.__name__}{body}"
    def state(self, context:dict) -> Flag:
        defaults = []
        for expr in self.defaults:
            default = yield from expr.evaluate(context)
            defaults.append(default)
        if self.defreturn is not None:
            defreturn = yield from self.defreturn.evaluate(context)
        else:
            defreturn = yield from self.SUPERDEFAULT.evaluate(context)
        context[self.name.val] = DataFunction(self.name, self.args, defaults, self.block, defreturn, context)
    @classmethod
    def pull(cls, parser, lexer:Lexer, indent:Indent) -> (bool, object):
        if not lexer.peek(Keyword["def"]):
            return (False, None)
        lexer.enter()
        lexer.pop() # "def"
        name = lexer.pop()
        if name is None or not name.be(Name):
            raise lexer.reject(SyntaxError("'def' keyword was not followed by a valid identifier"))
        args, dargs = cls.grab_args(parser, lexer, name)
        if lexer.peek(Special["->"]):
            lexer.pop() # "->"
            success, default = Exprs.pull(parser, lexer)
            if not success:
                raise lexer.reject(SyntaxError("'def "+name.val+" ... ->' expected an expression after '->' "))
        else:
            dummy = lexer.__class__("")
            dummy.push(cls.SUPERDEFAULT)
            default = Exprs.pull(parser, dummy)[1]
        colon = lexer.pop()
        if colon is None or not colon.be(Punct, ":"):
            raise lexer.reject(SyntaxError("function definition must always contain a : after the ARG sequence and possible DEFAULT expression ("+name.val+"); got "+repr(colon)))
        sub_indent = parser.get_next_indent(lexer, indent)
        success, block = Block.pull(parser, lexer, sub_indent)
        if not success:
            raise lexer.reject(SyntaxError("function definition must always be followed by a code block after ':'"))
        return (lexer.conclude(True), cls(name, args, dargs, block, default))
    @classmethod
    def grab_args(cls, parser, lexer:Lexer, name:Name) -> (list, list):
        lexer.enter()
        TAIL = "'"+name.val+"'"
        if not lexer.peek(Paren["["]):
            raise lexer.reject(SyntaxError("'def' statement requires a []-enclosed ARG sequence after the identifier "+TAIL))
        lexer.pop() # "["
        args = []
        dargs = []
        darg_seen = False
        CLOSER = Paren["]"]
        SEPP = Punct[","]
        if lexer.peek(CLOSER):
            lexer.pop() # "]"
        else:
            while True:
                success, name, default = cls.grab_arg(parser, lexer, CLOSER)
                if not success:
                    raise lexer.reject(SyntaxError("failed to parse function ARG sequence "+TAIL))
                args.append(name)
                if default is None:
                    if darg_seen:
                        raise lexer.reject(SyntaxError("no-default ARG appeared after a defaulted arg in function ARG sequence "+TAIL))
                else:
                    darg_seen = True
                    dargs.append(default)
                if lexer.peek(SEPP):
                    lexer.pop()
                    continue
                if lexer.peek(CLOSER):
                    lexer.pop()
                    break
                raise lexer.reject(SyntaxError("expected "+SEPP.val+" or "+CLOSER.val+" in function ARG sequence "+TAIL))
        return (args, dargs)
    @classmethod
    def grab_arg(cls, parser, lexer:Lexer, CLOSER:Paren) -> (bool, object, object):
        if lexer.peek(CLOSER):
            return (False, None, None)
        lexer.enter()
        name = lexer.pop()
        if name is None or not name.be(Name):
            raise lexer.reject(SyntaxError("expected a NAME in function ARG sequence."))
        if lexer.peek(Special["="]):
            lexer.pop()
            success, default = Exprs.pull(parser, lexer)
            if not success:
                raise lexer.reject(SyntaxError("could not parse expression to the right of '=' in function ARG sequence"))
        else:
            default = None
        return (lexer.conclude(True), name, default)
class Print(Statement):
    def __init__(self, expr:Expression):
        self.expr = expr
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.expr)})"
    @classmethod
    def pull(cls, parser, lexer:Lexer, indent:Indent) -> (bool, object):
        if not lexer.peek(Name("print")):
            return (False, None)
        lexer.enter()
        lexer.pop()
        success, expr = Exprs.pull(parser, lexer)
        if not success:
            raise lexer.reject(SyntaxError("print statement must be followed by an expr"))
        return (lexer.conclude(True), cls(expr))
    def state(self, context:dict) -> Flag:
        expr = yield from self.expr.evaluate(context)
        print(expr)
        print()
# control
class Else(Statement):
    '''utility class'''
    def __init__(self):
        raise TypeError("Do not instance Else")
    @classmethod
    def pull(cls, parser, lexer:Lexer, indent:Indent) -> (bool, object):
        if not lexer.peek(Keyword["else"]):
            return (False, Block.PASS)
        lexer.enter()
        lexer.pop()
        colon = lexer.pop()
        if colon is None or not colon.be(Punct, ":"):
            raise lexer.reject(SyntaxError("control substatement else expected a :"))
        sub_indent = parser.get_next_indent(lexer, indent)
        success, block = Block.pull(parser, lexer, sub_indent)
        return (lexer.conclude(success), block)
class Control(Statement):
    NAME = None
    HASCOND = True
    def __init__(self, cond:Expression, true:Block, false:Block):
        self.cond = cond
        self.true = true
        self.false = false
    def __repr__(self):
        body = (self.cond, self.true, self.false)
        return f"{self.__class__.__name__}{body}"
    @classmethod
    def pull(cls, parser, lexer:Lexer, indent:Indent) -> (bool, object):
        if not lexer.peek(Keyword[cls.NAME]):
            return (False, None)
        lexer.enter()
        lexer.pop()
        if cls.HASCOND:
            success, condition = Exprs.pull(parser, lexer)
            if not success:
                raise lexer.reject(SyntaxError("control statement "+cls.NAME+" expected a condition after its keyword"))
        else:
            condition = None
        colon = lexer.pop()
        if colon is None or not colon.be(Punct, ":"):
            raise lexer.reject(SyntaxError("control statement "+cls.NAME+" expected a : after its condition"))
        sub_indent = parser.get_next_indent(lexer, indent)
        success, block = Block.pull(parser, lexer, sub_indent)
        if not success:
            raise lexer.reject(SyntaxError("could not pull main block for control statement of type "+cls.NAME))
        _, elseblock = Else.pull(parser, lexer, indent)
        return (lexer.conclude(True), cls(condition, block, elseblock))
class If(Control):
    NAME = "if"
    def state(self, context:dict) -> Flag:
        cond = yield from self.cond.evaluate(context)
        if cond.to_bool():
            result = yield from self.true.process(context)
        else:
            result = yield from self.false.process(context)
        return result
class While(Control):
    NAME = "while"
    def state(self, context:dict) -> Flag:
        cond = yield from self.cond.evaluate(context)
        while cond.to_bool():
            result = yield from self.true.process(context)
            if result is None or result.be("continue"):
                pass
            elif result.be("break"):
                break
            else:
                return result
            cond = yield from self.cond.evaluate(context)
        else:
            result = yield from self.false.process(context)
            return result
class Dowhile(Control):
    NAME = "dowhile"
    def state(self, context:dict) -> Flag:
        cond = yield from self.cond.evaluate(context)
        while cond.to_bool():
            result = yield from self.true.process(context)
            if result is None or result.be("continue"):
                pass
            elif result.be("break"):
                break
            else:
                return result
            cond = yield from self.cond.evaluate(context)
        else:
            result = yield from self.false.process(context)
            return result
class Do(Control):
    NAME = "do"
    HASCOND = False
    def state(self, context:dict) -> Flag:
        result = yield from self.true.process(context)
        if result is None or result.be("continue"):
            result = yield from self.false.process(context)
            return result
        elif result.be("break"):
            return None
        else:
            return result

# interrupt
class Interrupt(Statement):
    NAME = None
    HASEXPR = False
    INST = None
    FLAG = None
    def __init__(self, expr:Expression=None):
        self.expr = expr
    def __repr__(self):
        if self.expr is None:
            return f"{self.__class__.__name__}.INST"
        return f"{self.__class__.__name__}({repr(self.expr)})"
    def state(self, context:dict) -> Flag:
        if self.HASEXPR and self.expr is not None:
            expr = yield from self.expr.evaluate(context)
            return Flag(self.NAME, expr)
        return self.FLAG
    @classmethod
    def pull(cls, parser, lexer:Lexer, indent:Indent) -> (bool, object):
        if not lexer.peek(Keyword[cls.NAME]):
            return (False, None)
        lexer.enter()
        lexer.pop()
        result = cls.INST
        if cls.HASEXPR:
            success, expr = Exprs.pull(parser, lexer)
            if success:
                result = cls(expr)
        return (lexer.conclude(True), result)
class Pass(Interrupt):
    NAME = "pass"
    INST = Block.PASS
class Continue(Interrupt):
    NAME = "continue"
    FLAG = Flag(NAME)
Continue.INST = Continue(None)
class Break(Interrupt):
    NAME = "break"
    FLAG = Flag(NAME)
Break.INST = Break(None)
class Return(Interrupt):
    NAME = "return"
    HASEXPR = True
    FLAG = Flag(NAME)
Return.INST = Return(None)

#####################
#####################
#####################
#                   #
#                   #
# !!! END AST       #
# !!! BEGIN PARSER  #
#                   #
#                   #
#####################
#####################
#####################

def seqyield(*args):
    for arg in args:
        yield from iter(arg)

class Parser:
    PRIORITY = [Call]
    CONSTANTS = [ConstInt, ConstStr, ConstComb]
    EXPR_ETC = [DataBuiltin, Variable, Parenthetical]
    EXPRS = [PRIORITY, CONSTANTS, EXPR_ETC]
    #
    CONTROLS = [If, While, Dowhile, Do]
    INTERRUPTS = [Pass, Continue, Break, Return]
    STMT_ETC = [Print, Def, Assign]
    STMTS = [CONTROLS, INTERRUPTS, STMT_ETC]
    @classmethod
    def parse(cls, code:str) -> (bool, Block):
        lexer = Lexer(code)
        return Block.pull(cls, lexer, Indent[""])
    @classmethod # called by Exprs
    def pop_expr(cls, lexer:Lexer, exclusions:tuple=None) -> (bool, Expression):
        if exclusions is None:
            exclusions = ()
        for GROUP in cls.EXPRS:
            for Expr in GROUP:
                if Expr in exclusions:
                    continue
                success, expr = Expr.pull(cls, lexer)
                if success:
                    return (True, expr)
        return (False, None)
    @classmethod
    def discard(cls, lexer:Lexer, indent:Indent):
        lexer.enter()
        last_indent = None
        while True:
            token = lexer.pop()
            if token is None:
                if last_indent is not None:
                    break # error somewhere else
                return lexer.accept()
            if token is not None:
                if token.be(Comment):
                    token = lexer.pop()
                    if not token.be(Indent):
                        raise lexer.reject(SyntaxError("Comments must always be followed by a newline (\\n)"))
                if token.be(Indent):
                    while token is not None and token.be(Indent):
                        last_indent = token
                        token = lexer.pop()
                    lexer.push(token)
                    if token is not None and token.be(Comment):
                        continue
                    break
            if last_indent is None:
                raise lexer.reject(SyntaxError("Unexpected token while searching for newline "+repr(token)))
            break
        lexer.push(last_indent)
        lexer.accept()
    @classmethod # called by Block
    def pop_stmt(cls, lexer:Lexer, indent:Indent) -> (bool, Statement):
        lexer.enter()
        cls.discard(lexer, indent)
        token = lexer.pop()
        if token is None or not token.be(Indent) or token.oversee(indent):
            return (lexer.conclude(False), None)
        if indent.oversee(token):
            raise lexer.reject(SyntaxError("Unexpected indent: "+repr(indent.val)+"; "+repr(token.val)+";"+repr(lexer.pop())+";"+repr(lexer.pop())))
        if token.val != indent.val:
            raise lexer.reject(SyntaxError("Unexpected mismatched indents: "+repr(indent.val)+"; "+repr(token.val)))
        success, result = cls.stmt_util(token, lexer)
        return (lexer.conclude(success), result)
    @classmethod
    def stmt_util(cls, token:Token, lexer:Lexer) -> (bool, Statement):
        lexer.enter()
        for GROUP in cls.STMTS:
            for Stmt in GROUP:
                success, stmt = Stmt.pull(cls, lexer, token)
                if success:
                    return (lexer.conclude(True), stmt)
        peek = lexer.pop()
        if peek is None:
            return (lexer.conclude(False), None)
        raise lexer.reject(SyntaxError("Could not parse statement after indentation "+repr(token)+"; next token: "+repr(lexer.pop())))
    @classmethod # called by statements that expect an indented block to follow
    def get_next_indent(cls, lexer:Lexer, indent:Indent) -> Indent:
        '''this function should be treated like a PEEK'''
        lexer.enter()
        # specifically, the next SUB indent
        cls.discard(lexer, indent)
        # .discard only returns when the next token is an indent
        token = lexer.pop()
        if token is None:
            raise lexer.reject(SyntaxError("unexpected EOF while probing the next indentation"))
        lexer.push(token)
        lexer.accept()
        if not token.be(Indent):
            raise SyntaxError("unexpected token while probing the next indentation: "+repr(token))
        if not indent.oversee(token):
            raise SyntaxError("mismatched indentation spotted while probing the next indentation: "+repr(indent.val)+"; "+repr(token.val))
        return token

##########################
##########################
##########################
#                        #
#                        #
# !!! END PARSER         #
# !!! BEGIN INTERPRETER  #
#                        #
#                        #
##########################
##########################
##########################

class Interpreter:
    def __init__(self, block:Block, context:dict=None):
        self.block = block
        self.context = {} if context is None else context
    def interpret(self):
        return ignore_iterator(self.block.process(self.context))
