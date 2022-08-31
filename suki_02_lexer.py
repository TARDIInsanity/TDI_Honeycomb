# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 21:30:26 2022

@author: github.com/TARDIInsanity
"""

from weakref import WeakValueDictionary
import TDI_parse_integer as lex_integer_2 # https://github.com/TARDIInsanity/TDI_parse_integer

class Token:
    __slots__ = ("val", "__weakref__")
    def __init__(self, val):
        self.val = val
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.val)})"
    def be(self, kind, val=None) -> bool:
        return isinstance(self, kind) and (val is None or self.val == val)
    def like(self, kin) -> bool:
        return isinstance(self, type(kin)) and self.val == kin.val
    @classmethod
    def pull(cls, code:str) -> (bool, object, str):
        raise NotImplementedError
class ClosedClass(Token):
    __slots__ = ()
    '''a class of tokens which has a fully enumerated finite set of options'''
    INST = {}
    NAME = sorted_set(())
    @staticmethod
    def invalid(token:str, char:str) -> bool:
        '''if followed by a certain char, invalidate the whole expression?'''
        return False
    @classmethod
    def pull(cls, code:str) -> (bool, object, str):
        segment = code
        for length in cls.NAME[None]:
            segment, char = segment[:length], segment[length:length+1]
            if segment not in cls.NAME[length]:
                continue
            if cls.invalid(segment, char):
                continue
            return (True, cls[segment], code[length:])
        return (False, None, code)
    def __repr__(self):
        return f"{self.__class__.__name__}[{repr(self.val)}]"
    def __class_getitem__(cls, name:str) -> Token:
        if name not in cls.INST:
            cls.INST[name] = cls(name)
        return cls.INST[name]
class Paren(ClosedClass):
    __slots__ = ()
    NAME = sorted_set("()[]{}")
class Punct(ClosedClass):
    __slots__ = ()
    NAME = sorted_set(".,:;")
class Special(ClosedClass):
    __slots__ = ()
    NAME = sorted_set({"=", "->"})
class Keyword(ClosedClass):
    __slots__ = ()
    NAME = sorted_set("comb, def, if, while, dowhile, do, else, break, continue, return".split(", "))
    @staticmethod
    def invalid(token:str, char:str) -> bool:
        return char.isalnum()

class Noncode(Token):
    __slots__ = ()
class Comment(Noncode):
    __slots__ = ("multiline",)
    LINE = sorted_set({"#"})
    MULTILINE = sorted_dict({"###":"###"})
    def __init__(self, comment:str, ismultiline:bool):
        super().__init__(comment)
        self.multiline = ismultiline
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.val)}, {self.multiline})"
    @classmethod
    def pull(cls, code:str) -> (bool, object, str):
        segment = code
        for length in cls.MULTILINE[None]:
            segment = segment[:length]
            if segment not in cls.MULTILINE[length]:
                continue
            closer = cls.MULTILINE[length][segment]
            index = code.find(closer, length)
            if index == -1:
                continue
            comment, code = code[length:index], code[index+len(closer):]
            print(comment)
            return (True, cls(comment, True), code)
        segment = code
        for length in cls.LINE[None]:
            segment = segment[:length]
            if segment not in cls.LINE[length]:
                continue
            index = code.find("\n", length)
            if index == -1:
                return (True, cls(code[1:]), "")
            comment, code = code[length:index], code[index:]
            print(comment)
            return (True, cls(comment, False), code)
        return (False, None, code)
class Indent(Noncode):
    __slots__ = ()
    INST = WeakValueDictionary()
    @classmethod
    def pull(cls, code:str) -> (bool, int, str):
        if not code.startswith("\n"):
            return (False, None, code)
        for index, char in enumerate(code):
            if index and char == "\n":
                break
            if not char.isspace():
                break
        else:
            index += 1
        return (True, cls[code[1:index]], code[index:])
    def __repr__(self):
        return f"{self.__class__.__name__}[{repr(self.val)}]"
    def __class_getitem__(cls, name:str) -> Token:
        if name not in cls.INST:
            result = cls(name)
            cls.INST[name] = result
            return result
        return cls.INST[name]
    def oversee(self, indent):
        return indent.val != self.val and indent.val.startswith(self.val)
class Name(Token):
    __slots__ = ()
    @classmethod
    def pull(cls, code:str) -> (bool, object, str):
        success, name, code = pull_ide(code)
        return (success, cls(name), code)
class Constant(Token):
    __slots__ = ()
class Integer(Constant):
    __slots__ = ()
    @classmethod
    def pull(cls, code:str) -> (bool, object, str):
        success, integer, code = IntPuller.pull_int(code)
        return (success, cls(integer), code)
class String(Constant):
    __slots__ = ()
    @classmethod
    def pull(cls, code:str) -> (bool, object, str):
        for opener in {chr(39), chr(34)}:
            if code.startswith(opener):
                success, string, code = pull_string(code, opener, opener, pyth_escape)
                return (success, cls(string), code)
        return (False, None, code)

class Lexer:
    TOKENS = [Comment, Indent, Special, Paren, Punct, Keyword, Name, Integer, String]
    def __init__(self, code:str):
        self.code = code
        self.buffer = []
    def strip(self):
        i = 0
        for i, char in enumerate(self.code):
            if char not in " \t":
                break
        else:
            i += 1
        if i:
            self.code = self.code[i:]
    def pop(self) -> Token:
        if self.buffer:
            return self.buffer.pop()
        if not self.code:
            return None
        self.strip()
        for token in self.TOKENS:
            success, token, self.code = token.pull(self.code)
            if success:
                return token
        return None
    def push(self, val:Token):
        self.buffer.append(val)

