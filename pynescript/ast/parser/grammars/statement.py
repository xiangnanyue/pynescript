from pyparsing import (
    Forward,
    Suppress,
    Opt,
)
from pyparsing import delimited_list

from pynescript import ast
from pynescript.ast.parser.tokens import (
    IMPORT,
    BREAK,
    CONTINUE,
    COMMA,
    SLASH,
    IDENTIFIER,
    AS,
    LPAREN,
    RPAREN,
    RIGHT_DOUBLE_ARROW,
    ASSIGN,
)
from pynescript.ast.parser.parser_elements import (
    ResultNameableForward as Forward,
    ConvertToNode,
)

assignment = Forward()
structure = Forward()
expression = Forward()
local_body = Forward()

common_statement = assignment | structure | expression
common_statement.set_name("common_statement")

import_statement = (
    Suppress(IMPORT)
    + IDENTIFIER
    + Suppress(SLASH)
    + IDENTIFIER
    + Suppress(SLASH)
    + IDENTIFIER
    + Suppress(AS)
    + IDENTIFIER
)
import_statement.set_name("import_statement")

default_value = expression

full_parameter_declaration = (
    IDENTIFIER("name") + Suppress(ASSIGN) + default_value("default_value")
)

name_only_parameter_declaration = IDENTIFIER("name")

parameter_declaration = ConvertToNode(ast.Parameter)(
    full_parameter_declaration | name_only_parameter_declaration
)

parameter_declaration.set_name("parameter_declaration")

parameter_list = delimited_list(parameter_declaration, COMMA)

function_declaration = ConvertToNode(ast.FunctionDef)(
    IDENTIFIER("name")
    + Suppress(LPAREN)
    + Opt(parameter_list("parameters"))
    + Suppress(RPAREN)
    + Suppress(RIGHT_DOUBLE_ARROW)
    + local_body("body")
)
function_declaration.set_name("function_declaration")

global_only_statement = import_statement | function_declaration

break_statement = ConvertToNode(ast.Break)(BREAK)
continue_statement = ConvertToNode(ast.Continue)(CONTINUE)

jump_statement = break_statement | continue_statement

local_only_statement = jump_statement

global_atomic_statement = global_only_statement | common_statement
local_atomic_statement = local_only_statement | common_statement

global_atomic_statement.set_name("global_atomic_statement")
local_atomic_statement.set_name("local_atomic_statement")

global_statement = delimited_list(global_atomic_statement, COMMA)
local_statement = delimited_list(local_atomic_statement, COMMA)

global_statement.set_name("global_statement")
local_statement.set_name("local_statement")

statement = (
    break_statement
    | continue_statement
    | import_statement
    | function_declaration
    | assignment
    | structure
    | expression
)
statement.set_name("statement")
