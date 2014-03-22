/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     IDENTIFIER = 258,
     FLOATCONSTANT = 259,
     INTCONSTANT = 260,
     STRING_CONST = 261,
     LBRACE = 262,
     RBRACE = 263,
     RES_KEYWORD = 264,
     TEX_KEYWORD = 265,
     RENDER_TARGET_KEYWORD = 266,
     SHADER_KEYWORD = 267,
     DEF = 268,
     EQUAL = 269,
     SEMI_COLON = 270
   };
#endif
/* Tokens.  */
#define IDENTIFIER 258
#define FLOATCONSTANT 259
#define INTCONSTANT 260
#define STRING_CONST 261
#define LBRACE 262
#define RBRACE 263
#define RES_KEYWORD 264
#define TEX_KEYWORD 265
#define RENDER_TARGET_KEYWORD 266
#define SHADER_KEYWORD 267
#define DEF 268
#define EQUAL 269
#define SEMI_COLON 270




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 40 "res_script_parser.y"
{
	struct {
		int line;
		union {
			hqint32 iconst;
			hqfloat64 fconst;
			char* string;
		};
	} lex;
	
	HQEngineResParserNode * node;
	HQEngineResParserNode::ValueType value;
}
/* Line 1529 of yacc.c.  */
#line 93 "gen_res_script_parser.hpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE hqengine_res_parser_lval;

