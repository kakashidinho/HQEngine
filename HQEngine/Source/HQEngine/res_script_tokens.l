/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

%{
#include <stdlib.h>

#include "HQEngineCommonInternal.h"
#include "HQEngineResParserCommon.h"
#include "gen_res_script_parser.hpp"

#define SAVE_LINE (yylval.lex.line = yylineno)

#if defined _MSC_VER
#	define fileno _fileno
#else
#endif

#define yylval hqengine_res_parser_lval
#define yyerror hqengine_res_parser_error

extern HQDataReaderStream * hqengine_res_parser_input_stream;
extern "C"  void yyerror(const char *s);

#define YY_INPUT(buf,result,max_size) (result = hqengine_res_parser_input_stream->ReadBytes(buf, 1, max_size)) 

extern "C" int yywrap() { return 1; }

void hqengine_res_parser_clean_up_lexer(); 

%}


%option   yylineno
%option	  never-interactive

%x	COMMENT_BLOCK
%x	COMMENT_LINE

D           [0-9]
L           [a-zA-Z_]
H           [a-fA-F0-9]
E           [Ee][+-]?{D}+
O           [0-7]
F           [hHfF]
I           [uUlL]
     
%%

[ \t\n\r]                 {   ;}

";"					  {return SEMI_COLON; }

"{"					  {return LBRACE;}
"}"					  {return RBRACE;}
"//"				  {BEGIN(COMMENT_LINE);}
"/*"				  {BEGIN(COMMENT_BLOCK);}

"resources"			  {	SAVE_LINE; return (RES_KEYWORD);}
"texture"			  {	SAVE_LINE; return (TEX_KEYWORD);}
"render_target"		  {	SAVE_LINE; return (RENDER_TARGET_KEYWORD);}
"shader"			  {	SAVE_LINE; return (SHADER_KEYWORD);}

"define"			  {	SAVE_LINE; return (DEF);}
"="					  {	SAVE_LINE; return (EQUAL);}	

{L}({L}|{D}|"-")*	  {  SAVE_LINE; yylval.lex.string = HQEngineHelper::GlobalPoolMallocString(yytext, yyleng);  return IDENTIFIER; }
{D}+{E}{F}?           {  SAVE_LINE; yylval.lex.fconst = strtod(yytext, NULL); return (FLOATCONSTANT); }
{D}+"."{D}*({E})?{F}? {  SAVE_LINE; yylval.lex.fconst = strtod(yytext, NULL); return (FLOATCONSTANT); }
"."{D}+({E})?{F}?     {  SAVE_LINE; yylval.lex.fconst = strtod(yytext, NULL); return (FLOATCONSTANT); }
{D}+{F}               {  SAVE_LINE; yylval.lex.fconst = strtod(yytext, NULL); return (FLOATCONSTANT); }

0[xX]{H}+{I}?         {  SAVE_LINE; yylval.lex.iconst = (hqint32)strtol(yytext, NULL, 0); return (INTCONSTANT); }
0{O}+{I}?             {  SAVE_LINE; yylval.lex.iconst = (hqint32)strtol(yytext, NULL, 0); return (INTCONSTANT); }
{D}+{I}?              {  SAVE_LINE; yylval.lex.iconst = (hqint32)strtol(yytext, NULL, 0); return (INTCONSTANT); }

\"[^\"]*\"			  { SAVE_LINE;  yylval.lex.string = HQEngineHelper::GlobalPoolMallocString(yytext + 1, yyleng - 2);  return STRING_CONST; }

.					  {	 
	char errMsg[256]; 
	sprintf(errMsg, "unexpected character %s", yytext); 
	yyerror (errMsg);  
	yyterminate(); 
}


<COMMENT_BLOCK>{
	"*/"      BEGIN(INITIAL);
	[^*\n]+   // eat comment in chunks
	"*"       // eat the lone star
	\n        
}
 
<COMMENT_LINE>{
	\n      {	BEGIN(INITIAL);	}
	[^\n]+   // eat comment in chunks
}

%%


void hqengine_res_parser_clean_up_lexer(){
#ifdef YY_CURRENT_BUFFER_LVALUE /*flex version found in other systems other than Windows*/
	hqengine_res_parser_lex_destroy();
#else
	if (YY_CURRENT_BUFFER)
	{
		yy_delete_buffer(YY_CURRENT_BUFFER);
		YY_CURRENT_BUFFER = NULL;
	}
#endif
}

void hqengine_res_parser_start_lexer()
{
#ifdef YY_CURRENT_BUFFER_LVALUE /*flex version found in other systems other than Windows*/
	yy_init_globals();
#else
	yyrestart(0);
#endif
}
