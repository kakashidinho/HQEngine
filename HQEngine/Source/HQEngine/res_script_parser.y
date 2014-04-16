/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

%{
	#include "HQEngineCommonInternal.h"
	#include "HQEngineResParserCommon.h"
	
	#include <sstream>
	
	#define YY_DECL int yylex()
	#define yylineno hqengine_res_parser_lineno
	
	std::stringstream	* hqengine_res_parser_log_stream = NULL;
	HQDataReaderStream * hqengine_res_parser_input_stream = NULL;
	HQEngineResParserNode * hqengine_res_parser_root_result = NULL;
	
	static HQLinkedList<HQEngineResParserNode*> g_allocatedNode;
	
	void hqengine_res_parser_recover_from_error();
	void hqengine_res_parser_clean_up();
	static HQEngineResParserNode * yynew_node(const char *type, int line);
	static HQEngineResParserNode * yynew_node();
	
    extern int yylex();
    extern void hqengine_res_parser_start_lexer();
    extern void hqengine_res_parser_clean_up_lexer();
    extern int yylineno;
    extern "C" void yyerror(const char *s) ;
%}

%error-verbose

%union {
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

%token <lex> IDENTIFIER FLOATCONSTANT INTCONSTANT STRING_CONST
%token <lex> LBRACE RBRACE RES_KEYWORD TEX_KEYWORD TEX_UAV_KEYWORD RENDER_TARGET_KEYWORD SHADER_KEYWORD DEF EQUAL SEMI_COLON

%type  <lex>  string_or_identifier
%type  <node> root resource_blocks resource_block block child_elems single_child assignment definition
%type  <value> identifier_or_value
%type  <value> string_or_identifier_or_value

%start root

%%
root: resource_blocks { hqengine_res_parser_root_result = $1; }
	;

resource_blocks: 
	resource_block { 
		$$ = yynew_node("resource_blocks", $1->GetSourceLine());
		$$->AddChild($1);
	}
	| resource_blocks resource_block {
		$$->AddChild($1);
	}
	;
resource_block:
	RES_KEYWORD block {
		$$ = $2;
		$$->SetType("resources");
		$$->SetSourceLine($1.line);
	}
	;
	
block : LBRACE child_elems RBRACE {
		$$ = $2;
	}
	;

child_elems:
	single_child { 
		$$ = yynew_node();
		$$->AddChild($1);
	}
	| child_elems single_child {
		$$->AddChild($2);
	}
	;
	
single_child:
	TEX_KEYWORD block {
		$$ = $2;
		$$->SetType("texture");
		$$->SetSourceLine($1.line);
	}
	| TEX_KEYWORD string_or_identifier block {
		$$ = $3;
		$$->SetType("texture");
		$$->SetSourceLine($1.line);
		$$->SetAttribute("name", $2.string);
	}
	| RENDER_TARGET_KEYWORD string_or_identifier block {
		$$ = $3;
		$$->SetType("render_target");
		$$->SetSourceLine($1.line);
		$$->SetAttribute("name", $2.string);
	}
	| SHADER_KEYWORD block {
		$$ = $2;
		$$->SetType("shader");
		$$->SetSourceLine($1.line);
	}
	| SHADER_KEYWORD string_or_identifier block {
		$$ = $3;
		$$->SetType("shader");
		$$->SetSourceLine($1.line);
		$$->SetAttribute("name", $2.string);
	}
	| TEX_UAV_KEYWORD string_or_identifier block {
		$$ = $3;
		$$->SetType("texture_uav");
		$$->SetSourceLine($1.line);
		$$->SetAttribute("name", $2.string);
	}
	| assignment { $$ = $1; }
	| definition { $$ = $1; }
	;

assignment: 
	IDENTIFIER EQUAL string_or_identifier_or_value {
		$$ = yynew_node($1.string, $1.line);
		$$->SetAttribute("value", $3);	
	}
	;
	
definition:
	DEF IDENTIFIER  {
		$$ = yynew_node("definition", $1.line);
		$$->SetAttribute("name", $2.string);	
	}
	| DEF IDENTIFIER EQUAL identifier_or_value{
		$$ = yynew_node("definition", $1.line);
		$$->SetAttribute("name", $2.string);
		
		
		const char *value = NULL;
		switch ($4.type)
		{
		case HQEngineCommonResParserNode::ValueType::INTEGER_TYPE:
			{
				char buf[256];
				sprintf(buf, "%d", $4.ivalue);
				value = HQEngineHelper::GlobalPoolMallocString(buf, sizeof(buf));
			}
			break;
		case HQEngineCommonResParserNode::ValueType::FLOAT_TYPE:
			{
				char buf[256];
				sprintf(buf, "%f", $4.fvalue);
				value = HQEngineHelper::GlobalPoolMallocString(buf, sizeof(buf));
			}
			break;
		case HQEngineCommonResParserNode::ValueType::STRING_TYPE:
			value = $4.string;
			break;
		}
		$$->SetAttribute("value", value);	
	}
	;

string_or_identifier_or_value:
	identifier_or_value {$$ = $1;}
	| STRING_CONST {$$.type = HQEngineCommonResParserNode::ValueType::STRING_TYPE;  $$.string = $1.string;}
	;

identifier_or_value:
	IDENTIFIER {$$.type = HQEngineCommonResParserNode::ValueType::STRING_TYPE;  $$.string = $1.string;}
	| FLOATCONSTANT  {$$.type = HQEngineCommonResParserNode::ValueType::FLOAT_TYPE;  $$.fvalue = $1.fconst;}
	| INTCONSTANT {$$.type = HQEngineCommonResParserNode::ValueType::INTEGER_TYPE;  $$.ivalue = $1.iconst;}
	;
	
string_or_identifier:
	IDENTIFIER  {$$ = $1;}
	| STRING_CONST {$$ = $1;}
	;	

%%

void yyerror(const char *s) 
{ 
	if (hqengine_res_parser_log_stream) 
	{
		*hqengine_res_parser_log_stream << " " << yylineno << " : " <<  s << std::endl; 
	}
}	

int hqengine_res_parser_scan()
{
	hqengine_res_parser_start_lexer();//start lexer
	
	hqengine_res_parser_root_result = NULL;
	
	int re = yyparse();//parse
	
	//clean up
	if (re)
		hqengine_res_parser_recover_from_error();
	else
		hqengine_res_parser_clean_up();
		
	return re;
}
	
void hqengine_res_parser_recover_from_error(){
	hqengine_res_parser_root_result = NULL;//invalidate result
	
	HQLinkedList<HQEngineResParserNode*>::Iterator ite;
	g_allocatedNode.GetIterator(ite);
	for (; !ite.IsAtEnd(); ++ite)
	{
		delete *ite;
	}
	
	hqengine_res_parser_clean_up();
}

void hqengine_res_parser_clean_up() {
	hqengine_res_parser_clean_up_lexer();//clean up lexer
	
	g_allocatedNode.RemoveAll();
	yylineno = 1;//reset line number
}

static HQEngineResParserNode * yynew_node(const char *type, int line)
{
	HQEngineResParserNode *new_node = HQ_NEW HQEngineResParserNode(type, line);
	g_allocatedNode.PushBack(new_node);//for keeping track of allocated nodes
	return new_node;
};

static HQEngineResParserNode * yynew_node()
{
	HQEngineResParserNode *new_node = HQ_NEW HQEngineResParserNode();
	g_allocatedNode.PushBack(new_node);//for keeping track of allocated nodes
	return new_node;
};