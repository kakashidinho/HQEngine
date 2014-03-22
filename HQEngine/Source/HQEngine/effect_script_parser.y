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
	#define yylineno hqengine_effect_parser_lineno
	
	struct YYLTYPE;
	
	std::stringstream	* hqengine_effect_parser_log_stream = NULL;
	HQDataReaderStream * hqengine_effect_parser_input_stream = NULL;
	HQEngineEffectParserNode * hqengine_effect_parser_root_result = NULL;
	static HQLinkedList<HQEngineEffectParserNode*> g_allocatedNode;
	
	void hqengine_effect_parser_recover_from_error();
	void hqengine_effect_parser_clean_up();
	static HQEngineEffectParserNode * yynew_node(const char *type, int line);
	static HQEngineEffectParserNode * yynew_node();
	
    extern int yylex();
    extern void hqengine_effect_parser_start_lexer();
    extern void hqengine_effect_parser_clean_up_lexer();
    extern int yylineno;
    extern "C" void yyerror(const char *s) ;
%}

%error-verbose

%union {
	struct {
		int line;
		union {
			int iconst;
			double fconst;
			char* string;
		};
	} lex;
	
	HQEngineEffectParserNode * node;
	HQEngineEffectParserNode::ValueType value;
}

%token <lex> IDENTIFIER FLOATCONSTANT INTCONSTANT STRING_CONST EQUAL SEMI_COLON
%token <lex> LBRACE RBRACE TECHNIQUES_GROUP_KEYWORD TECHNIQUE_KEYWORD PASS_KEYWORD BLEND_KEYWORD  STENCIL_KEYWORD
%token <lex> OUTPUTS_KEYWORD BORDER_COLOR_KEYWORD
%token <lex> TEXUNIT OUTPUT CUBE_FACE

%type  <node> root technique_blocks technique_block block child_elems single_child assignment
%type  <node> border_color_assignment  output_assignment
%type  <node> texture_unit_assignment
%type  <value> identifier_or_value
%type  <value> string_or_identifier_or_value
%type  <lex>  string_or_identifier

%start root

%%
root: technique_blocks { hqengine_effect_parser_root_result = $1; }
	;

technique_blocks: 
	technique_block { 
		$$ = yynew_node("technique_blocks", $1->GetSourceLine());
		$$->AddChild($1);
	}
	| technique_blocks technique_block {
		$$->AddChild($1);
	}
	;
technique_block:
	TECHNIQUES_GROUP_KEYWORD block {
		$$ = $2;
		$$->SetType("techniques");
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
	TECHNIQUE_KEYWORD string_or_identifier block {
		$$ = $3;
		$$->SetType("technique");
		$$->SetSourceLine($1.line);
		$$->SetAttribute("name", $2.string);
	}
	| PASS_KEYWORD string_or_identifier block {
		$$ = $3;
		$$->SetType("pass");
		$$->SetSourceLine($1.line);
		$$->SetAttribute("name", $2.string);
	}
	| BLEND_KEYWORD block {
		$$ = $2;
		$$->SetType("blend");
		$$->SetSourceLine($1.line);
	}
	| STENCIL_KEYWORD block {
		$$ = $2;
		$$->SetType("stencil");
		$$->SetSourceLine($1.line);
	}
	| OUTPUTS_KEYWORD block {
		$$ = $2;
		$$->SetType("custom_targets");
		$$->SetSourceLine($1.line);
	}
	| assignment { $$ = $1; }
	;

assignment:
	texture_unit_assignment   {$$ = $1;}
	| border_color_assignment  {$$ = $1;} 
	| output_assignment   {$$ = $1;}
	| IDENTIFIER EQUAL string_or_identifier_or_value {
		$$ = yynew_node($1.string, $1.line);
		$$->SetAttribute("value", $3);	
	}
	;
	
texture_unit_assignment:
	TEXUNIT block {
		$$ = $2;
		$$->SetType($1.string);
		$$->SetSourceLine($1.line);
	}
	| TEXUNIT EQUAL string_or_identifier {
		$$ = yynew_node($1.string, $1.line);
		HQEngineEffectParserNode * source_elem = yynew_node("source", $3.line);
		source_elem->SetAttribute("value", $3.string);
		$$->AddChild(source_elem);
	}
	;

border_color_assignment:
	BORDER_COLOR_KEYWORD EQUAL INTCONSTANT INTCONSTANT INTCONSTANT INTCONSTANT {
		$$ = yynew_node("border_color", $1.line);
		hquint32 color = (((hquint32)$3.iconst) & 0xff) << 24;
				color = color | (((hquint32)$4.iconst) & 0xff) << 16;
				color = color | (((hquint32)$5.iconst) & 0xff) << 8;
				color = color | ((hquint32)$6.iconst) & 0xff;
				
		$$->SetAttribute("value", (hqint32) color);
	}
	;

output_assignment:
	OUTPUT EQUAL string_or_identifier {
		$$ = yynew_node($1.string, $1.line);
		$$->SetAttribute("value", $3.string);
	}
	| OUTPUT EQUAL string_or_identifier CUBE_FACE {
		$$ = yynew_node($1.string, $1.line);
		$$->SetAttribute("value", $3.string);
		$$->SetAttribute("cube_face", $4.string);
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

void yyerror(const char *s) { 
	if (hqengine_effect_parser_log_stream) 
		*hqengine_effect_parser_log_stream << " " << yylineno << " : " << s << std::endl; 
}

int hqengine_effect_parser_scan()
{
	hqengine_effect_parser_start_lexer();//start lexer
	
	hqengine_effect_parser_root_result = NULL;//invalidate result
	
	int re = yyparse();//parse
	
	//clean up
	if (re)
		hqengine_effect_parser_recover_from_error();
	else
		hqengine_effect_parser_clean_up();
		
	return re;
}

void hqengine_effect_parser_recover_from_error(){
	hqengine_effect_parser_root_result = NULL;//invalidate result

	HQLinkedList<HQEngineEffectParserNode*>::Iterator ite;
	g_allocatedNode.GetIterator(ite);
	for (; !ite.IsAtEnd(); ++ite)
	{
		delete *ite;
	}
	
	hqengine_effect_parser_clean_up();
}

void hqengine_effect_parser_clean_up() {
	
	hqengine_effect_parser_clean_up_lexer();//clean up lexer
	
	g_allocatedNode.RemoveAll();
	yylineno = 1;//reset line number
}

static HQEngineEffectParserNode * yynew_node(const char *type, int line)
{
	HQEngineEffectParserNode *new_node = HQ_NEW HQEngineEffectParserNode(type, line);
	g_allocatedNode.PushBack(new_node);//for keeping track of allocated nodes
	return new_node;
}

static HQEngineEffectParserNode * yynew_node()
{
	HQEngineEffectParserNode *new_node = HQ_NEW HQEngineEffectParserNode();
	g_allocatedNode.PushBack(new_node);//for keeping track of allocated nodes
	return new_node;
}