#pragma once

#include <string>

enum TokenKind {
	END_OF_FILE = 0,
	SEMI_COLON = 1,
	COLON,
	LBRACE, RBRACE,
	LBRACKET, RBRACKET,
	FLOATCONSTANT,
	INTCONSTANT,
	SCALAR,
	VECTOR,
	MAT,
	IDENTIFIER,
	STRING_CONST,
	UNIFORM,
	HASH_SIGN,
	PRAGMA,
	PACK_MATRIX,
	LPAREN,
	RPAREN,
	COL_MAJOR,
	ROW_MAJOR,
	UNKNOWN
};

typedef struct  {
	std::string lexeme;
	int row;
	int col;
	size_t src_start_pos;
	size_t src_end_pos;
} YYLVALTYPE;

extern YYLVALTYPE * pyylval;

#define yylex hqengine_shader_d3d9_lexer_lex
int yylex();
void hqengine_shader_d3d9_lexer_start(const char * src);
void hqengine_shader_d3d9_lexer_clean_up();
int hqengine_shader_d3d9_lexer_input(void* buf, size_t max_size);