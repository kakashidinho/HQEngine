%{


#include "HQShaderD3D9_Lexer.h"

#include <stdlib.h>
#include <string.h>


#if defined _MSC_VER
#	define fileno _fileno
#else
#endif

YYLVALTYPE * pyylval;

static const char *hqengine_shader_d3d9_lexer_src = NULL;
static size_t hqengine_shader_d3d9_lexer_src_size = 0;
static size_t hqengine_shader_d3d9_lexer_src_pos = 0;
static size_t char_src_pos = 0;

#define yylval pyylval

#define INCRE_POS_COUNT  (char_src_pos += yyleng)
#define SAVE_POS  {INCRE_POS_COUNT; pyylval->src_start_pos = char_src_pos - yyleng;   pyylval->src_end_pos = char_src_pos;}
#define SAVE_TOKEN {SAVE_POS; pyylval->lexeme = yytext; pyylval->row = 1; pyylval->col = 1; }
#define SAVE_VEC_TOKEN(size) {SAVE_POS; pyylval->lexeme = yytext; pyylval->row = 1; pyylval->col = size; }
#define SAVE_MAT_TOKEN(_row, _col) {SAVE_POS; pyylval->lexeme = yytext; pyylval->row = _row; pyylval->col = _col; }

#define YY_INPUT(buf,result,max_size) (result = hqengine_shader_d3d9_lexer_input(buf, max_size)) 

extern "C" int yywrap() { return 1; }


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

[ \t\n\r]                 {  SAVE_POS ;}
"\n"					  {  SAVE_POS ;}

";"					  {SAVE_POS; return SEMI_COLON; }
":"					  {SAVE_POS; return COLON; }

"#"					  {SAVE_POS; return HASH_SIGN;}
"pragma"			  {SAVE_POS; return PRAGMA;}
"pack_matrix"		  {SAVE_POS; return PACK_MATRIX;}
"column_major"		  {SAVE_POS; return COL_MAJOR;}
"row_major"			  {SAVE_POS; return ROW_MAJOR;}
"("					  {SAVE_POS; return LPAREN;}
")"					  {SAVE_POS; return RPAREN;}

"{"					  {SAVE_POS; return LBRACE;}
"}"					  {SAVE_POS; return RBRACE;}
"["					  {SAVE_POS; return LBRACKET;}
"]"					  {SAVE_POS; return RBRACKET;}
"//"				  {SAVE_POS; BEGIN(COMMENT_LINE);}
"/*"				  {SAVE_POS; BEGIN(COMMENT_BLOCK);}

{D}+{E}{F}?           {  SAVE_TOKEN; return (FLOATCONSTANT); }
{D}+"."{D}*({E})?{F}? {  SAVE_TOKEN; return (FLOATCONSTANT); }
"."{D}+({E})?{F}?     {  SAVE_TOKEN; return (FLOATCONSTANT); }
{D}+{F}               {  SAVE_TOKEN; return (FLOATCONSTANT); }

0[xX]{H}+{I}?         {  SAVE_TOKEN; return (INTCONSTANT); }
0{O}+{I}?             {  SAVE_TOKEN; return (INTCONSTANT); }
{D}+{I}?              {  SAVE_TOKEN; return (INTCONSTANT); }

"uniform"			  {	SAVE_POS; return (UNIFORM);}
"float"|"int"|"bool"|"half"|"fixed" 			  {	SAVE_TOKEN; return (SCALAR);}
("float"|"int"|"bool"|"half"|"fixed")2 			  {	SAVE_VEC_TOKEN(2); return (VECTOR);}
("float"|"int"|"bool"|"half"|"fixed")3 			  {	SAVE_VEC_TOKEN(3); return (VECTOR);}
("float"|"int"|"bool"|"half"|"fixed")4 			  {	SAVE_VEC_TOKEN(4); return (VECTOR);}

("float"|"int"|"bool"|"half"|"fixed")2x2 		  {	SAVE_MAT_TOKEN(2, 2); return (MAT);}
("float"|"int"|"bool"|"half"|"fixed")2x3 		  {	SAVE_MAT_TOKEN(2, 3); return (MAT);}
("float"|"int"|"bool"|"half"|"fixed")2x4 		  {	SAVE_MAT_TOKEN(2, 4); return (MAT);}
("float"|"int"|"bool"|"half"|"fixed")3x2 		  {	SAVE_MAT_TOKEN(3, 2); return (MAT);}
("float"|"int"|"bool"|"half"|"fixed")3x3 		  {	SAVE_MAT_TOKEN(3, 3); return (MAT);}
("float"|"int"|"bool"|"half"|"fixed")3x4 		  {	SAVE_MAT_TOKEN(3, 4); return (MAT);}
("float"|"int"|"bool"|"half"|"fixed")4x2 		  {	SAVE_MAT_TOKEN(4, 2); return (MAT);}
("float"|"int"|"bool"|"half"|"fixed")4x3 		  {	SAVE_MAT_TOKEN(4, 3); return (MAT);}
("float"|"int"|"bool"|"half"|"fixed")4x4 		  {	SAVE_MAT_TOKEN(4, 4); return (MAT);}


{L}({L}|{D}|"-")*	  {  SAVE_TOKEN;  return IDENTIFIER; }

\"[^\"]*\"			  { SAVE_TOKEN;  return STRING_CONST; }

.					  {	SAVE_TOKEN; return UNKNOWN; }


<COMMENT_BLOCK>{
	"*/"      SAVE_POS; BEGIN(INITIAL);
	[^*\n]+   SAVE_POS; // eat comment in chunks
	"*"       SAVE_POS; // eat the lone star
	\n        SAVE_POS; 
}
 
<COMMENT_LINE>{
	\n      {	SAVE_POS; BEGIN(INITIAL);	}
	[^\n]+   SAVE_POS; // eat comment in chunks
}

%%

void hqengine_shader_d3d9_lexer_clean_up(){
#ifdef YY_CURRENT_BUFFER_LVALUE /*flex version found in other systems other than Windows*/
	hqengine_shader_d3d9_lexer_lex_destroy();
#else
	if (YY_CURRENT_BUFFER)
	{
		yy_delete_buffer(YY_CURRENT_BUFFER);
		YY_CURRENT_BUFFER = NULL;
	}
#endif
}

void hqengine_shader_d3d9_lexer_start(const char * src)
{
#ifdef YY_CURRENT_BUFFER_LVALUE /*flex version found in other systems other than Windows*/
	yy_init_globals();
#else
	yyrestart(0);
#endif
	hqengine_shader_d3d9_lexer_src = src;
	hqengine_shader_d3d9_lexer_src_pos = 0;
	hqengine_shader_d3d9_lexer_src_size = strlen(src);
	char_src_pos = 0;
}



int hqengine_shader_d3d9_lexer_input(void* buf, size_t max_size)
{
	size_t remain = hqengine_shader_d3d9_lexer_src_size - hqengine_shader_d3d9_lexer_src_pos;
	size_t bytesRead = remain < max_size ? remain : max_size;
	if (bytesRead > 0)
		memcpy(buf, hqengine_shader_d3d9_lexer_src + hqengine_shader_d3d9_lexer_src_pos, bytesRead);
	hqengine_shader_d3d9_lexer_src_pos += bytesRead;

	return bytesRead;
}