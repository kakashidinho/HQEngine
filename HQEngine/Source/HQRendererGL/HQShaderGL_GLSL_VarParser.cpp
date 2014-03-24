/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#define VERTEX_SEMANTIC_START 1000
#define COMMENT_START 200
#define COMMENTLINE 201
#define COMMENTBLOCK 300
#define SEMANTIC 36
#define PRECISION 7
#define DATA_TYPE 22
#define SAMPLER_TYPE 77
#define INTEGER_ATTR 150
#define SAMPLER_BUFFER 500
#define MATRIX_START 10000
#define ZERO_NUMBER_START 20000
#define DEC_NUMBER_START 30000

#define MOJOSHADER_NO_VERSION_INCLUDE
#include "../../../ThirdParty-mod/mojoshader/mojoshader.h" //for preprocessor

#include "HQShaderGL_GLSL_VarParser.h"

HQVarParserGL::HQVarParserGL(HQLoggableObject *manager )
{
	this->manager = manager;
}

HQVarParserGL::~HQVarParserGL()
{

}

char HQVarParserGL::nextChar()
{
	hquint32 readCharPos = currentChar++;
	return source[readCharPos];
}

void HQVarParserGL::Token(HQVarParserTokenKindGL kind, size_t src_start_pos , size_t src_end_pos )
{
	token.kind = kind;
	
	//default source location is [currentChar - length(lexeme)	---> currentChar ]
	if (src_end_pos == -1)
		token.src_end_pos = currentChar;
	else 
		token.src_end_pos = src_end_pos;

	if (src_start_pos == -1)
		token.src_start_pos = token.src_end_pos - token.lexeme.size();
	else
		token.src_start_pos = src_start_pos;

	breakLoop = true;
}

void HQVarParserGL::ChangeState(int newState)
{
	state = newState;
	token.lexeme += inputChar;
}
bool HQVarParserGL::IsWhiteSpaceNotNewLine()
{
	return (inputChar == ' '|| inputChar == '\t' || inputChar == '\r');
}
bool HQVarParserGL::IsWhiteSpace()
{
	return (inputChar == ' '|| inputChar == '\t' || inputChar == '\n' || inputChar == '\r');
}

bool HQVarParserGL::IsDigit()
{
	return (inputChar >= '0' && inputChar <= '9');
}

bool HQVarParserGL::IsDigitNonZero()
{
	return (inputChar >= '1' && inputChar <= '9');
}

bool HQVarParserGL::IsNonDigit()
{
	return ((inputChar >= 'a' && inputChar <= 'z') || (inputChar >= 'A' && inputChar <= 'Z') || inputChar == '_');
}
bool HQVarParserGL::Match(HQVarParserTokenKindGL kind)
{
	if (token.kind == kind)
	{
		nextToken();
		return true;
	}
	return false;
}

void HQVarParserGL::nextToken()
{
	state = 0;
	breakLoop = false;
	token.lexeme = "";

	while (!breakLoop) {
		inputChar = nextChar();
		switch (state)
		{
			case 0:
				switch (inputChar)
				{
					case '\0':
						Token(VTK_NULL, currentChar - 1);
						break;
					case '{':
						Token(VTK_LBRACE, currentChar - 1);
						break;
					case '}':
						Token(VTK_RBRACE, currentChar - 1);
						break;
					case '[':
						Token(VTK_LBRACKET, currentChar - 1);
						break;
					case ']':
						Token(VTK_RBRACKET, currentChar - 1);
						break;
					case ';':
						Token(VTK_SEMI_COLON, currentChar - 1);
						break;
					case 'm':
						ChangeState(1);//mediump or mat
						break;
					case 'h':
						ChangeState(8);//highp
						break;
					case 'l':
						ChangeState(13);//lowp
						break;
					case 'f':
						ChangeState(18);//float
						break;
					case 'i'://ivec / int /in /isampler*
						ChangeState(INTEGER_ATTR);
						break;
					case 'v':
						ChangeState(23);//vec
						break;
					case 'V'://maybe vertex attribute semantic
						ChangeState(VERTEX_SEMANTIC_START);
						break;
					case 'T':
						ChangeState(81);//TEXUNIT 
						break;
					case 'a':
						ChangeState(100);//attribute
						break;
					case 'u':
						ChangeState(61);//uniform / uint / uvec /usampler*
						break;
					case 's':
						ChangeState(69);//sampler*
						break;
					case '/':
						ChangeState(COMMENT_START);//comment
						break;
					default:
						if (inputChar == '\n')
						{
							currentLine ++;
							break;
						}
						else if (IsNonDigit()) {
							ChangeState(17);
							break;
						}
						else if (IsDigitNonZero())
						{
							ChangeState(DEC_NUMBER_START);
							break;
						}
						else if (inputChar == '0')
						{
							ChangeState(ZERO_NUMBER_START);
							break;
						}
						else if (IsWhiteSpaceNotNewLine())
							break;
						token.lexeme = inputChar;
						Token(VTK_UNKNOWN, currentChar - 1);
				}
				break;
			case ZERO_NUMBER_START:
				currentChar --;
				Token(VTK_INTEGER);
				break;
			case DEC_NUMBER_START:
				if (IsDigit())
					ChangeState(DEC_NUMBER_START);
				else
				{
					currentChar --;
					Token(VTK_INTEGER);
				}
				break;
			case VERTEX_SEMANTIC_START:
				switch(inputChar)
				{
					case 'P':
						ChangeState(29);//VPOSITION / VPSIZE
						break;
					case 'C':
						ChangeState(37);//VCOLOR
						break;
					case 'N':
						ChangeState(42);//VNORMAL
						break;
					case 'T':
						ChangeState(48);//VTEXCOORD / VTANGENT
						break;
					case 'B'://VBINORMAL / VBLENDWEIGHT / VBLENDINDICES
						ChangeState(112);
						break;
					default ://id
						currentChar --;
						state = 17;
				}
				break;
			case 1://maybe mediump or mat
				if (inputChar == 'e')
					ChangeState(2);
				else if (inputChar == 'a')
					ChangeState(MATRIX_START);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 2://maybe mediump
				if (inputChar == 'd')
					ChangeState(3);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 3://maybe mediump
				if (inputChar == 'i')
					ChangeState(4);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 4://maybe mediump
				if (inputChar == 'u')
					ChangeState(5);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 5://maybe mediump
				if (inputChar == 'm')
					ChangeState(6);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 6://maybe mediump
				if (inputChar == 'p')
					ChangeState(PRECISION);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case MATRIX_START: //maybe mat
				if (inputChar == 't')
					ChangeState(MATRIX_START + 1);
				else {//id
					currentChar--;
					state = 17;
				}
				break;
			case MATRIX_START + 1: //maybe mat
				if (inputChar == '2' || inputChar == '3' || inputChar == '4')
					ChangeState(MATRIX_START + 2);
				else {//id
					currentChar--;
					state = 17;
				}
				break;
			case MATRIX_START + 2://maybe mat_x*
				if (inputChar == 'x')
					ChangeState(MATRIX_START + 3);
				else if (IsWhiteSpace())
				{
					currentChar--;
					state = DATA_TYPE;
				}
				else {//id
					currentChar--;
					state = 17;
				}
				break;
			case MATRIX_START + 3://maybe mat_x_
				if (inputChar == '2' || inputChar == '3' || inputChar == '4')
					ChangeState(DATA_TYPE);
				else {//id
					currentChar--;
					state = 17;
				}
				break;
			case PRECISION://maybe precision keyword
				if (IsNonDigit() || IsDigit())//id
				{
					ChangeState(17);
				}
				else {//precision keyword
					currentChar--;
					Token(VTK_PRECISION);
				}

				break;
			case 8://maybe highp
				if (inputChar == 'i')
					ChangeState(9);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 9://maybe highp
				if (inputChar == 'g')
					ChangeState(10);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 10://maybe highp
				if (inputChar == 'h')
					ChangeState(11);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 11://maybe highp
				if (inputChar == 'p')
					ChangeState(PRECISION);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 13://maybe lowp
				if (inputChar == 'o')
					ChangeState(14);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 14://maybe lowp
				if (inputChar == 'w')
					ChangeState(15);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 15://maybe lowp
				if (inputChar == 'p')
					ChangeState(PRECISION);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 17://maybe id
				if (IsDigit() || IsNonDigit()) {
					ChangeState(17);
				}
				else {
					currentChar--;
					Token(VTK_ID);
				}

				break;
			case INTEGER_ATTR://ivec / int /in /isampler
				if (inputChar == 'v')//ivec
					ChangeState(23);
				else if ( inputChar == 's')//isampler
					ChangeState(69);
				else if (inputChar == 'n')//int / in
					ChangeState(INTEGER_ATTR + 1);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case INTEGER_ATTR + 1://int / in
				if (inputChar == 't')
					ChangeState(DATA_TYPE);
				else if (IsWhiteSpace())//in
				{
					currentChar--;
					Token(VTK_ATTRIBUTE);
				}
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case INTEGER_ATTR + 2://uint
				if (inputChar == 'n')
					ChangeState(INTEGER_ATTR + 3);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case INTEGER_ATTR + 3://uint 
				if (inputChar == 't')
					ChangeState(DATA_TYPE);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			
			case 18://maybe float
				if (inputChar == 'l')
					ChangeState(19);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 19://maybe float
				if (inputChar == 'o')
					ChangeState(20);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 20://maybe float
				if (inputChar == 'a')
					ChangeState(21);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 21://maybe float
				if (inputChar == 't')
					ChangeState(DATA_TYPE);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case DATA_TYPE://maybe data type
				if (IsDigit() || IsNonDigit())//id
					ChangeState(17);
				else {//data type
					currentChar--;
					Token(VTK_TYPE);
				}

				break;
			case 23://maybe vec
				if (inputChar == 'e')
					ChangeState(24);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 24://maybe vec
				if (inputChar == 'c')
					ChangeState(25);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 25://maybe vec2 | vec3 | vec4
				if (inputChar == '2' || inputChar == '3' || inputChar == '4')
					ChangeState(DATA_TYPE);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 29://maybe VPOSITION or VPSIZE
				if (inputChar == 'O')//VPOSITION
					ChangeState(30);
				else if (inputChar == 'S')//VPSIZE
					ChangeState(109);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 30://maybe VPOSITION
				if (inputChar == 'S')
					ChangeState(31);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 31://maybe VPOSITION
				if (inputChar == 'I')
					ChangeState(32);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 32://maybe VPOSITION
				if (inputChar == 'T')
					ChangeState(33);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 33://maybe VPOSITION
				if (inputChar == 'I')
					ChangeState(34);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 34://maybe VPOSITION
				if (inputChar == 'O')
					ChangeState(35);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 35://maybe VPOSITION
				if (inputChar == 'N')
					ChangeState(SEMANTIC);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case SEMANTIC://maybe semantic
				if (IsDigit() || IsNonDigit())//id
					ChangeState(17);
				else {//semantic
					currentChar--;
					Token(VTK_SEMANTIC);
				}

				break;
			case 109://maybe VPSIZE
				if (inputChar == 'I')
					ChangeState(110);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 110://maybe VPSIZE
				if (inputChar == 'Z')
					ChangeState(111);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 111://maybe VPSIZE
				if (inputChar == 'E')
					ChangeState(SEMANTIC);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 37://maybe VCOLOR
				if (inputChar == 'O')
					ChangeState(38);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 38://maybe VCOLOR
				if (inputChar == 'L')
					ChangeState(39);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 39://maybe VCOLOR
				if (inputChar == 'O')
					ChangeState(40);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 40://maybe VCOLOR
				if (inputChar == 'R')
					ChangeState(SEMANTIC);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 42://maybe VNORMAL
				if (inputChar == 'O')
					ChangeState(43);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 43://maybe VNORMAL
				if (inputChar == 'R')
					ChangeState(44);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 44://maybe VNORMAL
				if (inputChar == 'M')
					ChangeState(45);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 45://maybe VNORMAL
				if (inputChar == 'A')
					ChangeState(46);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 46://maybe VNORMAL
				if (inputChar == 'L')
					ChangeState(SEMANTIC);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 48://maybe VTEXCOORD or VTANGENT
				if (inputChar == 'E')
					ChangeState(49);
				else if (inputChar == 'A')//VTANGENT
					ChangeState(136);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 49://maybe VTEXCOORD 
				if (inputChar == 'X')
					ChangeState(50);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 50://maybe VTEXCOORD
				if (inputChar == 'C')//VTEXCOORD
					ChangeState(51);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 51://maybe VTEXCOORD
				if (inputChar == 'O')
					ChangeState(52);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 52://maybe VTEXCOORD
				if (inputChar == 'O')
					ChangeState(53);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 53://maybe VTEXCOORD
				if (inputChar == 'R')
					ChangeState(54);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 54://maybe VTEXCOORD
				if (inputChar == 'D')
					ChangeState(55);
				else {//id
					currentChar--;
					state = 17;
				}

				break;

			case 55://maybe VTEXCOORD0  - VTEXCOORD7
				if (inputChar >= '0' && inputChar <= '7')
					ChangeState(SEMANTIC);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 136 ://VTANGENT
				if (inputChar == 'N')
					ChangeState(137);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 137 ://VTANGENT
				if (inputChar == 'G')
					ChangeState(138);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 138 ://VTANGENT
				if (inputChar == 'E')
					ChangeState(139);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 139 ://VTANGENT
				if (inputChar == 'N')
					ChangeState(140);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 140 ://VTANGENT
				if (inputChar == 'T')
					ChangeState(SEMANTIC);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 112://VBINORMAL / VBLENDWEIGHT / VBLENDINDICES
				if (inputChar == 'I')
					ChangeState(113);
				else if (inputChar == 'L')
					ChangeState(120);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 113 ://VBINORMAL
				if (inputChar == 'N')
					ChangeState(114);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 114 ://VBINORMAL
				if (inputChar == 'O')
					ChangeState(115);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			
			case 115 ://VBINORMAL
				if (inputChar == 'R')
					ChangeState(116);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			
			case 116 ://VBINORMAL
				if (inputChar == 'M')
					ChangeState(117);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			
			case 117 ://VBINORMAL
				if (inputChar == 'A')
					ChangeState(118);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 118 ://VBINORMAL
				if (inputChar == 'L')
					ChangeState(SEMANTIC);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			
			case 120 ://VBLENDWEIGHT / VBLENDINDICES
				if (inputChar == 'E')
					ChangeState(121);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 121 ://VBLENDWEIGHT / VBLENDINDICES
				if (inputChar == 'N')
					ChangeState(122);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 122 ://VBLENDWEIGHT / VBLENDINDICES
				if (inputChar == 'D')
					ChangeState(123);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 123 ://VBLENDWEIGHT / VBLENDINDICES
				if (inputChar == 'W')//VBLENDWEIGHT
					ChangeState(124);
				else if (inputChar == 'I')//VBLENDINDICES
					ChangeState(130);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 124 ://VBLENDWEIGHT
				if (inputChar == 'E')
					ChangeState(125);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			
			case 125 ://VBLENDWEIGHT
				if (inputChar == 'I')
					ChangeState(126);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			
			case 126 ://VBLENDWEIGHT
				if (inputChar == 'G')
					ChangeState(127);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 127 ://VBLENDWEIGHT
				if (inputChar == 'H')
					ChangeState(128);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			
			case 128 ://VBLENDWEIGHT
				if (inputChar == 'T')
					ChangeState(SEMANTIC);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 130 ://VBLENDINDICES
				if (inputChar == 'N')
					ChangeState(131);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 131 ://VBLENDINDICES
				if (inputChar == 'D')
					ChangeState(132);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 132 ://VBLENDINDICES
				if (inputChar == 'I')
					ChangeState(133);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 133 ://VBLENDINDICES
				if (inputChar == 'C')
					ChangeState(134);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 134 ://VBLENDINDICES
				if (inputChar == 'E')
					ChangeState(135);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 135 ://VBLENDINDICES
				if (inputChar == 'S')
					ChangeState(SEMANTIC);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 61 ://maybe uniform / uint / uvec /usampler
				if (inputChar == 'n')
					ChangeState(62);
				else if ( inputChar == 's')//usampler
					ChangeState(69);
				else if (inputChar == 'i')//uint
					ChangeState(INTEGER_ATTR + 2);
				else if (inputChar == 'v')//uvec
					ChangeState(23);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 62 ://maybe uniform 
				if (inputChar == 'i')
					ChangeState(63);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 63 ://maybe uniform
				if (inputChar == 'f')
					ChangeState(64);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 64 ://maybe uniform
				if (inputChar == 'o')
					ChangeState(65);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 65 ://maybe uniform
				if (inputChar == 'r')
					ChangeState(66);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 66 ://maybe uniform
				if (inputChar == 'm')
					ChangeState(67);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 67://maybe uniform
				if (IsDigit() || IsNonDigit())//id
					ChangeState(17);
				else {//uniform
					currentChar--;
					Token(VTK_UNIFORM);
				}

				break;
			case 69: //maybe sampler
				if (inputChar == 'a')
					ChangeState(70);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 70: //maybe sampler
				if (inputChar == 'm')
					ChangeState(71);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 71: //maybe sampler
				if (inputChar == 'p')
					ChangeState(72);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 72: //maybe sampler
				if (inputChar == 'l')
					ChangeState(73);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 73: //maybe sampler
				if (inputChar == 'e')
					ChangeState(74);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 74: //maybe sampler
				if (inputChar == 'r')
					ChangeState(75);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 75: //maybe sampler1D || sampler2D || sampler3D  || samplerCUBE 
				if (inputChar == '1' || inputChar == '2' || inputChar == '3')//sampler1D || sampler2D || sampler3D
					ChangeState(76);
				else if (inputChar == 'C')//samplerCUBE
					ChangeState(78);
				else if (inputChar == 'B')//samplerBuffer
					ChangeState(SAMPLER_BUFFER);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 76://maybe sampler1D || sampler2D || sampler3D
				if (inputChar == 'D')//sampler1D || sampler2D || sampler3D
					ChangeState(SAMPLER_TYPE);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case SAMPLER_TYPE://maybe sampler type
				if (IsDigit() || IsNonDigit())//id
					ChangeState(17);
				else {//sampler type
					currentChar--;
					Token(VTK_SAMPLER);
				}

				break;
			case 78://maybe samplerCUBE
				if (inputChar == 'U')//samplerCUBE
					ChangeState(79);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 79://maybe samplerCUBE
				if (inputChar == 'B')//samplerCUBE
					ChangeState(80);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 80://maybe samplerCUBE
				if (inputChar == 'E')//samplerCUBE
					ChangeState(SAMPLER_TYPE);
				else {//id
					currentChar--;
					state = 17;
				}

				break;

			case SAMPLER_BUFFER://maybe samplerBuffer
				if (inputChar == 'u')//samplerBuffer
					ChangeState(SAMPLER_BUFFER + 1);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case SAMPLER_BUFFER + 1://maybe samplerBuffer
				if (inputChar == 'f')//samplerBuffer
					ChangeState(SAMPLER_BUFFER + 2);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case SAMPLER_BUFFER + 2://maybe samplerBuffer
				if (inputChar == 'f')//samplerBuffer
					ChangeState(SAMPLER_BUFFER + 3);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case SAMPLER_BUFFER + 3://maybe samplerBuffer
				if (inputChar == 'e')//samplerBuffer
					ChangeState(SAMPLER_BUFFER + 4);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case SAMPLER_BUFFER + 4://maybe samplerBuffer
				if (inputChar == 'r')//samplerBuffer
					ChangeState(SAMPLER_TYPE);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			
			case 81://maybe TEXUNIT
				if (inputChar == 'E')
					ChangeState(82);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 82://maybe TEXUNIT 
				if (inputChar == 'X')
					ChangeState(83);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 83://maybe TEXUNIT
				if (inputChar == 'U')//TEXUNIT
					ChangeState(84);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 84://maybe TEXUNIT
				if (inputChar == 'N')
					ChangeState(85);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 85://maybe TEXUNIT
				if (inputChar == 'I')
					ChangeState(86);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 86://maybe TEXUNIT
				if (inputChar == 'T')
					ChangeState(87);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 87://maybe TEXUNITx
				if (inputChar == '0')//TEXUNIT0
					ChangeState(89);
				else if (IsDigit())
					ChangeState(88);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 88://maybe TEXUNITx or TEXUNITxx
				if (IsDigit())//TEXUNITxx
					ChangeState(89);
				else if (IsNonDigit())//id
					ChangeState(17);
				else //TEXUNITx
				{
					currentChar--;
					Token(VTK_TEXUNIT);
				}

				break;
			case 89://maybe TEXUNITxx or TEXUNIT0
				if (IsDigit() || IsNonDigit())//id
					ChangeState(17);
				else //TEXUNITxx or TEXUNIT0
				{
					currentChar--;
					Token(VTK_TEXUNIT);
				}

				break;
			case 100://maybe attribute
				if (inputChar == 't')
					ChangeState(101);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 101://maybe attribute
				if (inputChar == 't')
					ChangeState(102);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 102://maybe attribute
				if (inputChar == 'r')
					ChangeState(103);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 103://maybe attribute
				if (inputChar == 'i')
					ChangeState(104);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 104://maybe attribute
				if (inputChar == 'b')
					ChangeState(105);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 105://maybe attribute
				if (inputChar == 'u')
					ChangeState(106);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 106://maybe attribute
				if (inputChar == 't')
					ChangeState(107);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 107://maybe attribute
				if (inputChar == 'e')
					ChangeState(108);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 108://maybe attribute
				if (IsDigit() || IsNonDigit())//id
					ChangeState(17);
				else {//attribute
					currentChar--;
					Token(VTK_ATTRIBUTE);
				}

				break;
			case COMMENT_START:
				if (inputChar == '/') {
					state = COMMENTLINE;
					token.lexeme = "";
				}
				else if (inputChar == '*')
				{
					state = COMMENTBLOCK;
					token.lexeme = "";
				}
				else
				{
					currentChar--;
					Token(VTK_UNKNOWN);
				}
				break;
			case COMMENTLINE:
				if (inputChar == '\n')
				{
					state = 0;
					currentLine ++;
				}
				break;
			case COMMENTBLOCK:
				if (inputChar == '*')//may terminate comment block
					state = COMMENTBLOCK + 1;
				else if (inputChar == '\n')
					currentLine ++;
				break;
			case COMMENTBLOCK + 1:
				if (inputChar == '/')//terminate comment block
					state = 0;
				else
				{
					if (inputChar == '\n')
						currentLine ++;
					state = COMMENTBLOCK;
				}
				break;
		}
	}
}

bool HQVarParserGL::Parse(const char* ori_source , 
				const HQShaderMacro * pDefines,
			   std::string& processed_source_out,
			   HQLinkedList<HQUniformBlockInfoGL>**ppUniformBlocks,
			   HQLinkedList<HQShaderAttrib>** ppAttribList ,
			   HQLinkedList<HQUniformSamplerGL>** ppUniformSamplerList
			   )
{
#ifdef HQ_OPENGLES
	processed_source_out	= "#define HQEXT_GLSL_ES\n";
#else
	processed_source_out	= "#define HQEXT_GLSL\n";
#endif
	processed_source_out += ori_source;

	int numDefines = 0;
	const HQShaderMacro *pD = pDefines;

	//calculate number of macros
	while (pD->name != NULL && pD->definition != NULL)
	{
		numDefines++;
		pD++;
	}
	//preprocess source first
	const MOJOSHADER_preprocessData * preprocessedData = MOJOSHADER_preprocess("source",
												processed_source_out.c_str(),
												processed_source_out.size(),
												(const MOJOSHADER_preprocessorDefine*)pDefines,
												numDefines,
												NULL,
												NULL,
												NULL,
												NULL,
												NULL);

	if (preprocessedData->output != NULL)
	{
		processed_source_out = preprocessedData->output;
	}
	else
		processed_source_out = source;//fallback to original source

	MOJOSHADER_freePreprocessData(preprocessedData);

	//create info lists
	if (ppUniformBlocks != NULL)
		this->pUniformBlocks = *ppUniformBlocks = HQ_NEW HQLinkedList<HQUniformBlockInfoGL>();
	if (ppAttribList != NULL)
		this->pAttribList = *ppAttribList = HQ_NEW HQLinkedList<HQShaderAttrib>();
	if (ppUniformSamplerList != NULL)
		this->pUniformSamplerList = *ppUniformSamplerList = HQ_NEW HQLinkedList<HQUniformSamplerGL>();

	this->source = processed_source_out.c_str();
	this->pPreprocessing_src = &processed_source_out;
	this->currentChar = 0;
	this->currentLine = 1;

	token.kind = VTK_UNKNOWN;
	bool noError = true;

	while (token.kind != VTK_NULL) {
		nextToken();
		/*
		if(token.kind == VTK_SEMANTIC)
		{
			manager->Log("(%u) error : '%s' is semantic keyword" ,currentLine , token.lexeme.c_str());
			noError = false;
		}
		else */if(token.kind == VTK_TEXUNIT)
		{
			manager->Log("(%u) error : '%s' is texture sampler unit binding keyword" ,currentLine , token.lexeme.c_str());
			noError = false;
		}
		else if (token.kind == VTK_UNIFORM)
		{
			size_t src_start_pos = token.src_start_pos;
			nextToken();
			if(ppUniformSamplerList != NULL && Match(VTK_SAMPLER))
				noError = ParseSamplerBinding() && noError;
			else if (ppUniformBlocks != NULL && token.kind == (VTK_ID))
				noError = ParseUniformBlock(src_start_pos) && noError;
		}
		else if(ppAttribList != NULL && token.kind == VTK_ATTRIBUTE)
		{
			noError = ParseSematicBinding() && noError;
		}
	}

	if (ppUniformBlocks != NULL)
		this->TransformUniformBlockDecls();

	return noError;
}

void HQVarParserGL::GetAttribIndex(HQShaderAttrib &attrib , const char *lexeme)
{
	if (!strcmp( lexeme , "VPOSITION")) {
		attrib.index = POS_ATTRIB;
	}
	else if (!strcmp(lexeme , "VCOLOR")) {
		attrib.index = COLOR_ATTRIB;
	}
	else if (!strcmp(lexeme , "VNORMAL")) {
		attrib.index = NORMAL_ATTRIB;
	}
	else if (!strncmp(lexeme , "VTEXCOORD" , strlen("VTEXCOORD"))) {
		attrib.index = TEXCOORD0_ATTRIB + (token.lexeme[strlen("VTEXCOORD")] - '0');
	}
	else if (!strcmp(lexeme , "VTANGENT")) {
		attrib.index = TANGENT_ATTRIB;
	}
	else if (!strcmp(lexeme , "VBINORMAL")) {
		attrib.index = BINORMAL_ATTRIB;
	}
	else if (!strcmp(lexeme , "VBLENDWEIGHT")) {
		attrib.index = BLENDWEIGHT_ATTRIB;
	}
	else if (!strcmp(lexeme , "VBLENDINDICES")) {
		attrib.index = BLENDINDICES_ATTRIB;
	}
	else if (!strcmp(lexeme , "VPSIZE")) {
		attrib.index = PSIZE_ATTRIB;
	}
}

bool HQVarParserGL::ParseSematicBinding()
{
	nextToken();
	HQShaderAttrib attrib;
	if(Match(VTK_PRECISION))
	{
		if(!Match(VTK_TYPE))
		{
			//manager->Log("(%u) semantic binding syntax error : unexpected token '%s'" , currentLine , token.lexeme.c_str());
			return true;//it's up to opengl compiler to check this error
		}
		if(token.kind != VTK_ID)
		{
			//manager->Log("(%u) semantic binding syntax error : unexpected token '%s'" , currentLine , token.lexeme.c_str());
			return true;//it's up to opengl compiler to check this error
		}
		attrib.name = token.lexeme;
		nextToken();

		if(token.kind != VTK_SEMANTIC)
		{
			//manager->Log("(%u) semantic binding syntax error : unexpected token '%s'" , currentLine , token.lexeme.c_str());
			return true;//it's up to opengl compiler to check this error
		}
		this->GetAttribIndex(attrib , token.lexeme.c_str());
		
	}
	else if(Match(VTK_TYPE))
	{
		if(token.kind != VTK_ID)
		{
			//manager->Log("(%u) semantic binding syntax error : unexpected token '%s'" , currentLine , token.lexeme.c_str());
			return true;//it's up to opengl compiler to check this error
		}
		attrib.name = token.lexeme;
		nextToken();

		if(token.kind != VTK_SEMANTIC)
		{
			//manager->Log("(%u) semantic binding syntax error : unexpected token '%s'" , currentLine , token.lexeme.c_str());
			return true;//it's up to opengl compiler to check this error
		}
		this->GetAttribIndex(attrib , token.lexeme.c_str());
	}
	else{
		//manager->Log("(%u) semantic binding syntax error : unexpected token '%s'" , currentLine , token.lexeme.c_str());
		return true;//it's up to opengl compiler to check this error
	}
	pAttribList->PushBack(attrib);
	return true;
}


bool HQVarParserGL::ParseSamplerBinding()
{
	HQUniformSamplerGL sampler;
	if(token.kind != VTK_ID)
	{
		manager->Log("(%u) sampler unit binding syntax error : unexpected token '%s'" , currentLine, token.lexeme.c_str());
		return false;
	}
	sampler.name = token.lexeme;
	nextToken();

	if(token.kind != VTK_TEXUNIT)
	{
		manager->Log("(%u) sampler unit binding syntax error : unexpected token '%s'" , currentLine, token.lexeme.c_str());
		return false;
	}
	
	int samplerUnit;
	sscanf(token.lexeme.c_str() , "TEXUNIT%d" , &samplerUnit);

	if (samplerUnit > 31)
	{
		manager->Log("(%u) sampler unit binding error : i in TEXUNITi is too large (i = %d).Max supported i is 31" , currentLine, samplerUnit);
		return false;
	}

	sampler.samplerUnit = samplerUnit;
	
	
	pUniformSamplerList->PushBack(sampler);
	return true;
}

bool HQVarParserGL::ParseTypeInfo(HQUniformBlkElemInfoGL& blockElemInfo)
{
	bool stop = false;
	if (strncmp(token.lexeme.c_str(), "vec", 3) == 0)
	{
		blockElemInfo.integer = false;
		blockElemInfo.row = 1;
		if (sscanf(token.lexeme.c_str(), "vec%d", &blockElemInfo.col) != 1)
			stop = true;
	}
	else if (strncmp(token.lexeme.c_str(), "bvec", 4) == 0
			|| strncmp(token.lexeme.c_str(), "ivec", 4) == 0)
	{
		blockElemInfo.integer = true;
		blockElemInfo.row = 1;
		if (sscanf(token.lexeme.c_str() + 1, "vec%d", &blockElemInfo.col) != 1)
			stop = true;
	}
	else if (strncmp(token.lexeme.c_str(), "mat", 3) == 0)
	{
		blockElemInfo.integer = false;
		if (sscanf(token.lexeme.c_str(), "mat%d", &blockElemInfo.row) != 1)
			stop = true;
		else if (sscanf(token.lexeme.c_str() + 4, "x%d", &blockElemInfo.col) != 1)
			blockElemInfo.col = blockElemInfo.row;
	}
	else 
		stop = true;

	return !stop;
}

bool HQVarParserGL::ParseUniformBlock(size_t src_start_pos)
{
	enum UniformBlockElemDeclState {
		UDECL_START,
		UDECL_PRECISION,
		UDECL_TYPE,
		UDECL_NAME,
		UDECL_ARRAY_SIZE_BEGIN,
		UDECL_ARRAY_SIZE,
		UDECL_ARRAY_SIZE_END,
		UBLOCK_END
	};

	HQVarParserTokenGL *ptoken = &this->token;
	size_t uniform_decl_start_pos = src_start_pos;
	int bufferIndex;
	if (sscanf(ptoken->lexeme.c_str(), "ubuffer%d", &bufferIndex) != 1)
		return true;//failed

	nextToken();
	if (ptoken->kind == VTK_LBRACE)
	{
		bool stop = false;
		bool succeeded = false;
		UniformBlockElemDeclState state = UDECL_START;
		HQUniformBlkElemInfoGL blockElemInfo;
		HQUniformBlockInfoGL uniformBlock;
		blockElemInfo.blockIndex = bufferIndex;
		uniformBlock.blockPrologue_start_pos = uniform_decl_start_pos;
		uniformBlock.blockPrologue_end_pos = ptoken->src_end_pos;

		while (ptoken->kind != VTK_NULL && !stop)
		{
			nextToken();
			int kind = ptoken->kind;
			switch (state)
			{
			case UDECL_START:
				if (kind == VTK_RBRACE)
				{
					uniformBlock.blockEpilogue_start_pos = ptoken->src_start_pos;
					state = UBLOCK_END;
				}
				else if (kind == VTK_PRECISION)
				{
					state = UDECL_PRECISION;
					blockElemInfo.src_start_pos = ptoken->src_start_pos;
				}
				else if (kind == VTK_TYPE)
				{
					state = UDECL_TYPE;
					blockElemInfo.src_start_pos = ptoken->src_start_pos;
					stop = !ParseTypeInfo(blockElemInfo);
				}
				else
					stop = true;
				break;
			case UDECL_PRECISION:
				if (kind == VTK_TYPE)
				{
					state = UDECL_TYPE;
					stop = !ParseTypeInfo(blockElemInfo);
				}
				else
					stop = true;
				break;
			case UDECL_TYPE:
				if (kind == VTK_ID)
				{
					state = UDECL_NAME;
					blockElemInfo.name = ptoken->lexeme;
				}
				else
					stop = true;
				break;
			case UDECL_NAME:
				if (kind == VTK_LBRACKET)
				{
					state = UDECL_ARRAY_SIZE_BEGIN;
				}
				else if (kind == VTK_SEMI_COLON)
				{
					blockElemInfo.arraySize = 1;
					uniformBlock.blockElems.PushBack(blockElemInfo);
					state = UDECL_START;
				}
				else 
					stop = true;
				break;
			case UDECL_ARRAY_SIZE_BEGIN:
				if (kind == VTK_INTEGER)
				{
					blockElemInfo.arraySize = (int)atol(ptoken->lexeme.c_str());
					state = UDECL_ARRAY_SIZE;
				}
				else
					stop = true;
				break;
			case UDECL_ARRAY_SIZE:
				if (kind == VTK_RBRACKET)
					state = UDECL_ARRAY_SIZE_END;
				else
					stop = true;
				break;
			case UDECL_ARRAY_SIZE_END:
				if (kind == VTK_SEMI_COLON)
				{
					uniformBlock.blockElems.PushBack(blockElemInfo);
					state = UDECL_START;
				}
				else
					stop = true;
				break;
			case UBLOCK_END:
				if (kind == VTK_SEMI_COLON)
				{
					succeeded = true;
					uniformBlock.blockEpilogue_end_pos = ptoken->src_end_pos;
				}
				stop = true;
				break;
			}//switch (state)

		}//while (ptoken->kind != VTK_NULL && !stop)

		if (succeeded)
		{
			//add the uniform block info to the list
			uniformBlock.index = bufferIndex;
			this->pUniformBlocks->PushBack(uniformBlock);

		}
	}//if (kind == LBRACE)

	return true;
}

void HQVarParserGL::TransformUniformBlockDecls()
{
	//Note: if we are doing these things here, that means the system doesn't support native uniform buffer object
	std::string & preprocessed_src = *this->pPreprocessing_src;
	HQLinkedList<HQUniformBlockInfoGL> ::Iterator block_ite;
	//first compute transformed string size and remove uniform block starting and ending declaration
	size_t newSize = preprocessed_src.size();
	for (this->pUniformBlocks->GetIterator(block_ite); !block_ite.IsAtEnd(); ++block_ite)
	{
		HQLinkedList<HQUniformBlkElemInfoGL>::Iterator blockElemIte;
		for (block_ite->blockElems.GetIterator(blockElemIte); !blockElemIte.IsAtEnd(); ++blockElemIte)
		{
			newSize += 8;//length of additional "uniform "
			
			//now remove the block prologue and epilogue because compiler doesn't support it
			for (size_t i = block_ite->blockPrologue_start_pos; i < block_ite->blockPrologue_end_pos; ++i)
				preprocessed_src[i] = ' ';
			for (size_t i = block_ite->blockEpilogue_start_pos; i < block_ite->blockEpilogue_end_pos; ++i)
				preprocessed_src[i] = ' ';
		}
	}

	preprocessed_src.reserve(newSize);

	//now append "uniform" keyword to each uniform block's element
	size_t offset = 0;
	for (this->pUniformBlocks->GetIterator(block_ite); !block_ite.IsAtEnd(); ++block_ite)
	{

		//append "uniform" keyword to each uniform block's element
		HQLinkedList<HQUniformBlkElemInfoGL>::Iterator blockElemIte;
		for (block_ite->blockElems.GetIterator(blockElemIte); !blockElemIte.IsAtEnd(); ++blockElemIte)
		{
			preprocessed_src.insert(blockElemIte->src_start_pos + offset, "uniform ");
			offset += 8;//the next starting postion is shifted by length of "uniform "
		}
	}
}