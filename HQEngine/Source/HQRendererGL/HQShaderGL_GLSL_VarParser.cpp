/*
 *  ShaderGLES_VarParser.cpp
 *
 *  Copyright 2011 Le Hoang Quyen. All rights reserved.
 *
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

#include "HQShaderGL_GLSL_VarParser.h"

HQVarParserGL::HQVarParserGL(HQLoggableObject *manager )
{
	this->manager = manager;
}

char HQVarParserGL::nextChar()
{
	return source[currentChar++];
}

void HQVarParserGL::Token(HQVarParserTokenKindGL kind)
{
	token.kind = kind;
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
						Token(VTK_NULL);
						break;
					case 'm':
						ChangeState(1);//mediump
						break;
					case 'h':
						ChangeState(8);//highp
						break;
					case 'l':
						ChangeState(13);//lowp
						break;
					case 'f':
						ChangeState(18);//hq_float32
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
						else if (IsWhiteSpaceNotNewLine())
							break;
						token.lexeme = inputChar;
						Token(VTK_UNKNOWN);
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
			case 1://maybe mediump
				if (inputChar == 'e')
					ChangeState(2);
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
			
			case 18://maybe hq_float32
				if (inputChar == 'l')
					ChangeState(19);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 19://maybe hq_float32
				if (inputChar == 'o')
					ChangeState(20);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 20://maybe hq_float32
				if (inputChar == 'a')
					ChangeState(21);
				else {//id
					currentChar--;
					state = 17;
				}

				break;
			case 21://maybe hq_float32
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

bool HQVarParserGL::Parse(const char* source , 
			   HQLinkedList<HQShaderAttrib>** ppAttribList ,
			   HQLinkedList<HQUniformSamplerGL>** ppUniformSamplerList
			   )
{
	if (ppAttribList != NULL)
		this->pAttribList = *ppAttribList = new HQLinkedList<HQShaderAttrib>();
	if (ppUniformSamplerList != NULL)
		this->pUniformSamplerList = *ppUniformSamplerList = new HQLinkedList<HQUniformSamplerGL>();
	this->source = (char*)source;
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
		else if (ppUniformSamplerList != NULL && token.kind == VTK_UNIFORM)
		{
			noError = ParseSamplerBinding() && noError;
		}
		else if(ppAttribList != NULL && token.kind == VTK_ATTRIBUTE)
		{
			noError = ParseSematicBinding() && noError;
		}
	}

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
	nextToken();
	HQUniformSamplerGL sampler;
	if(Match(VTK_SAMPLER))
	{
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
	}
	
	else{//not uniform sampler2D declaration.It's OK
		return true;
	}
	pUniformSamplerList->PushBack(sampler);
	return true;
}
