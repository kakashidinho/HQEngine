/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_GL_VAR_PARSER_H
#define HQ_SHADER_GL_VAR_PARSER_H

#include "../HQLoggableObject.h"


#define POS_ATTRIB 0
#define COLOR_ATTRIB 1
#define NORMAL_ATTRIB 2
#define TANGENT_ATTRIB 3
#define BINORMAL_ATTRIB 4
#define BLENDWEIGHT_ATTRIB 5
#define BLENDINDICES_ATTRIB 6
#define TEXCOORD0_ATTRIB 7
#define TEXCOORD1_ATTRIB 8
#define TEXCOORD2_ATTRIB 9
#define TEXCOORD3_ATTRIB 10
#define TEXCOORD4_ATTRIB 11
#define TEXCOORD5_ATTRIB 12
#define TEXCOORD6_ATTRIB 13
#define TEXCOORD7_ATTRIB 14
#define PSIZE_ATTRIB 15
#define MAX_TEXCOORD_ATTRIB 8

/*----------HQext GLSL variable parser-----------------*/
enum HQVarParserTokenKindGL
{
	VTK_ATTRIBUTE,
	VTK_UNIFORM,
	VTK_PRECISION,
	VTK_TYPE,
	VTK_SAMPLER,
	VTK_ID,
	VTK_SEMANTIC,
	VTK_TEXUNIT,
	VTK_NULL,
	VTK_UNKNOWN
};

struct HQVarParserTokenGL
{
	HQVarParserTokenKindGL kind;
	std::string lexeme;
};

struct HQShaderAttrib
{
	std::string name;
	int index;
};

struct HQUniformSamplerGL
{
	std::string name;
	int samplerUnit;
};

struct HQVarParserGL 
{
private:
	hq_uint32 currentLine;
	hq_uint32 currentChar;
	char inputChar ;
	bool breakLoop;
	int state;
	char *source;
	
	HQVarParserTokenGL token;
	
	HQLinkedList<HQShaderAttrib>* pAttribList;
	HQLinkedList<HQUniformSamplerGL>* pUniformSamplerList;
	
	HQLoggableObject *manager;
	
	bool IsWhiteSpaceNotNewLine();
	bool IsWhiteSpace();
	bool IsDigit();
	bool IsNonDigit();
	void Token(HQVarParserTokenKindGL kind);
	void ChangeState(int newState);
	char nextChar();
	void nextToken();
	bool Match(HQVarParserTokenKindGL kind);
	
	void GetAttribIndex(HQShaderAttrib &attrib , const char *lexeme);
	
	bool ParseSematicBinding();
	bool ParseSamplerBinding();
public:
	HQVarParserGL(HQLoggableObject *manager);
	bool Parse(const char* source , 
			   HQLinkedList<HQShaderAttrib>** ppAttribList ,//if <ppAttribList> not NULL , after this method returns , pointer that <ppAttribList> points to must be freed manually
			   HQLinkedList<HQUniformSamplerGL>** ppUniformSamplerList//if <ppUniformSamplerList> not NULL , after this method returns , pointer that <ppUniformSamplerList> points to must be freed manually
			   );
};

#endif
