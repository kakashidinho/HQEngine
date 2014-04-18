/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_GL_VAR_PARSER_H
#define HQ_SHADER_GL_VAR_PARSER_H

#include "../HQLoggableObject.h"
#include "../HQFileManager.h"
#include "../HQLinkedList.h"
#include "../HQRendererCoreType.h"

#include <string>

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
	VTK_SEMI_COLON,
	VTK_LBRACE,
	VTK_RBRACE,
	VTK_LBRACKET,
	VTK_RBRACKET,
	VTK_INTEGER,
	VTK_UNKNOWN
};

struct HQVarParserTokenGL
{
	HQVarParserTokenKindGL kind;
	std::string lexeme;
	size_t src_start_pos, src_end_pos;
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

struct HQUniformBlkElemInfoGL{
	std::string name;
	int row;
	int col;
	int arraySize;
	int blockIndex;
	bool integer;
	size_t src_start_pos;
};
struct HQUniformBlockInfoGL{
	hquint32 index;
	HQLinkedList<HQUniformBlkElemInfoGL> blockElems;

	size_t blockPrologue_start_pos, blockPrologue_end_pos;
	size_t blockEpilogue_start_pos, blockEpilogue_end_pos;
};

struct HQVarParserGL 
{
public:

	HQVarParserGL(HQLoggableObject *logger);
	~HQVarParserGL();
	bool Parse(const char* source,
				HQFileManager *includeManager,
				const std::string & predefinedMacrosString,
			   const HQShaderMacro * pDefines,
			   std::string& processed_source_out,
			   bool native_UBO_supported,//is uniform buffer object supported natively
			   HQLinkedList<HQUniformBlockInfoGL>**ppUniformBlocks,//if not NULL, the returned pointer must be freed. Note: only pass non NULL value when native UBO is not supported
			   HQLinkedList<HQShaderAttrib>** ppAttribList ,//if <ppAttribList> not NULL , after this method returns , pointer that <ppAttribList> points to must be freed manually
			   HQLinkedList<HQUniformSamplerGL>** ppUniformSamplerList//if <ppUniformSamplerList> not NULL , after this method returns , pointer that <ppUniformSamplerList> points to must be freed manually
			   );
private:
	hq_uint32 currentLine;
	hq_uint32 currentChar;
	char inputChar ;
	bool breakLoop;
	int state;
	const char *source;
	std::string *pPreprocessing_src;

	HQVarParserTokenGL token;
	
	HQLinkedList<HQShaderAttrib>* pAttribList;
	HQLinkedList<HQUniformSamplerGL>* pUniformSamplerList;
	HQLinkedList<HQUniformBlockInfoGL>* pUniformBlocks;
	
	HQLoggableObject *logger;
	
	bool IsWhiteSpaceNotNewLine();
	bool IsWhiteSpace();
	bool IsDigit();
	bool IsDigitNonZero();
	bool IsNonDigit();
	void Token(HQVarParserTokenKindGL kind, size_t src_start_pos = -1, size_t src_end_pos = -1);
	void ChangeState(int newState);
	char nextChar();
	void nextToken();
	bool Match(HQVarParserTokenKindGL kind);
	
	void GetAttribIndex(HQShaderAttrib &attrib , const char *lexeme);
	
	bool ParseSematicBinding();
	bool ParseUniformBlock(size_t src_start_pos);
	void TransformUniformBlockDecls();//transform uniform blocks declaration so that they can be used in system that don't support native UBO
	bool ParseTypeInfo(HQUniformBlkElemInfoGL& uniformBlkElem);
	bool ParseSamplerBinding();
};

#endif
