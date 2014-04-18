/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_VAR_PARSER_D3D9_H
#define HQ_SHADER_VAR_PARSER_D3D9_H

#include "../HQLinkedList.h"
#include "../HQRendererCoreType.h"
#include "../HQFileManager.h"
#include <string>

/*----------------------ShaderVarParserInfo-----------------------*/
struct UniformBlkElemInfo {
	std::string name;
	int row;
	int col;
	int arraySize;
	int blockIndex;
	bool integer;
};

struct UniformBlock{
	int index;
	HQLinkedList<UniformBlkElemInfo> blockElems;

	size_t blockPrologue_start_pos, blockPrologue_end_pos;
	size_t blockEpilogue_start_pos, blockEpilogue_end_pos;
};

struct HQShaderVarParserD3D9 {
	HQShaderVarParserD3D9(const char * src, const HQShaderMacro* pDefines, HQFileManager* includeFileManager);
	~HQShaderVarParserD3D9();
	HQLinkedList<UniformBlock> uniformBlocks;//list of declared uniform blocks

	const char* GetPreprocessedSrc() const { return m_preprocessedSrc.size() == 0 ? NULL : m_preprocessedSrc.c_str(); };
private:
	void ParseUniform();
	void ParseDirective();

	bool m_isColMajor;
	std::string m_preprocessedSrc;
};

#endif