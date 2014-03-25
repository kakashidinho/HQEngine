/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQShaderVarParserD3D9.h"
#include "HQShaderD3D9_Lexer.h"

#ifndef MOJOSHADER_NO_VERSION_INCLUDE
#define MOJOSHADER_NO_VERSION_INCLUDE
#endif
#include "../../../ThirdParty-mod/mojoshader/mojoshader.h" //for preprocessor


/*----------------------ShaderVarParserInfo implementation -----------------------*/

HQShaderVarParserD3D9::HQShaderVarParserD3D9(const char *src, const HQShaderMacro* pDefines)
: m_isColMajor (false)
{
	int numDefines = 0;
	const HQShaderMacro *pD = pDefines;
	
	//calculate number of macros
	while (pD != NULL && pD->name != NULL && pD->definition != NULL)
	{
		numDefines++;
		pD++;
	}

	//preprocess source first
	const MOJOSHADER_preprocessData * preprocessedData = MOJOSHADER_preprocess("source",
												src,
												strlen(src),
												(const MOJOSHADER_preprocessorDefine*)pDefines,
												numDefines,
												NULL,
												NULL,
												NULL,
												NULL,
												NULL);

	if (preprocessedData->output != NULL)
	{

		//prepare lexer
		int lex_return;
		TokenKind kind;
		YYLVALTYPE token;
		pyylval = &token;

		m_preprocessedSrc = preprocessedData->output;

		hqengine_shader_d3d9_lexer_start(m_preprocessedSrc.c_str());

		while ((lex_return = yylex()) > 0)
		{
			kind = (TokenKind)lex_return;

			if (kind == UNIFORM)
			{
				ParseUniform();
			}
			else if (kind == HASH_SIGN)
			{
				ParseDirective();
			}

		}//while ((lex_return = yylex()) > 0)

		hqengine_shader_d3d9_lexer_clean_up();
	}//if (preprocessedData->output != NULL)

	MOJOSHADER_freePreprocessData(preprocessedData);

}

HQShaderVarParserD3D9::~HQShaderVarParserD3D9()
{
}

void HQShaderVarParserD3D9::ParseDirective()
{
	if (yylex() == PRAGMA 
		&& yylex() == PACK_MATRIX
		&& yylex() == LPAREN
		&& yylex() == COL_MAJOR
		&& yylex() == RPAREN)
	{
		m_isColMajor = true;	
	}
}

void HQShaderVarParserD3D9::ParseUniform()
{
	enum UniformBlockElemDeclState {
		UDECL_START,
		UDECL_TYPE,
		UDECL_NAME,
		UDECL_ARRAY_SIZE_BEGIN,
		UDECL_ARRAY_SIZE,
		UDECL_ARRAY_SIZE_END,
		UBLOCK_END,
		UBLOCK_SEMANTIC,
		UBLOCK_SEMANTIC_NAME_ARRAY_STYLE,
		UBLOCK_SEMANTIC_NAME_ARRAY_STYLE_INDEX_START,
		UBLOCK_SEMANTIC_NAME_ARRAY_STYLE_INDEX,
		UBLOCK_SEMANTIC_END
	};

	YYLVALTYPE *ptoken = pyylval;

	size_t uniform_decl_start_pos = ptoken->src_start_pos;
	TokenKind kind = (TokenKind)yylex();

	if (kind == IDENTIFIER)//may be uniform block
	{
		kind = (TokenKind)yylex();
		if (kind == LBRACE)
		{
			bool stop = false;
			UniformBlockElemDeclState state = UDECL_START;
			int blockIndex = -1;
			UniformBlkElemInfo blockElemInfo;
			UniformBlock uniformBlock;
			uniformBlock.blockPrologue_start_pos = uniform_decl_start_pos;
			uniformBlock.blockPrologue_end_pos = ptoken->src_end_pos;

			while (!stop && (kind = (TokenKind)yylex()) > 0)
			{
				switch (state)
				{
				case UDECL_START:
					if (kind == RBRACE)
					{
						uniformBlock.blockEpilogue_start_pos = ptoken->src_start_pos;
						state = UBLOCK_END;
					}
					else if (kind == SCALAR || kind == VECTOR || kind == MAT)
					{
						state = UDECL_TYPE;
						blockElemInfo.row = ptoken->row;
						blockElemInfo.col = ptoken->col;
						if (strncmp(ptoken->lexeme.c_str(), "int", 3) == 0)
							blockElemInfo.integer = true;
						else if (strncmp(ptoken->lexeme.c_str(), "bool", 4) == 0)
							blockElemInfo.integer = true;
						else
							blockElemInfo.integer = false;
					}
					else
						stop = true;
					break;
				case UDECL_TYPE:
					if (kind == IDENTIFIER)
					{
						state = UDECL_NAME;
						blockElemInfo.name = ptoken->lexeme;
					}
					else
						stop = true;
					break;
				case UDECL_NAME:
					if (kind == LBRACKET)
					{
						state = UDECL_ARRAY_SIZE_BEGIN;
					}
					else if (kind == SEMI_COLON)
					{
						blockElemInfo.arraySize = 1;
						uniformBlock.blockElems.PushBack(blockElemInfo);
						state = UDECL_START;
					}
					else 
						stop = true;
					break;
				case UDECL_ARRAY_SIZE_BEGIN:
					if (kind == INTCONSTANT)
					{
						blockElemInfo.arraySize = (int)atol(ptoken->lexeme.c_str());
						state = UDECL_ARRAY_SIZE;
					}
					else
						stop = true;
					break;
				case UDECL_ARRAY_SIZE:
					if (kind == RBRACKET)
						state = UDECL_ARRAY_SIZE_END;
					else
						stop = true;
					break;
				case UDECL_ARRAY_SIZE_END:
					if (kind == SEMI_COLON)
					{
						uniformBlock.blockElems.PushBack(blockElemInfo);
						state = UDECL_START;
					}
					else
						stop = true;
					break;
				case UBLOCK_END:
					if (kind == SEMI_COLON)
					{
						stop = true;
						blockIndex = 0;
					}
					else if (kind == COLON)
						state = UBLOCK_SEMANTIC;
					break;
				case UBLOCK_SEMANTIC:
					if (kind == IDENTIFIER)
					{
						if (strcmp(ptoken->lexeme.c_str(), "BUFFER") == 0)
							state = UBLOCK_SEMANTIC_NAME_ARRAY_STYLE;
						else
						{
							sscanf(ptoken->lexeme.c_str(), "BUFFER%d", &blockIndex);
							state = UBLOCK_SEMANTIC_END;
						}
					}
					break;
				case UBLOCK_SEMANTIC_NAME_ARRAY_STYLE:
					if (kind == LBRACKET)
					{
						state = UBLOCK_SEMANTIC_NAME_ARRAY_STYLE_INDEX_START;
					}
					else 
						stop = true;
					break;
				case UBLOCK_SEMANTIC_NAME_ARRAY_STYLE_INDEX_START:
					if (kind == INTCONSTANT)
					{
						blockIndex = (int)atol(ptoken->lexeme.c_str());
						state = UBLOCK_SEMANTIC_NAME_ARRAY_STYLE_INDEX;
					}
					else 
						stop = true;
					break;
				case UBLOCK_SEMANTIC_NAME_ARRAY_STYLE_INDEX:
					if (kind == RBRACKET)
					{
						state = UBLOCK_SEMANTIC_END;
					}
					else 
						stop = true;
					break;
				case UBLOCK_SEMANTIC_END:
					if (kind == SEMI_COLON)
					{
						stop = true;
						uniformBlock.blockEpilogue_end_pos = ptoken->src_end_pos;
					}
					else
						blockIndex = -1;//invalid uniform block
					break;
				}//switch (state)
			}//while ((kind = (TokenKind)yylex()) > 0 && !stop)

			if (blockIndex >= 0)
			{
				//add the uniform block info to the list
				uniformBlock.index = blockIndex;
				UniformBlock* pAddedBlock = &this->uniformBlocks.PushBack(uniformBlock)->m_element;

				HQLinkedList<UniformBlkElemInfo>::Iterator ite;
				for (pAddedBlock->blockElems.GetIterator(ite); !ite.IsAtEnd(); ++ite)
				{

					ite->blockIndex = blockIndex;
					if (ite->row > 1 && m_isColMajor)//flip row and column in column major mode
					{
						int temp = ite->row;
						ite->row = ite->col;
						ite->col = temp;
					}
				}
				//now remove the block prologue and epilogue because cg compiler doesn't support it
				for (size_t i = uniformBlock.blockPrologue_start_pos; i < uniformBlock.blockPrologue_end_pos; ++i)
					m_preprocessedSrc[i] = ' ';
				for (size_t i = uniformBlock.blockEpilogue_start_pos; i < uniformBlock.blockEpilogue_end_pos; ++i)
					m_preprocessedSrc[i] = ' ';
			}//if (blockIndex >= 0)
		}//if (kind == LBRACE)
	}//if (kind == IDENTIFIER)
}
