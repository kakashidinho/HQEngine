/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "hlsl2glsl.h"
#include "glsl_optimizer.h"

#include "../../../../HQEngine/Source/HQEngine/HQDefaultFileManager.h"
#include "../../../../HQEngine/Source/HQRendererD3D9/HQShaderVarParserD3D9.h"
#include "../../../../HQEngine/Source/HQClosedStringHashTable.h"

//TO DO: add ES 3.0


const int gNumAttriSems = 16;

const char* gAttriSematics[][2] = {
	"VPOSITION", "xlat_attrib_VPOSITION",
	"VCOLOR", "xlat_attrib_VCOLOR",
	"VNORMAL", "xlat_attrib_VNORMAL",
	"VTANGENT", "xlat_attrib_VTANGENT",
	"VBINORMAL", "xlat_attrib_VBINORMAL",
	"VBLENDWEIGHT", "xlat_attrib_VBLENDWEIGHT",
	"VBLENDINDICES", "xlat_attrib_VBLENDINDICES",
	"VTEXCOORD0", "xlat_attrib_VTEXCOORD0",
	"VTEXCOORD1", "xlat_attrib_VTEXCOORD1",
	"VTEXCOORD2", "xlat_attrib_VTEXCOORD2",
	"VTEXCOORD3", "xlat_attrib_VTEXCOORD3",
	"VTEXCOORD4", "xlat_attrib_VTEXCOORD4",
	"VTEXCOORD5", "xlat_attrib_VTEXCOORD5",
	"VTEXCOORD6", "xlat_attrib_VTEXCOORD6",
	"VTEXCOORD7", "xlat_attrib_VTEXCOORD7",
	"VPSIZE", "xlat_attrib_VPSIZE",
};


HQSharedPtr<HQShaderVarParserD3D9> g_preprocessor;
HQSharedPtr<HQDefaultFileManager> g_includeFileManager = new HQDefaultFileManager();

//transform the cg source to HQEngined version of glsl 
bool compileSrc(const char* src_file, const char *entry, 
				int version, bool esVersion, bool separate_shader,
				const char* profile,
				const std::string & macros,
				std::string & compiled_code_out);

//optimize source
bool optimizeGLSL(const char *source, 
							EShLanguage language,
							ETargetVersion version,
							std::string & optimized_code_out);

//final step
bool postProcess(ShHandle hlsl2glslParser, std::string& inout, bool vertexShader, bool separate_shader, ETargetVersion version);

void insertCode(std::string &inout, const std::string &code);

void emitGLSLUniformBlkElem(std::stringstream& stream, const UniformBlkElemInfo & blkElem);

void printUsage(FILE* outStream = stdout);

//initialize predefined macros
void initPredefinedMacros(std::stringstream & macros);

//main function
int main (int argc, char** argv){
	if (argc < 2)
	{
		printUsage(stderr);
		return -3;
	}
	const char *outputFile = NULL;
	const char *inputFile = argv[argc - 1];
	const char *entryFunc = NULL;
	const char *processedCodeOutFile = NULL;
	std::string preprocessedCode;

	std::stringstream additional_definitions;
	initPredefinedMacros(additional_definitions);

	const char* profile = NULL;
	bool esVersion = false;
	bool separate_shader = false;
	int version = -1;

	//parsing the command line arguments
	for (int i = 1; i < argc - 1; ++i)
	{
		if (strcmp(argv[i], "-entry") == 0 )
		{
			if (i < argc - 1)
				entryFunc = argv[++i];
		}
		else if (strcmp(argv[i], "-o") == 0 )//output file
		{
			if (i < argc - 1)
				outputFile = argv[++i];
		}
		else if (strcmp(argv[i], "-p") == 0 )//output preprocessed code
		{
			if (i < argc - 1)
				processedCodeOutFile = argv[++i];
		}
		else if (strcmp(argv[i], "-profile") == 0 )//output profile
		{
			if (i < argc - 1){
				profile = argv[++i];
			}
		}
		else if (strcmp(argv[i], "-glsles") == 0)//compile to GLSL es
		{
			esVersion = true;
		}
		else if (strcmp(argv[i], "-separate") == 0)//re-declare built-in blocks
		{
			separate_shader = true;
		}
		else if (strncmp(argv[i], "-version", 7) == 0)
		{
			if (i < argc - 1)
			{
				sscanf(argv[i + 1], "%d", &version);
				++i;
			}
		}
		else if (strncmp(argv[i], "-D", 2) == 0){
			char *startDef = argv[i] + 2;
			const char* delim = strstr(startDef, "=");
			additional_definitions << "#define "; 
			if (delim != NULL)
			{
				additional_definitions.write(startDef, delim - startDef);//macro name
				additional_definitions << " " << delim + 1;//macro value
			}
			else
			{
				additional_definitions << startDef;
			}
			
			additional_definitions << "\n";
		}
	}//for (int i = 0; i < argc; ++i)
	


	if (profile != NULL && entryFunc != NULL)
	{
		std::string compiled_code;
		//start compiling
		bool re = compileSrc(inputFile, 
			entryFunc,
			version, 
			esVersion, 
			separate_shader,
			profile, 
			additional_definitions.str(), 
			compiled_code);

		//output preprocessed code
		if (processedCodeOutFile != NULL)
		{
			
			FILE *outstream =  fopen(processedCodeOutFile, "wb");

			if (outstream == NULL)
			{
				printf("Cannot write to %s\n", processedCodeOutFile);
			}
			else
			{
				const char *processedCode = g_preprocessor->GetPreprocessedSrc();
				if (processedCode != NULL)
					fwrite(processedCode, strlen(processedCode), 1, outstream);
				fclose(outstream);
			}
		}

		//write result
		if (re)
		{
			fprintf(stdout, "Compilation Succeeded!\n");

			//write to output file or standard output
			FILE *outstream = stdout;
			if (outputFile != NULL)
			{
				outstream = fopen(outputFile, "wb");

				if (outstream == NULL)
				{
					printf("Cannot write to %s\n", outputFile);
					outstream = stdout;
				}
			}

			fwrite(compiled_code.c_str(), compiled_code.size(), 1, outstream);

			if (outstream != stdout)
			{
				fclose(outstream);
			}
		}
		else{
			fprintf(stderr, "Compilation Failed!\n");
		}
	}
	else
	{
		printUsage(stderr);
	}

	return 0;
}

void printUsage(FILE* outStream)
{
	fprintf(outStream, "Usage:	-entry <entry function> -profile <glslv | glslf> [options] <input file>\n");
	fprintf(outStream, "		options:\n");
	fprintf(outStream, "				-glsles						: Produces GLSL ES code\n");
	fprintf(outStream, "				-o <output file>			: output file\n");
	fprintf(outStream, "				-version <number>			: GLSL version number. Must be 100, 110, 120\n");
	fprintf(outStream, "				-Dname[=value]				: Set a preprocessor macro\n");
	fprintf(outStream, "				-p	<preprocessed file>		: output a preprocessed code\n");
	fprintf(outStream, "				-separate					: insert built-in blocks' re-declaraion for separate shader object\n");
}

//initialize predefined macros
void initPredefinedMacros(std::stringstream & macros){
#if 0
	macros << "#define VPOSITION POSITION\n";
	macros << "#define VCOLOR COLOR\n";
	macros << "#define VNORMAL NORMAL\n";
	macros << "#define VTEXCOORD0 TEXCOORD0\n";
	macros << "#define VTEXCOORD1 TEXCOORD1\n";
	macros << "#define VTEXCOORD2 TEXCOORD2\n";
	macros << "#define VTEXCOORD3 TEXCOORD3\n";
	macros << "#define VTEXCOORD4 TEXCOORD4\n";
	macros << "#define VTEXCOORD5 TEXCOORD5\n";
	macros << "#define VTEXCOORD6 TEXCOORD6\n";
	macros << "#define VTEXCOORD7 TEXCOORD7\n";
	macros << "#define VTANGENT TANGENT\n";
	macros << "#define VBINORMAL BINORMAL\n";
	macros << "#define VBLENDWEIGHT BLENDWEIGHT\n";
	macros << "#define VBLENDINDICES BLENDINDICES\n";
	macros << "#define VPSIZE PSIZE\n";
#endif
}

//transform the cg source to HQEngined version of glsl 
bool compileSrc(const char* src_file, const char *entry, 
				int version, bool esVersion, bool separate_shader,
				const char* profile,
				const std::string & macros,
				std::string & compiled_code_out)
{
	std::ifstream filestream(src_file, std::ifstream::binary);

	if (filestream.good() == false)
	{
		fprintf(stderr, "'%s' not found!\n", src_file);
		return false;
	}

	EShLanguage glShaderType = EShLangCount;
	ETargetVersion etVersion = ETargetVersionCount;

	if(strcmp(profile, "glslv") == 0)
	{
		glShaderType = EShLangVertex;
	}
	else if(strcmp(profile, "glslf") == 0)
	{
		glShaderType = EShLangFragment;
	}

	if (esVersion){
		switch (version)
		{
		default:
			etVersion = ETargetGLSL_ES_100;
			compiled_code_out = "#version 100\nprecision mediump int;\nprecision mediump float;\n";
			break;
		}
	}
	else{
		switch (version)
		{
		case 120:
			etVersion = ETargetGLSL_120;
			break;
		default:
			etVersion = ETargetGLSL_110;
			break;
		}
	}
	
	if (etVersion == ETargetVersionCount){
		fprintf(stderr, "Error! invalid version!\n");
		printUsage(stderr);
		return false;
	}

	//read string from file

	filestream.seekg(0, filestream.end);
	size_t size = filestream.tellg();
	filestream.seekg(0, filestream.beg);

	size_t lineDirectiveSize = strlen("#line 0\n");
	char * source = new char[size + 1 + macros.size() + lineDirectiveSize];
	memcpy(source, macros.c_str(), macros.size());
	memcpy(source + macros.size(), "#line 0\n",  lineDirectiveSize);//add line directive
	filestream.read(source + macros.size()  + lineDirectiveSize, size);
	source[size + macros.size() + lineDirectiveSize] = 0;

	filestream.close();

	/*--------preprocess code-------------*/
	//get containing directory of source file
	{
		std::string source_file_std_string = src_file;
		std::string containingFoler;

		//find last '/' or '\\'
		size_t pos1 = source_file_std_string.find_last_of('/');
		size_t pos2 = source_file_std_string.find_last_of('\\');
		size_t last_slash;
		if (pos1 != std::string::npos)
		{
			last_slash = pos1;
			if (pos2 != std::string::npos && pos2 > pos1)
				last_slash = pos2;
		}
		else
			last_slash = pos2;

		if (last_slash != std::string::npos)
		{
			containingFoler = source_file_std_string.substr(0, last_slash);
			g_includeFileManager->AddFirstSearchPath(containingFoler.c_str());
		}
	}

	g_preprocessor = new HQShaderVarParserD3D9(source, NULL, g_includeFileManager.GetRawPointer());

	const char *codeToCompile = g_preprocessor->GetPreprocessedSrc();
	if (codeToCompile == NULL)
		codeToCompile = source;//fallback to original source

	//init hlsl2glsl compiler
	Hlsl2Glsl_Initialize(NULL, NULL, NULL);

	// create the parser
    ShHandle  parser = Hlsl2Glsl_ConstructCompiler(glShaderType);

	//enable varying variables
	Hlsl2Glsl_UseUserVaryings(parser, true);

	int res = 0;
	int options = 0;
    // parse the file
	res = Hlsl2Glsl_Parse (parser, codeToCompile, profile, etVersion, options);
	

    // convert from cg to glsl
    res = res && Hlsl2Glsl_Translate( parser,  entry,  etVersion, options);

	//print info
    const char*  parserLog = Hlsl2Glsl_GetInfoLog(parser);
	fprintf(stdout, parserLog);

	bool success = true;
    // check for error
    if (res == 0)
    {

		success = false;
    }
	else{
		const char *glsl_code = Hlsl2Glsl_GetShader(parser); 
		std::string opt_code;
		if (!optimizeGLSL(glsl_code, glShaderType, etVersion, opt_code))
			success = false;
		else
		{
			compiled_code_out += opt_code;
			if (!postProcess(parser, compiled_code_out, glShaderType == EShLangVertex, separate_shader, etVersion))
				success = false;
		}
	}


	//clean up
	Hlsl2Glsl_DestructCompiler(parser);

	Hlsl2Glsl_Shutdown();

	delete[] source;

	return success;
}


//optimize source
bool optimizeGLSL(const char *source, 
							EShLanguage language,
							ETargetVersion version,
							std::string & optimized_code_out)
{
	glslopt_ctx*  glsl_optContext;
	switch (version)
	{
	case ETargetGLSL_ES_100:
		glsl_optContext = glslopt_initialize(kGlslTargetOpenGLES20);
		break;
	default:
		glsl_optContext = glslopt_initialize(kGlslTargetOpenGL);
		break;
	}

	glslopt_shader_type shaderType;
	switch (language)
	{
	case EShLangVertex:
		shaderType = kGlslOptShaderVertex ;
		break;
	default:
		shaderType = kGlslOptShaderFragment;
	}

	bool success = true;
	glslopt_shader* shader = glslopt_optimize(glsl_optContext, shaderType, source, 0);

    if(glslopt_get_status(shader))
    {
		optimized_code_out = glslopt_get_output(shader);
    }
	else
	{
		const char * log = glslopt_get_log(shader);

		fprintf(stderr, "internal error: %s\n", log);

		success = false;

	}
    glslopt_shader_delete(shader);

	glslopt_cleanup(glsl_optContext);

	return success;
}

//final step, add some HQEngine semantics
bool postProcess(ShHandle hlsl2glslParser, std::string& inout, bool vertexShader, bool separate_shader, ETargetVersion etVersion)
{
	if (vertexShader)
	{
		//add HQEngine specific semantic to attributes
		for (int i = 0; i < gNumAttriSems ; ++i)
		{
			const char *semantic = gAttriSematics[i][0];
			const char *attribName = gAttriSematics[i][1];

			size_t pos1 = inout.find(attribName);
			if (pos1 != std::string::npos)
			{
				size_t semLen = strlen(semantic);
				size_t nameLen = strlen(attribName);
				inout.replace(pos1, nameLen, std::string(attribName) + " " + semantic);
			}
		}//for (int i = 0; i < gNumAttriSems ; ++i)

		if (separate_shader)
		{
			std::string additionalCode = 
"#ifdef HQEXT_GLSL_SEPARATE_SHADER\n\
#if __VERSION__ > 140\n\
out gl_PerVertex{\n\
#endif\n\
	vec4 gl_Position;\n\
	float gl_PointSize;\n\
#if __VERSION__ > 140\n\
};\n\
#endif\n\
#endif\n";
			insertCode(inout, additionalCode);
		}//if (separate_shader)
	}//if (vertexShader)

	//now add semantics to uniforms
	HQClosedStringHashTable<bool> activeUniforms;//active uniform table

	const ShUniformInfo* uniforms = Hlsl2Glsl_GetUniformInfo(hlsl2glslParser);
	int numUniforms = Hlsl2Glsl_GetUniformCount(hlsl2glslParser);
	for (int i = 0; i < numUniforms; ++i)
	{
		activeUniforms.Add(uniforms[i].name, true);

		if ((uniforms[i].type == EShTypeSampler2D || uniforms[i].type == EShTypeSamplerCube
			 || uniforms[i].type == EShTypeSampler3D 
			  || uniforms[i].type == EShTypeSampler1D) &&
			uniforms[i].semantic != NULL )
		{
			int unit = -1;
			if (sscanf(uniforms[i].semantic, "TEXUNIT%d", &unit) == 1)
			{
				size_t pos1 = inout.find(uniforms[i].name);
				if (pos1 != std::string::npos)
				{
					size_t semLen = strlen(uniforms[i].semantic);
					size_t nameLen = strlen(uniforms[i].name);
					inout.replace(pos1, nameLen, std::string(uniforms[i].name) + " " + uniforms[i].semantic);
				}
			}
		}
	}//for (int i = 0; i < numUniforms; ++i)

	//add uniform blocks
	char uniformBlockPrologue[] = "uniform ubuffer9999999999\n{\n       ";
	char uniformBlockEpilogue[] = "};\n";
	std::stringstream block_decl_stream;
	HQLinkedList<UniformBlock>::Iterator ite;
	size_t search_pos = 0;
	for (g_preprocessor->uniformBlocks.GetIterator(ite); !ite.IsAtEnd(); ++ite)
	{
		sprintf(uniformBlockPrologue, "uniform ubuffer%d\n{\n", ite->index);
		block_decl_stream.str("");
		block_decl_stream << uniformBlockPrologue;

		HQLinkedList<UniformBlkElemInfo>::Iterator elemIte;
		size_t insertPoint = std::string::npos;

		for (ite->blockElems.GetIterator(elemIte); !elemIte.IsAtEnd(); ++elemIte)
		{
			//emit uniform block's element declaration
			emitGLSLUniformBlkElem(block_decl_stream, *elemIte);

			//find in the constant table
			bool found;
			bool active = activeUniforms.GetItem(elemIte->name, found);
			if (found)//found
			{
				search_pos = inout.find(elemIte->name, search_pos);
				if (search_pos == std::string::npos)
				{
					fprintf(stderr, "internal error: could not found %s\n", elemIte->name);
					return false;
				}
				size_t uniform_pos = inout.rfind("uniform", search_pos);
				if (insertPoint == std::string::npos)//uniform block declaration's insert point
				{
					insertPoint = uniform_pos;
				}
				
				size_t semicolon_pos = inout.find(";", search_pos);
				//remove "uniform" declaration because we will insert uniform block declaration
				for (size_t i = uniform_pos; i <= semicolon_pos; ++i)
					inout[i] = ' ';

			}//if (found)
		}//for (ite->blockElems.GetIterator(elemIte); !elemIte.IsAtEnd(); ++elemIte)
		
		block_decl_stream << uniformBlockEpilogue;//close block declaration

		if (insertPoint != std::string::npos)//this block is active so we write block declaration
		{
			inout.replace(insertPoint, 1, block_decl_stream.str());
		}
	}//for (g_preprocessor->uniformBlocks.GetIterator(ite); !ite.IsAtEnd(); ++ite)

	return true;
}

#define IS_WHITE_SPACE(c) (c == ' ' || c == '\t' || c == '\r')

void insertCode(std::string &inout, const std::string &code){
	enum State {
		LINE_START,
		PREPROCESSOR_START,
		IGNORE_LINE
	};
	//insert "statement" to the source code, but only after every #extension and #version lines since these lines must appear before any none preprocessor lines
	size_t line = 1;
	size_t insert_pos = 0;
	size_t line_insert = 1;
	State state = LINE_START;
	
	for (size_t i = 0; i < inout.size(); ++i)
	{
		char c = inout[i];
		if (c == '\n')
		{
			line ++;
			state = LINE_START;
		}
		else
		{
			switch (state)
			{
			case LINE_START:
				if (c == '#')
					state = PREPROCESSOR_START;
				else if (!IS_WHITE_SPACE(c))//not white space
					state = IGNORE_LINE;
				break;
			case PREPROCESSOR_START:
				if (c == 'e' || c == 'v')//may be "extension" or "version"
				{
					if (inout.compare(i, 9, "extension") == 0 && i + 9 < inout.size() && IS_WHITE_SPACE(inout[i + 9]))
					{
						//found #extension
						insert_pos = inout.find("\n", i + 9) + 1;
						i = insert_pos - 1;
						line ++;
						line_insert = line;
						state = LINE_START;
					}
					else if (inout.compare(i, 7, "version") == 0 && i + 7 < inout.size() && IS_WHITE_SPACE(inout[i + 7]))
					{
						//found #version
						insert_pos = inout.find("\n", i + 7) + 1;
						i = insert_pos - 1;
						line ++;
						line_insert = line;
						state = LINE_START;
					}
				}
				else if (!IS_WHITE_SPACE(c))//not white space
				{
					state = IGNORE_LINE;
				}
				break;
		
			}//switch (state)
		}//else of if (c == '\n')
			
	}//for (size_t i = 0; i < inout.size(); ++i)

	if (insert_pos < inout.size())
	{
		inout.insert(insert_pos, code);
	}
	
}

void emitGLSLUniformBlkElem(std::stringstream& stream, const UniformBlkElemInfo & blkElem)
{
	stream << "\t";
	if (blkElem.row > 1)//matrix
	{
		if (blkElem.row != blkElem.col)//non squared matrix is transformed to array of vectors by hlsl2glsl
		{
			stream << "vec" << blkElem.col;
		}
		else {
			stream << "mat" << blkElem.col;
		}
	}
	else {
		//vector
		if (blkElem.integer)
		{
			if (blkElem.col == 1)
				stream << "int";
			else
				stream << "ivec" << blkElem.col;
		}
		else
		{
			if (blkElem.col == 1)
				stream << "float";
			else
				stream << "vec" << blkElem.col;
		}
	}

	stream << " " << blkElem.name;
	if (blkElem.row > 1 && blkElem.row != blkElem.col)//non squared matrix is transformed to array of vectors by hlsl2glsl
	{
		stream << "[" << blkElem.arraySize * blkElem.row << "]";
	}
	else if (blkElem.arraySize > 1)
		stream << "[" << blkElem.arraySize << "]";

	stream << ";\n";
}