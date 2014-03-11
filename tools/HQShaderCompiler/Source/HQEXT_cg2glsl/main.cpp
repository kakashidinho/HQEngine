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

//transform the cg source to HQEngined version of glsl 
bool compileSrc(const char* src_file, const char *entry, 
				int version, bool esVersion,
				const char* profile,
				const std::string & macros,
				std::string & compiled_code_out);

//optimize source
bool optimizeGLSL(const char *source, 
							EShLanguage language,
							ETargetVersion version,
							std::string & optimized_code_out);

//final step
bool postProcess(ShHandle hlsl2glslParser, std::string& inout, bool vertexShader);

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

	std::stringstream additional_definitions;
	initPredefinedMacros(additional_definitions);

	const char* profile = NULL;
	bool esVersion = false;
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
		bool re = compileSrc(inputFile, entryFunc, version, esVersion, profile, additional_definitions.str(), compiled_code);

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
	fprintf(outStream, "				-glsles				: Produces GLSL ES code\n");
	fprintf(outStream, "				-o <output file>	: output file\n");
	fprintf(outStream, "				-version <number>	: GLSL version number. Must be 100, 110, 120\n");
	fprintf(outStream, "				-Dname[=value]		: Set a preprocessor macro\n");
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
				int version, bool esVersion,
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

	//init hlsl2glsl compiler
	Hlsl2Glsl_Initialize(NULL, NULL, NULL);

	// create the parser
    ShHandle  parser = Hlsl2Glsl_ConstructCompiler(glShaderType);

	//enable varying variables
	Hlsl2Glsl_UseUserVaryings(parser, true);

	int res = 0;
	int options = 0;
    // parse the file
    res = Hlsl2Glsl_Parse (parser, source, profile, etVersion, options);
	

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
			if (!postProcess(parser, compiled_code_out, glShaderType == EShLangVertex))
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
bool postProcess(ShHandle hlsl2glslParser, std::string& inout, bool vertexShader)
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
		}
	}//if (vertexShader)

	//now add semantics to uniforms
	const ShUniformInfo* uniforms = Hlsl2Glsl_GetUniformInfo(hlsl2glslParser);
	int numUniforms = Hlsl2Glsl_GetUniformCount(hlsl2glslParser);
	for (int i = 0; i < numUniforms; ++i)
	{
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

	return true;
}