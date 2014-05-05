#define _CRT_SECURE_NO_WARNINGS
extern "C"
{
#include "hlslcc.h"
};
#include <string>
#include <stdio.h>

#define GL_VERTEX_SHADER 0x8B31

const int gNumAttriSems = 16;

const char* gAttriSematics[][2] = {
	"VPOSITION", "VPOSITION0",
	"VCOLOR", "VCOLOR0",
	"VNORMAL", "VNORMAL0",
	"VTANGENT", "VTANGENT0",
	"VBINORMAL", "VBINORMAL0",
	"VBLENDWEIGHT", "VBLENDWEIGHT0",
	"VBLENDINDICES", "VBLENDINDICES0",
	"VTEXCOORD0", "VTEXCOORD0",
	"VTEXCOORD1", "VTEXCOORD1",
	"VTEXCOORD2", "VTEXCOORD2",
	"VTEXCOORD3", "VTEXCOORD3",
	"VTEXCOORD4", "VTEXCOORD4",
	"VTEXCOORD5", "VTEXCOORD5",
	"VTEXCOORD6", "VTEXCOORD6",
	"VTEXCOORD7", "VTEXCOORD7",
	"VPSIZE", "VPSIZE0",
};
void printUsage(FILE* outStream = stdout);
void postProcess(std::string& inout, int shaderType);
GLLang languageFromString(const char* str);

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		printUsage(stderr);
		return -3;
	}
	GLLang lang = LANG_DEFAULT;
	const char *outputFile = NULL;
	const char *inputFile = argv[argc - 1];

	//parsing the command line arguments
	for (int i = 1; i < argc - 1; ++i)
	{
		if (strcmp(argv[i], "-o") == 0 )//output file
		{
			if (i < argc - 1)
				outputFile = argv[++i];
		}
		else if (strncmp(argv[i], "-version", 7) == 0)
		{
			if (i < argc - 1)
			{
				lang = languageFromString (argv[i + 1]);
				++i;
			}
		}
	}

	GLSLShader result;
	unsigned int flags = HLSLCC_FLAG_UNIFORM_BUFFER_OBJECT | HLSLCC_FLAG_INOUT_SEMANTIC_NAMES 
		| HLSLCC_FLAG_DONT_BIND_VERTEX_ATTRIBUTE;
    //compile
	int compiledOK = TranslateHLSLFromFile(inputFile, flags, lang, NULL, &result);

	if(compiledOK)
	{
		std::string compiled_code = result.sourceCode;

		//post process
		postProcess(compiled_code, result.shaderType);

		//write to ouput
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

		FreeGLSLShader(&result);
	}
	else
	{
		fprintf(stderr, "Compilation Failed!\n");
		//TO DO
		return -1;
	}

	return 0;
}

void printUsage(FILE* outStream)
{
	fprintf(outStream, "Usage:	[options] <input file>\n");
	fprintf(outStream, "		options:\n");
	fprintf(outStream, "				-version <version number>	: GLSL version number. Can be es100, 430, etc\n");
	fprintf(outStream, "				-o <output file>			: output file\n");
}

GLLang languageFromString(const char* str)
{
    if(strcmp(str, "es100")==0)
    {
        return LANG_ES_100;
    }
    if(strcmp(str, "es300")==0)
    {
        return LANG_ES_300;
    }
	if(strcmp(str, "es310")==0)
	{
		return LANG_ES_310;
	}
    if(strcmp(str, "120")==0)
    {
        return LANG_120;
    }
    if(strcmp(str, "130")==0)
    {
        return LANG_130;
    }
    if(strcmp(str, "140")==0)
    {
        return LANG_140;
    }
    if(strcmp(str, "150")==0)
    {
        return LANG_150;
    }
    if(strcmp(str, "330")==0)
    {
        return LANG_330;
    }
    if(strcmp(str, "400")==0)
    {
        return LANG_400;
    }
    if(strcmp(str, "410")==0)
    {
        return LANG_410;
    }
    if(strcmp(str, "420")==0)
    {
        return LANG_420;
    }
    if(strcmp(str, "430")==0)
    {
        return LANG_430;
    }
    if(strcmp(str, "440")==0)
    {
        return LANG_440;
    }
    return LANG_DEFAULT;
}

void postProcess(std::string& inout, int shaderType)
{
	if (shaderType == GL_VERTEX_SHADER)
	{
		//add HQEngine specific semantic to vertex attributes
		for (int i = 0; i < gNumAttriSems ; ++i)
		{
			const char *semantic = gAttriSematics[i][0];
			const char *attribName = gAttriSematics[i][1];

			size_t pos1 = inout.find(attribName);
			if (pos1 != std::string::npos)
			{
				//add sematic at the end of declaration
				size_t semLen = strlen(semantic);
				size_t nameLen = strlen(attribName);
				inout.replace(pos1, nameLen, std::string("vinput_") + attribName + " " + semantic);

				pos1 += nameLen + semLen + 8;
				//replace name at every occurances
				while ((pos1 = inout.find(attribName, pos1)) != std::string::npos){
					inout.replace(pos1, nameLen, std::string("vinput_") + attribName);
					pos1 += nameLen + 7;
				}
			}
		}//for (int i = 0; i < gNumAttriSems ; ++i)
	}//if (shaderType == GL_VERTEX_SHADER)
}

