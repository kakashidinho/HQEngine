/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "HQShaderGL_GLSLController_inline.h"
#include "HQShaderGL_UBO.h"

#include <string>

#if defined DEBUG || defined _DEBUG
#define LOG_SRC_ERR 0
#else
#define LOG_SRC_ERR 0
#endif

const char semanticKeywords[] =
	"\
#define VPOSITION\n\
#define VCOLOR\n\
#define VNORMAL\n\
#define VTEXCOORD0\n\
#define VTEXCOORD1\n\
#define VTEXCOORD2\n\
#define VTEXCOORD3\n\
#define VTEXCOORD4\n\
#define VTEXCOORD5\n\
#define VTEXCOORD6\n\
#define VTEXCOORD7\n\
#define VTANGENT\n\
#define VBINORMAL\n\
#define VBLENDWEIGHT\n\
#define VBLENDINDICES\n\
#define VPSIZE\n"
;

const char samplerKeywords[] =
"\
#define TEXUNIT0\n\
#define TEXUNIT1\n\
#define TEXUNIT2\n\
#define TEXUNIT3\n\
#define TEXUNIT4\n\
#define TEXUNIT5\n\
#define TEXUNIT6\n\
#define TEXUNIT7\n\
#define TEXUNIT8\n\
#define TEXUNIT9\n\
#define TEXUNIT10\n\
#define TEXUNIT11\n\
#define TEXUNIT12\n\
#define TEXUNIT13\n\
#define TEXUNIT14\n\
#define TEXUNIT15\n\
#define TEXUNIT16\n\
#define TEXUNIT17\n\
#define TEXUNIT18\n\
#define TEXUNIT19\n\
#define TEXUNIT20\n\
#define TEXUNIT21\n\
#define TEXUNIT22\n\
#define TEXUNIT23\n\
#define TEXUNIT24\n\
#define TEXUNIT25\n\
#define TEXUNIT26\n\
#define TEXUNIT27\n\
#define TEXUNIT28\n\
#define TEXUNIT29\n\
#define TEXUNIT30\n\
#define TEXUNIT31\n";

#define IS_WHITE_SPACE(c) (c == ' ' || c == '\t' || c == '\r')

static void InsertUBO_STD140_LayoutControl(std::string &inout){
	enum State {
		LINE_START,
		PREPROCESSOR_START,
		IGNORE_LINE
	};
	//insert "layout(std140) uniform;" to the source code, but only after every #extension lines since these lines must appear before any none preprocessor lines
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
				if (c == 'e')//may be "extension"
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
		char statement_to_insert[] = "layout(std140) uniform;\n#line 9999999999\n";
		sprintf(statement_to_insert, "layout(std140) uniform;\n#line %u\n", (hquint32)line_insert - 1);
		inout.insert(insert_pos, statement_to_insert);
	}
	
}

/*-----------HQBaseGLSLShaderController----------------------*/

HQBaseGLSLShaderController::HQBaseGLSLShaderController()
{
	pVParser = new HQVarParserGL(g_pShaderMan);
}
HQBaseGLSLShaderController::~HQBaseGLSLShaderController()
{
	SafeDelete(pVParser);
}

void HQBaseGLSLShaderController::GetPredefineMacroGLSL(std::string & macroDef , const HQShaderMacro * pDefines, std::string &version_string)
{
	if(pDefines == NULL)
		return;
	const HQShaderMacro *pD = pDefines;
	macroDef = "";
	while (pD->name != NULL && pD->definition != NULL)
	{
		if (!strcmp(pD->name , "version") && version_string.size() == 0)
		{
			version_string = "#version ";
            version_string += pD->definition;
            version_string += '\n';
		}
		else
		{
			macroDef += "#define ";
			macroDef += pD->name;
			macroDef += " ";
            macroDef += pD->definition;
            macroDef += '\n';
		}
        pD++;
	}
}

HQReturnVal HQBaseGLSLShaderController::CreateShaderFromStreamGLSL(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 HQShaderObjectGL **ppShaderObjectOut)
{
	long size = dataStream->TotalSize();

	char *source = new char[size + 1];

	dataStream->ReadBytes(source,size ,1);
	source[size] = '\0';

	if (dataStream->GetName() != NULL)
			g_pShaderMan->Log("GLSL shader is beging compiled from stream %s ...", dataStream->GetName());

	HQReturnVal re = this->CreateShaderFromMemoryGLSL(type , source , pDefines, ppShaderObjectOut);

	delete[] source;

	if (HQFailed(re))
	{
		if (dataStream->GetName() != NULL)
			g_pShaderMan->Log("GLSL shader compile from stream %s error !", dataStream->GetName());
		else
			g_pShaderMan->Log("GLSL shader compile from stream error !");
	}

	return re;
}


HQReturnVal HQBaseGLSLShaderController::CreateShaderFromMemoryGLSL(HQShaderType type,
									 const char* source,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 HQShaderObjectGL **ppShaderObjectOut)
{
	if(source == NULL)
	{
		return HQ_FAILED;
	}

	GLenum shaderType;
	bool selfContained = false;
	switch (type)
	{
	case HQ_VERTEX_SHADER:
		shaderType = GL_VERTEX_SHADER; break;
	case HQ_PIXEL_SHADER:
		shaderType = GL_FRAGMENT_SHADER; break;
#ifdef GL_GEOMETRY_SHADER
	case HQ_GEOMETRY_SHADER:
		if (GLEW_EXT_geometry_shader4 || GLEW_VERSION_3_2)
		{
			shaderType = GL_GEOMETRY_SHADER;
			break;
		}
#endif
#ifdef GL_COMPUTE_SHADER
	case HQ_COMPUTE_SHADER:
		if (GLEW_VERSION_4_3)
		{
			shaderType = GL_COMPUTE_SHADER;
			selfContained = true;
			break;
		}
#endif
	default:
		g_pShaderMan->Log("Error : Unsupported GLSL shader type=%u", (hquint32)type);
		return HQ_FAILED;
	}


	HQShaderObjectGL *sobject = HQ_NEW HQShaderObjectGL();
	sobject->type = type;
	sobject->selfContainProgram = selfContained;
	sobject->isGLSL = true;
	bool parseSuccess ;
	std::string processed_src;

	HQLinkedList<HQUniformBlockInfoGL>** ppUBlocksInfo = NULL;
	ppUBlocksInfo = &sobject->pUniformBlocks;
	bool native_UBO_supported = GLEW_VERSION_3_1 == GL_TRUE;

#ifdef HQ_OPENGLES
	std::string preDefined = "#define HQEXT_GLSL_ES\n";
#else
	std::string preDefined = "#define HQEXT_GLSL\n";
#endif

	this->AppendPredefinedMacros(preDefined);//implement dependent

	/*--------------preprocess source---------------*/
	if(type==HQ_VERTEX_SHADER)
	{
		parseSuccess = pVParser->Parse(source, g_pShaderMan->GetIncludeFileManager(), preDefined, pDefines, processed_src, native_UBO_supported, ppUBlocksInfo, &sobject->pAttribList, &sobject->pUniformSamplerList);
	}
	else
	{
		parseSuccess = pVParser->Parse(source, g_pShaderMan->GetIncludeFileManager(), preDefined, pDefines, processed_src, native_UBO_supported, ppUBlocksInfo, NULL, &sobject->pUniformSamplerList);
	}

	if (!parseSuccess)
	{
		delete sobject;
		return HQ_FAILED;
	}

	sobject->shader = glCreateShader(shaderType);

	std::string version_string = "";
	/*------ Remove #version in the source---------*/ 
	{
		size_t pos1 = processed_src.find("#");
		if (pos1 != std::string::npos)
		{
			size_t pos2 = processed_src.find("version", pos1);
			if (pos2 != std::string::npos)
			{
				bool found = true;
				for (size_t i = pos1 + 1; i < pos2; ++i)//make sure only white spaces are between "#" and "version"
				{
					char c = processed_src[i];
					if (c != ' ' && c != '\t' && c != '\r')
					{
						found = false;
						break;
					}
				}
				
				if (found)
				{
					size_t pos3 = processed_src.find("\n", pos2 + 7);
					version_string.assign(processed_src.c_str() + pos1, pos3 - pos1 + 1); 
					for (size_t i = pos1; i < pos3; ++i)
					{
						processed_src[i] = ' ';//remove from the source
					}
				}

			}//if (pos2 != std::string::npos)
		}//if (pos1 != std::string::npos)
	}

	/*---create macro definition list---------*/
	std::string macroDefList;
	this->GetPredefineMacroGLSL(macroDefList , pDefines, version_string);

	//get version number

	int version_number = 0;
	if (sscanf(version_string.c_str(), "#version %d", &version_number) != 1)
		sscanf(version_string.c_str(), "# version %d", &version_number);//try one more time
	std::string UBO_extension_line = "";
	//work around uniform buffer objects
	if (sobject->pUniformBlocks != NULL && sobject->pUniformBlocks->GetSize() > 0)
	{
		if (native_UBO_supported)
		{
			SafeDelete(sobject->pUniformBlocks);//no need for this information
			InsertUBO_STD140_LayoutControl(processed_src);//set std140 layout as default
#ifdef HQ_OPENGLES
			if (version_number < 300)
			{
				version_string = "#version 300 es\n";//300 is minimum version for uniform buffer objects
				g_pShaderMan->Log("GLSL shader compile warning: shader contains uniform buffer blocks but they are not supported. Switching to version 300 es...", version_number); 
				
			}
#else//#ifdef HQ_OPENGLES
			if (version_number < 140)
			{
				version_string = "#version 140\n";//140 is minimum version for uniform buffer objects
				g_pShaderMan->Log("GLSL shader compile warning: shader contains uniform buffer blocks but they are not supported. Switching to version 140...", version_number); 
			}
#endif//#ifdef HQ_OPENGLES
		}//if (native_UBO_supported)
	}//if (sobject->pUniformBlocks != NULL && sobject->pUniformBlocks->GetSize() > 0)


	/*--------set shader source---------*/
	const GLchar* sourceArray[] = {
		version_string.c_str(),
		UBO_extension_line.c_str(),
		semanticKeywords,
		samplerKeywords,
		preDefined.c_str(),
		macroDefList.c_str(),
		"#line 0 0\n",
		processed_src.c_str()
	};

	if (type != HQ_VERTEX_SHADER)
		sourceArray[2] = "";//only vertex shader need semantic definitions

	glShaderSource(sobject->shader, 8, (const GLchar**)sourceArray, NULL);
	glCompileShader(sobject->shader);

	GLint compileOK;
	glGetShaderiv(sobject->shader , GL_COMPILE_STATUS , &compileOK);
	
	bool error = (sobject->shader==0 || compileOK == GL_FALSE);
	{
		int infologLength = 0;
		int charsWritten  = 0;
		char *infoLog;
		glGetShaderiv(sobject->shader, GL_INFO_LOG_LENGTH,&infologLength);
		if (infologLength > 0)
		{
#if LOG_SRC_ERR
			int src_length;
			char * glsource;
			glGetShaderiv(sobject->shader, GL_SHADER_SOURCE_LENGTH,&src_length);
			glsource = (char*)malloc(src_length);
			glGetShaderSource(sobject->shader, src_length, NULL, glsource);
#endif
			infoLog = (char *)malloc(infologLength);
	        	glGetShaderInfoLog(sobject->shader, infologLength, &charsWritten, infoLog);
			if (error)
#if LOG_SRC_ERR
				g_pShaderMan->Log("GLSL shader compile error: %s. Source code=<%s>",infoLog, glsource);
#else
				g_pShaderMan->Log("GLSL shader compile error: %s",infoLog);
#endif
			else
				g_pShaderMan->Log("GLSL shader compile info: %s",infoLog);


	        	free(infoLog);
#if LOG_SRC_ERR
			free(glsource);
#endif
	    	}
		if (error)
		{
			delete sobject;
			return HQ_FAILED;
		}
	}
	
	//create self-contained shader program
	if (sobject->selfContainProgram)
	{
		HQReturnVal re = this->ConvertToSelfContainedShaderProgram(sobject);
		if (HQFailed(re))
		{
			delete sobject;
			return re;
		}
	}

	*ppShaderObjectOut = sobject;

	return HQ_OK;
}

//convert shader object to self-contained shader program
HQReturnVal HQBaseGLSLShaderController::ConvertToSelfContainedShaderProgram(HQShaderObjectGL *sobject)
{
	//backup shader object
	GLuint originalGlShaderObj = sobject->shader;
	//create a self-contained program
	sobject->shader = glCreateProgram();

	glAttachShader(sobject->shader, originalGlShaderObj);

	this->OnLinkingProgram(sobject->shader);//implementation dependent, may convert program to separable program

	//fist linking
	glLinkProgram(sobject->shader);

	if (sobject->type == HQ_VERTEX_SHADER)
	{
		//bind attributes' locations
		this->BindAttributeLocationGLSL(sobject->shader, *sobject->pAttribList);

		//re-link
		glLinkProgram(sobject->shader);
	}

	//no longer need this shader object
	glDeleteShader(originalGlShaderObj);

	//check for link error
	GLint OK;
	glGetProgramiv(sobject->shader, GL_LINK_STATUS, &OK);

	if (OK == GL_FALSE)
	{
		int infologLength = 0;
		int charsWritten = 0;
		char *infoLog;
		glGetProgramiv(sobject->shader, GL_INFO_LOG_LENGTH, &infologLength);
		if (infologLength > 0)
		{
			infoLog = (char *)malloc(infologLength);
			glGetProgramInfoLog(sobject->shader, infologLength, &charsWritten, infoLog);
			g_pShaderMan->Log("GLSL self-contained program link error: %s", infoLog);
			free(infoLog);
		}
		return HQ_FAILED;
	}


	//---------bind sampler units and uniform blocks-------------------
	GLuint currentActiveProgram;
	glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*)&currentActiveProgram);
	glUseProgram(sobject->shader);


	//bind sampler units
	this->BindSamplerUnitGLSL(sobject->shader,
		*sobject->pUniformSamplerList);

	//bind uniform blocks
	this->BindUniformBlockGLSL(sobject->shader);

	//back to current program
	glUseProgram(currentActiveProgram);

	return HQ_OK;
}

HQReturnVal HQBaseGLSLShaderController::CreateProgramGLSL(
							HQBaseShaderProgramGL *pNewProgramObj,
							  HQSharedPtr<HQShaderObjectGL>& pVShader,
							  HQSharedPtr<HQShaderObjectGL>& pGShader,
							  HQSharedPtr<HQShaderObjectGL>& pFShader)
{

	hq_uint32 flags=0;//cờ thể hiện trong program có những loại shader nào

	if(pVShader != NULL)
	{
		if(pVShader->type!=HQ_VERTEX_SHADER//shader có id <vertexShaderID> không phải vertex shader
			|| pVShader->isGLSL == false)//chỉ chấp nhận shader compile từ GL shading language
			return HQ_FAILED_WRONG_SHADER_TYPE;
		flags|=useV;
	}
	if(pFShader != NULL)
	{
		if(pFShader->type!=HQ_PIXEL_SHADER//shader có id <pixelShaderID> không phải pixel shader
			|| pFShader->isGLSL == false)//chỉ chấp nhận shader compile từ GL shading language
			return HQ_FAILED_WRONG_SHADER_TYPE;
		flags|=useF;
	}
	if(pGShader != NULL)
	{
		if(pGShader->type!=HQ_GEOMETRY_SHADER//shader có id <geometryShaderID> không phải geometry shader
			|| pGShader->isGLSL == false)//chỉ chấp nhận shader compile từ GL shading language
			return HQ_FAILED_WRONG_SHADER_TYPE;
		flags|=useG;
	}

	if(flags != useVF && flags != useVGF)//need both valid vertex & pixel shader
		return HQ_FAILED;

	pNewProgramObj->programGLHandle = glCreateProgram();

	if (pNewProgramObj->programGLHandle == 0)
	{
		return HQ_FAILED;
	}

	if(flags & useV)
		glAttachShader(pNewProgramObj->programGLHandle, pVShader->shader);
	if(flags & useG)
		glAttachShader(pNewProgramObj->programGLHandle, pGShader->shader);
	if(flags & useF)
		glAttachShader(pNewProgramObj->programGLHandle, pFShader->shader);

	this->OnLinkingProgram(pNewProgramObj->programGLHandle);//implementation dependent, may convert program to separable program

	//first link
	glLinkProgram(pNewProgramObj->programGLHandle);

	//bind program attributes; locations
	this->BindAttributeLocationGLSL(pNewProgramObj->programGLHandle, *pVShader->pAttribList);

	//final link
	glLinkProgram(pNewProgramObj->programGLHandle);

	GLint OK;
	glGetProgramiv(pNewProgramObj->programGLHandle, GL_LINK_STATUS, &OK);

	if(OK == GL_FALSE)
	{
		int infologLength = 0;
	    int charsWritten  = 0;
	    char *infoLog;
		glGetProgramiv(pNewProgramObj->programGLHandle, GL_INFO_LOG_LENGTH, &infologLength);
	    if (infologLength > 0)
	    {
	        infoLog = (char *)malloc(infologLength);
			glGetProgramInfoLog(pNewProgramObj->programGLHandle, infologLength, &charsWritten, infoLog);
			g_pShaderMan->Log("GLSL program link error: %s", infoLog);
	        free(infoLog);
	    }
		return HQ_FAILED;
	}


	//bind sampler unit
	GLuint currentActiveProgram;
	glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*)&currentActiveProgram);
	glUseProgram(pNewProgramObj->programGLHandle);

	if(flags & useV)
		this->BindSamplerUnitGLSL(pNewProgramObj,
								  *pVShader->pUniformSamplerList);
	if(flags & useG)
		this->BindSamplerUnitGLSL(pNewProgramObj,
								  *pGShader->pUniformSamplerList);
	if(flags & useF)
		this->BindSamplerUnitGLSL(pNewProgramObj,
								  *pFShader->pUniformSamplerList);
	//bind uniform block
	this->BindUniformBlockGLSL(pNewProgramObj->programGLHandle);

	//back to current program
	glUseProgram(currentActiveProgram);

#if 0//TO DO: doesn't make sense to validate program here. it should be validated just before drawing
	//check if this program is valid
	glValidateProgram(pNewProgramObj->programGLHandle);
	glGetProgramiv(pNewProgramObj->programGLHandle, GL_VALIDATE_STATUS, &OK);
	if(OK == GL_FALSE)
	{
		int infologLength = 0;
		int charsWritten  = 0;
		char *infoLog;
		glGetProgramiv(pNewProgramObj->programGLHandle, GL_INFO_LOG_LENGTH, &infologLength);
		if (infologLength > 0)
		{
			infoLog = (char *)malloc(infologLength);
			glGetProgramInfoLog(pNewProgramObj->programGLHandle, infologLength, &charsWritten, infoLog);
			g_pShaderMan->Log("GLSL program validate error: %s", infoLog);
			free(infoLog);
		}
		return HQ_FAILED;
	}
#endif

	if (!g_pShaderMan->AddItem(pNewProgramObj))
	{
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}
/*-------------------------*/
void HQBaseGLSLShaderController::BindAttributeLocationGLSL(GLuint program , HQLinkedList<HQShaderAttrib>& attribList)
{
	HQLinkedList<HQShaderAttrib>::Iterator ite;
	attribList.GetIterator(ite);
	for (;!ite.IsAtEnd(); ++ite) {
		if (glGetAttribLocation(program, ite->name.c_str()) != -1) {
			glBindAttribLocation(program, ite->index, ite->name.c_str());
		}
	}
}

void HQBaseGLSLShaderController::BindUniformBlockGLSL(GLuint program)
{
#ifdef HQ_GL_UNIFORM_BUFFER_DEFINED
	if (GLEW_VERSION_3_1)
	{
		char blockName[10];
		GLuint index;
		for (int i = 0 ; i < MAX_UNIFORM_BUFFER_SLOTS ; ++i)
		{
			sprintf(blockName , "ubuffer%d" , i );
			index = glGetUniformBlockIndex(program , blockName);
			if (index != GL_INVALID_INDEX)
				glUniformBlockBinding(program , index , i);
		}
	}
#endif
}

void HQBaseGLSLShaderController::BindSamplerUnitGLSL(GLuint program, HQLinkedList<HQUniformSamplerGL>& samplerList)
{
	HQLinkedList<HQUniformSamplerGL>::Iterator ite;
	samplerList.GetIterator(ite);
	for (; !ite.IsAtEnd(); ++ite) {
		GLint index = glGetUniformLocation(program, ite->name.c_str());
		if (index != -1) {
			glUniform1i(index, ite->samplerUnit);
		}
	}
}

void HQBaseGLSLShaderController::BindSamplerUnitGLSL(HQBaseShaderProgramGL* pProgram , HQLinkedList<HQUniformSamplerGL>& samplerList)
{
	HQLinkedList<HQUniformSamplerGL>::Iterator ite;
	samplerList.GetIterator(ite);
	for (;!ite.IsAtEnd(); ++ite) {
		const HQShaderParameterGL* param = pProgram->GetParameter(ite->name.c_str()); 
		if (param != NULL) {
			glUniform1i(param->location , ite->samplerUnit);
		}
	}
}



#ifdef HQ_GLSL_SHADER_PIPELINE_DEFINED
/*-----------HQBaseGLSLShaderPipelineController----------------------*/
GLuint ge_shader_pipeline = 0;


HQBaseGLSLShaderPipelineController::HQBaseGLSLShaderPipelineController()
: HQBaseGLSLShaderController()
{
	if (ge_shader_pipeline == 0)
		glGenProgramPipelines(1, &ge_shader_pipeline);
	glBindProgramPipeline(ge_shader_pipeline);
#if HQ_GLSL_SHADER_VERIFY_PIPELINE
	GLint pipeline;
	glGetIntegerv(GL_PROGRAM_PIPELINE_BINDING, &pipeline);
#endif
}
HQBaseGLSLShaderPipelineController::~HQBaseGLSLShaderPipelineController()
{
	glBindProgramPipeline(0);
	glDeleteProgramPipelines(1, &ge_shader_pipeline);

	ge_shader_pipeline = 0;
}


void HQBaseGLSLShaderPipelineController::OnLinkingProgram(GLuint program)
{
	//convert to separable program
	glProgramParameteri(program, GL_PROGRAM_SEPARABLE, GL_TRUE);
}

void HQBaseGLSLShaderPipelineController::AppendPredefinedMacros(std::string & predefinedMacros)
{
	predefinedMacros += "#define HQEXT_GLSL_SEPARATE_SHADER\n";
}

#endif //#ifdef HQ_GLSL_SHADER_PIPELINE_DEFINED
