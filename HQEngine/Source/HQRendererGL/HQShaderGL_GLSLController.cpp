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


/*------------HQShaderProgramGL_GLSL---------------------*/

HQShaderParameterGL* HQShaderProgramGL_GLSL::TryCreateParameterObject(const char *parameterName)
{
	GLint paramGLLoc =glGetUniformLocation(this->programGLHandle,parameterName);//get parameter handle

	if(paramGLLoc==-1)//không tìm thấy
	{
		return  NULL;
	}
	HQShaderParameterGL* pNewParameter = new HQShaderParameterGL();
	pNewParameter->texUnit=-1;
	pNewParameter->location = paramGLLoc;

	
	return pNewParameter;
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

void HQBaseGLSLShaderController::GetPredefineMacroGLSL(std::string & macroDef , const HQShaderMacro * pDefines, bool ignoreVersion)
{
	if(pDefines == NULL)
		return;
	const HQShaderMacro *pD = pDefines;
	macroDef = "";
	while (pD->name != NULL && pD->definition != NULL)
	{
		if (!strcmp(pD->name , "version"))
		{
			macroDef += "#version ";
		}
		else
		{
			macroDef += "#define ";
			macroDef += pD->name;
			macroDef += " ";
		}
		macroDef += pD->definition;
		macroDef += '\n';
		pD++;
	}
}

HQReturnVal HQBaseGLSLShaderController::CreateShaderFromFileGLSL(HQShaderType type,
									 const char* fileName,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 HQShaderObjectGL **ppShaderObjectOut)
{
	FILE *f = fopen(fileName,"rb");
	if(f == NULL)
	{
		g_pShaderMan->Log("GLSL shader compile from file %s error : file not found !",fileName);
		return HQ_FAILED;
	}
	fseek(f, 0  , SEEK_END);
	long size = ftell(f);
	rewind(f);

	char *source = new char[size + 1];
	if(source == NULL)
	{
		fclose(f);
		return HQ_FAILED;
	}

	fread(source,size ,1,f);
	source[size] = '\0';
	fclose(f);

	HQReturnVal re = this->CreateShaderFromMemoryGLSL(type , source , pDefines, ppShaderObjectOut);

	delete[] source;

	if (HQFailed(re))
		g_pShaderMan->Log("GLSL shader compile from file %s error !",fileName);

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
	if(type==HQ_VERTEX_SHADER)
		shaderType = GL_VERTEX_SHADER;
	else if (type==HQ_PIXEL_SHADER)
		shaderType = GL_FRAGMENT_SHADER;
	else if ((GLEW_EXT_geometry_shader4 || GLEW_VERSION_3_2 ) && type==HQ_GEOMETRY_SHADER)
		shaderType = GL_GEOMETRY_SHADER_EXT;
	else
	{
		return HQ_FAILED;
	}

	HQShaderObjectGL *sobject= new HQShaderObjectGL();
	sobject->isGLSL = true;
	bool parseSuccess ;
	if(type==HQ_VERTEX_SHADER)
	{
		parseSuccess  = pVParser->Parse(source , &sobject->pAttribList , &sobject->pUniformSamplerList);
	}
	else
	{
		parseSuccess = pVParser->Parse(source , NULL , &sobject->pUniformSamplerList);
	}

	if (!parseSuccess)
	{
		delete sobject;
		return HQ_FAILED;
	}

	sobject->shader = glCreateShader(shaderType);

	std::string processed_src = source;
	std::string version_string = "";
	/*------ Remove #version---------*/ 
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
					version_string.assign(source + pos1, pos3 - pos1 + 1); 
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
	this->GetPredefineMacroGLSL(macroDefList , pDefines, version_string.size() != 0);


	/*--------set shader source---------*/
	const GLchar* sourceArray[] = {
		version_string.c_str(),
		macroDefList.c_str(),
#ifdef GLES
		"#define HQEXT_GLSL_ES\n",
#else
		"#define HQEXT_GLSL\n",
#endif
		semanticKeywords,
		samplerKeywords,
		"#line 0 0\n",
		processed_src.c_str()
	};

	if (type != HQ_VERTEX_SHADER)
		sourceArray[2] = "";//only vertex shader need semantic definitions

	glShaderSource(sobject->shader, 7, (const GLchar**)sourceArray, NULL);
	glCompileShader(sobject->shader);

	GLint compileOK;
	glGetShaderiv(sobject->shader , GL_COMPILE_STATUS , &compileOK);
	if(sobject->shader==0 || compileOK == GL_FALSE)
	{
		int infologLength = 0;
	    int charsWritten  = 0;
	    char *infoLog;
	    glGetShaderiv(sobject->shader, GL_INFO_LOG_LENGTH,&infologLength);
	    if (infologLength > 0)
	    {
	        infoLog = (char *)malloc(infologLength);
	        glGetShaderInfoLog(sobject->shader, infologLength, &charsWritten, infoLog);
			g_pShaderMan->Log("GLSL shader compile error: %s",infoLog);
	        free(infoLog);
	    }
		delete sobject;
		return HQ_FAILED;
	}
	sobject->type=type;
	
#if 0//can't debug shader if use the following line
	//hide predefine macros
	glShaderSource(sobject->shader, 1 , (const GLchar**)&source, NULL);
#endif
	*ppShaderObjectOut = sobject;

	return HQ_OK;
}

HQReturnVal HQBaseGLSLShaderController::CreateProgramGLSL(
							  HQSharedPtr<HQShaderObjectGL>& pVShader,
							  HQSharedPtr<HQShaderObjectGL>& pGShader,
							  HQSharedPtr<HQShaderObjectGL>& pFShader,
							  const char** uniformParameterNames,
							  hq_uint32 *pID)
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

	HQBaseShaderProgramGL *pProgram=new HQShaderProgramGL_GLSL();

	pProgram->programGLHandle = glCreateProgram();

	if(pProgram->programGLHandle==0)
	{
		SafeDelete(pProgram);
		return HQ_FAILED;
	}

	if(flags & useV)
		glAttachShader(pProgram->programGLHandle, pVShader->shader);
	if(flags & useG)
		glAttachShader(pProgram->programGLHandle, pGShader->shader);
	if(flags & useF)
		glAttachShader(pProgram->programGLHandle, pFShader->shader);

	glLinkProgram(pProgram->programGLHandle);

	this->BindAttributeLocationGLSL(pProgram->programGLHandle , *pVShader->pAttribList);

	glLinkProgram(pProgram->programGLHandle);

	GLint OK;
	glGetProgramiv(pProgram->programGLHandle , GL_LINK_STATUS , &OK);

	if(OK == GL_FALSE)
	{
		int infologLength = 0;
	    int charsWritten  = 0;
	    char *infoLog;
	    glGetProgramiv(pProgram->programGLHandle, GL_INFO_LOG_LENGTH,&infologLength);
	    if (infologLength > 0)
	    {
	        infoLog = (char *)malloc(infologLength);
	        glGetProgramInfoLog(pProgram->programGLHandle, infologLength, &charsWritten, infoLog);
			g_pShaderMan->Log("GLSL program link error: %s", infoLog);
	        free(infoLog);
	    }
		SafeDelete(pProgram);
		return HQ_FAILED;
	}




	pProgram->isGLSL = true;

	//create paramters list
	if(uniformParameterNames!=NULL)
	{
		int i=0;
		while(uniformParameterNames[i]!=NULL)
		{
			pProgram->GetParameter(uniformParameterNames[i]);

			i++;
		}
	}

	hq_uint32 newProgramID;
	if(!g_pShaderMan->AddItem(pProgram,&newProgramID))
	{
		delete pProgram;
		return HQ_FAILED_MEM_ALLOC;
	}
	if (pID != NULL)
		*pID = newProgramID;

	//bind sampler unit
	hq_uint32 currentActiveProgram = g_pShaderMan->GetActiveProgram();
	g_pShaderMan->ActiveProgram(newProgramID);

	if(flags & useV)
		this->BindSamplerUnitGLSL(pProgram ,
								  *pVShader->pUniformSamplerList);
	if(flags & useG)
		this->BindSamplerUnitGLSL(pProgram ,
								  *pGShader->pUniformSamplerList);
	if(flags & useF)
		this->BindSamplerUnitGLSL(pProgram ,
								  *pFShader->pUniformSamplerList);
	//bind uniform block
	this->BindUniformBlockGLSL(pProgram->programGLHandle);

	g_pShaderMan->ActiveProgram(currentActiveProgram);

	//check if this program is valid
	glValidateProgram(pProgram->programGLHandle);
	glGetProgramiv(pProgram->programGLHandle, GL_VALIDATE_STATUS, &OK);
	if(OK == GL_FALSE)
	{
		int infologLength = 0;
		int charsWritten  = 0;
		char *infoLog;
		glGetProgramiv(pProgram->programGLHandle, GL_INFO_LOG_LENGTH,&infologLength);
		if (infologLength > 0)
		{
			infoLog = (char *)malloc(infologLength);
			glGetProgramInfoLog(pProgram->programGLHandle, infologLength, &charsWritten, infoLog);
			g_pShaderMan->Log("GLSL program validate error: %s", infoLog);
			free(infoLog);
		}

		g_pShaderMan->Remove(newProgramID);
		return HQ_FAILED;
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
#ifndef GLES
	if (GLEW_VERSION_3_1 || GLEW_ARB_uniform_buffer_object)
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

/*----------HQGLSLShaderController----------------*/

HQReturnVal HQGLSLShaderController::CreateShaderFromFile(HQShaderType type,
										const char* fileName,
										const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										bool isPreCompiled,
										const char* entryFunctionName,
										HQShaderObjectGL **ppShaderObjectOut)
{
	return this->CreateShaderFromFileGLSL(type,fileName,pDefines ,ppShaderObjectOut);
}

HQReturnVal HQGLSLShaderController::CreateShaderFromMemory(HQShaderType type,
										  const char* pSourceData,
										  const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										  bool isPreCompiled,
										  const char* entryFunctionName,
										  HQShaderObjectGL **ppShaderObjectOut)
{
	return this->CreateShaderFromMemoryGLSL(type , pSourceData,pDefines , ppShaderObjectOut);
}

HQReturnVal HQGLSLShaderController::CreateShaderFromFile(HQShaderType type,
								 HQShaderCompileMode compileMode,
								 const char* fileName,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 HQShaderObjectGL **ppShaderObjectOut)
{
	switch (compileMode)
	{
	case HQ_SCM_GLSL:case HQ_SCM_GLSL_DEBUG:
		return this->CreateShaderFromFileGLSL(type , fileName,pDefines,ppShaderObjectOut);

	default:
		return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}
}

HQReturnVal HQGLSLShaderController::CreateShaderFromMemory(HQShaderType type,
								 HQShaderCompileMode compileMode,
								 const char* pSourceData,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 HQShaderObjectGL **ppShaderObjectOut)
{
	switch (compileMode)
	{
	case HQ_SCM_GLSL:case HQ_SCM_GLSL_DEBUG:
		return this->CreateShaderFromMemoryGLSL(type , pSourceData,pDefines,ppShaderObjectOut);
	default:
		return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}

}

HQReturnVal HQGLSLShaderController::CreateProgram(	bool isGLSL ,
								hq_uint32 vertexShaderID,
								hq_uint32 pixelShaderID,
								hq_uint32 geometryShaderID,
								HQSharedPtr<HQShaderObjectGL>& pVShader,
								HQSharedPtr<HQShaderObjectGL>& pGShader,
								HQSharedPtr<HQShaderObjectGL>& pFShader,
								const char** uniformParameterNames,
								hq_uint32 *pID)
{
	hquint32 programID;
	HQReturnVal re = this->CreateProgramGLSL(pVShader , pGShader , pFShader , uniformParameterNames , &programID);
	
	if (pID != NULL)
		*pID = programID;

	//store shaders' IDs
	if (!HQFailed(re))
	{
		HQSharedPtr<HQBaseShaderProgramGL> pProgram = g_pShaderMan->GetItemPointer(programID);
		pProgram->vertexShaderID = vertexShaderID;
		pProgram->geometryShaderID = geometryShaderID;
		pProgram->pixelShaderID = pixelShaderID;
	}

	return re;
}

