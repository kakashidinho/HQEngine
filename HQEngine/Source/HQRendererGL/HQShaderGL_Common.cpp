/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "HQShaderGL_Common.h"
#if defined DEVICE_LOST_POSSIBLE
#include "HQDeviceGL.h"
#endif


HQBaseCommonShaderManagerGL* g_pShaderMan = NULL;

/*---------HQShaderObjectGL---------------*/
HQShaderObjectGL::HQShaderObjectGL()
{
	this->isGLSL = false;
	this->shader = 0;
	this->pUniformBlocks = NULL;
	this->pAttribList = NULL;
	this->pUniformSamplerList = NULL;
}
HQShaderObjectGL::~HQShaderObjectGL()
{
	if(isGLSL)
	{
		if (this->shader != 0)
		{
#if defined DEVICE_LOST_POSSIBLE
			if (!g_pOGLDev->IsDeviceLost())
#endif
				glDeleteShader(this->shader);
		}
		SafeDelete(this->pUniformBlocks);
		SafeDelete(this->pAttribList);
		SafeDelete(this->pUniformSamplerList);
	}

}



/*---------HQBaseShaderProgramGL--------------*/
HQBaseShaderProgramGL::HQBaseShaderProgramGL() 
{
}
HQBaseShaderProgramGL::~HQBaseShaderProgramGL()
{
	if(!isGLSL)
	{
		//legacy code was here. we used Cg before.
	}
	else
	{	
		if(programGLHandle != 0)
		{
#if defined DEVICE_LOST_POSSIBLE
			if (!g_pOGLDev->IsDeviceLost())
#endif
				glDeleteProgram(programGLHandle);
		}
	}
}

HQShaderObject * HQBaseShaderProgramGL::GetShader(HQShaderType type)
{
	switch (type)
	{
	case HQ_VERTEX_SHADER:
		return vertexShader;
	case HQ_PIXEL_SHADER:
		return pixelShader;
	case HQ_GEOMETRY_SHADER:
		return geometryShader;
	}

	return NULL;
}

HQShaderParameterGL* HQBaseShaderProgramGL::TryCreateParameterObject(const char *parameterName)
{
	if (isGLSL)
	{
		GLint paramGLLoc = glGetUniformLocation(this->programGLHandle, parameterName);//get parameter handle

		if (paramGLLoc == -1)//không tìm thấy
		{
			return  NULL;
		}
		HQShaderParameterGL* pNewParameter = new HQShaderParameterGL();
		pNewParameter->texUnit = -1;
		pNewParameter->location = paramGLLoc;


		return pNewParameter;
	}
	else
		return NULL;
}

hq_uint32 HQBaseShaderProgramGL::TryCreateParamObjAndAddToParamsList(const char *parameterName)
{
	HQShaderParameterGL* pNewParameter = this->TryCreateParameterObject(parameterName);
		
	if(pNewParameter == NULL)//không tìm thấy
	{
		return  HQ_NOT_AVAIL_ID;
	}
	
	hq_uint32 paramIndex;
	if(!this->parameters.AddItem(pNewParameter , &paramIndex))
	{
		delete pNewParameter;
		return HQ_NOT_AVAIL_ID;
	}
	else
	{
		this->parameterIndexes.Add(parameterName , paramIndex );
	}
	return paramIndex;
}


/*------------HQBaseCommonShaderManagerGL--------------------*/

HQBaseCommonShaderManagerGL::HQBaseCommonShaderManagerGL(HQLogStream* logFileStream , const char * logPrefix , bool flushLog)
:HQLoggableObject(logFileStream , logPrefix , flushLog , 1024) 
{
	activeProgram=NULL;

	g_pShaderMan = this;
}
HQBaseCommonShaderManagerGL::~HQBaseCommonShaderManagerGL()
{

	g_pShaderMan = NULL;
}


/*------------------------*/

bool HQBaseCommonShaderManagerGL::IsUsingVShader() //có đang dùng vertex shader không,hay đang dùng fixed function
{
	if (activeProgram == NULL)
		return false;

	return activeProgram->vertexShader != NULL;
}
bool HQBaseCommonShaderManagerGL::IsUsingGShader()//có đang dùng geometry shader không,hay đang dùng fixed function
{
	if (activeProgram == NULL)
		return false;

	return activeProgram->geometryShader != NULL;
}
bool HQBaseCommonShaderManagerGL::IsUsingPShader() //có đang dùng pixel/fragment shader không,hay đang dùng fixed function
{
	if (activeProgram == NULL)
		return false;

	return activeProgram->pixelShader != NULL;
}


/*------------------------*/

HQReturnVal HQBaseCommonShaderManagerGL::DestroyProgram(HQShaderProgram* programID)
{
	HQSharedPtr<HQBaseShaderProgramGL> pProgram = this->GetItemPointer(programID);
	if(pProgram == NULL)
		return HQ_FAILED;
	if(programID==activeProgram.GetRawPointer())
	{
#if defined DEVICE_LOST_POSSIBLE
		if (!g_pOGLDev->IsDeviceLost())//must not call opengl when device is lost
#endif
			this->ActiveProgram(NULL);
		
		this->activeProgram = NULL;
	}
	this->Remove(programID);
	return HQ_OK;
}

void HQBaseCommonShaderManagerGL::DestroyAllProgram()
{
#if defined DEVICE_LOST_POSSIBLE
	if (!g_pOGLDev->IsDeviceLost())//must not call opengl when device is lost
#endif
		this->ActiveProgram(NULL);

	this->RemoveAll();
	this->activeProgram = NULL;
}

HQReturnVal HQBaseCommonShaderManagerGL::DestroyShader(HQShaderObject* shaderID)
{
	return (HQReturnVal)this->shaderObjects.Remove(shaderID);
}

void HQBaseCommonShaderManagerGL::DestroyAllShader()
{
	this->shaderObjects.RemoveAll();
}

void HQBaseCommonShaderManagerGL::DestroyAllResource()
{
#ifndef HQ_OPENGLES
	DestroyAllUniformBuffers();
#endif
	DestroyAllProgram();
	DestroyAllShader();
}




/*--------------------------------------*/


hq_uint32 HQBaseCommonShaderManagerGL::GetParameterIndex(HQShaderProgram* programID ,
											const char *parameterName)
{
	HQSharedPtr<HQBaseShaderProgramGL> pProgram	= this->GetItemPointer(programID);
	if(pProgram == NULL)
		return HQ_NOT_AVAIL_ID;
	
	return pProgram->GetParameterIndex(parameterName);
}
/*--------------------------------------*/
