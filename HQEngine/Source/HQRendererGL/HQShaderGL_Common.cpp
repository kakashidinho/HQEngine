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
#ifndef GLES
	this->program = NULL;
#else
	this->shader = 0;
#endif
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
		SafeDelete(this->pAttribList);
		SafeDelete(this->pUniformSamplerList);
	}
#ifndef GLES
	else if (this->program)
		cgDestroyProgram(this->program);
#endif
}



/*---------HQBaseShaderProgramGL--------------*/
HQBaseShaderProgramGL::HQBaseShaderProgramGL() 
{
#ifndef GLES
	program = NULL;
#endif
}
HQBaseShaderProgramGL::~HQBaseShaderProgramGL()
{
	if(!isGLSL)
	{
#ifndef GLES
		if(program != NULL)
			cgDestroyProgram(program);
#endif
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

hq_uint32 HQBaseShaderProgramGL::TryCreateParamObjAndAddToParamsList(const char *parameterName)
{
	HQShaderParameterGL* pNewParameter = this->TryCreateParameterObject(parameterName);
		
	if(pNewParameter == NULL)//không tìm thấy
	{
		return  NOT_AVAIL_ID;
	}
	
	hq_uint32 paramIndex;
	if(!this->parameters.AddItem(pNewParameter , &paramIndex))
	{
		delete pNewParameter;
		return NOT_AVAIL_ID;
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
	activeProgram=NOT_USE_SHADER;

	g_pShaderMan = this;
}
HQBaseCommonShaderManagerGL::~HQBaseCommonShaderManagerGL()
{

	g_pShaderMan = NULL;
}


/*------------------------*/

bool HQBaseCommonShaderManagerGL::IsUsingVShader() //có đang dùng vertex shader không,hay đang dùng fixed function
{
	HQSharedPtr<HQBaseShaderProgramGL> pProgram = this->GetItemPointerNonCheck(activeProgram);
	if (pProgram == NULL)
		return false;

	return pProgram->vertexShaderID != NOT_USE_VSHADER;
}
bool HQBaseCommonShaderManagerGL::IsUsingGShader()//có đang dùng geometry shader không,hay đang dùng fixed function
{
	HQSharedPtr<HQBaseShaderProgramGL> pProgram = this->GetItemPointerNonCheck(activeProgram);
	if (pProgram == NULL)
		return false;

	return pProgram->geometryShaderID != NOT_USE_GSHADER;
}
bool HQBaseCommonShaderManagerGL::IsUsingPShader() //có đang dùng pixel/fragment shader không,hay đang dùng fixed function
{
	HQSharedPtr<HQBaseShaderProgramGL> pProgram = this->GetItemPointerNonCheck(activeProgram);
	if (pProgram == NULL)
		return false;

	return pProgram->pixelShaderID != NOT_USE_PSHADER;
}


/*------------------------*/

HQReturnVal HQBaseCommonShaderManagerGL::DestroyProgram(hq_uint32 programID)
{
	HQSharedPtr<HQBaseShaderProgramGL> pProgram = this->GetItemPointer(programID);
	if(pProgram == NULL)
		return HQ_FAILED;
	if(programID==activeProgram)
	{
#if defined DEVICE_LOST_POSSIBLE
		if (!g_pOGLDev->IsDeviceLost())//must not call opengl when device is lost
#endif
			this->ActiveProgram(NOT_USE_SHADER);
		
		this->activeProgram = NOT_USE_SHADER;
	}
	this->Remove(programID);
	return HQ_OK;
}

void HQBaseCommonShaderManagerGL::DestroyAllProgram()
{
#if defined DEVICE_LOST_POSSIBLE
	if (!g_pOGLDev->IsDeviceLost())//must not call opengl when device is lost
#endif
		this->ActiveProgram(NOT_USE_SHADER);

	this->RemoveAll();
	this->activeProgram = NOT_USE_SHADER;
}

HQReturnVal HQBaseCommonShaderManagerGL::DestroyShader(hq_uint32 shaderID)
{
	return (HQReturnVal)this->shaderObjects.Remove(shaderID);
}

void HQBaseCommonShaderManagerGL::DestroyAllShader()
{
	this->shaderObjects.RemoveAll();
}

void HQBaseCommonShaderManagerGL::DestroyAllResource()
{
#ifndef GLES
	DestroyAllUniformBuffers();
#endif
	DestroyAllProgram();
	DestroyAllShader();
}

hq_uint32 HQBaseCommonShaderManagerGL::GetShader(hq_uint32 programID, HQShaderType shaderType)
{
	HQSharedPtr<HQBaseShaderProgramGL> pProgram = this->GetItemPointer(programID);

	switch (shaderType)
	{
	case HQ_VERTEX_SHADER:
		return (pProgram != NULL)? pProgram->vertexShaderID : NOT_USE_VSHADER;
	case HQ_GEOMETRY_SHADER:
		return (pProgram != NULL)? pProgram->geometryShaderID : NOT_USE_GSHADER;
	case HQ_PIXEL_SHADER:
		return (pProgram != NULL)? pProgram->pixelShaderID : NOT_USE_PSHADER;
	}

	return NOT_USE_VSHADER;
}



/*--------------------------------------*/


hq_uint32 HQBaseCommonShaderManagerGL::GetParameterIndex(hq_uint32 programID ,
											const char *parameterName)
{
	HQSharedPtr<HQBaseShaderProgramGL> pProgram	= this->GetItemPointer(programID);
	if(pProgram == NULL)
		return NOT_AVAIL_ID;
	
	return pProgram->GetParameterIndex(parameterName);
}
/*--------------------------------------*/

#ifndef GLES

HQReturnVal HQBaseCommonShaderManagerGL::CreateUniformBuffer(hq_uint32 size , void *initData , bool isDynamic , hq_uint32 *pBufferIDOut)
{
	return HQ_FAILED;
}
HQReturnVal HQBaseCommonShaderManagerGL::DestroyUniformBuffer(hq_uint32 bufferID)
{
	return HQ_FAILED;
}
void HQBaseCommonShaderManagerGL::DestroyAllUniformBuffers()
{
}
HQReturnVal HQBaseCommonShaderManagerGL::SetUniformBuffer(hq_uint32 slot ,  hq_uint32 bufferID )
{
	return HQ_FAILED;
}
HQReturnVal HQBaseCommonShaderManagerGL::MapUniformBuffer(hq_uint32 bufferID , void **ppData)
{
	return HQ_FAILED;
}
HQReturnVal HQBaseCommonShaderManagerGL::UnmapUniformBuffer(hq_uint32 bufferID)
{
	return HQ_FAILED;
}
HQReturnVal HQBaseCommonShaderManagerGL::UpdateUniformBuffer(hq_uint32 bufferID, const void * pData)
{
	return HQ_FAILED;
}
#endif
