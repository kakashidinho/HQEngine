#include "HQDeviceD3D9PCH.h"
#include "HQShaderD3D9.h"
#include <string.h>

char *semantics[17]= {
	"-DVPOSITION=POSITION",
	"-DVCOLOR=COLOR",
	"-DVNORMAL=NORMAL",
	"-DVTEXCOORD0=TEXCOORD0",
	"-DVTEXCOORD1=TEXCOORD1",
	"-DVTEXCOORD2=TEXCOORD2",
	"-DVTEXCOORD3=TEXCOORD3",
	"-DVTEXCOORD4=TEXCOORD4",
	"-DVTEXCOORD5=TEXCOORD5",
	"-DVTEXCOORD6=TEXCOORD6",
	"-DVTEXCOORD7=TEXCOORD7",
	"-DVTANGENT=TANGENT",
	"-DVBINORMAL=BINORMAL",
	"-DVBLENDWEIGHT=BLENDWEIGHT",
	"-DVBLENDINDICES=BLENDINDICES",
	"-DVPSIZE=PSIZE",
	NULL
};

HQShaderManagerD3D9* pShaderMan=NULL;
void cgErrorCallBack(void)
{
	CGerror err=cgGetError();
	if(err == cgD3D9Failed)
	{
		HRESULT hr=cgD3D9GetLastError();
		pShaderMan->Log("D3D9 error %s",cgD3D9TranslateHRESULT(hr));
	}
	else
		pShaderMan->Log("%s",cgGetErrorString(err));
}

/*---------HQShaderManagerD3D9--------------*/
HQShaderManagerD3D9::HQShaderManagerD3D9(LPDIRECT3DDEVICE9 pD3DDevice,HQLogStream* logFileStream,bool flushLog)
: HQLoggableObject(logFileStream , "D3D9 Shader Manager :" , flushLog , 1024)
#if defined _DEBUG || defined DEBUG
	,activeProgramID(NOT_USE_SHADER)
#endif
{
	this->pD3DDevice=pD3DDevice;

	this->cgContext=cgCreateContext();
	
	cgD3D9SetDevice(pD3DDevice);

	this->cgVertexProfile=cgD3D9GetLatestVertexProfile();
	this->cgPixelProfile=cgD3D9GetLatestPixelProfile();
	
	//fixed function controlling values
	this->firstStageOp[0] = D3DTOP_MODULATE;
	this->firstStageOp[1] = D3DTOP_SELECTARG1;

	pShaderMan=this;
#if defined(DEBUG)||defined(_DEBUG)
	cgSetErrorCallback(cgErrorCallBack);
#endif
	Log("Init done!");
}

HQShaderManagerD3D9::~HQShaderManagerD3D9()
{
	this->DestroyAllResource();
	cgD3D9SetDevice(0);
	cgDestroyContext(cgContext);
	Log("Released!");
	pShaderMan=NULL;
}

void HQShaderManagerD3D9::OnResetDevice()
{
	if (activeVShader != NULL)
		cgD3D9BindProgram(activeVShader->program);
	if (activePShader != NULL)
		cgD3D9BindProgram(activePShader->program);

}

/*------------------------*/

bool HQShaderManagerD3D9::IsUsingVShader() //có đang dùng vertex shader không,hay đang dùng fixed function
{
	if (activeProgram == NULL)
		return false;

	return activeProgram->vertexShader != NULL;
}
bool HQShaderManagerD3D9::IsUsingGShader()//có đang dùng geometry shader không,hay đang dùng fixed function
{
	return false;
}
bool HQShaderManagerD3D9::IsUsingPShader() //có đang dùng pixel/fragment shader không,hay đang dùng fixed function
{
	if (activeProgram == NULL)
		return false;

	return activeProgram->pixelShader != NULL;
}
/*------------------------*/
HQReturnVal HQShaderManagerD3D9::ActiveProgram(hq_uint32 programID)
{
	if (programID == NOT_USE_SHADER)
	{
		if (activeProgram == NULL)
			return HQ_OK;
		if(activeVShader!=NULL)
		{
			cgD3D9UnbindProgram(activeVShader->program);
		}
		if(activePShader!=NULL)
		{
			cgD3D9UnbindProgram(activePShader->program);
		}
		activeProgram=HQSharedPtr<HQShaderProgramD3D9>::null;

		activeVShader=HQSharedPtr<HQShaderObjectD3D9>::null;
		activePShader=HQSharedPtr<HQShaderObjectD3D9>::null;

#if defined _DEBUG || defined DEBUG
		activeProgramID = NOT_USE_SHADER;
#endif
		
		return HQ_OK;
	}
	else{
		HQSharedPtr<HQShaderProgramD3D9> pProgram = this->GetItemPointer(programID);

		if( pProgram == activeProgram)
			return HQ_OK;
#if defined _DEBUG || defined DEBUG
		if (pProgram == NULL)
			return HQ_FAILED_INVALID_ID;
#endif

		if(activeVShader!=pProgram->vertexShader)//vertex shader object đang active khác vertex shader object trong program
		{
			if(activeVShader==NULL)//đang dùng fixed function vertex processor
			{
				activeVShader=pProgram->vertexShader;
				cgD3D9BindProgram(activeVShader->program);
			}
			else if(pProgram->vertexShader==NULL)//đang dùng shader nhưng program chuẩn bị active lại dùng fixed function vertex processor
			{
				cgD3D9UnbindProgram(activeVShader->program);
				activeVShader=HQSharedPtr<HQShaderObjectD3D9>::null;
			}
			else//đang dùng vertex shader khác vertex shader trong program chuẩn bị active 
			{
				activeVShader=pProgram->vertexShader;
				cgD3D9BindProgram(activeVShader->program);
			}
		}
		if(activePShader!=pProgram->pixelShader)//pixel shader object đang active khác pixel shader object trong program
		{
			if(activePShader==NULL)//đang dùng fixed function pixel processor
			{
				activePShader=pProgram->pixelShader;
				cgD3D9BindProgram(activePShader->program);
			}
			else if(pProgram->pixelShader == NULL)//đang dùng shader nhưng program chuẩn bị active lại dùng fixed function pixel processor
			{
				cgD3D9UnbindProgram(activePShader->program);
				activePShader=HQSharedPtr<HQShaderObjectD3D9>::null;
			}
			else//đang dùng pixel shader khác pixel shader trong program chuẩn bị active 
			{
				activePShader=pProgram->pixelShader;
				cgD3D9BindProgram(activePShader->program);
			}
		}
		activeProgram=pProgram;
#if defined _DEBUG || defined DEBUG
		activeProgramID = programID;
#endif
	}
	return HQ_OK;
}
/*------------------------*/
HQReturnVal HQShaderManagerD3D9::DestroyProgram(hq_uint32 programID)
{
	HQSharedPtr<HQShaderProgramD3D9> pProgram = this->GetItemPointer(programID);
	if(pProgram == NULL)
		return HQ_FAILED;
	if(pProgram==activeProgram)
	{
		this->ActiveProgram(NOT_USE_SHADER);
	}
	this->Remove(programID);
	return HQ_OK;
}

void HQShaderManagerD3D9::DestroyAllProgram()
{
	this->ActiveProgram(NOT_USE_SHADER);
	
	this->RemoveAll();
}

void HQShaderManagerD3D9::DestroyAllResource()
{
	this->DestroyAllUniformBuffers();
	this->DestroyAllProgram();
	this->DestroyAllShader();
}


HQReturnVal HQShaderManagerD3D9::DestroyShader(hq_uint32 shaderID)
{
	return (HQReturnVal)this->shaderObjects.Remove(shaderID);
}

void HQShaderManagerD3D9::DestroyAllShader()
{
	this->shaderObjects.RemoveAll();
}
/*--------------------------*/

char ** HQShaderManagerD3D9::GetPredefineMacroArguments(const HQShaderMacro * pDefines)
{
	if(pDefines == NULL)
		return semantics;
	int numDefines = 0;
	int nameLen = 0;
	int definitionLen = 0;
	const HQShaderMacro *pD = pDefines;
	
	//calculate number of macros
	while (pD->name != NULL && pD->definition != NULL)
	{
		numDefines++;
		pD++;
	}
	if(numDefines == 0)
		return semantics;
	/*------create arguments---------*/
	char ** args = new char *[numDefines + 17];
	for (int i = 0 ; i < 16 ; ++i)
		args[i] = semantics[i];

	args[numDefines + 16] = NULL;

	pD = pDefines;
	
	int i = 0;
	while (pD->name != NULL && pD->definition != NULL)
	{
		nameLen = strlen(pD->name);
		definitionLen = strlen(pD->definition);
		
	
		if (definitionLen != 0)
		{
			args[16 + i] = new char[nameLen + definitionLen + 4];
			sprintf(args[16 + i] , "-D%s=%s" , pD->name , pD->definition);
		}
		else
		{
			args[16 + i] = new char[nameLen + 4];
			sprintf(args[16 + i] , "-D%s" , pD->name);
		}
		pD++;
		i++;
	}

	return args;
}

void HQShaderManagerD3D9::DeAlloc(char **ppC)
{
	if(ppC == NULL || ppC == semantics)
		return;
	char ** ppD = ppC + 16;
	while(*ppD != NULL)
	{
		SafeDeleteArray(*ppD);
		ppD ++;
	}

	SafeDeleteArray(ppC);
}


HQReturnVal HQShaderManagerD3D9::CreateShaderFromFileEx(HQShaderType type,
									 const char* fileName,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 const char **args,
									 bool debugMode , 
									 hq_uint32 *pID)
{
	CGprofile profile;
	if(type==HQ_VERTEX_SHADER)
		profile=this->cgVertexProfile;
	else if (type==HQ_PIXEL_SHADER)
		profile=this->cgPixelProfile;
	else
	{
		return HQ_FAILED;
	}
	HQShaderObjectD3D9 *sobject= new HQShaderObjectD3D9();
	sobject->program=cgCreateProgramFromFile(this->cgContext,isPreCompiled? CG_OBJECT : CG_SOURCE ,
												 fileName,profile,entryFunctionName,args);
	if(sobject->program==NULL)
	{
		delete sobject;
		return HQ_FAILED;
	}
	sobject->type=type;
	
	if (cgIsProgramCompiled(sobject->program) == CG_FALSE)
		cgCompileProgram(sobject->program);
	
	
	if(debugMode)
		cgD3D9LoadProgram(sobject->program,false,D3DXSHADER_DEBUG);
	else
		cgD3D9LoadProgram(sobject->program,false,0);

	if (!this->shaderObjects.AddItem(sobject , pID))
	{
		delete sobject;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::CreateShaderFromMemoryEx(HQShaderType type,
									 const char* pSourceData,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 const char **args,
									 bool debugMode,
									 hq_uint32 *pID)
{
	

	CGprofile profile;
	if(type==HQ_VERTEX_SHADER)
		profile=this->cgVertexProfile;
	else if (type==HQ_PIXEL_SHADER)
		profile=this->cgPixelProfile;
	else
	{
		return HQ_FAILED;
	}
	HQShaderObjectD3D9 *sobject= new HQShaderObjectD3D9();
	sobject->program=cgCreateProgram(this->cgContext,isPreCompiled? CG_OBJECT : CG_SOURCE ,
												 pSourceData,profile,entryFunctionName,args);
	if(sobject->program==NULL)
	{
		delete sobject;
		return HQ_FAILED;
	}
	sobject->type=type;

	if (cgIsProgramCompiled(sobject->program) == CG_FALSE)
		cgCompileProgram(sobject->program);

	if(debugMode)
		cgD3D9LoadProgram(sobject->program,false,D3DXSHADER_DEBUG);
	else
		cgD3D9LoadProgram(sobject->program,false,0);

	if (!this->shaderObjects.AddItem(sobject , pID))
	{
		delete sobject;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}
HQReturnVal HQShaderManagerD3D9::CreateShaderFromFile(HQShaderType type,
										const char* fileName,
										const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										bool isPreCompiled,
										const char* entryFunctionName,
										hq_uint32 *pID)
{
	char ** args = this->GetPredefineMacroArguments(pDefines);
	HQReturnVal re = this->CreateShaderFromFileEx(type,fileName,isPreCompiled , entryFunctionName ,(const char**)args ,false , pID);
	this->DeAlloc(args);
	return re;
}

HQReturnVal HQShaderManagerD3D9::CreateShaderFromMemory(HQShaderType type,
										  const char* pSourceData,
										  const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										  bool isPreCompiled,
										  const char* entryFunctionName,
										  hq_uint32 *pID)
{
	char ** args = this->GetPredefineMacroArguments(pDefines);
	HQReturnVal re =  this->CreateShaderFromMemoryEx(type , pSourceData,isPreCompiled,entryFunctionName,(const char**)args , false, pID);
	this->DeAlloc(args);
	return re;
}

HQReturnVal HQShaderManagerD3D9::CreateShaderFromFile(HQShaderType type,
								 HQShaderCompileMode compileMode,
								 const char* fileName,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 hq_uint32 *pID)
{
	HQReturnVal re;
	char ** args = this->GetPredefineMacroArguments(pDefines);
	switch (compileMode)
	{
	case HQ_SCM_CG:
		re = this->CreateShaderFromFileEx(type , fileName,false , entryFunctionName, (const char**)args , false,pID);
		break;
	case HQ_SCM_CG_DEBUG:
		re = this->CreateShaderFromFileEx(type , fileName,false , entryFunctionName, (const char**)args , true,pID);
		break;
	default:
		re = HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}
	this->DeAlloc(args);
	return re;
}

HQReturnVal HQShaderManagerD3D9::CreateShaderFromMemory(HQShaderType type,
								 HQShaderCompileMode compileMode,
								 const char* pSourceData,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 hq_uint32 *pID)
{
	HQReturnVal re;
	char ** args = this->GetPredefineMacroArguments(pDefines);
	switch (compileMode)
	{
	case HQ_SCM_CG:
		re = this->CreateShaderFromMemoryEx(type , pSourceData,false , entryFunctionName, (const char**)args ,false,pID);
	case HQ_SCM_CG_DEBUG:
		re = this->CreateShaderFromMemoryEx(type , pSourceData,false , entryFunctionName, (const char**)args , true,pID);
	default:
		re = HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}
	this->DeAlloc(args);
	return re;
}


HQReturnVal HQShaderManagerD3D9::CreateProgram(hq_uint32 vertexShaderID,
							  hq_uint32 pixelShaderID,
							  hq_uint32 geometryShaderID,
							  const char** uniformParameterNames,
							  hq_uint32 *pID)
{
	if(vertexShaderID==NOT_USE_VSHADER && pixelShaderID==NOT_USE_PSHADER && geometryShaderID==NOT_USE_GSHADER)
		return HQ_FAILED_SHADER_PROGRAM_NEED_SHADEROBJECT;//tất cả đều là fixed function => báo lỗi
	
	HQSharedPtr<HQShaderObjectD3D9> pVShader = HQSharedPtr<HQShaderObjectD3D9> :: null;
	HQSharedPtr<HQShaderObjectD3D9> pPShader = HQSharedPtr<HQShaderObjectD3D9> :: null;
	if (vertexShaderID != NOT_USE_VSHADER)
		pVShader = this->shaderObjects.GetItemPointer(vertexShaderID);
	if (pixelShaderID != NOT_USE_PSHADER)
		pPShader = this->shaderObjects.GetItemPointer(pixelShaderID);
	
	if(pVShader == NULL || pPShader == NULL)
		return HQ_FAILED_INVALID_ID;
	if(vertexShaderID!=NOT_USE_VSHADER && pVShader->type!=HQ_VERTEX_SHADER)//shader có id <vertexShaderID> không phải vertex shader
		return HQ_FAILED_WRONG_SHADER_TYPE;
	if(pixelShaderID!=NOT_USE_PSHADER && pPShader->type!=HQ_PIXEL_SHADER)//shader có id <pixelShaderID> không phải pixel shader
		return HQ_FAILED_WRONG_SHADER_TYPE;

	HQShaderProgramD3D9 *pShader=new HQShaderProgramD3D9();

	//store shaders' pointers
	pShader->vertexShader = pVShader;
	pShader->pixelShader = pPShader;

	//store shaders' IDs
	pShader->vertexShaderID = (pVShader != NULL)? vertexShaderID : NOT_USE_VSHADER;
	pShader->pixelShaderID = (pPShader != NULL)? pixelShaderID : NOT_USE_PSHADER;
	
	hq_uint32 newProgramID;
	if(!this->AddItem(pShader,&newProgramID))
	{
		delete pShader;
		return HQ_FAILED_MEM_ALLOC;
	}
	//create paramters list
	if(uniformParameterNames!=NULL)
	{
		int i=0;
		while(uniformParameterNames[i]!=NULL)
		{
			GetUniformParam(this->GetItemPointerNonCheck(newProgramID) , 
				uniformParameterNames[i]);

			i++;
		}
	}
	if(pID != NULL)
		*pID = newProgramID;
	return HQ_OK;
}

hq_uint32 HQShaderManagerD3D9::GetShader(hq_uint32 programID, HQShaderType shaderType)
{
	HQSharedPtr<HQShaderProgramD3D9> pProgram = this->GetItemPointer(programID);

	switch (shaderType)
	{
	case HQ_VERTEX_SHADER:
		return (pProgram != NULL)? pProgram->vertexShaderID : NOT_USE_VSHADER;
	case HQ_PIXEL_SHADER:
		return (pProgram != NULL)? pProgram->pixelShaderID : NOT_USE_PSHADER;
	}

	return NOT_USE_GSHADER;
}

/*-----------------------*/
hq_uint32 HQShaderManagerD3D9::GetParameterIndex(HQSharedPtr<HQShaderProgramD3D9> &pProgram , 
											const char *parameterName)
{
	hq_uint32 *pIndex = pProgram->parameterIndexes.GetItemPointer(parameterName);
	if (pIndex == NULL)//không có
	{
		CGparameter param = NULL;
		//cố tìm trong vertex shader
		if(pProgram->vertexShader!=NULL)
			param=cgGetNamedParameter(pProgram->vertexShader->program,parameterName);//get parameter handle
			
		if(param==NULL && (pProgram->pixelShader!=NULL))//nếu không tìm thấy,tìm trong pixel shader
			param=cgGetNamedParameter(pProgram->pixelShader->program,parameterName);
		if(param==NULL)//vẫn không tìm thấy
		{
			return NOT_AVAIL_ID;
		}
		//đã tìm thấy =>thêm vào danh sách
		HQParameterD3D9* pNewParameter = new HQParameterD3D9();
		pNewParameter->parameter=param;
		pNewParameter->type=cgGetParameterType(param);//get parameter type
	
		hq_uint32 paramIndex;
		if(!pProgram->parameters.AddItem(pNewParameter , &paramIndex))
		{
			delete pNewParameter;
			return NOT_AVAIL_ID;
		}
		else
		{
			pProgram->parameterIndexes.Add(parameterName , paramIndex);
		}
		return paramIndex;
	}
	return *pIndex;
}
inline CGparameter HQShaderManagerD3D9::GetUniformParam(HQSharedPtr<HQShaderProgramD3D9>& pProgram,const char* parameterName)
{
	hq_uint32 paramIndex = this->GetParameterIndex(pProgram , parameterName);

	if(paramIndex == NOT_AVAIL_ID)
		return NULL;

	return pProgram->parameters.GetItemPointerNonCheck(paramIndex)->parameter;
}

/*-----------------------*/

HQReturnVal HQShaderManagerD3D9::SetUniformInt(const char* parameterName,
					 const int* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	

	CGparameter param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValueir(param,(int)numElements,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniform2Int(const char* parameterName,
					 const int* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	

	CGparameter param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValueir(param,(int)numElements * 2,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniform3Int(const char* parameterName,
					 const int* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	

	CGparameter param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValueir(param,(int)numElements * 3,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniform4Int(const char* parameterName,
					 const int* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	

	CGparameter param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValueir(param,(int)numElements * 4,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniformFloat(const char* parameterName,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	

	CGparameter param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValuefr(param,(int)numElements,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniform2Float(const char* parameterName,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	

	CGparameter param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValuefr(param,(int)numElements * 2,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniform3Float(const char* parameterName,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	

	CGparameter param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValuefr(param,(int)numElements * 3,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniform4Float(const char* parameterName,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	

	CGparameter param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValuefr(param,(int)numElements * 4,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniformMatrix(const char* parameterName,
					 const HQBaseMatrix4* pMatrices,
					 hq_uint32 numMatrices)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	

	CGparameter param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	HRESULT hr=cgD3D9SetUniformMatrixArray(param,0,numMatrices,(D3DMATRIX*)pMatrices);
	if(FAILED(hr))
	{
		//Log("%s",cgD3D9TranslateHRESULT(hr));
		return HQ_FAILED;
	}

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniformMatrix(const char* parameterName,
					 const HQBaseMatrix3x4* pMatrices,
					 hq_uint32 numMatrices)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	

	CGparameter param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	HRESULT hr=cgD3D9SetUniformArray(param,0,numMatrices,pMatrices);
	if(FAILED(hr))
	{
		//Log("%s",cgD3D9TranslateHRESULT(hr));
		return HQ_FAILED;
	}

	return HQ_OK;
}

/*-------parameter index version---------------*/

#if defined _DEBUG || defined DEBUG
#define GETPARAM(paramIndex) (activeProgram->parameters.GetItemRawPointer(paramIndex))
#else
#define GETPARAM(paramIndex) (activeProgram->parameters.GetItemRawPointerNonCheck(paramIndex))
#endif

hq_uint32 HQShaderManagerD3D9::GetParameterIndex(hq_uint32 programID , 
											const char *parameterName)
{
	HQSharedPtr<HQShaderProgramD3D9> pProgram	= this->GetItemPointer(programID);
	if(pProgram == NULL)
		return NOT_AVAIL_ID;
	return this->GetParameterIndex(pProgram , parameterName);
}

HQReturnVal HQShaderManagerD3D9::SetUniformInt(hq_uint32 parameterIndex,
					 const int* pValues,
					 hq_uint32 numElements)
{
	/*-------fixed function shader---------*/
	if(activeProgram==NULL)
		return this->SetRenderState(parameterIndex , pValues);

	
	
	/*-----programmable shader-------------*/
	HQParameterD3D9* param = GETPARAM(parameterIndex);
#if defined _DEBUG || defined DEBUG	
	if(param == NULL)
	{
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValueir(param->parameter,(int)numElements,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniform2Int(hq_uint32 parameterIndex,
					 const int* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	
	
	HQParameterD3D9* param = GETPARAM(parameterIndex);
#if defined _DEBUG || defined DEBUG	
	if(param == NULL)
	{
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValueir(param->parameter,(int)numElements * 2,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniform3Int(hq_uint32 parameterIndex,
					 const int* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	
	
	HQParameterD3D9* param = GETPARAM(parameterIndex);
#if defined _DEBUG || defined DEBUG	
	if(param == NULL)
	{
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValueir(param->parameter,(int)numElements * 3,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniform4Int(hq_uint32 parameterIndex,
					 const int* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	
	
	HQParameterD3D9* param = GETPARAM(parameterIndex);
#if defined _DEBUG || defined DEBUG	
	if(param == NULL)
	{
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValueir(param->parameter,(int)numElements * 4,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniformFloat(hq_uint32 parameterIndex,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	
	
	HQParameterD3D9* param = GETPARAM(parameterIndex);
#if defined _DEBUG || defined DEBUG	
	if(param == NULL)
	{
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValuefr(param->parameter,(int)numElements,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniform2Float(hq_uint32 parameterIndex,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	
	
	HQParameterD3D9* param = GETPARAM(parameterIndex);
#if defined _DEBUG || defined DEBUG	
	if(param == NULL)
	{
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValuefr(param->parameter,(int)numElements * 2,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniform3Float(hq_uint32 parameterIndex,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	
	
	HQParameterD3D9* param = GETPARAM(parameterIndex);
#if defined _DEBUG || defined DEBUG	
	if(param == NULL)
	{
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValuefr(param->parameter,(int)numElements * 3,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniform4Float(hq_uint32 parameterIndex,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	
	
	HQParameterD3D9* param = GETPARAM(parameterIndex);
#if defined _DEBUG || defined DEBUG	
	if(param == NULL)
	{
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	cgSetParameterValuefr(param->parameter,(int)numElements * 4,pValues);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniformMatrix(hq_uint32 parameterIndex,
					 const HQBaseMatrix4* pMatrices,
					 hq_uint32 numMatrices)
{
	//fixed function shader
	if(activeProgram==NULL)
	{
		HRESULT hr = pD3DDevice->SetTransform((D3DTRANSFORMSTATETYPE)parameterIndex , (const D3DMATRIX*) pMatrices);
#if defined _DEBUG || defined DEBUG
		if (FAILED(hr))
			return HQ_FAILED;
#endif
		return HQ_OK;
	}
	
	/*------programmable shader---------*/
	HQParameterD3D9* param = GETPARAM(parameterIndex);
#if defined _DEBUG || defined DEBUG	
	if(param == NULL)
	{
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	//cgSetParameterValuefr(param->parameter,(int)numMatrices * 16,(hq_float32*)pMatrices);

	
	HRESULT hr =cgD3D9SetUniformMatrixArray(param->parameter,0,numMatrices,(D3DMATRIX*)pMatrices);
#if defined _DEBUG || defined DEBUG
	if(FAILED(hr))
	{
		this->Log("Error : SetUniformMatrix() failed . %s",cgD3D9TranslateHRESULT(hr));
		return HQ_FAILED;
	}
#endif
	
	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::SetUniformMatrix(hq_uint32 parameterIndex,
					 const HQBaseMatrix3x4* pMatrices,
					 hq_uint32 numMatrices)
{
#if defined _DEBUG || defined DEBUG
	if(activeProgram==NULL)
		return HQ_FAILED;
#endif
	
	
	
	HQParameterD3D9* param = GETPARAM(parameterIndex);
#if defined _DEBUG || defined DEBUG	
	if(param == NULL)
	{
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	//cgSetParameterValuefr(param->parameter,(int)numMatrices * 12,(hq_float32*)pMatrices);

	
	HRESULT hr =cgD3D9SetUniformArray(param->parameter,0,numMatrices,pMatrices);
#if defined _DEBUG || defined DEBUG
	if(FAILED(hr))
	{
		this->Log("Error : SetUniformMatrix() failed . %s",cgD3D9TranslateHRESULT(hr));
		return HQ_FAILED;
	}
#endif
	
	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D9::CreateUniformBuffer(hq_uint32 size , void *initData , bool isDynamic , hq_uint32 *pBufferIDOut)
{
	return HQ_FAILED;
}
HQReturnVal HQShaderManagerD3D9::DestroyUniformBuffer(hq_uint32 bufferID)
{
	return HQ_FAILED;
}
void HQShaderManagerD3D9::DestroyAllUniformBuffers()
{

}
HQReturnVal HQShaderManagerD3D9::SetUniformBuffer(hq_uint32 slot ,  hq_uint32 bufferID )
{
	return HQ_FAILED;
}
HQReturnVal HQShaderManagerD3D9::MapUniformBuffer(hq_uint32 bufferID , void **ppData)
{
	return HQ_FAILED;
}
HQReturnVal HQShaderManagerD3D9::UnmapUniformBuffer(hq_uint32 bufferID)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D9::UpdateUniformBuffer(hq_uint32 bufferID, const void * pData)
{
	return HQ_FAILED;
}