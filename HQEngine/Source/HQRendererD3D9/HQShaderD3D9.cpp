/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

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
	,activeProgramID(HQ_NOT_USE_SHADER)
#endif
{
	this->pD3DDevice=pD3DDevice;

	this->cgContext=cgCreateContext();
	
	cgD3D9SetDevice(pD3DDevice);

#if 1
	this->cgVertexProfile = CG_PROFILE_HLSLV;
	this->cgPixelProfile = CG_PROFILE_HLSLF;
#else
	this->cgVertexProfile=cgD3D9GetLatestVertexProfile();
	this->cgPixelProfile=cgD3D9GetLatestPixelProfile();
#endif
	
	//fixed function controlling values
	this->firstStageOp[0] = D3DTOP_MODULATE;
	this->firstStageOp[1] = D3DTOP_SELECTARG1;

	pShaderMan=this;
#if defined(DEBUG)||defined(_DEBUG)
	cgSetErrorCallback(cgErrorCallBack);
#endif

	cgSetContextBehavior(this->cgContext, CG_BEHAVIOR_3100); 

	const char* version = cgGetString(CG_VERSION);
	Log("Init done! Cg library version is %s", version);
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
		pD3DDevice->SetVertexShader(activeVShader->vshader);
	if (activePShader != NULL)
		pD3DDevice->SetPixelShader(activePShader->pshader);

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
	if (programID == HQ_NOT_USE_SHADER)
	{
		if (activeProgram == NULL)
			return HQ_OK;
		if(activeVShader!=NULL)
		{
			pD3DDevice->SetVertexShader(NULL);
		}
		if(activePShader!=NULL)
		{
			pD3DDevice->SetPixelShader(NULL);
		}
		activeProgram=HQSharedPtr<HQShaderProgramD3D9>::null;

		activeVShader=HQSharedPtr<HQShaderObjectD3D9>::null;
		activePShader=HQSharedPtr<HQShaderObjectD3D9>::null;

#if defined _DEBUG || defined DEBUG
		activeProgramID = HQ_NOT_USE_SHADER;
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
			if(pProgram->vertexShader==NULL)//đang dùng shader nhưng program chuẩn bị active lại dùng fixed function vertex processor
			{
				pD3DDevice->SetVertexShader(NULL);
				activeVShader=HQSharedPtr<HQShaderObjectD3D9>::null;
			}
			else//đang dùng vertex shader khác vertex shader trong program chuẩn bị active 
			{
				activeVShader=pProgram->vertexShader;
				pD3DDevice->SetVertexShader(activeVShader->vshader);
			}
		}
		if(activePShader!=pProgram->pixelShader)//pixel shader object đang active khác pixel shader object trong program
		{
			if(pProgram->pixelShader == NULL)//đang dùng shader nhưng program chuẩn bị active lại dùng fixed function pixel processor
			{
				pD3DDevice->SetPixelShader(NULL);
				activePShader=HQSharedPtr<HQShaderObjectD3D9>::null;
			}
			else//đang dùng pixel shader khác pixel shader trong program chuẩn bị active 
			{
				activePShader=pProgram->pixelShader;
				pD3DDevice->SetPixelShader(activePShader->pshader);
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
		this->ActiveProgram(HQ_NOT_USE_SHADER);
	}
	this->Remove(programID);
	return HQ_OK;
}

void HQShaderManagerD3D9::DestroyAllProgram()
{
	this->ActiveProgram(HQ_NOT_USE_SHADER);
	
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


HQReturnVal HQShaderManagerD3D9::CreateShaderFromStreamEx(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 const char **args,
									 bool debugMode , 
									 hq_uint32 *pID)
{
	const char nullStreamName[] = "";
	const char *streamName = dataStream->GetName() != NULL? dataStream->GetName(): nullStreamName;
#if defined DEBUG || defined _DEBUG
	//debugMode = true; force debugging
#endif

	if (isPreCompiled)
	{
		//precompiled shader is not supported for now.
		this->Log("Shader compile from stream %s error ! Precompiled shader is not supported.",streamName);
		return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}
	CGprofile profile; 
	if(type==HQ_VERTEX_SHADER)
		profile = this->cgVertexProfile;
	else if (type==HQ_PIXEL_SHADER)
		profile = this->cgPixelProfile ;
	else
	{
		return HQ_FAILED;
	}
	char * streamContent = new char [dataStream->TotalSize() + 1];
	dataStream->ReadBytes(streamContent, dataStream->TotalSize(), 1);
	streamContent[dataStream->TotalSize()] = '\0';

	HQShaderObjectD3D9 *sobject= new HQShaderObjectD3D9();
	//translate cg to hlsl
	CGprogram cgprogram=cgCreateProgram(this->cgContext, CG_SOURCE,
												 streamContent, profile, entryFunctionName,args);
	
	delete[] streamContent;
	
	if(cgprogram==NULL)
	{
		delete sobject;
		return HQ_FAILED;
	}
	sobject->type=type;
	
	//now compile hlsl code

	{
		HRESULT hr;
		//check for errors
		ID3DXBuffer* byteCode = NULL;
		ID3DXBuffer* errorMsg = NULL;
		ID3DXConstantTable* constTable = NULL;
		DWORD compileFlags = debugMode? (D3DXSHADER_DEBUG | D3DXSHADER_SKIPOPTIMIZATION): 0;
		compileFlags |= D3DXSHADER_ENABLE_BACKWARDS_COMPATIBILITY;
		const char * d3dprofile = type == HQ_VERTEX_SHADER? D3DXGetVertexShaderProfile(pD3DDevice): D3DXGetPixelShaderProfile(pD3DDevice);

		//get output from cg
		const char * compiled_src = cgGetProgramString(cgprogram, CG_COMPILED_PROGRAM);
		bool writeToTemp = false;
		if (debugMode && dataStream->GetName() != NULL)
		{
			//write translated code to temp file
			std::string tempFileName =  dataStream->GetName() ;
			tempFileName += ".";
			tempFileName += d3dprofile;
			tempFileName += ".temp";
			FILE* tempf = fopen(tempFileName.c_str(), "wb");
			if (tempf == NULL)
				writeToTemp = false;
			else
			{
				writeToTemp = true;
				fwrite(compiled_src, strlen(compiled_src), 1, tempf);//write translated code to temp file
				fclose(tempf);

				//compile by d3dx
				hr = D3DXCompileShaderFromFileA(
						tempFileName.c_str(),
						NULL,
						NULL,
						"main",
						d3dprofile,
						compileFlags,
						&byteCode,
						&errorMsg,
						&constTable
						);
			}
		}

		if (!writeToTemp)
		{
			//compile by d3dx
			hr = D3DXCompileShader(
					compiled_src,
					strlen(compiled_src),
					NULL,
					NULL,
					"main",
					d3dprofile,
					compileFlags,
					&byteCode,
					&errorMsg,
					&constTable
					);
		}

		cgDestroyProgram(cgprogram);//no thing more to do with cg program

		if (FAILED(hr))
		{
			if (errorMsg)
				this->Log("Shader compile from stream %s error ! Error message \"%s\"",streamName, errorMsg->GetBufferPointer());
			else
				this->Log("Shader compile from stream %s error !", streamName);

			delete sobject;
			SafeRelease(constTable);
			SafeRelease(byteCode);
			SafeRelease(errorMsg);
			return HQ_FAILED;
		}
		//succeeded
		if (type == HQ_VERTEX_SHADER)
			pD3DDevice->CreateVertexShader((DWORD*) byteCode->GetBufferPointer(), &sobject->vshader);
		else
			pD3DDevice->CreatePixelShader((DWORD*) byteCode->GetBufferPointer(), &sobject->pshader);

		sobject->consTable = constTable;

		SafeRelease(byteCode);
		SafeRelease(errorMsg);

	}

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
	

	if (isPreCompiled)
	{
		//precompiled shader is not supported for now.
		this->Log("Shader compile from memory error ! Precompiled shader is not supported.");
		return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}
	CGprofile profile; 
	if(type==HQ_VERTEX_SHADER)
		profile = this->cgVertexProfile;
	else if (type==HQ_PIXEL_SHADER)
		profile = this->cgPixelProfile ;
	else
	{
		return HQ_FAILED;
	}


	HQShaderObjectD3D9 *sobject= new HQShaderObjectD3D9();
	//translate cg to hlsl
	CGprogram cgprogram=cgCreateProgram(this->cgContext, CG_SOURCE,
												 pSourceData, profile, entryFunctionName,args);
	
	
	if(cgprogram==NULL)
	{
		delete sobject;
		return HQ_FAILED;
	}
	sobject->type=type;
	
	//now compile hlsl code

	{
		HRESULT hr;
		//check for errors
		ID3DXBuffer* byteCode = NULL;
		ID3DXBuffer* errorMsg = NULL;
		ID3DXConstantTable* constTable = NULL;
		DWORD compileFlags = debugMode? (D3DXSHADER_DEBUG): 0;
		const char * d3dprofile = type == HQ_VERTEX_SHADER? D3DXGetVertexShaderProfile(pD3DDevice): D3DXGetPixelShaderProfile(pD3DDevice);

		//get output from cg
		const char * compiled_src = cgGetProgramString(cgprogram, CG_COMPILED_PROGRAM);

		//compile by d3dx
		hr = D3DXCompileShader(
				compiled_src,
				strlen(compiled_src),
				NULL,
				NULL,
				"main",
				d3dprofile,
				compileFlags,
				&byteCode,
				&errorMsg,
				&constTable
				);

		cgDestroyProgram(cgprogram);//no more thing to do with cg program

		if (FAILED(hr))
		{
			if (errorMsg)
				this->Log("Shader compile from memory error ! Error message \"%s\"",errorMsg->GetBufferPointer());
			else
				this->Log("Shader compile from memory error !");

			SafeRelease(constTable);
			delete sobject;
			SafeRelease(byteCode);
			SafeRelease(errorMsg);
			return HQ_FAILED;
		}
		//succeeded
		if (type == HQ_VERTEX_SHADER)
			pD3DDevice->CreateVertexShader((DWORD*) byteCode->GetBufferPointer(), &sobject->vshader);
		else
			pD3DDevice->CreatePixelShader((DWORD*) byteCode->GetBufferPointer(), &sobject->pshader);

		sobject->consTable = constTable;

		SafeRelease(byteCode);
		SafeRelease(errorMsg);

	}

	if (!this->shaderObjects.AddItem(sobject , pID))
	{
		delete sobject;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}
HQReturnVal HQShaderManagerD3D9::CreateShaderFromStream(HQShaderType type,
										HQDataReaderStream* dataStream,
										const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										bool isPreCompiled,
										const char* entryFunctionName,
										hq_uint32 *pID)
{
	char ** args = this->GetPredefineMacroArguments(pDefines);
	HQReturnVal re = this->CreateShaderFromStreamEx(type,dataStream,isPreCompiled , entryFunctionName ,(const char**)args ,false , pID);
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

HQReturnVal HQShaderManagerD3D9::CreateShaderFromStream(HQShaderType type,
								 HQShaderCompileMode compileMode,
								 HQDataReaderStream* dataStream,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 hq_uint32 *pID)
{
	HQReturnVal re;
	char ** args = this->GetPredefineMacroArguments(pDefines);
	switch (compileMode)
	{
	case HQ_SCM_CG:
		re = this->CreateShaderFromStreamEx(type , dataStream,false , entryFunctionName, (const char**)args , false,pID);
		break;
	case HQ_SCM_CG_DEBUG:
		re = this->CreateShaderFromStreamEx(type , dataStream,false , entryFunctionName, (const char**)args , true,pID);
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
	if(vertexShaderID==HQ_NOT_USE_VSHADER && pixelShaderID==HQ_NOT_USE_PSHADER && geometryShaderID==HQ_NOT_USE_GSHADER)
		return HQ_FAILED_SHADER_PROGRAM_NEED_SHADEROBJECT;//tất cả đều là fixed function => báo lỗi
	
	HQSharedPtr<HQShaderObjectD3D9> pVShader = HQSharedPtr<HQShaderObjectD3D9> :: null;
	HQSharedPtr<HQShaderObjectD3D9> pPShader = HQSharedPtr<HQShaderObjectD3D9> :: null;
	if (vertexShaderID != HQ_NOT_USE_VSHADER)
		pVShader = this->shaderObjects.GetItemPointer(vertexShaderID);
	if (pixelShaderID != HQ_NOT_USE_PSHADER)
		pPShader = this->shaderObjects.GetItemPointer(pixelShaderID);
	
	if(pVShader == NULL || pPShader == NULL)
		return HQ_FAILED_INVALID_ID;
	if(vertexShaderID!=HQ_NOT_USE_VSHADER && pVShader->type!=HQ_VERTEX_SHADER)//shader có id <vertexShaderID> không phải vertex shader
		return HQ_FAILED_WRONG_SHADER_TYPE;
	if(pixelShaderID!=HQ_NOT_USE_PSHADER && pPShader->type!=HQ_PIXEL_SHADER)//shader có id <pixelShaderID> không phải pixel shader
		return HQ_FAILED_WRONG_SHADER_TYPE;

	HQShaderProgramD3D9 *pShader=new HQShaderProgramD3D9();

	//store shaders' pointers
	pShader->vertexShader = pVShader;
	pShader->pixelShader = pPShader;

	//store shaders' IDs
	pShader->vertexShaderID = (pVShader != NULL)? vertexShaderID : HQ_NOT_USE_VSHADER;
	pShader->pixelShaderID = (pPShader != NULL)? pixelShaderID : HQ_NOT_USE_PSHADER;
	
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
		return (pProgram != NULL)? pProgram->vertexShaderID : HQ_NOT_USE_VSHADER;
	case HQ_PIXEL_SHADER:
		return (pProgram != NULL)? pProgram->pixelShaderID : HQ_NOT_USE_PSHADER;
	}

	return HQ_NOT_USE_GSHADER;
}

/*-----------------------*/
hquint32 HQShaderManagerD3D9::GetD3DConstantStartRegister(ID3DXConstantTable* table, const char* name)
{
	//TO DO:
	//find nested element in a struct, array element
	D3DXHANDLE constant = table->GetConstantByName(NULL, name);

	if (constant == NULL)
	{
		//try one more with "$" prefix
		if (name[0] != '$')
		{
			std::string funcionConstInputName = "$";
			funcionConstInputName += name;
			constant = table->GetConstantByName(NULL, funcionConstInputName.c_str());
		}
	}
	if (constant == NULL)
		return 0xffffffff;
	D3DXCONSTANT_DESC desc;
	UINT numDesc = 1;
	table->GetConstantDesc(constant, &desc, &numDesc);

	return desc.RegisterIndex;
}


hq_uint32 HQShaderManagerD3D9::GetParameterIndex(HQSharedPtr<HQShaderProgramD3D9> &pProgram , 
											const char *parameterName)
{
	hq_uint32 *pIndex = pProgram->parameterIndexes.GetItemPointer(parameterName);
	if (pIndex == NULL)//không có
	{
		std::string cgParameterName = "_";
		cgParameterName += parameterName;
		hquint32 paramRegInVS = 0xffffffff, paramRegInPS = 0xffffffff;
		//cố tìm trong vertex shader
		if(pProgram->vertexShader!=NULL)
			paramRegInVS = this->GetD3DConstantStartRegister(pProgram->vertexShader->consTable ,cgParameterName.c_str());//get parameter handle
			
		if(pProgram->pixelShader!=NULL)//tìm tiếp trong pixel shader
			paramRegInPS = this->GetD3DConstantStartRegister(pProgram->pixelShader->consTable ,cgParameterName.c_str());
		if(paramRegInVS == 0xffffffff && paramRegInPS==0xffffffff)//không tìm thấy
		{
			return HQ_NOT_AVAIL_ID;
		}
		//đã tìm thấy =>thêm vào danh sách
		HQParameterD3D9* pNewParameter = new HQParameterD3D9();
		pNewParameter->parameterReg[0]= paramRegInVS;
		pNewParameter->parameterReg[1]= paramRegInPS;
		
		hq_uint32 paramIndex;
		if(!pProgram->parameters.AddItem(pNewParameter , &paramIndex))
		{
			delete pNewParameter;
			return HQ_NOT_AVAIL_ID;
		}
		else
		{
			pProgram->parameterIndexes.Add(parameterName , paramIndex);
		}
		return paramIndex;
	}
	return *pIndex;
}
inline HQSharedPtr<HQParameterD3D9> HQShaderManagerD3D9::GetUniformParam(HQSharedPtr<HQShaderProgramD3D9>& pProgram,const char* parameterName)
{
	hq_uint32 paramIndex = this->GetParameterIndex(pProgram , parameterName);

	if(paramIndex == HQ_NOT_AVAIL_ID)
		return NULL;

	return pProgram->parameters.GetItemPointerNonCheck(paramIndex);
}

/*-----------------------*/
template <size_t vecSize> inline void HQShaderManagerD3D9::SetD3DVShaderConstantI(
													hquint32 startReg, 
													 const int* pValues,
													 hq_uint32 numElements)
{
	int temp[4];
	for (hquint32 i = 0; i < numElements; ++i)
	{
		memcpy(temp, pValues + vecSize * i, vecSize * sizeof(int));
		pD3DDevice->SetVertexShaderConstantI(startReg + 4 * i, temp, 1);
	}
}

template <> inline void HQShaderManagerD3D9::SetD3DVShaderConstantI<4>(
										hquint32 startReg, 
										const int* pValues,
										hq_uint32 numElements)
{
	pD3DDevice->SetVertexShaderConstantI(startReg, pValues, numElements);
}


template <size_t vecSize> inline void HQShaderManagerD3D9::SetD3DVShaderConstantF(
												hquint32 startReg, 
												 const float* pValues,
												 hq_uint32 numElements)
{
	float temp[4];
	for (hquint32 i = 0; i < numElements; ++i)
	{
		memcpy(temp, pValues + vecSize * i, vecSize * sizeof(float));
		pD3DDevice->SetVertexShaderConstantF(startReg + 4 * i, temp, 1);
	}
}

template <> inline void HQShaderManagerD3D9::SetD3DVShaderConstantF<4>(
										hquint32 startReg, 
										 const float* pValues,
										 hq_uint32 numElements)
{
	pD3DDevice->SetVertexShaderConstantF(startReg, pValues, numElements);
}

template <size_t vecSize> inline void HQShaderManagerD3D9::SetD3DPShaderConstantI(
													hquint32 startReg, 
													 const int* pValues,
													 hq_uint32 numElements)
{
	int temp[4];
	for (hquint32 i = 0; i < numElements; ++i)
	{
		memcpy(temp, pValues + vecSize * i, vecSize * sizeof(int));
		pD3DDevice->SetPixelShaderConstantI(startReg + 4 * i, temp, 1);
	}
}

template <> inline void HQShaderManagerD3D9::SetD3DPShaderConstantI<4>(
										hquint32 startReg, 
										const int* pValues,
										hq_uint32 numElements)
{
	pD3DDevice->SetPixelShaderConstantI(startReg, pValues, numElements);
}


template <size_t vecSize> inline void HQShaderManagerD3D9::SetD3DPShaderConstantF(
												hquint32 startReg, 
												 const float* pValues,
												 hq_uint32 numElements)
{
	float temp[4];
	for (hquint32 i = 0; i < numElements; ++i)
	{
		memcpy(temp, pValues + vecSize * i, vecSize * sizeof(float));
		pD3DDevice->SetPixelShaderConstantF(startReg + 4 * i, temp, 1);
	}
}

template <> inline void HQShaderManagerD3D9::SetD3DPShaderConstantF<4>(
										hquint32 startReg, 
										 const float* pValues,
										 hq_uint32 numElements)
{
	pD3DDevice->SetPixelShaderConstantF(startReg, pValues, numElements);
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
	

	HQSharedPtr<HQParameterD3D9> param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantI<1>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantI<1>(param->parameterReg[1], pValues, numElements);

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
	

	HQSharedPtr<HQParameterD3D9> param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantI<2>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantI<2>(param->parameterReg[1], pValues, numElements);

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
	

	HQSharedPtr<HQParameterD3D9> param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantI<3>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantI<3>(param->parameterReg[1], pValues, numElements);

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
	

	HQSharedPtr<HQParameterD3D9> param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantI<4>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantI<4>(param->parameterReg[1], pValues, numElements);

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
	

	HQSharedPtr<HQParameterD3D9> param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantF<1>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantF<1>(param->parameterReg[1], pValues, numElements);

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
	

	HQSharedPtr<HQParameterD3D9> param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantF<2>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantF<2>(param->parameterReg[1], pValues, numElements);

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
	

	HQSharedPtr<HQParameterD3D9> param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantF<3>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantF<3>(param->parameterReg[1], pValues, numElements);

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
	

	HQSharedPtr<HQParameterD3D9> param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantF<4>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantF<4>(param->parameterReg[1], pValues, numElements);

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
	

	HQSharedPtr<HQParameterD3D9> param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif

	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantF<4>(param->parameterReg[0], (float*)pMatrices, 4 * numMatrices);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantF<4>(param->parameterReg[1], (float*)pMatrices, 4 * numMatrices);


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
	

	HQSharedPtr<HQParameterD3D9> param = this->GetUniformParam(activeProgram , parameterName);
#if defined _DEBUG || defined DEBUG
	if(param == NULL)
	{
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgramID);
		return HQ_FAILED_SHADER_PARAMETER_NOT_FOUND;
	}
#endif

	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantF<4>(param->parameterReg[0], (float*)pMatrices, 3 * numMatrices);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantF<4>(param->parameterReg[1], (float*)pMatrices, 3 * numMatrices);
	
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
		return HQ_NOT_AVAIL_ID;
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
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantI<1>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantI<1>(param->parameterReg[1], pValues, numElements);
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
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantI<2>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantI<2>(param->parameterReg[1], pValues, numElements);

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
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantI<3>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantI<3>(param->parameterReg[1], pValues, numElements);

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
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantI<4>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantI<4>(param->parameterReg[1], pValues, numElements);
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
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantF<1>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantF<1>(param->parameterReg[1], pValues, numElements);

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
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantF<2>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantF<2>(param->parameterReg[1], pValues, numElements);

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
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantF<3>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantF<3>(param->parameterReg[1], pValues, numElements);

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
	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantF<4>(param->parameterReg[0], pValues, numElements);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantF<4>(param->parameterReg[1], pValues, numElements);

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

	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantF<4>(param->parameterReg[0], (float*)pMatrices, 4 * numMatrices);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantF<4>(param->parameterReg[1], (float*)pMatrices, 4 * numMatrices);
	
	
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

	
	if (param->parameterReg[0] != 0xffffffff)
		this->SetD3DVShaderConstantF<4>(param->parameterReg[0], (float*)pMatrices, 3 * numMatrices);
	if (param->parameterReg[1] != 0xffffffff)
		this->SetD3DPShaderConstantF<4>(param->parameterReg[1], (float*)pMatrices, 3 * numMatrices);
	
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


