/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D11PCH.h"
#include "HQShaderD3D11.h"
#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#include "d3dx11.h"
#else
#include "../HQEngine/winstore/HQWinStoreFileSystem.h"
#endif
#include <string.h>
#include <stdio.h>

#define HQ_NON_DYNAMIC_U_BUFFER_CAN_BE_UPDATED 1

#if !HQ_D3D_CLEAR_VP_USE_GS
struct ClearBufferParameters
{
	HQColor color;
	hq_float32 depth;
	hq_float32 padding[3];
};
#endif

struct ID3DBlobImpl: public ID3DBlob
{
	ID3DBlobImpl()
		:data(NULL), dataSize(0), ownData(true), refCout(1)
	{
	}
	ID3DBlobImpl(hqubyte *_data, hquint32 _dataSize, bool _ownData)
		:data(_data), dataSize(_dataSize), ownData(_ownData), refCout(1)
	{
	}
	
	ID3DBlobImpl(const hqubyte *_data, hquint32 _dataSize)
		:data(const_cast<hqubyte *>(_data)), dataSize(_dataSize), ownData(false), refCout(1)
	{
	}

	~ID3DBlobImpl()
	{
		if (data && ownData)
			delete[] data;
	}
	

	hqubyte *data;
	hquint32 dataSize;
	bool ownData;
	ULONG refCout;
	
	void setData(hqubyte *_data, hquint32 _dataSize, bool _ownData = false)
	{
		data = _data;
		dataSize = _dataSize;
		ownData = _ownData;
	}
	virtual LPVOID STDMETHODCALLTYPE GetBufferPointer() {return data;}
	virtual SIZE_T STDMETHODCALLTYPE GetBufferSize() {return dataSize;} 
	virtual HRESULT STDMETHODCALLTYPE QueryInterface( 
                /* [in] */ REFIID riid,
                /* [iid_is][out] */ __RPC__deref_out void __RPC_FAR *__RPC_FAR *ppvObject)
	{
#if 0
		if (riid == __uuidof(ID3DBlobImpl))
		{
			if (ppvObject != NULL)
				*ppvObject = this;

			return S_OK;
		}
#endif

		return S_FALSE;
	}

    virtual ULONG STDMETHODCALLTYPE AddRef( void)
	{
		return ++refCout;
	}

    virtual ULONG STDMETHODCALLTYPE Release( void)
	{
		refCout--;
		if (refCout == 0)
		{
			HQ_DELETE (this);
			return 0;
		}

		return refCout;
	}
};

static char *cgSemantics[]= {
#if HQ_DEFINE_SEMANTICS
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
#endif
	NULL
};

static const size_t cgSemanticsSize = sizeof(cgSemantics)/ sizeof(char*);
static const size_t numCgPreDefined = cgSemanticsSize - 1;

#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
static D3D10_SHADER_MACRO hlslSemantics[] =
{
#if HQ_DEFINE_SEMANTICS
	"VPOSITION","POSITION",
	"VCOLOR","COLOR",
	"VNORMAL","NORMAL",
	"VTEXCOORD0","TEXCOORD0",
	"VTEXCOORD1","TEXCOORD1",
	"VTEXCOORD2","TEXCOORD2",
	"VTEXCOORD3","TEXCOORD3",
	"VTEXCOORD4","TEXCOORD4",
	"VTEXCOORD5","TEXCOORD5",
	"VTEXCOORD6","TEXCOORD6",
	"VTEXCOORD7","TEXCOORD7",
	"VTANGENT","TANGENT",
	"VBINORMAL","BINORMAL",
	"VBLENDWEIGHT","BLENDWEIGHT",
	"VBLENDINDICES","BLENDINDICES",
	"VPSIZE","PSIZE",
#endif
	NULL , NULL
};

static const size_t hlslSemanticsSize = sizeof(hlslSemantics)/ sizeof(D3D10_SHADER_MACRO);
static const size_t numHlslPreDefined = hlslSemanticsSize - 1;

#endif//#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

HQShaderManagerD3D11* pShaderMan=NULL;

/*---------Shader object-------------------*/
HQShaderObjectD3D11::HQShaderObjectD3D11()
{
	pD3DShader = NULL;
	pByteCodeInfo = NULL;
}
HQShaderObjectD3D11::~HQShaderObjectD3D11()
{
	SafeRelease(pD3DShader);
	SafeRelease(pByteCodeInfo);
}
/*---------Shader Constant Buffer----------*/
HQShaderConstBufferD3D11::HQShaderConstBufferD3D11(bool dynamic, hq_uint32 _size)
:isDynamic(dynamic), size(_size)
{
	pD3DBuffer = 0;
}
HQShaderConstBufferD3D11::HQShaderConstBufferD3D11(ID3D11Buffer *pD3DBuffer, bool dynamic, hq_uint32 _size)
:isDynamic(dynamic), size(_size)
{
	this->pD3DBuffer = pD3DBuffer;
}
HQShaderConstBufferD3D11::~HQShaderConstBufferD3D11()
{
	SafeRelease(pD3DBuffer);
}
/*-----------------------------------------*/
#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
void cgErrorCallBack(void)
{
	CGerror err=cgGetError();
	HRESULT hr=cgD3D11GetLastError();
	if(FAILED(hr))
		pShaderMan->Log("D3D11 error %s",cgD3D11TranslateHRESULT(hr));
	else
		pShaderMan->Log("%s",cgGetErrorString(err));
}
#endif//#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
/*---------HQShaderManagerD3D11--------------*/
HQShaderManagerD3D11::HQShaderManagerD3D11(ID3D11Device * pD3DDevice ,
										   ID3D11DeviceContext* pD3DContext,
										   D3D_FEATURE_LEVEL featureLevel,
										   HQLogStream* logFileStream , bool flushLog)
: HQLoggableObject(logFileStream , "D3D11 Shader Manager :" , flushLog , 1024)
{
	this->pD3DDevice=pD3DDevice;
	this->pD3DContext = pD3DContext;

#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	this->cgContext=cgCreateContext();
	const char* version = cgGetString(CG_VERSION);
	this->Log("Cg library loaded. Version is %s", version);

	cgD3D11SetDevice(this->cgContext , pD3DDevice);
	
	this->featureLevel = featureLevel;
	switch(featureLevel)
	{
	case D3D_FEATURE_LEVEL_11_0:
		this->cgVertexProfile=CG_PROFILE_VS_5_0;
		this->cgPixelProfile=CG_PROFILE_PS_5_0;
		this->cgGeometryProfile=CG_PROFILE_GS_5_0;
		break;
	case D3D_FEATURE_LEVEL_10_0:
		this->cgVertexProfile=CG_PROFILE_VS_4_0;
		this->cgPixelProfile=CG_PROFILE_PS_4_0;
		this->cgGeometryProfile=CG_PROFILE_GS_4_0;
		break;
	case D3D_FEATURE_LEVEL_9_3:
		this->cgVertexProfile=CG_PROFILE_VS_2_0;
		this->cgPixelProfile=CG_PROFILE_PS_2_0;
		this->cgGeometryProfile=CG_PROFILE_UNKNOWN;
		break;
	}

#endif//#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

	pShaderMan=this;

#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#if defined(DEBUG)||defined(_DEBUG)
	cgSetErrorCallback(cgErrorCallBack);
#endif
#endif//#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

/*------create shaders for clearing viewport------*/
	ID3DBlob *pBlob = 0;
	ID3DBlob *pError = 0;

#if !HQ_D3D_CLEAR_VP_USE_BYTE_CODE
	UINT flags;
#if defined(DEBUG)||defined(_DEBUG)
		flags = D3D10_SHADER_DEBUG | D3D10_SHADER_SKIP_OPTIMIZATION;
#else
		flags = D3D10_SHADER_OPTIMIZATION_LEVEL3;
#endif
	hq_uint32 len = strlen(g_clearShaderCode);
#endif//if !HQ_D3D_CLEAR_VP_USE_BYTE_CODE

#if HQ_D3D_CLEAR_VP_USE_GS
	D3DX11CompileFromMemory(g_clearShaderCode , len ,NULL , NULL ,NULL,
							"VS" , "vs_4_0" , flags , 0 , NULL,
							&pBlob , &pError , NULL);
	this->CreateShader(HQ_VERTEX_SHADER , pBlob , &this->clearVShader);

	D3DX11CompileFromMemory(g_clearShaderCode , len ,NULL , NULL ,NULL,
							"GS" , "gs_4_0" , flags , 0 , NULL,
							&pBlob , &pError , NULL);
	this->CreateShader(HQ_GEOMETRY_SHADER , pBlob , &this->clearGShader);
	SafeRelease(pBlob);
	SafeRelease(pError);

	D3DX11CompileFromMemory(g_clearShaderCode , len ,NULL , NULL ,NULL,
							"PS" , "ps_4_0" , flags , 0 , NULL,
							&pBlob , &pError , NULL);
	this->CreateShader(HQ_PIXEL_SHADER , pBlob , &this->clearPShader);
	SafeRelease(pBlob);
	SafeRelease(pError);
#else//#if !HQ_D3D_CLEAR_VP_USE_GS
#if !HQ_D3D_CLEAR_VP_USE_BYTE_CODE
	D3DX11CompileFromMemory(g_clearShaderCode , len ,NULL , NULL ,NULL,
							"VS" , "vs_4_0_level_9_1" , flags , 0 , NULL,
							&pBlob , &pError , NULL);
#else
	pBlob = HQ_NEW ID3DBlobImpl(HQClearViewportShaderCodeD3D1x_VS,
							sizeof(HQClearViewportShaderCodeD3D1x_VS));
#endif

	this->CreateShader(HQ_VERTEX_SHADER , pBlob , &this->clearVShader);
	SafeRelease(pBlob);
	SafeRelease(pError);

#if !HQ_D3D_CLEAR_VP_USE_BYTE_CODE
	D3DX11CompileFromMemory(g_clearShaderCode , len ,NULL , NULL ,NULL,
							"PS" , "ps_4_0_level_9_1" , flags , 0 , NULL,
							&pBlob , &pError , NULL);
#else
	pBlob = HQ_NEW ID3DBlobImpl(HQClearViewportShaderCodeD3D1x_PS,
							sizeof(HQClearViewportShaderCodeD3D1x_PS));
#endif

	this->CreateShader(HQ_PIXEL_SHADER , pBlob , &this->clearPShader);
	SafeRelease(pBlob);
	SafeRelease(pError);

	this->CreateUniformBuffer(sizeof(ClearBufferParameters), NULL, true, &this->clearShaderParameters);
#endif//#if HQ_D3D_CLEAR_VP_USE_GS

	/*------------------------*/

	InitFFEmu();
	/*------------------------*/

	Log("Init done!");
}

HQShaderManagerD3D11::~HQShaderManagerD3D11()
{

	this->DestroyAllResource();
#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	cgD3D11SetDevice(this->cgContext , NULL);
	cgDestroyContext(cgContext);
#endif//#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

	this->ReleaseFFEmu();

	Log("Released!");
	pShaderMan=NULL;
}
/*------------------------*/

bool HQShaderManagerD3D11::IsUsingVShader() //có đang dùng vertex shader không,hay đang dùng fixed function
{
	if (this->IsFFEmuActive())
		return false;

	return activeProgram->isUseVS();
}
bool HQShaderManagerD3D11::IsUsingGShader()//có đang dùng geometry shader không,hay đang dùng fixed function
{
	if (this->IsFFEmuActive())
		return false;

	return activeProgram->isUseGS();
}
bool HQShaderManagerD3D11::IsUsingPShader() //có đang dùng pixel/fragment shader không,hay đang dùng fixed function
{
	if (this->IsFFEmuActive())
		return false;

	return activeProgram->isUsePS();
}

ID3DBlob *HQShaderManagerD3D11::GetCompiledVertexShader(hq_uint32 vertexShaderID)
{
	HQSharedPtr<HQShaderObjectD3D11> pVShader = this->shaderObjects.GetItemPointer(vertexShaderID);
	if (pVShader == NULL || pVShader->type != HQ_VERTEX_SHADER)
		return NULL;

	return pVShader->pByteCodeInfo;
}
ID3DBlob *HQShaderManagerD3D11::GetCompiledClearVShader()
{
	return clearVShader.pByteCodeInfo;
}

/*------------------------*/
HQReturnVal HQShaderManagerD3D11::ActiveProgram(hq_uint32 programID)
{
	if (programID == HQ_NOT_USE_SHADER)
	{
		ActiveFFEmu();//active fixed function shader
	}
	else
	{
		DeActiveFFEmu();//deactive fixed function shader

		HQSharedPtr<HQShaderProgramD3D11> pProgram = this->GetItemPointer(programID);
		
#if defined _DEBUG || defined DEBUG
		if (pProgram == NULL)
			return HQ_FAILED_INVALID_ID;
#endif

		ID3D11VertexShader* pVS = NULL;
		ID3D11GeometryShader* pGS = NULL;
		ID3D11PixelShader* pPS = NULL;

		if (pProgram->isUseVS())
			pVS = (ID3D11VertexShader*)pProgram->vertexShader->pD3DShader;
		if (pProgram->isUseGS())
			pGS = (ID3D11GeometryShader*)pProgram->geometryShader->pD3DShader;
		if (pProgram->isUsePS())
			pPS = (ID3D11PixelShader*)pProgram->pixelShader->pD3DShader;

		if (pProgram->vertexShader != activeVShader)
		{
			pD3DContext->VSSetShader(pVS , NULL , 0);
			activeVShader=pProgram->vertexShader;
		}
		if (pProgram->geometryShader != activeGShader)
		{
			pD3DContext->GSSetShader(pGS , NULL , 0);
			activeGShader=pProgram->geometryShader;
		}
		if (pProgram->pixelShader != activePShader)
		{
			pD3DContext->PSSetShader(pPS , NULL , 0);
			activePShader=pProgram->pixelShader;
		}

		this->activeProgram = pProgram;
	}
	return HQ_OK;
}

/*------------------------*/
HQReturnVal HQShaderManagerD3D11::DestroyProgram(hq_uint32 programID)
{
	HQSharedPtr<HQShaderProgramD3D11> pProgram = this->GetItemPointer(programID);
	if(pProgram == NULL)
		return HQ_FAILED;
	if (IsFFProgram(programID))
		return HQ_FAILED;//prevent deletion of default fixed function shader

	if(pProgram==activeProgram)
	{
		this->ActiveProgram(HQ_NOT_USE_SHADER);
	}
	this->Remove(programID);
	return HQ_OK;
}

void HQShaderManagerD3D11::DestroyAllProgram()
{
	this->ActiveProgram(HQ_NOT_USE_SHADER);
	
	HQItemManager<HQShaderProgramD3D11>::Iterator ite;
	this->GetIterator(ite);

	while (!ite.IsAtEnd())
	{
		this->DestroyProgram(ite.GetID());

		++ite;
	}
}

void HQShaderManagerD3D11::DestroyAllResource()
{
	this->DestroyAllUniformBuffers();
	this->DestroyAllProgram();
	this->DestroyAllShader();
}


HQReturnVal HQShaderManagerD3D11::DestroyShader(hq_uint32 shaderID)
{
	if (IsFFShader(shaderID))
		return HQ_FAILED;//prevent deletion of default fixed function shader

	return (HQReturnVal)this->shaderObjects.Remove(shaderID);
}

void HQShaderManagerD3D11::DestroyAllShader()
{
	HQItemManager<HQShaderObjectD3D11>::Iterator ite;
	this->shaderObjects.GetIterator(ite);

	while (!ite.IsAtEnd())
	{
		this->DestroyShader(ite.GetID());

		++ite;
	}
}

HQReturnVal HQShaderManagerD3D11::DestroyUniformBuffer(hq_uint32 bufferID)
{
	if (IsFFConstBuffer(bufferID))//prevent deletion of fixed function's own const buffer
		return HQ_FAILED;
	return (HQReturnVal)shaderConstBuffers.Remove(bufferID);
}
void HQShaderManagerD3D11::DestroyAllUniformBuffers()
{
	HQItemManager<HQShaderConstBufferD3D11>::Iterator ite;
	this->shaderConstBuffers.GetIterator(ite);

	while (!ite.IsAtEnd())
	{
		this->DestroyUniformBuffer(ite.GetID());

		++ite;
	}
}
/*--------------------------*/

HQReturnVal HQShaderManagerD3D11::CreateShader(HQShaderType type , ID3D10Blob *pBlob , HQShaderObjectD3D11 * sobject)
{
	sobject->type=type;
	
	HRESULT hr; 
	try
	{
		switch(type)
		{
		case HQ_VERTEX_SHADER:
			hr = pD3DDevice->CreateVertexShader(pBlob->GetBufferPointer() , pBlob->GetBufferSize() , 
				NULL , (ID3D11VertexShader **)&sobject->pD3DShader);
			sobject->pByteCodeInfo = pBlob;
			pBlob->AddRef();
			break;
		case HQ_PIXEL_SHADER:
			hr = pD3DDevice->CreatePixelShader(pBlob->GetBufferPointer() , pBlob->GetBufferSize() , 
				NULL , (ID3D11PixelShader **)&sobject->pD3DShader);
			break;
		case HQ_GEOMETRY_SHADER:
			hr = pD3DDevice->CreateGeometryShader(pBlob->GetBufferPointer() , pBlob->GetBufferSize() , 
				NULL ,(ID3D11GeometryShader **)&sobject->pD3DShader);
			break;
		}
	}
	catch (...)
	{
		hr = S_FALSE;
	}
	if(FAILED(hr))
		return HQ_FAILED;
	return HQ_OK;
}

int HQShaderManagerD3D11::GetNumMacros(const HQShaderMacro * pDefines)
{
	int numDefines = 0;
	const HQShaderMacro *pD = pDefines;
	
	//calculate number of macros
	while (pD->name != NULL && pD->definition != NULL)
	{
		numDefines++;
		pD++;
	}

	return numDefines;
}

char ** HQShaderManagerD3D11::GetPredefineMacroArgumentsCg(const HQShaderMacro * pDefines)
{
	if(pDefines == NULL)
		return cgSemantics;
	int numDefines ;
	int nameLen = 0;
	int definitionLen = 0;
	const HQShaderMacro *pD = pDefines;
	
	numDefines = this->GetNumMacros(pDefines);

	if(numDefines == 0)
		return cgSemantics;
	/*------create arguments---------*/
	char ** args = HQ_NEW char *[numDefines + cgSemanticsSize];
	for (int i = 0 ; i < numCgPreDefined ; ++i)
		args[i] = cgSemantics[i];

	args[numDefines + numCgPreDefined] = NULL;

	pD = pDefines;
	
	int i = 0;
	while (pD->name != NULL && pD->definition != NULL)
	{
		nameLen = strlen(pD->name);
		definitionLen = strlen(pD->definition);
		
	
		if (definitionLen != 0)
		{
			args[numCgPreDefined + i] = HQ_NEW char[nameLen + definitionLen + 4];
			sprintf(args[numCgPreDefined + i] , "-D%s=%s" , pD->name , pD->definition);
		}
		else
		{
			args[numCgPreDefined + i] = HQ_NEW char[nameLen + 4];
			sprintf(args[numCgPreDefined + i] , "-D%s" , pD->name);
		}
		pD++;
		i++;
	}

	return args;
}

void HQShaderManagerD3D11::DeAllocArgsCg(char **ppC)
{
	if(ppC == NULL || ppC == cgSemantics)
		return;
	char ** ppD = ppC + numCgPreDefined;
	while(*ppD != NULL)
	{
		SafeDeleteArray(*ppD);
		ppD ++;
	}

	SafeDeleteArray(ppC);
}

#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
D3D10_SHADER_MACRO* HQShaderManagerD3D11::GetPredefineMacroArgumentsHLSL(const HQShaderMacro * pDefines)
{
	if(pDefines == NULL)
		return hlslSemantics;
	int numDefines ;
	int nameLen = 0;
	int definitionLen = 0;
	const HQShaderMacro *pD = pDefines;
	
	numDefines = this->GetNumMacros(pDefines);

	if(numDefines == 0)
		return hlslSemantics;
	/*------create macro array---------*/
	D3D10_SHADER_MACRO* macros = HQ_NEW D3D10_SHADER_MACRO[numDefines + hlslSemanticsSize];
	for (int i = 0 ; i < numHlslPreDefined ; ++i)
	{
		macros[i].Definition = hlslSemantics[i].Definition;
		macros[i].Name = hlslSemantics[i].Name;
	}

	macros[numDefines + numHlslPreDefined].Name = macros[numDefines + numHlslPreDefined].Definition = NULL;

	pD = pDefines;
	
	int i = 0;
	while (pD->name != NULL && pD->definition != NULL)
	{
		macros[i + numHlslPreDefined].Definition = pD->definition;
		macros[i + numHlslPreDefined].Name = pD->name;
		pD++;
		i++;
	}

	return macros;
}
void HQShaderManagerD3D11::DeAllocMacrosHLSL(D3D10_SHADER_MACRO *pM)
{
	if (pM == NULL || pM == hlslSemantics)
		return;
	delete[] pM;
}

#endif//#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

HQReturnVal HQShaderManagerD3D11::CreateShaderFromStreamCg(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 bool debugMode , 
									 hq_uint32 *pID)
{
#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	CGprofile profile;
	if(type==HQ_VERTEX_SHADER)
		profile=this->cgVertexProfile;
	else if (type==HQ_PIXEL_SHADER)
		profile=this->cgPixelProfile;
	else
	{
		profile=this->cgGeometryProfile;
	}

	char * streamContent = new char [dataStream->TotalSize() + 1];
	dataStream->ReadBytes(streamContent, dataStream->TotalSize(), 1);
	streamContent[dataStream->TotalSize()] = '0';
	
	char ** predefineMacroArgs = this->GetPredefineMacroArgumentsCg(pDefines);
	CGprogram program=cgCreateProgram(this->cgContext,isPreCompiled? CG_OBJECT : CG_SOURCE ,
												 streamContent,profile,entryFunctionName,(const char **)predefineMacroArgs);

	this->DeAllocArgsCg(predefineMacroArgs);
	delete [] streamContent;

	if(program==NULL)
	{
		return HQ_FAILED;
	}
	
	
	if(debugMode)
		cgD3D11LoadProgram(program,D3D10_SHADER_DEBUG | D3D10_SHADER_SKIP_OPTIMIZATION);
	else
		cgD3D11LoadProgram(program,D3D10_SHADER_OPTIMIZATION_LEVEL3);
	
	ID3D10Blob *pBlob = cgD3D11GetCompiledProgram(program);

	HQShaderObjectD3D11 *sobject= HQ_NEW HQShaderObjectD3D11();
	
	if (this->CreateShader(type , pBlob , sobject)!= HQ_OK)
	{
		HQ_DELETE (sobject);
		cgDestroyProgram(program);
		return HQ_FAILED;
	}
	if (!this->shaderObjects.AddItem(sobject , pID))
	{
		HQ_DELETE (sobject);
		return HQ_FAILED_MEM_ALLOC;
	}

	cgDestroyProgram(program);

	return HQ_OK;
#else
	return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
#endif//#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
}

HQReturnVal HQShaderManagerD3D11::CreateShaderFromMemoryCg(HQShaderType type,
									 const char* pSourceData,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 bool debugMode ,
									 hq_uint32 *pID)
{
	
#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	CGprofile profile;
	if(type==HQ_VERTEX_SHADER)
		profile=this->cgVertexProfile;
	else if (type==HQ_PIXEL_SHADER)
		profile=this->cgPixelProfile;
	else
	{
		profile=this->cgGeometryProfile;
	}

	char ** predefineMacroArgs = this->GetPredefineMacroArgumentsCg(pDefines);
	CGprogram program=cgCreateProgram(this->cgContext,isPreCompiled? CG_OBJECT : CG_SOURCE ,
												 pSourceData,profile,entryFunctionName,(const char **)predefineMacroArgs);

	this->DeAllocArgsCg(predefineMacroArgs);

	if(program==NULL)
	{
		return HQ_FAILED;
	}
	
	
	if(debugMode)
		cgD3D11LoadProgram(program,D3D10_SHADER_DEBUG | D3D10_SHADER_SKIP_OPTIMIZATION);
	else
		cgD3D11LoadProgram(program,D3D10_SHADER_OPTIMIZATION_LEVEL3);
	
	ID3D10Blob *pBlob = cgD3D11GetCompiledProgram(program);

	HQShaderObjectD3D11 *sobject= HQ_NEW HQShaderObjectD3D11();

	if (this->CreateShader(type , pBlob , sobject)!= HQ_OK)
	{
		HQ_DELETE (sobject);
		cgDestroyProgram(program);
		return HQ_FAILED;
	}
	if (!this->shaderObjects.AddItem(sobject , pID))
	{
		HQ_DELETE (sobject);
		return HQ_FAILED_MEM_ALLOC;
	}

	cgDestroyProgram(program);

	return HQ_OK;
#else
	return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
#endif//#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
}

HQReturnVal HQShaderManagerD3D11::CreateShaderFromStreamHLSL(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 const char* entryFunctionName,
									 bool debugMode,
									 hq_uint32 *pID)
{
#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)	
	char profile[] = "vs_4_0_level_9_3";
	switch (this->featureLevel)
	{
	case D3D_FEATURE_LEVEL_10_0:
		profile[3] = '4';
		profile[6] = '\0';//remove "_level_9_3"
		break;
	case D3D_FEATURE_LEVEL_11_0:
		profile[3] = '5';
		profile[6] = '\0';//remove "_level_9_3"
		break;
	case D3D_FEATURE_LEVEL_9_3:
		if (type == HQ_GEOMETRY_SHADER)
		{
			Log("Error : geometry shader is not supported");
			return HQ_FAILED;
		}
		break;
	}

	switch(type)
	{
	case HQ_PIXEL_SHADER:
		profile[0] = 'p';
		break;
	case HQ_GEOMETRY_SHADER:
		profile[0] = 'g';
		break;
	}

	const char nullStreamName[] = "";
	const char *streamName = dataStream->GetName() != NULL? dataStream->GetName(): nullStreamName;

	char * streamContent = new char [dataStream->TotalSize() + 1];
	dataStream->ReadBytes(streamContent, dataStream->TotalSize(), 1);
	streamContent[dataStream->TotalSize()] = '0';

	ID3D10Blob *pBlob = 0;
	ID3D10Blob *pError = 0;
	
	UINT flags;
	if(debugMode)
		flags = D3D10_SHADER_DEBUG | D3D10_SHADER_SKIP_OPTIMIZATION;
	else
		flags = D3D10_SHADER_OPTIMIZATION_LEVEL3;

	HQShaderObjectD3D11 *sobject= HQ_NEW HQShaderObjectD3D11();
	
	D3D10_SHADER_MACRO *macros = this->GetPredefineMacroArgumentsHLSL(pDefines);	

	if(FAILED(D3DX11CompileFromMemory(streamContent, dataStream->TotalSize(), dataStream->GetName() , macros , NULL , entryFunctionName , profile , 
							flags , 0 , NULL,
							&pBlob , &pError , NULL)) || 
		this->CreateShader(type , pBlob , sobject)!= HQ_OK)
	{
		if(pError)
			Log("HLSL shader compile from  stream %s error: %s" , streamName ,pError->GetBufferPointer());
		SafeRelease(pBlob);
		SafeRelease(pError);
		
		this->DeAllocMacrosHLSL(macros);
		delete[] streamContent;

		HQ_DELETE (sobject);
		return HQ_FAILED;;
	}

	SafeRelease(pBlob);
	SafeRelease(pError);
	this->DeAllocMacrosHLSL(macros);
	delete[] streamContent;

	if (!this->shaderObjects.AddItem(sobject , pID))
	{
		HQ_DELETE (sobject);
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
#else
	return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
#endif//#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
}

HQReturnVal HQShaderManagerD3D11::CreateShaderFromMemoryHLSL(HQShaderType type,
									 const char* pSourceData,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 const char* entryFunctionName,
									 bool debugMode ,
									 hq_uint32 *pID)
{
#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)	
	char profile[] = "vs_4_0_level_9_3";
	switch (this->featureLevel)
	{
	case D3D_FEATURE_LEVEL_10_0:
		profile[3] = '4';
		profile[6] = '\0';//remove "_level_9_3"
		break;
	case D3D_FEATURE_LEVEL_11_0:
		profile[3] = '5';
		profile[6] = '\0';//remove "_level_9_3"
		break;
	case D3D_FEATURE_LEVEL_9_3:
		if (type == HQ_GEOMETRY_SHADER)
		{
			Log("Error : geometry shader is not supported");
			return HQ_FAILED;
		}
		break;
	}

	switch(type)
	{
	case HQ_PIXEL_SHADER:
		profile[0] = 'p';
		break;
	case HQ_GEOMETRY_SHADER:
		profile[0] = 'g';
		break;
	}

	ID3D10Blob *pBlob = 0;
	ID3D10Blob *pError = 0;

	UINT flags;
	if(debugMode)
		flags = D3D10_SHADER_DEBUG | D3D10_SHADER_SKIP_OPTIMIZATION;
	else
		flags = D3D10_SHADER_OPTIMIZATION_LEVEL3;

	HQShaderObjectD3D11 *sobject= HQ_NEW HQShaderObjectD3D11();
	
	D3D10_SHADER_MACRO *macros = this->GetPredefineMacroArgumentsHLSL(pDefines);

	if(FAILED(D3DX11CompileFromMemory(pSourceData , strlen(pSourceData),NULL , macros ,NULL,
							entryFunctionName , profile , flags , 0 , NULL,
							&pBlob , &pError , NULL)) || 
		this->CreateShader(type , pBlob , sobject)!= HQ_OK)
	{
		if(pError)
			Log("HLSL shader compile from  memory error: %s" , pError->GetBufferPointer());
		SafeRelease(pBlob);
		SafeRelease(pError);
		this->DeAllocMacrosHLSL(macros);
		HQ_DELETE (sobject);
		return HQ_FAILED;;
	}

	SafeRelease(pBlob);
	SafeRelease(pError);
	this->DeAllocMacrosHLSL(macros);

	if (!this->shaderObjects.AddItem(sobject , pID))
	{
		HQ_DELETE (sobject);
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
#else
	return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
#endif//#if !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
}

HQReturnVal HQShaderManagerD3D11::CreateShaderFromStream(HQShaderType type,
										HQDataReaderStream* dataStream,
										const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										bool isPreCompiled,
										const char* entryFunctionName,
										hq_uint32 *pID)
{
	return this->CreateShaderFromStreamCg(type,dataStream,pDefines,isPreCompiled , entryFunctionName ,false , pID);
}

HQReturnVal HQShaderManagerD3D11::CreateShaderFromMemory(HQShaderType type,
										  const char* pSourceData,
										  const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										  bool isPreCompiled,
										  const char* entryFunctionName,
										  hq_uint32 *pID)
{
	return this->CreateShaderFromMemoryCg(type , pSourceData,pDefines,isPreCompiled,entryFunctionName,false, pID);
}

HQReturnVal HQShaderManagerD3D11::CreateShaderFromStream(HQShaderType type,
								 HQShaderCompileMode compileMode,
								 HQDataReaderStream* dataStream,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 hq_uint32 *pID)
{
	switch (compileMode)
	{
	case HQ_SCM_CG:
		return this->CreateShaderFromStreamCg(type , dataStream , pDefines,false , entryFunctionName,false,pID);
	case HQ_SCM_HLSL_10:
		return this->CreateShaderFromStreamHLSL(type , dataStream , pDefines , entryFunctionName,false , pID);
	case HQ_SCM_CG_DEBUG:
		return this->CreateShaderFromStreamCg(type , dataStream , pDefines,false , entryFunctionName,true,pID);
	case HQ_SCM_HLSL_10_DEBUG:
		return this->CreateShaderFromStreamHLSL(type , dataStream , pDefines , entryFunctionName,true , pID);
	default:
		return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}
}

HQReturnVal HQShaderManagerD3D11::CreateShaderFromMemory(HQShaderType type,
								 HQShaderCompileMode compileMode,
								 const char* pSourceData,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 hq_uint32 *pID)
{
	switch (compileMode)
	{
	case HQ_SCM_CG:
		return this->CreateShaderFromMemoryCg(type , pSourceData , pDefines,false , entryFunctionName,false ,pID);
	case HQ_SCM_HLSL_10:
		return this->CreateShaderFromMemoryHLSL(type , pSourceData , pDefines , entryFunctionName ,false , pID);
	case HQ_SCM_CG_DEBUG:
		return this->CreateShaderFromMemoryCg(type , pSourceData , pDefines,false , entryFunctionName,true ,pID);
	case HQ_SCM_HLSL_10_DEBUG:
		return this->CreateShaderFromMemoryHLSL(type , pSourceData , pDefines , entryFunctionName,true , pID);
	default:
		return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}
}


HQReturnVal HQShaderManagerD3D11::CreateShaderFromByteCodeStream(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 hq_uint32 *pID)
{
	hqubyte8 *pData = NULL;
	hquint32 dataSize = 0;
	
	dataSize = dataStream->TotalSize();
	pData = HQ_NEW hqubyte8[dataSize];
	dataStream->ReadBytes(pData, dataSize, 1);

	HQReturnVal re = this->CreateShaderFromByteCode(type, pData, dataSize, pID);

	return re;
}

HQReturnVal HQShaderManagerD3D11::CreateShaderFromByteCode(HQShaderType type,
									const hqubyte8* byteCodeData,
									hq_uint32 byteCodeLength,
									hq_uint32 *pID)
{
	//clone data buffer
	hqubyte8 *cloneData = HQ_NEW hqubyte8[byteCodeLength];
	memcpy(cloneData, byteCodeData, byteCodeLength);

	return this->CreateShaderFromByteCode(type, cloneData, byteCodeLength, pID);
}

HQReturnVal HQShaderManagerD3D11::CreateShaderFromByteCode(HQShaderType type,
									hqubyte8* byteCodeData,
									hq_uint32 byteCodeLength,
									hq_uint32 *pID)
{
	ID3DBlobImpl *pBlob = HQ_NEW ID3DBlobImpl(byteCodeData, byteCodeLength, true);

	HQShaderObjectD3D11 *sobject= HQ_NEW HQShaderObjectD3D11();

	if(this->CreateShader(type , pBlob , sobject)!= HQ_OK)
	{
		SafeRelease(pBlob);
		HQ_DELETE (sobject);
		return HQ_FAILED;;
	}

	SafeRelease(pBlob);

	if (!this->shaderObjects.AddItem(sobject , pID))
	{
		HQ_DELETE (sobject);
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D11::CreateProgram(hq_uint32 vertexShaderID,
							  hq_uint32 pixelShaderID,
							  hq_uint32 geometryShaderID,
							  const char** uniformParameterNames,
							  hq_uint32 *pID)
{
	if(vertexShaderID==HQ_NOT_USE_VSHADER && pixelShaderID==HQ_NOT_USE_PSHADER && geometryShaderID==HQ_NOT_USE_GSHADER)
		return HQ_FAILED_SHADER_PROGRAM_NEED_SHADEROBJECT;//tất cả đều là fixed function => báo lỗi
	HQSharedPtr<HQShaderObjectD3D11> pVShader = HQSharedPtr<HQShaderObjectD3D11> :: null;
	HQSharedPtr<HQShaderObjectD3D11> pPShader = HQSharedPtr<HQShaderObjectD3D11> :: null;
	HQSharedPtr<HQShaderObjectD3D11> pGShader = HQSharedPtr<HQShaderObjectD3D11> :: null;
	if (vertexShaderID != HQ_NOT_USE_VSHADER)
		pVShader = this->shaderObjects.GetItemPointer(vertexShaderID);
	if (pixelShaderID != HQ_NOT_USE_PSHADER)
		pPShader = this->shaderObjects.GetItemPointer(pixelShaderID);
	if (geometryShaderID != HQ_NOT_USE_GSHADER)
		pGShader = this->shaderObjects.GetItemPointer(geometryShaderID);
	
	if(pVShader == NULL && pPShader == NULL && pGShader == NULL)//tất cả đều null
		return HQ_FAILED_INVALID_ID;

	if(pVShader!=NULL && pVShader->type!=HQ_VERTEX_SHADER)//shader có id <vertexShaderID> không phải vertex shader
		return HQ_FAILED_WRONG_SHADER_TYPE;
	if(pPShader!=NULL && pPShader->type!=HQ_PIXEL_SHADER)//shader có id <pixelShaderID> không phải pixel shader
		return HQ_FAILED_WRONG_SHADER_TYPE;
	if(pGShader!=NULL && pGShader->type!=HQ_GEOMETRY_SHADER)//shader có id <geometryShaderID> không phải geometry shader
		return HQ_FAILED_WRONG_SHADER_TYPE;


	HQShaderProgramD3D11 *pProgram=new HQShaderProgramD3D11();

	//store shaders' pointers
	pProgram->vertexShader = pVShader;
	pProgram->pixelShader = pPShader;
	pProgram->geometryShader = pGShader;

	//store shaders' IDs
	pProgram->vertexShaderID = pProgram->isUseVS()? vertexShaderID : HQ_NOT_USE_VSHADER;
	pProgram->geometryShaderID = pProgram->isUseGS()? geometryShaderID : HQ_NOT_USE_GSHADER;
	pProgram->pixelShaderID = pProgram->isUsePS()? pixelShaderID : HQ_NOT_USE_PSHADER;

	if(!this->AddItem(pProgram,pID))
	{
		HQ_DELETE (pProgram);
		return HQ_FAILED_MEM_ALLOC;
	}
	return HQ_OK;
}


hq_uint32 HQShaderManagerD3D11::GetShader(hq_uint32 programID, HQShaderType shaderType)
{
	HQSharedPtr<HQShaderProgramD3D11> pProgram = this->GetItemPointer(programID);

	switch (shaderType)
	{
	case HQ_VERTEX_SHADER:
		return (pProgram != NULL)? pProgram->vertexShaderID : HQ_NOT_USE_VSHADER;
	case HQ_GEOMETRY_SHADER:
		return (pProgram != NULL)? pProgram->geometryShaderID : HQ_NOT_USE_GSHADER;
	case HQ_PIXEL_SHADER:
		return (pProgram != NULL)? pProgram->pixelShaderID : HQ_NOT_USE_PSHADER;
	}

	return HQ_NOT_USE_VSHADER;
}

/*-----------------------*/

HQReturnVal HQShaderManagerD3D11::SetUniformInt(const char* parameterName,
					 const int* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniform2Int(const char* parameterName,
					 const int* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniform3Int(const char* parameterName,
					 const int* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniform4Int(const char* parameterName,
					 const int* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniformFloat(const char* parameterName,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniform2Float(const char* parameterName,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniform3Float(const char* parameterName,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniform4Float(const char* parameterName,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniformMatrix(const char* parameterName,
					 const HQBaseMatrix4* pMatrices,
					 hq_uint32 numMatrices)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniformMatrix(const char* parameterName,
					 const HQBaseMatrix3x4* pMatrices,
					 hq_uint32 numMatrices)
{
	return HQ_FAILED;
}


/*-------parameter index version---------------*/

hq_uint32 HQShaderManagerD3D11::GetParameterIndex(hq_uint32 programID , 
											const char *parameterName)
{
	return HQ_NOT_AVAIL_ID;
}

HQReturnVal HQShaderManagerD3D11::SetUniformInt(hq_uint32 parameterIndex,
					 const int* pValues,
					 hq_uint32 numElements)
{
	if (this->IsFFEmuActive())
		return this->SetFFRenderState((HQFFRenderState) parameterIndex, pValues);
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniform2Int(hq_uint32 parameterIndex,
					 const int* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniform3Int(hq_uint32 parameterIndex,
					 const int* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniform4Int(hq_uint32 parameterIndex,
					 const int* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniformFloat(hq_uint32 parameterIndex,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniform2Float(hq_uint32 parameterIndex,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniform3Float(hq_uint32 parameterIndex,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniform4Float(hq_uint32 parameterIndex,
					 const hq_float32* pValues,
					 hq_uint32 numElements)
{
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniformMatrix(hq_uint32 parameterIndex,
					 const HQBaseMatrix4* pMatrices,
					 hq_uint32 numMatrices)
{
	if (this->IsFFEmuActive())
		return this->SetFFTransform((HQFFTransformMatrix) parameterIndex, pMatrices);
	return HQ_FAILED;
}

HQReturnVal HQShaderManagerD3D11::SetUniformMatrix(hq_uint32 parameterIndex,
					 const HQBaseMatrix3x4* pMatrices,
					 hq_uint32 numMatrices)
{
	return HQ_FAILED;
}

/*------------------------*/
HQReturnVal HQShaderManagerD3D11::CreateUniformBuffer(hq_uint32 size , void *initData , bool isDynamic , hq_uint32 *pBufferIDOut)
{
	D3D11_BUFFER_DESC cbDesc;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	if (isDynamic)
	{
		cbDesc.Usage = D3D11_USAGE_DYNAMIC;
		cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	}
	else
	{
#if HQ_NON_DYNAMIC_U_BUFFER_CAN_BE_UPDATED
		cbDesc.Usage = D3D11_USAGE_DEFAULT;
		cbDesc.CPUAccessFlags = 0;
#else
		cbDesc.Usage = D3D11_USAGE_IMMUTABLE;
		cbDesc.CPUAccessFlags = 0;
#endif
	}
    cbDesc.MiscFlags = 0;
    cbDesc.ByteWidth = size;
	cbDesc.StructureByteStride = 0;

	//make sure that buffer size is multiples of 16
	UINT remain = (cbDesc.ByteWidth % 16);
	if (remain != 0)
	{
		cbDesc.ByteWidth += 16 - remain;
	}

	ID3D11Buffer *pNewD3DBuffer;

	D3D11_SUBRESOURCE_DATA d3dinitdata;
	d3dinitdata.pSysMem = initData;
	
	if(FAILED(pD3DDevice->CreateBuffer(&cbDesc , (initData != NULL)? &d3dinitdata :  NULL , &pNewD3DBuffer)))
		return HQ_FAILED;
	HQShaderConstBufferD3D11 *pNewBuffer = HQ_NEW HQShaderConstBufferD3D11(pNewD3DBuffer, isDynamic, size);
	if(!shaderConstBuffers.AddItem(pNewBuffer , pBufferIDOut))
	{
		HQ_DELETE (pNewBuffer);
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}
HQReturnVal HQShaderManagerD3D11::SetUniformBuffer(hq_uint32 index ,  hq_uint32 bufferID )
{
	HQSharedPtr<HQShaderConstBufferD3D11> pBuffer = shaderConstBuffers.GetItemPointer(bufferID);
	
	hq_uint32 slot = index & 0x0fffffff;
	
	if (slot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)
		return HQ_FAILED;

	hq_uint32 shaderStage = index & 0xf0000000;

	HQSharedPtr<HQShaderConstBufferD3D11>* ppCurrentBuffer;

	switch(shaderStage)
	{
	case HQ_VERTEX_SHADER:
		ppCurrentBuffer = this->uBufferSlots[0] + slot;
		if (pBuffer != *ppCurrentBuffer)
		{
			if (pBuffer != NULL)
				pD3DContext->VSSetConstantBuffers(slot , 1 , &pBuffer->pD3DBuffer);
			/*TO DO: crash
			else
				pD3DContext->VSSetConstantBuffers(slot , 1 , NULL);*/
			*ppCurrentBuffer = pBuffer;
		}
		break;
	case HQ_GEOMETRY_SHADER:
		ppCurrentBuffer = this->uBufferSlots[1] + slot;
		if (pBuffer != *ppCurrentBuffer)
		{
			if (pBuffer != NULL)
				pD3DContext->GSSetConstantBuffers(slot , 1 , &pBuffer->pD3DBuffer);
			/*TO DO: crash
			else
				pD3DContext->GSSetConstantBuffers(slot , 1 , NULL)*/;
			*ppCurrentBuffer = pBuffer;
		}
		break;
	case HQ_PIXEL_SHADER:
		ppCurrentBuffer = this->uBufferSlots[2] + slot;
		if (pBuffer != *ppCurrentBuffer)
		{
			if (pBuffer != NULL)
				pD3DContext->PSSetConstantBuffers(slot , 1 , &pBuffer->pD3DBuffer);
			/*TO DO: crash
			else
				pD3DContext->PSSetConstantBuffers(slot , 1 , NULL)*/;
			*ppCurrentBuffer = pBuffer;
		}
		break;
	}

	return HQ_OK;
}
HQReturnVal HQShaderManagerD3D11::MapUniformBuffer(hq_uint32 bufferID , void **ppData)
{
	HQShaderConstBufferD3D11* pBuffer = shaderConstBuffers.GetItemRawPointer(bufferID);

#if defined _DEBUG || defined DEBUG

	if(pBuffer == NULL)
		return HQ_FAILED;
	
	if (!ppData)
		return HQ_FAILED;
#endif
	D3D11_MAPPED_SUBRESOURCE mappedSubResource;
	pD3DContext->Map(pBuffer->pD3DBuffer , 0 , D3D11_MAP_WRITE_DISCARD , 0 , &mappedSubResource);
	*ppData = mappedSubResource.pData;

	return HQ_OK;
}
HQReturnVal HQShaderManagerD3D11::UnmapUniformBuffer(hq_uint32 bufferID)
{
	HQShaderConstBufferD3D11* pBuffer = shaderConstBuffers.GetItemRawPointer(bufferID);
	
#if defined _DEBUG || defined DEBUG	
	if(pBuffer == NULL)
		return HQ_FAILED;
#endif

	pD3DContext->Unmap(pBuffer->pD3DBuffer , 0);

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D11::UpdateUniformBuffer(hq_uint32 bufferID, const void * pData)
{
	HQShaderConstBufferD3D11* pBuffer = shaderConstBuffers.GetItemRawPointer(bufferID);
#if defined _DEBUG || defined DEBUG	
	if (pBuffer == NULL)
		return HQ_FAILED;
	if (pBuffer->isDynamic == true)
	{
		this->Log("Error : dynamic buffer can't be updated using UpdateUniformBuffer method!");
		return HQ_FAILED_NOT_DYNAMIC_RESOURCE;
	}
#endif

#if !HQ_NON_DYNAMIC_U_BUFFER_CAN_BE_UPDATED
	//update entire buffer
	D3D11_MAPPED_SUBRESOURCE mappedSubResource;
	if (SUCCEEDED(pD3DContext->Map(pBuffer->pD3DBuffer , 0 , D3D11_MAP_WRITE_DISCARD , 0 , &mappedSubResource)))
	{
		//copy
		memcpy(mappedSubResource.pData, pData, pBuffer->size);

		pD3DContext->Unmap(pBuffer->pD3DBuffer , 0);
	}
#else
	//update entire buffer
	pD3DContext->UpdateSubresource(pBuffer->pD3DBuffer, 0, NULL, pData, 0, 0);
#endif

	return HQ_OK;
}

/*--------------------------------*/
void HQShaderManagerD3D11::BeginClearViewport()
{
	pD3DContext->VSSetShader((ID3D11VertexShader*)this->clearVShader.pD3DShader , NULL , 0);
	pD3DContext->GSSetShader((ID3D11GeometryShader*)this->clearGShader.pD3DShader, NULL , 0);
	pD3DContext->PSSetShader((ID3D11PixelShader*)this->clearPShader.pD3DShader, NULL , 0);
#if !HQ_D3D_CLEAR_VP_USE_GS
	HQSharedPtr<HQShaderConstBufferD3D11> pBuffer = shaderConstBuffers.GetItemPointerNonCheck(this->clearShaderParameters);
	pD3DContext->VSSetConstantBuffers(0, 1, &pBuffer->pD3DBuffer);
#endif
}

#if !HQ_D3D_CLEAR_VP_USE_GS
void HQShaderManagerD3D11::ChangeClearVPParams(HQColor clearColor, hqfloat32 clearDepth)
{
	ClearBufferParameters *parameters = NULL;

	this->MapUniformBuffer(this->clearShaderParameters, (void**)&parameters);
	parameters->color = clearColor;
	parameters->depth = clearDepth;
	this->UnmapUniformBuffer(this->clearShaderParameters);
}
#endif

void HQShaderManagerD3D11::EndClearViewport()
{
	if (this->activeVShader != NULL)
		pD3DContext->VSSetShader((ID3D11VertexShader*)this->activeVShader->pD3DShader, NULL , 0);
	else
		pD3DContext->VSSetShader(NULL , NULL , 0);

	if (this->activeGShader != NULL)
		pD3DContext->GSSetShader((ID3D11GeometryShader*)this->activeGShader->pD3DShader, NULL , 0);
	else
		pD3DContext->GSSetShader(NULL, NULL , 0);
	
	if (this->activePShader != NULL)
		pD3DContext->PSSetShader((ID3D11PixelShader*)this->activePShader->pD3DShader, NULL , 0);
	else
		pD3DContext->PSSetShader(NULL, NULL , 0);

#if !HQ_D3D_CLEAR_VP_USE_GS
	if (this->uBufferSlots[0][0] != NULL)
		pD3DContext->VSSetConstantBuffers(0, 1, &this->uBufferSlots[0][0]->pD3DBuffer);
	else
	{
		/*
		TO DO: crash
		ID3D11Buffer *pNULL = NULL;
		pD3DContext->VSSetConstantBuffers(0, 1, &pNULL);
		*/
	}
#endif
}

