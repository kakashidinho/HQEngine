/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D9PCH.h"
#include "HQShaderD3D9.h"

#include "HQShaderVarParserD3D9.h"

#include <string.h>
#include <sstream>

#define NUM_PREDEFINED_ARGS 16


static char *semantics[17]= {
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

#define PREPROCESS_BY_CG 0
#if PREPROCESS_BY_CG
static const char g_preprocessOption[] = "-E";
static const char g_compileOption[] = "-d3d";
#endif

static bool g_disableErrorCallback = false;

HQShaderManagerD3D9* g_pShaderMan=NULL;

//returned pointer must be deleted manually. last character in returned data is zero
static void * ReadFileData(HQFileManager* fileManager, const char * fileName, hquint32 &size){
	HQDataReaderStream * stream = fileManager->OpenFileForRead(fileName);
	if (!stream)
		return NULL;
	hqubyte8* pData;
	pData = HQ_NEW hqubyte8[stream->TotalSize() + 1];

	stream->ReadBytes(pData, stream->TotalSize(), 1);
	pData[stream->TotalSize()] = '\0';


	//return size
	size = stream->TotalSize() + 1;

	stream->Release();

	//return data
	return pData;
}
/*---------------------cg include handler----------------*/
static void cgIncludeCallback(CGcontext context, const char *filename)
{
	if (g_pShaderMan->GetIncludeFileManager() == NULL)
		return;
	hquint32 size;
	void *pData = ReadFileData(g_pShaderMan->GetIncludeFileManager(), filename, size);

	if (pData)
	{
		cgSetCompilerIncludeString(context, filename, (const char*)pData);
		delete ((hqubyte8*)pData);
	}
}

/*----------------cg error callback----------------*/
static void cgErrorCallBack(void)
{
	if (g_disableErrorCallback)
		return;
	CGerror err=cgGetError();
	if(err == cgD3D9Failed)
	{
		HRESULT hr=cgD3D9GetLastError();
		g_pShaderMan->Log("D3D9 error %s",cgD3D9TranslateHRESULT(hr));
	}
	else
		g_pShaderMan->Log("%s",cgGetErrorString(err));
}


#if !HQ_TRANSLATE_CG_TO_HLSL
/*--------------class HQConstantTableD3D9 -------*/
struct HQConstantTableD3D9  {
public:
	struct ConstInfo {
		DWORD regIndex;
		hquint32 numVectors;
		bool isInteger;
	};
	struct BufferSlotInfo{
		hquint32 index;
		HQLinkedList<ConstInfo> constants;//list of constants that consume datat from this buffer slot
	};

	typedef HQLinkedList<BufferSlotInfo > BufferSlotList;

	HQConstantTableD3D9(const char *cgCompiledCode, HQShaderVarParserD3D9 &preprocessedInfo);

	void Release() { delete this; }//identical method with D3D interfaces

	void AddConstant(const std::string & name, DWORD regIndex);
	DWORD GetConstantRegIndex(const char *name, bool& found);

	BufferSlotList constBufferSlotList;//list of constant buffer slots associated with this shader
protected:
	~HQConstantTableD3D9() {}

	void InitConstBufferSlotList(HQShaderVarParserD3D9 &preprocessedInfo);

	hash_map_type<std::string, DWORD> table;
};
#endif

/*--------------------HQShaderManagerD3D9::BufferSlotInfo---------------------*/

struct HQShaderManagerD3D9::BufferSlotInfo{
	BufferSlotInfo() : dirtyFlags(1) {}

	hquint32 dirtyFlags;//dirty flags
	HQSharedPtr<HQShaderConstBufferD3D9> buffer;
	HQShaderConstBufferD3D9::BufferSlotList::LinkedListNodeType* bufferLink;//this link can be used for fast removing the slot from buffer's bound slots
};


#if !HQ_TRANSLATE_CG_TO_HLSL
/*--------------class HQConstantTableD3D9 implementation -------*/
HQConstantTableD3D9::HQConstantTableD3D9(const char *compiledCgCode, HQShaderVarParserD3D9& preproccessedConstInfo) {
	/*example of constant in compiled code
	* //var float3x4 rotation :  : c[0], 3 : 1 : 1
	*/
	enum State{
		BEGIN_STATE,
		FIRST_SLASH,
		COMMENT,
		VAR1,
		VAR2,
		VAR3,
		VAR,
		EAT_INPUT,
		TYPE1,
		TYPE,
		TYPE_END,
		NAME,
		NAME_END,
		COLON1,
		COLON2_1,
		COLON2_1_TOKEN,
		COLON2_1_TOKEN_END_WITH_COLON,
		COLON2_1_COMMA,
		COLON2_2,
		COLON3,
		COLON4,
		COLON4_TOKEN
	};

#define WHITE_SPACE(character) (character == ' ' || character == '\t' || character == '\r')

	State state = BEGIN_STATE;
	std::stringstream tokenStr;
	std::string name;
	DWORD regIndex = 0xffffffff;
	int i = 0;
	char c = 0; 
	while ((c = compiledCgCode[i++]) != '\0')
	{
		switch (state)
		{
		case BEGIN_STATE:
			if (c == '/')
				state = FIRST_SLASH;
			break;
		case EAT_INPUT:
			if (c == '\n')
				state = BEGIN_STATE;
			break;
		case FIRST_SLASH:
			if (c == '/')
				state = COMMENT;
			else
				state = EAT_INPUT;
			break;
		case COMMENT:
			if (c == '\n')
				state = BEGIN_STATE;
			else if (c == 'v')
				state = VAR1;
			break;
		case VAR1:
			if (c == 'a')
				state = VAR2;
			else
				state = EAT_INPUT;
			break;
		case VAR2:
			if (c == 'r')
				state = VAR3;
			else
				state = EAT_INPUT;
			break;
		case VAR3:
			if (WHITE_SPACE(c))
				state = VAR;
			else
				state = EAT_INPUT;
			break;
		case VAR:
			if (c == '\n')
				state = BEGIN_STATE;
			else if (!WHITE_SPACE(c))
			{
				tokenStr.str("");
				tokenStr << c;
				state = TYPE1;
			}
			break;
		case TYPE1:
			if (c == '\n')
				state = BEGIN_STATE;
			else if (!WHITE_SPACE(c))
				state = TYPE;
			break;
		case TYPE:
			if (WHITE_SPACE(c))
				state = TYPE_END;
			else if (c == '\n')
				state = BEGIN_STATE;
			break;
		case TYPE_END:
			if (c == '\n')
				state = BEGIN_STATE;
			else if (!WHITE_SPACE(c))
			{
				state = NAME;
				tokenStr.str("");
				tokenStr << c;
			}
			break;
		case NAME:
			if (c == '\n')
				state = BEGIN_STATE;
			else if (!WHITE_SPACE(c))
			{
				tokenStr << c;
			}
			else
			{
				name = tokenStr.str();
				state = NAME_END;
			}

			break;
		case NAME_END:
			if (c == '\n')
				state = BEGIN_STATE;
			else if (c == ':')
				state = COLON1;

			break;
		case COLON1:
			if (c == '\n')
				state = BEGIN_STATE;
			else if (c == ':')
			{
				tokenStr.str("");//clear token string
				state = COLON2_1;
			}

			break;
		case COLON2_1:
			if (c == ',')
			{
				state = COLON2_1_COMMA;
			}
			else if (c == ':')
				state = EAT_INPUT;
			else if (!WHITE_SPACE(c))
			{
				state = COLON2_1_TOKEN;
				tokenStr << c;
			}
			else if (c == '\n')
				state = BEGIN_STATE;
			break;
		case COLON2_1_TOKEN:
			if (c == '\n')
				state = BEGIN_STATE;
			else if (c == ':')
				state = COLON2_1_TOKEN_END_WITH_COLON;
			else if (c == ',')
			{
				state = COLON2_1_COMMA;
			}
			else 
				tokenStr << c;
			break;
		case COLON2_1_COMMA: case COLON2_1_TOKEN_END_WITH_COLON:
			{
				--i;//re-read this input character again
				std::string token = tokenStr.str();
				if (sscanf(token.c_str(), "c[%d]", &regIndex) == 1)
				{
					if (state == COLON2_1_TOKEN_END_WITH_COLON)
						state = COLON3;
					else
						state = COLON2_2;
				}
				else
					state = EAT_INPUT;
			}
			break;
		case COLON2_2:
			if (c == ':')
				state = COLON3;
			else if (c == '\n')
				state = BEGIN_STATE;
			break;
		case COLON3:
			if (c == ':')
			{
				state = COLON4;
				tokenStr.str("");//clear token
			}
			else if (c == '\n')
				state = BEGIN_STATE;
			break;
		case COLON4:
			if (c == '\n')
				state = BEGIN_STATE;
			else if (!WHITE_SPACE(c))
			{
				state = COLON4_TOKEN;
				tokenStr << c;
			}
			break;
		case COLON4_TOKEN:
			if (c == '\n' || WHITE_SPACE(c))
			{
				if (tokenStr.str().compare("1") == 0)
				{
					//found constant variable
					if (regIndex != 0xfffffff)
						AddConstant(name, regIndex);
				}
				state = BEGIN_STATE;
				tokenStr.str("");//clear token
			}
			else
				tokenStr << c;
			break;
			
		}//switch (state)
	}//while ((c = compiledCgCode[0]) != '\0')
	

	this->InitConstBufferSlotList(preproccessedConstInfo);
}

void HQConstantTableD3D9::InitConstBufferSlotList(HQShaderVarParserD3D9& preprocessedInfo)
{
	HQLinkedList<UniformBlock>::Iterator ite;
	for (preprocessedInfo.uniformBlocks.GetIterator(ite); !ite.IsAtEnd(); ++ite)
	{
		if (ite->index < 16)//ignore out of bound buffer index
		{
			BufferSlotInfo* pUsedSlot = NULL;
			HQLinkedList<UniformBlkElemInfo>::Iterator elemIte;
			for (ite->blockElems.GetIterator(elemIte); !elemIte.IsAtEnd(); ++elemIte)
			{
				//find in the constant table
				bool found;
				DWORD regIndex = this->GetConstantRegIndex(elemIte->name.c_str(), found);
				if (found)//found
				{
					if (pUsedSlot == NULL)
					{
						//at least one constant is valid, so we added a new buffer slot
						BufferSlotInfo emptyItem;
						pUsedSlot = &this->constBufferSlotList.PushBack(emptyItem)->m_element;
						pUsedSlot->index = ite->index;
					}
					ConstInfo constInfo;
					constInfo.regIndex = regIndex;
					constInfo.isInteger = elemIte->integer;
					constInfo.numVectors = elemIte->row * elemIte->arraySize;
					pUsedSlot->constants.PushBack(constInfo);//insert this constant register to the list of constants that consume this buffer slot
				}
			}//for (ite->blockElems.GetIterator(elemIte); !elemIte.IsAtEnd(); ++elemIte)
		}//if (ite->index < 16)
	}//for (preprocessedInfo.uniformBlocks.GetIterator(ite); !ite.IsAtEnd(); ++ite)
}

void HQConstantTableD3D9::AddConstant(const std::string &name, DWORD regIndex)
{
	this->table[name] = regIndex;
	size_t bracket_pos = name.find('[');
	if (bracket_pos != std::string::npos)
	{
		int index = -1;
		if (sscanf(name.c_str() + bracket_pos, "[%d]", &index) == 1 && index == 0)
		{
			std::string array_name = name.substr(0, bracket_pos);//array's register index should be the same as its first element
			this->table[array_name] = regIndex;
		}
	}
}

DWORD HQConstantTableD3D9::GetConstantRegIndex(const char *name, bool& found)
{
	hash_map_type<std::string, DWORD>::const_iterator ite = this->table.find(name);
	if (ite == this->table.end())
	{
		found = false;
		return 0xffffffff;
	}
	else
	{
		found = true;
		return ite->second;
	}


}

#endif

/*------------HQShaderObjectD3D9---------------*/
HQShaderObjectD3D9::HQShaderObjectD3D9()
{
	vshader = NULL;
	consTable = NULL;
}
HQShaderObjectD3D9::~HQShaderObjectD3D9()
{
	if (consTable)
		consTable->Release();
	if (vshader)
		vshader->Release();
}

/*----------HQShaderConstBufferD3D9-----------*/
HQShaderConstBufferD3D9::HQShaderConstBufferD3D9(HQSysMemBuffer::Listener* listener, bool isDynamic, hq_uint32 size)
: HQSysMemBuffer(listener),
boundSlots(HQ_NEW HQPoolMemoryManager(sizeof(BufferSlotList::LinkedListNodeType), 32))
{
	//must allocate a buffer with size is multiple of 16 byte
	size_t realSize = size;
	size_t remain = size % 16;
	if (remain > 0)
		realSize += (16 - remain);

	this->AllocRawBuffer(realSize);//allocate raw buffer with <realSize>
	this->size = size;//but only <size> bytes are mappable

	this->isDynamic = isDynamic;
}

HQShaderConstBufferD3D9::~HQShaderConstBufferD3D9()
{
}

/*---------HQShaderManagerD3D9--------------*/
HQShaderManagerD3D9::HQShaderManagerD3D9(LPDIRECT3DDEVICE9 pD3DDevice,HQLogStream* logFileStream,bool flushLog)
: HQLoggableObject(logFileStream , "D3D9 Shader Manager :" , flushLog , 1024)
{
	this->pD3DDevice=pD3DDevice;

	this->vshaderConstSlots = HQ_NEW BufferSlotInfo[16];
	this->pshaderConstSlots = HQ_NEW BufferSlotInfo[16];

	this->cgContext=cgCreateContext();
	
	cgD3D9SetDevice(pD3DDevice);

#if HQ_TRANSLATE_CG_TO_HLSL
	this->cgVertexProfile = CG_PROFILE_HLSLV;
	this->cgPixelProfile = CG_PROFILE_HLSLF;
#else
	this->cgVertexProfile=cgD3D9GetLatestVertexProfile();
	this->cgPixelProfile=cgD3D9GetLatestPixelProfile();
#endif
	
	//fixed function controlling values
	this->firstStageOp[0] = D3DTOP_MODULATE;
	this->firstStageOp[1] = D3DTOP_SELECTARG1;

	g_pShaderMan=this;
#if defined(DEBUG)||defined(_DEBUG)
	cgSetErrorCallback(cgErrorCallBack);
#endif

	cgSetCompilerIncludeCallback(this->cgContext, cgIncludeCallback);

	cgSetContextBehavior(this->cgContext, CG_BEHAVIOR_3100); 

	this->includeFileManager = NULL;

	const char* version = cgGetString(CG_VERSION);
	Log("Init done! Cg library version is %s", version);
}

HQShaderManagerD3D9::~HQShaderManagerD3D9()
{
	this->RemoveAllResource();
	delete[] this->vshaderConstSlots;
	delete[] this->pshaderConstSlots;
	cgD3D9SetDevice(0);
	cgDestroyContext(cgContext);
	Log("Released!");
	g_pShaderMan=NULL;
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
HQReturnVal HQShaderManagerD3D9::ActiveProgram(HQShaderProgram* programID)
{
	if (programID == NULL)
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
				this->MarkAllBufferSlotsDirtyForVShader(activeVShader.GetRawPointer());
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
				this->MarkAllBufferSlotsDirtyForPShader(activePShader.GetRawPointer());
			}
		}
		activeProgram=pProgram;
	}
	return HQ_OK;
}
/*------------------------*/
HQReturnVal HQShaderManagerD3D9::RemoveProgram(HQShaderProgram* programID)
{
	HQSharedPtr<HQShaderProgramD3D9> pProgram = this->GetItemPointer(programID);
	if(pProgram == NULL)
		return HQ_FAILED;
	if(pProgram==activeProgram)
	{
		this->ActiveProgram(NULL);
	}
	this->Remove(programID);
	return HQ_OK;
}

void HQShaderManagerD3D9::RemoveAllProgram()
{
	this->ActiveProgram(NULL);
	
	this->RemoveAll();
}

void HQShaderManagerD3D9::RemoveAllResource()
{
	this->RemoveAllUniformBuffers();
	this->RemoveAllProgram();
	this->RemoveAllShader();
}


HQReturnVal HQShaderManagerD3D9::RemoveShader(HQShaderObject * shaderID)
{return (HQReturnVal)this->shaderObjects.Remove(shaderID);
}

void HQShaderManagerD3D9::RemoveAllShader()
{
	this->shaderObjects.RemoveAll();
}
/*--------------------------*/

char ** HQShaderManagerD3D9::GetCompileArguments(const HQShaderMacro * pDefines, bool debug)
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
	//first argument is reserved for preprocess option
	char ** args = new char *[numDefines + NUM_PREDEFINED_ARGS + 3];
	for (int i = 0 ; i < NUM_PREDEFINED_ARGS ; ++i)
		args[i] = semantics[i];

	//optimization's option
	args[numDefines + NUM_PREDEFINED_ARGS] = HQ_NEW char[4];
	if (debug)
	{
#if 1
		strcpy(args[numDefines + NUM_PREDEFINED_ARGS], "-O0");
		args[numDefines + NUM_PREDEFINED_ARGS + 1] = HQ_NEW char[7];
		strcpy(args[numDefines + NUM_PREDEFINED_ARGS + 1], "-debug");
#else
		args[numDefines + NUM_PREDEFINED_ARGS] = NULL;
		args[numDefines + NUM_PREDEFINED_ARGS + 1] = NULL;
#endif
	}
	else
	{
		strcpy(args[numDefines + NUM_PREDEFINED_ARGS], "-O3");
		args[numDefines + NUM_PREDEFINED_ARGS + 1] = NULL;
	}
	args[numDefines + NUM_PREDEFINED_ARGS + 2] = NULL;//last element must be null

	pD = pDefines;
	
	int i = 0;
	while (pD->name != NULL && pD->definition != NULL)
	{
		nameLen = strlen(pD->name);
		definitionLen = strlen(pD->definition);
		
	
		if (definitionLen != 0)
		{
			args[NUM_PREDEFINED_ARGS + i] = new char[nameLen + definitionLen + 4];
			sprintf(args[NUM_PREDEFINED_ARGS + i] , "-D%s=%s" , pD->name , pD->definition);
		}
		else
		{
			args[NUM_PREDEFINED_ARGS + i] = new char[nameLen + 4];
			sprintf(args[NUM_PREDEFINED_ARGS + i] , "-D%s" , pD->name);
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
	char ** ppD = ppC + NUM_PREDEFINED_ARGS;
	while(*ppD != NULL)
	{
		SafeDeleteArray(*ppD);
		ppD ++;
	}

	SafeDeleteArray(ppC);
}
/*---------------------------*/

HQReturnVal HQShaderManagerD3D9::CreateShaderFromStreamEx(HQShaderType type,
									 HQDataReaderStream* dataStream,
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 const HQShaderMacro *pDefines,
									 const char **args,
									 bool debugMode , 
									 HQShaderObject **pID)
{
	const char nullStreamName[] = "";
	const char *streamName = dataStream->GetName() != NULL? dataStream->GetName(): nullStreamName;
#if defined DEBUG || defined _DEBUG
	//debugMode = true; //force debugging
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

	//obtain variables' infos from source
	HQShaderVarParserD3D9 preprocessedConstInfo(streamContent, pDefines, this->includeFileManager);
	
	const char *actualSource = preprocessedConstInfo.GetPreprocessedSrc();
	if (actualSource == NULL)
		actualSource = streamContent;

	//now compile
	CGprogram cgprogram = cgCreateProgram(this->cgContext, CG_SOURCE, actualSource, profile, entryFunctionName,args);
	
	//clear include data
	cgSetCompilerIncludeString(this->cgContext, NULL, NULL);

	delete[] streamContent;

	if (cgprogram == NULL )
	{
		this->Log("Shader compile from stream %s error !",streamName);
		return HQ_FAILED;
	}
	
	//now compile d3d code

	
	HRESULT hr;
	//check for errors
	ID3DXBuffer* byteCode = NULL;
	ID3DXBuffer* errorMsg = NULL;
	DWORD compileFlags = debugMode ? (D3DXSHADER_DEBUG) : 0;
#if HQ_TRANSLATE_CG_TO_HLSL
	compileFlags |= debugMode? (D3DXSHADER_SKIPOPTIMIZATION): 0;
	compileFlags |= D3DXSHADER_ENABLE_BACKWARDS_COMPATIBILITY;
#endif
	const char * d3dprofile = type == HQ_VERTEX_SHADER? D3DXGetVertexShaderProfile(pD3DDevice): D3DXGetPixelShaderProfile(pD3DDevice);

	//get output from cg
	const char * compiled_src = cgGetProgramString(cgprogram, CG_COMPILED_PROGRAM);
	bool writeToTemp = false;
#if HQ_TRANSLATE_CG_TO_HLSL
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
					NULL
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
				NULL
				);
	}
#else //#if HQ_TRANSLATE_CG_TO_HLSL
	hr = D3DXAssembleShader(compiled_src,
		strlen(compiled_src),
		NULL,
		NULL,
		compileFlags,
		&byteCode,
		&errorMsg
		);
#endif

	if (FAILED(hr))
	{
		if (errorMsg)
			this->Log("Shader compile from stream %s error ! Error message \"%s\"",streamName, errorMsg->GetBufferPointer());
		else
			this->Log("Shader compile from stream %s error !", streamName);

		SafeRelease(byteCode);
		SafeRelease(errorMsg); 
		cgDestroyProgram(cgprogram);//no thing more to do with cg program
		return HQ_FAILED;
	}
	//succeeded
	HQShaderObjectD3D9 *sobject= new HQShaderObjectD3D9();
	sobject->type=type;

	if (type == HQ_VERTEX_SHADER)
		pD3DDevice->CreateVertexShader((DWORD*) byteCode->GetBufferPointer(), &sobject->vshader);
	else
		pD3DDevice->CreatePixelShader((DWORD*) byteCode->GetBufferPointer(), &sobject->pshader);
#if HQ_TRANSLATE_CG_TO_HLSL
	D3DXGetShaderConstantTableEx((DWORD*)byteCode->GetBufferPointer(), D3DXCONSTTABLE_LARGEADDRESSAWARE, &sobject->consTable);
#else
	sobject->consTable = HQ_NEW HQConstantTableD3D9(compiled_src, preprocessedConstInfo);
#endif
	SafeRelease(byteCode);
	SafeRelease(errorMsg);
	cgDestroyProgram(cgprogram);//no thing more to do with cg program

	

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
									 const HQShaderMacro *pDefines,
									 const char **args,
									 bool debugMode,
									 HQShaderObject ** pID)
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

	//obtain variables' infos from source
	HQShaderVarParserD3D9 preprocessedConstInfo(pSourceData, pDefines, this->includeFileManager);

	const char *actualSource = preprocessedConstInfo.GetPreprocessedSrc();
	if (actualSource == NULL)
		actualSource = pSourceData;
	
	//now compile
	CGprogram cgprogram = cgCreateProgram(this->cgContext, CG_SOURCE, actualSource, profile, entryFunctionName,args);
	//clear include data
	cgSetCompilerIncludeString(this->cgContext, NULL, NULL);

	if (cgprogram == NULL )
	{
		this->Log("Shader compile from memory error !");
		return HQ_FAILED;
	}
	
	//now compile d3d code

	HRESULT hr;
	//check for errors
	ID3DXBuffer* byteCode = NULL;
	ID3DXBuffer* errorMsg = NULL;
	DWORD compileFlags = debugMode? (D3DXSHADER_DEBUG): 0;
	const char * d3dprofile = type == HQ_VERTEX_SHADER? D3DXGetVertexShaderProfile(pD3DDevice): D3DXGetPixelShaderProfile(pD3DDevice);

	//get output from cg
	const char * compiled_src = cgGetProgramString(cgprogram, CG_COMPILED_PROGRAM);
#if HQ_TRANSLATE_CG_TO_HLSL
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
			NULL
			);
#else
	hr = D3DXAssembleShader(compiled_src,
		strlen(compiled_src),
		NULL,
		NULL,
		compileFlags,
		&byteCode,
		&errorMsg
		);
#endif

	if (FAILED(hr))
	{
		if (errorMsg)
			this->Log("Shader compile from memory error ! Error message \"%s\"",errorMsg->GetBufferPointer());
		else
			this->Log("Shader compile from memory error !");

		SafeRelease(byteCode);
		SafeRelease(errorMsg);
		cgDestroyProgram(cgprogram);//no more thing to do with cg program
		return HQ_FAILED;
	}
	//succeeded
	HQShaderObjectD3D9 *sobject= new HQShaderObjectD3D9();
	sobject->type=type;

	if (type == HQ_VERTEX_SHADER)
		pD3DDevice->CreateVertexShader((DWORD*) byteCode->GetBufferPointer(), &sobject->vshader);
	else
		pD3DDevice->CreatePixelShader((DWORD*) byteCode->GetBufferPointer(), &sobject->pshader);

#if HQ_TRANSLATE_CG_TO_HLSL
	D3DXGetShaderConstantTableEx((DWORD*)byteCode->GetBufferPointer(), D3DXCONSTTABLE_LARGEADDRESSAWARE, &sobject->consTable);
#else
	sobject->consTable = HQ_NEW HQConstantTableD3D9(compiled_src, preprocessedConstInfo);
#endif
	SafeRelease(byteCode);
	SafeRelease(errorMsg);
	cgDestroyProgram(cgprogram);//no more thing to do with cg program

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
										HQShaderObject **pID)
{
	char ** args = this->GetCompileArguments(pDefines, false);
	HQReturnVal re = this->CreateShaderFromStreamEx(type,dataStream,isPreCompiled , entryFunctionName , pDefines, (const char**)args ,false , pID);
	this->DeAlloc(args);
	return re;
}

HQReturnVal HQShaderManagerD3D9::CreateShaderFromMemory(HQShaderType type,
										  const char* pSourceData,
										  const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										  bool isPreCompiled,
										  const char* entryFunctionName,
										  HQShaderObject **pID)
{
	char ** args = this->GetCompileArguments(pDefines, false);
	HQReturnVal re =  this->CreateShaderFromMemoryEx(type , pSourceData,isPreCompiled,entryFunctionName, pDefines, (const char**)args , false, pID);
	this->DeAlloc(args);
	return re;
}

HQReturnVal HQShaderManagerD3D9::CreateShaderFromStream(HQShaderType type,
								 HQShaderCompileMode compileMode,
								 HQDataReaderStream* dataStream,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 HQShaderObject **pID)
{
	HQReturnVal re;
	char ** args = NULL;
	switch (compileMode)
	{
	case HQ_SCM_CG:
		args = this->GetCompileArguments(pDefines, false);
		re = this->CreateShaderFromStreamEx(type , dataStream,false , entryFunctionName, pDefines, (const char**)args , false,pID);
		break;
	case HQ_SCM_CG_DEBUG:
		args = this->GetCompileArguments(pDefines, true);
		re = this->CreateShaderFromStreamEx(type , dataStream,false , entryFunctionName, pDefines, (const char**)args , true,pID);
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
								 HQShaderObject **pID)
{
	HQReturnVal re;
	char ** args = NULL;
	switch (compileMode)
	{
	case HQ_SCM_CG:
		args = this->GetCompileArguments(pDefines, false);
		re = this->CreateShaderFromMemoryEx(type , pSourceData,false , entryFunctionName, pDefines, (const char**)args ,false,pID);
	case HQ_SCM_CG_DEBUG:
		args = this->GetCompileArguments(pDefines, true);
		re = this->CreateShaderFromMemoryEx(type , pSourceData,false , entryFunctionName, pDefines, (const char**)args , true,pID);
	default:
		re = HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}
	this->DeAlloc(args);
	return re;
}


HQReturnVal HQShaderManagerD3D9::CreateProgram(HQShaderObject * vertexShaderID,
	HQShaderObject * pixelShaderID,
	HQShaderObject * geometryShaderID,
	HQShaderProgram **pID)
{
	if(vertexShaderID==NULL && pixelShaderID==NULL && geometryShaderID==NULL)
		return HQ_FAILED_SHADER_PROGRAM_NEED_SHADEROBJECT;//tất cả đều là fixed function => báo lỗi
	
	HQSharedPtr<HQShaderObjectD3D9> pVShader = HQSharedPtr<HQShaderObjectD3D9> :: null;
	HQSharedPtr<HQShaderObjectD3D9> pPShader = HQSharedPtr<HQShaderObjectD3D9> :: null;
	if (vertexShaderID != NULL)
		pVShader = this->shaderObjects.GetItemPointer(vertexShaderID);
	if (pixelShaderID != NULL)
		pPShader = this->shaderObjects.GetItemPointer(pixelShaderID);
	
	if(pVShader == NULL || pPShader == NULL)
		return HQ_FAILED_INVALID_ID;
	if(vertexShaderID!=NULL && pVShader->type!=HQ_VERTEX_SHADER)//shader có id <vertexShaderID> không phải vertex shader
		return HQ_FAILED_WRONG_SHADER_TYPE;
	if(pixelShaderID!=NULL && pPShader->type!=HQ_PIXEL_SHADER)//shader có id <pixelShaderID> không phải pixel shader
		return HQ_FAILED_WRONG_SHADER_TYPE;

	HQShaderProgramD3D9 *pShader=new HQShaderProgramD3D9();

	//store shaders' pointers
	pShader->vertexShader = pVShader;
	pShader->pixelShader = pPShader;
	
	HQShaderProgram* newProgramID;
	if(!this->AddItem(pShader,&newProgramID))
	{
		delete pShader;
		return HQ_FAILED_MEM_ALLOC;
	}
	if(pID != NULL)
		*pID = newProgramID;
	return HQ_OK;
}

/*-----------------------*/
hquint32 HQShaderManagerD3D9::GetD3DConstantStartRegister(HQConstantTableD3D9* table, const char* name)
{
	//TO DO:
	//find nested element in a struct, array element

#if HQ_TRANSLATE_CG_TO_HLSL
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
#else
	bool found = false;
	DWORD registerIndex = table->GetConstantRegIndex(name, found);
	if (!found)
		return 0xffffffff;
	return registerIndex;
#endif
}


hq_uint32 HQShaderManagerD3D9::GetParameterIndex(HQSharedPtr<HQShaderProgramD3D9> &pProgram , 
											const char *parameterName)
{
	hq_uint32 *pIndex = pProgram->parameterIndexes.GetItemPointer(parameterName);
	if (pIndex == NULL)//không có
	{
#if HQ_TRANSLATE_CG_TO_HLSL
		std::string cgParameterName = "_";
		cgParameterName += parameterName;
#else
		std::string cgParameterName = parameterName;
#endif
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
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgram->GetID());
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
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgram->GetID());
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
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgram->GetID());
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
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgram->GetID());
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
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgram->GetID());
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
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgram->GetID());
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
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgram->GetID());
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
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgram->GetID());
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
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgram->GetID());
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
		Log("error :parameter \"%s\" not found from program (%u)!",parameterName,activeProgram->GetID());
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

hq_uint32 HQShaderManagerD3D9::GetParameterIndex(HQShaderProgram* programID , 
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

HQReturnVal HQShaderManagerD3D9::CreateUniformBuffer(hq_uint32 size , void *initData , bool isDynamic , HQUniformBuffer **pBufferIDOut)
{
	HQShaderConstBufferD3D9* pNewBuffer = HQ_NEW HQShaderConstBufferD3D9(this, isDynamic, size); 
	if (initData != NULL)
		pNewBuffer->Update(0, size, initData);
	if(!shaderConstBuffers.AddItem(pNewBuffer , pBufferIDOut))
	{
		HQ_DELETE (pNewBuffer);
		return HQ_FAILED_MEM_ALLOC;
	}
	return HQ_OK;
}
HQReturnVal HQShaderManagerD3D9::RemoveUniformBuffer(HQUniformBuffer* bufferID)
{
	return (HQReturnVal)shaderConstBuffers.Remove(bufferID);
}
void HQShaderManagerD3D9::RemoveAllUniformBuffers()
{
	this->shaderConstBuffers.RemoveAll();
}

void HQShaderManagerD3D9::MarkBufferSlotDirty(hquint32 index)
{
	hq_uint32 slot = index & 0x0fffffff;

	hq_uint32 shaderStage = index & 0xf0000000;
	BufferSlotInfo *pBufferSlot;
	switch (shaderStage)
	{
	case HQ_VERTEX_SHADER:
		pBufferSlot = this->vshaderConstSlots + slot;
		pBufferSlot->dirtyFlags = 1;//notify all shaders
		break;
	case HQ_PIXEL_SHADER:
		pBufferSlot = this->pshaderConstSlots + slot;
		pBufferSlot->dirtyFlags = 1;//notify all shaders
		break;
	}

}

void HQShaderManagerD3D9::MarkAllBufferSlotsDirtyForVShader(HQShaderObjectD3D9* shader)
{
	HQConstantTableD3D9 * constTable = shader->consTable;
	BufferSlotInfo *pBaseBufferSlot = this->vshaderConstSlots;

	//mark every associated buffer slots as dirty for this shader
	HQConstantTableD3D9::BufferSlotList::Iterator slot_ite;
	for (constTable->constBufferSlotList.GetIterator(slot_ite); !slot_ite.IsAtEnd(); ++slot_ite)
	{
		pBaseBufferSlot[slot_ite->index].dirtyFlags = 1;
	}
}

void HQShaderManagerD3D9::MarkAllBufferSlotsDirtyForPShader(HQShaderObjectD3D9* shader)
{
	HQConstantTableD3D9 * constTable = shader->consTable;
	BufferSlotInfo *pBaseBufferSlot = this->pshaderConstSlots;

	//mark every associated buffer slots as dirty for this shader
	HQConstantTableD3D9::BufferSlotList::Iterator slot_ite;
	for (constTable->constBufferSlotList.GetIterator(slot_ite); !slot_ite.IsAtEnd(); ++slot_ite)
	{
		pBaseBufferSlot[slot_ite->index].dirtyFlags = 1;
	}
}


void HQShaderManagerD3D9::Commit()
{
	HQConstantTableD3D9 * constTable;
	//vertex shader
	if (this->activeVShader != NULL)
	{
		constTable = this->activeVShader->consTable;
		//check for dirty constant buffer slot
		HQConstantTableD3D9::BufferSlotList::Iterator slot_ite;
		for (constTable->constBufferSlotList.GetIterator(slot_ite); !slot_ite.IsAtEnd(); ++slot_ite)
		{
			BufferSlotInfo &bufferSlot = this->vshaderConstSlots[slot_ite->index];
			HQShaderConstBufferD3D9 *constBuffer = bufferSlot.buffer.GetRawPointer();

			if (bufferSlot.dirtyFlags == 1 && constBuffer != NULL)
			{
				//this slot is dirty. need to update constant data
				const hqubyte8 * pData = (const hqubyte8*)constBuffer->GetRawBuffer();
				hquint32 totalSize = constBuffer->GetSize();
				hquint32 offset = 0;
				//for each constant
				HQLinkedList<HQConstantTableD3D9::ConstInfo>::Iterator const_ite;
				for (slot_ite->constants.GetIterator(const_ite); 
					!const_ite.IsAtEnd() && offset < totalSize;
					++const_ite)
				{
					//for now, each element consumes one vector
					if (const_ite->isInteger)
						this->SetD3DVShaderConstantI<4>(const_ite->regIndex, (const hqint32*)(pData + offset), const_ite->numVectors);
					else
						this->SetD3DVShaderConstantF<4>(const_ite->regIndex, (const hqfloat32*)(pData + offset), const_ite->numVectors);

					offset += 4 * sizeof(hqfloat32)* const_ite->numVectors;
				}

				bufferSlot.dirtyFlags = 0;//mark as not dirty

			}//if (bufferSlot.dirtyFlags == 1 && constBuffer != NULL)
		}//for (constTable->constBufferSlotList.GetIterator(slot_ite); !slot_ite.IsAtEnd(); ++slot_ite)
	}//if (this->activeVShader != NULL)

	//pixel shader
	if (this->activePShader != NULL)
	{
		constTable = this->activePShader->consTable;
		//check for dirty constant buffer slot
		HQConstantTableD3D9::BufferSlotList::Iterator slot_ite;
		for (constTable->constBufferSlotList.GetIterator(slot_ite); !slot_ite.IsAtEnd(); ++slot_ite)
		{
			BufferSlotInfo &bufferSlot = this->pshaderConstSlots[slot_ite->index];
			HQShaderConstBufferD3D9 *constBuffer = bufferSlot.buffer.GetRawPointer();

			if (bufferSlot.dirtyFlags == 1 && constBuffer != NULL)
			{
				//this slot is dirty. need to update constant data
				const hqubyte8 * pData = (const hqubyte8*)constBuffer->GetRawBuffer();
				hquint32 totalSize = constBuffer->GetSize();
				hquint32 offset = 0;
				//for each constant
				HQLinkedList<HQConstantTableD3D9::ConstInfo>::Iterator const_ite;
				for (slot_ite->constants.GetIterator(const_ite);
					!const_ite.IsAtEnd() && offset < totalSize;
					++const_ite)
				{
					//for now, each element consumes one vector
					if (const_ite->isInteger)
						this->SetD3DPShaderConstantI<4>(const_ite->regIndex, (const hqint32*)(pData + offset), const_ite->numVectors);
					else
						this->SetD3DPShaderConstantF<4>(const_ite->regIndex, (const hqfloat32*)(pData + offset), const_ite->numVectors);

					offset += 4 * sizeof(hqfloat32)* const_ite->numVectors;
				}

				bufferSlot.dirtyFlags = 0;//mark as not dirty

			}//if (bufferSlot.dirtyFlags == 1 && constBuffer != NULL)
		}//for (constTable->constBufferSlotList.GetIterator(slot_ite); !slot_ite.IsAtEnd(); ++slot_ite)
	}//if (this->activePShader != NULL)
}


HQReturnVal HQShaderManagerD3D9::SetUniformBuffer(hq_uint32 index ,  HQUniformBuffer* bufferID )
{
	HQSharedPtr<HQShaderConstBufferD3D9> pBuffer = shaderConstBuffers.GetItemPointer(bufferID);
	
	hq_uint32 slot = index & 0x0fffffff;
	hq_uint32 shaderStage = index & 0xf0000000;
	
#if defined _DEBUG || defined DEBUG
	if (slot >= 16)
	{
		Log("SetUniformBuffer() Error : buffer slot=%u is out of range!", slot);
		return HQ_FAILED;
	}
#endif
	
	BufferSlotInfo *pBufferSlot;
	switch(shaderStage)
	{
	case HQ_VERTEX_SHADER:
		pBufferSlot = this->vshaderConstSlots + slot;
		if (pBufferSlot->buffer != pBuffer)
		{
			if (pBufferSlot->buffer != NULL)
			{
				pBufferSlot->buffer->boundSlots.RemoveAt(pBufferSlot->bufferLink);//remove the link with the old buffer
			}
			pBufferSlot->buffer = pBuffer;
			pBufferSlot->bufferLink = pBuffer->boundSlots.PushBack(index);

			pBufferSlot->dirtyFlags = 1;//notify all dependent shaders
		}
		break;
	case HQ_PIXEL_SHADER:
		pBufferSlot = this->pshaderConstSlots + slot;
		if (pBufferSlot->buffer != pBuffer)
		{
			if (pBufferSlot->buffer != NULL)
			{
				pBufferSlot->buffer->boundSlots.RemoveAt(pBufferSlot->bufferLink);//remove the link with the old buffer
			}
			pBufferSlot->buffer = pBuffer;
			pBufferSlot->bufferLink = pBuffer->boundSlots.PushBack(index);

			pBufferSlot->dirtyFlags = 1;//notify all dependent shaders
		}
		break;
	default:
		Log("Error : {index} parameter passing to SetUniformBuffer() method didn't bitwise OR with HQ_VERTEX_SHADER/HQ_PIXEL_SHADER!");
		return HQ_FAILED;
	}

	return HQ_OK;
}

void HQShaderManagerD3D9:: BufferChangeEnded(HQSysMemBuffer* pConstBuffer)
{
	HQShaderConstBufferD3D9* pBuffer = static_cast<HQShaderConstBufferD3D9*>(pConstBuffer);

	HQShaderConstBufferD3D9::BufferSlotList::Iterator ite;
	for (pBuffer->boundSlots.GetIterator(ite); !ite.IsAtEnd(); ++ite)
		this->MarkBufferSlotDirty(*ite);
}


