/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "HQShaderGL_CgController_inline.h"

#ifndef GLES

char *semantics[17]= {
	"-DVPOSITION=ATTR0",
	"-DVCOLOR=ATTR1",
	"-DVNORMAL=ATTR2",
	"-DVTANGENT=ATTR3",
	"-DVBINORMAL=ATTR4",
	"-DVBLENDWEIGHT=ATTR5",
	"-DVBLENDINDICES=ATTR6",
	"-DVTEXCOORD0=ATTR7",
	"-DVTEXCOORD1=ATTR8",
	"-DVTEXCOORD2=ATTR9",
	"-DVTEXCOORD3=ATTR10",
	"-DVTEXCOORD4=ATTR11",
	"-DVTEXCOORD5=ATTR12",
	"-DVTEXCOORD6=ATTR13",
	"-DVTEXCOORD7=ATTR14",
	"-DVPSIZE=ATTR15",
	NULL
};

#ifndef CG_IMPLICIT_LINK

HQ_DECL_CG_FUNC_PTR( cgGetError ) ;
HQ_DECL_CG_FUNC_PTR( cgSetErrorCallback );
HQ_DECL_CG_FUNC_PTR( cgGetErrorString ) ;
HQ_DECL_CG_FUNC_PTR( cgGetNamedParameter ) ;
HQ_DECL_CG_FUNC_PTR( cgGetParameterType ) ;
HQ_DECL_CG_FUNC_PTR( cgCreateContext ) ;
HQ_DECL_CG_FUNC_PTR( cgDestroyContext ) ;
HQ_DECL_CG_FUNC_PTR( cgCreateProgramFromFile ) ;
HQ_DECL_CG_FUNC_PTR( cgCreateProgram ) ;
HQ_DECL_CG_FUNC_PTR( cgCombinePrograms2 ) ;
HQ_DECL_CG_FUNC_PTR( cgCombinePrograms3 ) ;
HQ_DECL_CG_FUNC_PTR( cgDestroyProgram ) ;
HQ_DECL_CG_FUNC_PTR( cgSetParameterValueir ) ;
HQ_DECL_CG_FUNC_PTR( cgSetParameterValuefr ) ;
HQ_DECL_CG_FUNC_PTR( cgGLGetLatestProfile ) ;
HQ_DECL_CG_FUNC_PTR( cgGLIsProfileSupported ) ;
HQ_DECL_CG_FUNC_PTR( cgGLSetOptimalOptions ) ;
HQ_DECL_CG_FUNC_PTR( cgGLSetDebugMode ) ;
HQ_DECL_CG_FUNC_PTR( cgGLLoadProgram ) ;
HQ_DECL_CG_FUNC_PTR( cgGLUnbindProgram ) ;
HQ_DECL_CG_FUNC_PTR( cgGLDisableProfile ) ;
HQ_DECL_CG_FUNC_PTR( cgGLBindProgram ) ;
HQ_DECL_CG_FUNC_PTR( cgGetArrayTotalSize);
HQ_DECL_CG_FUNC_PTR( cgSetParameter1iv );
HQ_DECL_CG_FUNC_PTR( cgSetParameter2iv );
HQ_DECL_CG_FUNC_PTR( cgSetParameter3iv );
HQ_DECL_CG_FUNC_PTR( cgSetParameter4iv );
HQ_DECL_CG_FUNC_PTR( cgSetParameter1fv );
HQ_DECL_CG_FUNC_PTR( cgSetParameter2fv );
HQ_DECL_CG_FUNC_PTR( cgSetParameter3fv );
HQ_DECL_CG_FUNC_PTR( cgSetParameter4fv );

#endif//ifndef CG_IMPLICIT_LINK

//cg error report callback function
void cgErrorCallBack(void)
{
	CGerror err=cgGetError();
	if (g_pShaderMan != NULL)
		g_pShaderMan->Log("%s",cgGetErrorString(err));
}


/*------------HQShaderProgramGL_Cg---------------------*/

HQShaderParameterGL* HQShaderProgramGL_Cg::TryCreateParameterObject(const char *parameterName)
{
	CGparameter param = NULL;
	param=cgGetNamedParameter(this->program,parameterName);//get parameter handle
	if(param==NULL)//không tìm thấy
	{
		return  NULL;
	}

	HQShaderParameterGL* pNewParameter = new HQShaderParameterGL();
	pNewParameter->parameter=param;
	pNewParameter->type=cgGetParameterType(param);//get parameter type


	return pNewParameter;
}




/*----------HQBaseCgShaderController----------------*/

HQBaseCgShaderController::HQBaseCgShaderController()
{
	this->InitCgLibrary();

	this->cgContext=cgCreateContext();

	this->cgVertexProfile=cgGLGetLatestProfile(CG_GL_VERTEX );
	this->cgFragmentProfile=cgGLGetLatestProfile(CG_GL_FRAGMENT );
	this->cgGeometryProfile=cgGLGetLatestProfile(CG_GL_GEOMETRY );


	if(cgGLIsProfileSupported(CG_PROFILE_GLSLV)==CG_TRUE)
		this->cgVertexProfile=CG_PROFILE_GLSLV;
	if(cgGLIsProfileSupported(CG_PROFILE_GLSLF)==CG_TRUE)
		this->cgFragmentProfile=CG_PROFILE_GLSLF;
	if(cgGLIsProfileSupported(CG_PROFILE_GLSLG)==CG_TRUE)
		this->cgGeometryProfile=CG_PROFILE_GLSLG;

#if defined(DEBUG)||defined(_DEBUG)

	cgSetErrorCallback(cgErrorCallBack);
	cgGLSetDebugMode( CG_TRUE );

#else

	cgGLSetOptimalOptions(cgVertexProfile);
	cgGLSetOptimalOptions(cgGeometryProfile);
	cgGLSetOptimalOptions(cgFragmentProfile);
	cgGLSetDebugMode( CG_FALSE );

#endif
}
HQBaseCgShaderController::~HQBaseCgShaderController()
{
	cgDestroyContext(cgContext);

#ifndef CG_IMPLICIT_LINK
	if(cgLibHandle != NULL)
	{
#ifdef WIN32
		FreeLibrary(cgLibHandle);
#else
		dlclose(cgLibHandle);
#endif
		cgLibHandle = NULL;
	}
#	ifndef __APPLE__
	if(cgGLLibHandle != NULL)
	{
#		ifdef WIN32
		FreeLibrary(cgGLLibHandle);
#		else
		dlclose(cgGLLibHandle);
#		endif
		cgGLLibHandle = NULL;
	}
#	endif
#endif //ifndef CG_IMPLICIT_LINK
}


void HQBaseCgShaderController::InitCgLibrary()
{
#ifndef CG_IMPLICIT_LINK

#ifdef WIN32
	cgLibHandle =  LoadLibraryA("cg.dll");
	cgGLLibHandle =  LoadLibraryA("cgGL.dll");
#elif defined __APPLE__
	//get current working directory 
	NSString * cwd = [[NSString alloc] initWithString: [[NSFileManager defaultManager] currentDirectoryPath ] ];
	//get framework path
	NSString *exePath = [[NSString alloc] initWithString: [[NSBundle mainBundle] executablePath ] ];
	NSString *fwPath;
	unichar lastChar = [exePath characterAtIndex: [exePath length] - 1 ];
	if ( lastChar == '/')
		fwPath = [exePath stringByAppendingString: @"../Frameworks"];
	else
		fwPath = [exePath stringByAppendingString: @"/../Frameworks"];
	
	//set framework path to current directory
	[[NSFileManager defaultManager] changeCurrentDirectoryPath: fwPath];
		
	cgLibHandle = dlopen("Cg.framework/Versions/1.0/Cg",RTLD_LAZY);//open shared lib
	
	//set to old current directory
	[[NSFileManager defaultManager] changeCurrentDirectoryPath: cwd];
	[cwd release];
	[exePath release];
	
#else
	cgLibHandle = dlopen("libCg.so",RTLD_LAZY);
	cgGLLibHandle = dlopen("libCgGL.so",RTLD_LAZY);
#endif

	if (
		cgLibHandle == NULL
#ifndef __APPLE__
		|| cgGLLibHandle == NULL
#endif		
		)
	{
		g_pShaderMan->Log("Error : can't load Cg library!");
		throw std::bad_alloc();
	}

#ifdef __APPLE__
#	define cgGLLibHandle cgLibHandle //mac osx version of cg library doesn't have separate cg and cgGL module
#endif

	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgGetError ) ;
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgSetErrorCallback ) ;
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgGetErrorString ) ;
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgGetNamedParameter ) ;
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgGetParameterType ) ;
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgCreateContext ) ;
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgDestroyContext ) ;
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgCreateProgramFromFile ) ;
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgCreateProgram ) ;
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgCombinePrograms2 ) ;
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgCombinePrograms3 ) ;
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgDestroyProgram ) ;
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgSetParameterValueir ) ;
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgSetParameterValuefr ) ;
	HQ_GET_CG_FUNC_PTR( cgGLLibHandle , cgGLGetLatestProfile ) ;
	HQ_GET_CG_FUNC_PTR( cgGLLibHandle , cgGLIsProfileSupported ) ;
	HQ_GET_CG_FUNC_PTR( cgGLLibHandle , cgGLSetOptimalOptions ) ;
	HQ_GET_CG_FUNC_PTR( cgGLLibHandle , cgGLSetDebugMode ) ;
	HQ_GET_CG_FUNC_PTR( cgGLLibHandle , cgGLLoadProgram ) ;
	HQ_GET_CG_FUNC_PTR( cgGLLibHandle , cgGLUnbindProgram ) ;
	HQ_GET_CG_FUNC_PTR( cgGLLibHandle , cgGLDisableProfile ) ;
	HQ_GET_CG_FUNC_PTR( cgGLLibHandle , cgGLBindProgram ) ;

	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgGetArrayTotalSize);
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgSetParameter1iv );
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgSetParameter2iv );
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgSetParameter3iv );
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgSetParameter4iv );
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgSetParameter1fv );
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgSetParameter2fv );
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgSetParameter3fv );
	HQ_GET_CG_FUNC_PTR( cgLibHandle , cgSetParameter4fv );

#endif//ifndef CG_IMPLICIT_LINK
}

char ** HQBaseCgShaderController::GetPredefineMacroArgumentsCg(const HQShaderMacro * pDefines)
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

void HQBaseCgShaderController::DeAllocCgArgs(char **ppC)
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

HQReturnVal HQBaseCgShaderController::CreateShaderFromFileCg(HQShaderType type,
									 const char* fileName,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObjectGL **ppShaderObjectOut)
{
	CGprofile profile;
	if(type==HQ_VERTEX_SHADER)
		profile=this->cgVertexProfile;
	else if (type==HQ_PIXEL_SHADER)
		profile=this->cgFragmentProfile;
	else if (type==HQ_GEOMETRY_SHADER)
		profile=this->cgGeometryProfile;
	else
	{
		return HQ_FAILED;
	}
	HQShaderObjectGL *sobject=new HQShaderObjectGL();
	sobject->type=type;

	char **args = this->GetPredefineMacroArgumentsCg(pDefines);
	sobject->program=cgCreateProgramFromFile(this->cgContext,isPreCompiled? CG_OBJECT : CG_SOURCE ,
												 fileName,profile,entryFunctionName,(const char**) args);

	this->DeAllocCgArgs(args);

	if(sobject->program==NULL)
	{
		delete sobject;
		return HQ_FAILED;
	}


	*ppShaderObjectOut = sobject;

	return HQ_OK;
}

HQReturnVal HQBaseCgShaderController::CreateShaderFromMemoryCg(HQShaderType type,
									 const char* pSourceData,
									 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
									 bool isPreCompiled,
									 const char* entryFunctionName,
									 HQShaderObjectGL **ppShaderObjectOut)
{
	CGprofile profile;
	if(type==HQ_VERTEX_SHADER)
		profile=this->cgVertexProfile;
	else if (type==HQ_PIXEL_SHADER)
		profile=this->cgFragmentProfile;
	else if (type==HQ_GEOMETRY_SHADER)
		profile=this->cgGeometryProfile;
	else
	{
		return HQ_FAILED;
	}
	HQShaderObjectGL *sobject=new HQShaderObjectGL();
	sobject->type=type;

	char **args = this->GetPredefineMacroArgumentsCg(pDefines);
	sobject->program=cgCreateProgram(this->cgContext,isPreCompiled? CG_OBJECT : CG_SOURCE ,
												 pSourceData,profile,entryFunctionName,(const char**) args);

	this->DeAllocCgArgs(args);

	if(sobject->program==NULL)
	{
		delete sobject;
		return HQ_FAILED;
	}
	*ppShaderObjectOut = sobject;

	return HQ_OK;
}

HQReturnVal HQBaseCgShaderController::CreateProgramCg(
							  HQSharedPtr<HQShaderObjectGL>& pVShader,
							  HQSharedPtr<HQShaderObjectGL>& pGShader,
							  HQSharedPtr<HQShaderObjectGL>& pFShader,
							  const char** uniformParameterNames,
							  hq_uint32 *pID)
{
	hq_uint32 flags=0;//cờ thể hiện trong program có những loại shader nào

	if(pVShader!=NULL)
	{

		if(pVShader->type!=HQ_VERTEX_SHADER//shader có id <vertexShaderID> không phải vertex shader
			|| pVShader->isGLSL == true)//chỉ chấp nhận shader compile từ Cg language
			return HQ_FAILED_WRONG_SHADER_TYPE;
		flags|=useV;
	}
	if(pFShader != NULL)
	{
		if(pFShader->type!=HQ_PIXEL_SHADER//shader có id <pixelShaderID> không phải pixel shader
			|| pFShader->isGLSL == true)//chỉ chấp nhận shader compile từ Cg language
			return HQ_FAILED_WRONG_SHADER_TYPE;
		flags|=useF;
	}
	if(pGShader != NULL)
	{
		if(pGShader->type!=HQ_GEOMETRY_SHADER//shader có id <geometryShaderID> không phải geometry shader
			|| pGShader->isGLSL == true)//chỉ chấp nhận shader compile từ Cg language
			return HQ_FAILED_WRONG_SHADER_TYPE;
		flags|=useG;
	}

	HQBaseShaderProgramGL *pProgram=new HQShaderProgramGL_Cg();

	switch (flags)
	{
	case useVF:
		pProgram->program=cgCombinePrograms2(pVShader->program,
												pFShader->program);
		break;
	case useVGF:
		pProgram->program=cgCombinePrograms3(pVShader->program,
												pGShader->program,
												pFShader->program);
		break;
	}

	if(pProgram->program==NULL)
	{
		SafeDelete(pProgram);
		return HQ_FAILED;
	}

	cgGLLoadProgram(pProgram->program);

	pProgram->isGLSL = false;

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

	if(!g_pShaderMan->AddItem(pProgram,pID))
	{
		delete pProgram;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

/*----------HQCgShaderController----------------*/

HQReturnVal HQCgShaderController::CreateShaderFromFile(HQShaderType type,
										const char* fileName,
										const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										bool isPreCompiled,
										const char* entryFunctionName,
										HQShaderObjectGL **ppShaderObjectOut)
{
	return this->CreateShaderFromFileCg(type,fileName,pDefines ,isPreCompiled , entryFunctionName ,ppShaderObjectOut);
}

HQReturnVal HQCgShaderController::CreateShaderFromMemory(HQShaderType type,
										  const char* pSourceData,
										  const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
										  bool isPreCompiled,
										  const char* entryFunctionName,
										  HQShaderObjectGL **ppShaderObjectOut)
{
	return this->CreateShaderFromMemoryCg(type , pSourceData,pDefines ,isPreCompiled,entryFunctionName, ppShaderObjectOut);
}

HQReturnVal HQCgShaderController::CreateShaderFromFile(HQShaderType type,
								 HQShaderCompileMode compileMode,
								 const char* fileName,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 HQShaderObjectGL **ppShaderObjectOut)
{
	switch (compileMode)
	{
	case HQ_SCM_CG:case HQ_SCM_CG_DEBUG:
		return this->CreateShaderFromFileCg(type , fileName ,pDefines,false , entryFunctionName, ppShaderObjectOut);
	default:
		return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}
}

HQReturnVal HQCgShaderController::CreateShaderFromMemory(HQShaderType type,
								 HQShaderCompileMode compileMode,
								 const char* pSourceData,
								 const HQShaderMacro * pDefines,//pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần <name> và <definition>là NULL để chỉ kết thúc dãy
								 const char* entryFunctionName,
								 HQShaderObjectGL **ppShaderObjectOut)
{
	switch (compileMode)
	{
	case HQ_SCM_CG:case HQ_SCM_CG_DEBUG:
		return this->CreateShaderFromMemoryCg(type , pSourceData ,pDefines,false , entryFunctionName, ppShaderObjectOut);
	default:
		return HQ_FAILED_SHADER_SOURCE_IS_NOT_SUPPORTED;
	}

}

HQReturnVal HQCgShaderController::CreateProgram(	bool isGLSL ,
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
	HQReturnVal re = this->CreateProgramCg(pVShader , pGShader , pFShader , uniformParameterNames , &programID);
	
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


#endif//ifndef GLES
