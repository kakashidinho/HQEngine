/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

/*
Note: Initialization code is a bit ugly since it is written a long time ago

*/

#include "HQDeviceGLPCH.h"
#include "HQDeviceGL.h"
#ifdef HQ_OPENGLES
#include "HQVertexStreamManagerPreShaderGL.h"
#else
//TO DO : add to GLES 3.0 later
#include "HQVertexStreamManagerGL_VAO.h"
#endif

#include <string.h>
#include <string>


#if defined _DEBUG || defined DEBUG
#include <assert.h>
#endif


/*----------------------------------*/
#ifdef WIN32
#define gl_GetProcAddress wglGetProcAddress
#elif defined(HQ_LINUX_PLATFORM)
#include <malloc.h>
#endif

void GLAPIENTRY DummyProc1(GLenum v)
{
}


#ifdef HQ_LINUX_PLATFORM
pfnglxgetprocaddress pglGetProcAddress=NULL;
glxFuncPointer DummyProc2(const GLubyte* arg)
{
    return NULL;
}
/*
 need this function because stupid gcc can't cast const char* to const hq_ubyte8*
 when passing argument "procName" directly to pglGetProcAddress
*/
glxFuncPointer gl_GetProcAddress(const char* procName)
{
    if(pglGetProcAddress==NULL)
        return NULL;
    return pglGetProcAddress((const GLubyte*)procName);
}


#endif

/*-------------------memory barrier----------------------*/
typedef void (GLAPIENTRY * PFNGLMEMORYBARRIER) (GLbitfield barriers);
void GLAPIENTRY glMemoryBarrierDummy(GLbitfield barriers){}
PFNGLMEMORYBARRIER glMemoryBarrierWrapper;

/*--------------------dispatch compute----------*/
typedef void(GLAPIENTRY * PFNGLDISPATCHCOMPUTE) (GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);
typedef void(GLAPIENTRY * PFNGLDISPATCHCOMPUTEINDIRECT) (GLintptr offset);
void GLAPIENTRY glDispatchComputeDummy(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z) {}
void GLAPIENTRY glDispatchComputeIndirectDummy(GLintptr offset) {}
PFNGLDISPATCHCOMPUTE glDispatchComputeWrapper;
PFNGLDISPATCHCOMPUTEINDIRECT glDispatchComputeIndirectWrapper;

/*------------------draw indirect---------------*/
typedef void (GLAPIENTRY * PFNGLDRAWARRAYSINDIRECT) (GLenum mode, const GLvoid *indirect);
typedef void (GLAPIENTRY * PFNGLDRAWELEMENTSINDIRECT) (GLenum mode, GLenum type, const GLvoid *indirect);

void GLAPIENTRY glDrawArraysIndirectDummy(GLenum mode, const GLvoid *indirect) {}
void GLAPIENTRY glDrawElementsIndirectDummy(GLenum mode, GLenum type, const GLvoid *indirect) {}
PFNGLDRAWARRAYSINDIRECT glDrawArraysIndirectWrapper;
PFNGLDRAWELEMENTSINDIRECT glDrawElementsIndirectWrapper;

/*----------------------------------*/


HQDeviceGL* g_pOGLDev=NULL;



//***********************************
//create device
//***********************************
extern "C" {
#ifdef WIN32
HQReturnVal CreateDevice(HMODULE pDll,LPHQRenderDevice *ppDev,bool flushDebugLog , bool debugLayer)
#elif defined HQ_LINUX_PLATFORM
HQReturnVal CreateDevice(Display *display,LPHQRenderDevice *ppDev,bool flushDebugLog , bool debugLayer)
#else
HQReturnVal CreateDevice(LPHQRenderDevice *ppDev,bool flushDebugLog , bool debugLayer)
#endif
{
	if(g_pOGLDev != NULL)
		return HQ_DEVICE_ALREADY_EXISTS;


#ifdef WIN32
	*ppDev=new HQDeviceGL(pDll , flushDebugLog);
#elif defined HQ_LINUX_PLATFORM
	*ppDev=new HQDeviceGL(display , flushDebugLog);
#else
	*ppDev=new HQDeviceGL(flushDebugLog);
#endif
	return HQ_OK;
}
}
//**************************************
//safe release device
//**************************************
extern "C" {
HQReturnVal ReleaseDevice(LPHQRenderDevice * ppDev)
{
    HQReturnVal re = HQ_OK;
    if(g_pOGLDev!=NULL)
    {
        re = (*ppDev)->Release();
    }
    *ppDev = NULL;
    return re;
}
}
/*----------------------
HQDeviceGL
----------------------*/
#ifdef WIN32
HQDeviceGL::HQDeviceGL(HMODULE _pDll, bool flushLog )
#elif defined HQ_LINUX_PLATFORM
HQDeviceGL::HQDeviceGL(Display *dpy, bool flushLog)
#else
HQDeviceGL::HQDeviceGL(bool flushLog)
#endif
#ifdef HQ_OPENGLES
:HQBaseRenderDevice("OpenGL ES" , "GL Render Device :" , flushLog)
#else
:HQBaseRenderDevice("OpenGL" , "GL Render Device :" , flushLog)
#endif
{

	this->sWidth=800;
	this->sHeight=600;

#ifdef WIN32
	this->pDll=_pDll;

	this->hDC=0;
	this->hRC=0;

	this->pEnum = HQ_NEW HQDeviceEnumGL(_pDll);
#elif defined (HQ_LINUX_PLATFORM)
	this->dpy = dpy;
	this->glc=NULL;
    	this->pEnum = HQ_NEW HQDeviceEnumGL(dpy);
#elif defined HQ_IPHONE_PLATFORM
	this->glc = nil;
    	this->pEnum = HQ_NEW HQDeviceEnumGL();
#elif defined HQ_MAC_PLATFORM
	this->glc = nil;
    this->pEnum = HQ_NEW HQDeviceEnumGL();
#elif defined HQ_ANDROID_PLATFORM
	this->glc = NULL;
	this->jeglConfig = NULL;
	this->pEnum = NULL;//it will be ceated in Init();
#endif

	this->primitiveMode = GL_TRIANGLES;
	this->primitiveLookupTable[HQ_PRI_TRIANGLES] = GL_TRIANGLES;
	this->primitiveLookupTable[HQ_PRI_TRIANGLE_STRIP] = GL_TRIANGLE_STRIP;
	this->primitiveLookupTable[HQ_PRI_POINT_SPRITES] = GL_POINTS;
	this->primitiveLookupTable[HQ_PRI_LINES] = GL_LINES;
	this->primitiveLookupTable[HQ_PRI_LINE_STRIP] = GL_LINE_STRIP;

	this->textureMan=NULL;
	this->vStreamMan=NULL;
	this->renderTargetMan = NULL;
	this->shaderMan = NULL;
	this->stateMan = NULL;

	this->usingCoreProfile = false;//core profile will not supported some deprecated features

#ifdef GL_DISPATCH_INDIRECT_BUFFER
	this->boundDispatchIndirectBuffer = this->boundDrawIndirectBuffer = 0;
#endif

	g_pOGLDev=this;
}
//***********************************
//destructor,release
//***********************************
HQDeviceGL::~HQDeviceGL(){

	SafeDeleteTypeCast(HQBaseShaderManagerGL*, shaderMan);
	SafeDeleteTypeCast(HQVertexStreamManagerGL*, vStreamMan);
	SafeDeleteTypeCast(HQBaseRenderTargetManager*, renderTargetMan);
	SafeDeleteTypeCast(HQStateManagerGL*, stateMan);
	SafeDeleteTypeCast(HQTextureManagerGL*, textureMan);

#ifdef WIN32
	if(!this->IsWindowed())//fullscreen
		ChangeDisplaySettings(NULL,0);
	if(hRC)
	{
		wglMakeCurrent(NULL,NULL);
		wglDeleteContext(hRC);
		hRC=NULL;
	}
	if(hDC)
	{
		ReleaseDC(winfo.hwind,hDC);
		hDC=NULL;
	}
#elif defined (HQ_LINUX_PLATFORM)
    if(!this->IsWindowed())//fullsceen
    {
	//restore old screen display mode
#	if defined HQ_USE_XFREE86_VIDMODE
        XF86VidModeSwitchToMode(this->dpy, XDefaultScreen(this->dpy), pEnum->currentScreenDisplayMode);
        XF86VidModeSetViewPort(this->dpy, XDefaultScreen(this->dpy), 0, 0);
#	else
	XRRSetScreenConfig(this->dpy, pEnum->screenConfig, RootWindow(this->dpy, 0), 
				  pEnum->currentScreenSizeIndex, 
				  pEnum->currentScreenRotation, 
				  CurrentTime);
#	endif
    }
    if(this->dpy!=NULL)
    {
	if (GLXEW_VERSION_1_3)
		glXMakeContextCurrent(this->dpy, None, None, NULL);
	else
        	glXMakeCurrent(this->dpy, None, NULL);
    }
    if(glc!=NULL)
        glXDestroyContext(this->dpy, glc);
    int re;
    re = XDestroyWindow(this->dpy, winfo.win);
    re = XFreeColormap(this->dpy, winfo.cmap);
#elif defined HQ_IPHONE_PLATFORM

	if (glc != nil) {
		[glc release];
	}
#elif defined HQ_MAC_PLATFORM
	if (glc != nil) {
		if ([NSOpenGLContext currentContext] == glc)
			[NSOpenGLContext clearCurrentContext];
		[glc release];
	}
	
#elif defined HQ_ANDROID_PLATFORM
	SafeDelete(glc);
	
	if (this->jeglConfig != NULL)
	{
		JNIEnv *jenv = GetCurrentThreadJEnv();
		jenv->DeleteGlobalRef(this->jeglConfig);
	}
	
#endif
	SafeDelete(pEnum);


	Log("Released!");
}
HQReturnVal HQDeviceGL::Release(){
    if(g_pOGLDev != NULL)
    {
        delete this;
        g_pOGLDev = NULL;
    }
	return HQ_OK;
}
//***********************************
//init
//***********************************
HQReturnVal HQDeviceGL::Init(HQRenderDeviceInitInput input ,const char* settingFileDir,HQLogStream* logFileStream, const char *additionalSettings)
{

	/*
	These code are a bit ugly since they are written a long time ago, by an inexperienced student :)
	*/

	if(this->IsRunning())
	{
		Log("Already init!");
		return HQ_FAILED;
	}
	if (input == NULL)
	{
		Log("Init() is called with invalid parameters!");
		return HQ_FAILED;
	}

	this->SetLogStream(logFileStream);

	//scan addtional options
	int shaderManagerType = HQ_GLSL_SHADER_MANAGER;
	std::string core_profile = "";
	
	if (additionalSettings != NULL)
	{
		size_t len = strlen(additionalSettings);
	
		char *options = new char[len + 1];
		char *token;
		options[len] = '\0';
		strncpy(options , additionalSettings , len);
		
		token = strtok(options , " ");
		while (token != NULL)
		{
			if (!strncmp(token, "Core-GL", 7))
				core_profile = token + 7;
			token = strtok(NULL , " ");
		}

		delete[] options;
	}

#ifdef WIN32
	//get device context
	hDC=GetDC(input);
	if (hDC == NULL)
		return HQ_FAILED;

	pEnum->SetDC(hDC);

	//window info
	winfo.hwind = input;
	winfo.styles = (GetWindowLongPtrA(input,GWL_STYLE) & (~WS_MAXIMIZEBOX));
	winfo.styles &= (~WS_THICKFRAME);

	winfo.hparent = (HWND)GetWindowLongPtrA(input,GWLP_HWNDPARENT);

	RECT rect;
	GetWindowRect(input,&rect);
	if(winfo.hparent != 0)
	{
		POINT topleft;
		topleft.x = rect.left;
		topleft.y = rect.top;

		ScreenToClient(winfo.hparent,&topleft);
		winfo.x = topleft.x;
		winfo.y = topleft.y;
	}
	else
	{
		winfo.x = rect.left;
		winfo.y = rect.top;
	}

#elif defined (HQ_LINUX_PLATFORM)
	winfo.parent = input->parent;
    	winfo.x = input->x;
   	winfo.y = input->y;

#elif defined HQ_ANDROID_PLATFORM

	this->pEnum = new HQDeviceEnumGL(input->jegl , input->jdisplay, input->apiLevel);

	InitEGLMethodAndAttribsIDs();

#endif

#ifndef HQ_OPENGLES
	pEnum->EnumAllDisplayModes();//enumerate every supported display mode and pixel format
#endif

#ifdef HQ_MAC_PLATFORM
	this->pEnum->ParseSettingFile(settingFileDir ,
							[input->nsView frame].size.width , 
							[input->nsView frame].size.height,
							input->isWindowed);
#else
	pEnum->ParseSettingFile(settingFileDir);
#endif

	if(settingFileDir)
		this->CopySettingFileDir(settingFileDir);
#ifndef HQ_OPENGLES
	sWidth=pEnum->selectedResolution->width;
	sHeight=pEnum->selectedResolution->height;

	if(pEnum->windowed)
	{
		flags |=WINDOWED;
	}
	else//fullscreen
	{
#ifdef WIN32
		if(ChangeDisplaySettings(&pEnum->selectedResolution->w32DisplayMode,
			CDS_FULLSCREEN)!=DISP_CHANGE_SUCCESSFUL)//failed
			flags |=WINDOWED;
		else
#endif
            flags &= (~WINDOWED);
	}
#endif

#ifdef WIN32
	//resize window
	RECT Rect={0,0,sWidth,sHeight};
	if(this->flags & WINDOWED)
	{
		SetWindowLongPtrW(winfo.hwind,GWL_STYLE,winfo.styles);
		AdjustWindowRect(&Rect,winfo.styles,FALSE);
		SetWindowPos(winfo.hwind, HWND_TOP, winfo.x, winfo.y,
			Rect.right-Rect.left, Rect.bottom-Rect.top, SWP_NOZORDER);
	}
	else
	{
		SetParent(winfo.hwind,NULL);
		SetWindowLongPtrW(winfo.hwind,GWL_STYLE,WS_POPUP);
		AdjustWindowRect(&Rect,WS_POPUP,FALSE);
		SetWindowPos(winfo.hwind, HWND_TOP, 0, 0,
			Rect.right-Rect.left, Rect.bottom-Rect.top, SWP_NOZORDER);
	}

	if(!hDC)
	{
		Log("GetDC() failed!");
		return HQ_FAILED_INIT_DEVICE;
	}
#endif

	int result = this->CreateContext(input, core_profile.c_str());


#ifdef WIN32
	if(result!=0)
	{
		if((flags & WINDOWED)==0)//fullscreen
			ChangeDisplaySettings(NULL,0);
		ReleaseDC(winfo.hwind,hDC);
		if(result==1)
			Log("SetPixel() failed!");
		else if(result==-1)
			Log("ChoosePixel() failed!");
		else if(result==-2)
			Log("wglChoosePixelARB() failed!");
		else if (result == -3)
			Log("Create render context failed!");
		return HQ_FAILED_INIT_DEVICE;
	}

	if(wglMakeCurrent(hDC,hRC)==FALSE)
	{
		if((flags & WINDOWED)==0)//fullscreen
			ChangeDisplaySettings(NULL,0);
		wglDeleteContext(hRC);
		ReleaseDC(winfo.hwind,hDC);
		Log("Make current render context failed!");
		return HQ_FAILED_INIT_DEVICE;
	}
#elif defined (HQ_LINUX_PLATFORM)

	if(result!=0)
	{
		if(result==-2)
			Log("glXChooseVisual failed!");
		else if(result==-1)
		{
			Log("Create render context failed!");
			XDestroyWindow(this->dpy, winfo.win);
		}
		return HQ_FAILED_INIT_DEVICE;
	}
    	//window title
    	XStoreName(this->dpy, winfo.win, input->title);
	bool madeCurrent = false;
	if (GLXEW_VERSION_1_3)
		madeCurrent = glXMakeContextCurrent(this->dpy, winfo.glxWin, winfo.glxWin, glc);
	else
		madeCurrent = glXMakeCurrent(this->dpy, winfo.win, glc);
	if(madeCurrent == false)
	{
		glXDestroyContext(this->dpy, glc);
		XDestroyWindow(this->dpy, winfo.win);
		Log("Make current render context failed!");
		return HQ_FAILED_INIT_DEVICE;
	}

	if(GLXEW_VERSION_1_4)
		pglGetProcAddress=&glXGetProcAddress;
	else if(GLXEW_ARB_get_proc_address)
		pglGetProcAddress=&glXGetProcAddressARB;
	else
		pglGetProcAddress=&DummyProc2;

	input->window=winfo.win;

#elif defined HQ_IPHONE_PLATFORM
	if (result != 0) {
		Log("Create render context failed!");
		return HQ_FAILED_INIT_DEVICE;
	}

	CGRect screenRect = [[UIScreen mainScreen] bounds];
	
	if (input->landscapeMode)
	{
		this->sWidth = screenRect.size.height;
		this->sHeight = screenRect.size.width;
	}
	else {
		this->sWidth = screenRect.size.width;
		this->sHeight = screenRect.size.height;
	}


#elif defined (HQ_MAC_PLATFORM)

	if(result!=0)
	{
		if(result==1)
			Log("Init pixel format failed!");
        else if (result == 2)
            Log("Create render context failed!");
		return HQ_FAILED_INIT_DEVICE;
	}

	[glc makeCurrentContext];
	
	if (!input->isWindowed)//fullscreen
	{
		//moving window to upper left corner
		NSRect rect = [[input->nsView window ] frame];
		rect.origin.x = 0;
		rect.origin.y = [[NSScreen mainScreen] frame].size.height - rect.size.height;
		[[input->nsView window ] setFrame : rect display : YES];
		[glc update];
		CGDisplaySetDisplayMode( CGMainDisplayID() , pEnum->selectedResolution->cgDisplayMode , NULL);//change screen size
		[[input->nsView window ] makeKeyAndOrderFront:nil];
	}
#elif defined HQ_ANDROID_PLATFORM
	if(result!=0)
	{
		if(result==1)
			Log("Init pixel format failed!");
        else if (result == 2)
            Log("Create render context failed!");
		return HQ_FAILED_INIT_DEVICE;
	}


	//get surface size
	this->sWidth = (hquint32) this->glc->GetSurfaceWidth();
	this->sHeight = (hquint32) this->glc->GetSurfaceHeight();

	this->glc->MakeCurrent();

#endif
	
	//check openGL capabilities
	pEnum->CheckCapabilities();

	//save setting
	pEnum->SaveSettingFile(settingFileDir);

#ifndef HQ_OPENGLES
	if(pEnum->selectedMulSampleType > 0)
	{
		glEnable(GL_MULTISAMPLE_ARB);
		if(!GLEW_VERSION_3_0 && GLEW_NV_multisample_filter_hint)
			glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);
	}
	else glDisable(GL_MULTISAMPLE_ARB);
#endif
	this->OnFinishInitDevice(shaderManagerType);

	this->flags |= RUNNING;
	Log("Successfully created! Renderer : %s, OpenGL version : %s" , glGetString(GL_RENDERER) , glGetString(GL_VERSION));
	return HQ_OK;
}

#if defined DEVICE_LOST_POSSIBLE
bool HQDeviceGL::IsDeviceLost()
{
	bool lost = this->glc->IsLost();
	if(lost)
	{
		if ((this->flags & DEVICE_LOST) == 0)
		{
			this->flags |= DEVICE_LOST;
			this->OnLost();
		}
	}
	else if (this->flags & DEVICE_LOST)//reset
	{
		this->flags &= (~DEVICE_LOST);
		this->OnReset();
	}

	return lost;
}

void HQDeviceGL::OnLost()
{
	static_cast<HQVertexStreamManagerGL*> (this->vStreamMan)->OnLost();

	//delete shaders and textures
	static_cast<HQBaseShaderManagerGL*>(this->shaderMan)->OnLost();
	static_cast<HQTextureManagerGL*>(this->textureMan)->OnLost();

	this->glc->OnLost();
}

void HQDeviceGL::OnReset()
{
	this->glc->OnReset();
	this->glc->MakeCurrent();

	static_cast<HQVertexStreamManagerGL*> (this->vStreamMan)->OnReset();
	static_cast<HQStateManagerGL*> (this->stateMan)->OnReset();
	static_cast<HQBaseShaderManagerGL*>(this->shaderMan)->OnReset();
	static_cast<HQBaseRenderTargetManagerGL*>(this->renderTargetMan)->OnReset();
}

#endif

void HQDeviceGL::OnFinishInitDevice(int shaderManagerType)
{
	this->EnableVSyncNonSave(pEnum->vsync != 0);

#ifndef HQ_OPENGLES

	if (!GLEW_VERSION_3_0)
		glEnable(GL_POINT_SPRITE);//always enable in v3.0 and above
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	if (!GLEW_VERSION_3_0)
	{
		for(hq_uint32 i = 0; i < pEnum->caps.nFFTextureUnits;++i)
		{
			glActiveTexture(gl_texture(i));
			glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
		}
        
        glActiveTexture(GL_TEXTURE0);
	}

#endif

	this->currentVP.x = this->currentVP.y = 0;
	this->currentVP.width = sWidth;
	this->currentVP.height = sHeight;

	this->shaderMan = HQCreateShaderManager(shaderManagerType , this->m_pLogStream , this->m_flushLog);
	this->textureMan = new HQTextureManagerGL(pEnum->caps.nShaderSamplerUnits, pEnum->caps.nImageUnits, this->m_pLogStream, this->m_flushLog);
	this->stateMan = new HQStateManagerGL(static_cast<HQTextureManagerGL*> (this->textureMan) , pEnum->caps.maxAF , this->m_pLogStream , this->m_flushLog);

	//create vertex stream manager
#ifdef HQ_OPENGLES
	if (!GLEW_VERSION_2_0)
	{
		this->vStreamMan = new HQVertexStreamManagerNoShaderGL(pEnum->caps.maxVertexAttribs , this->m_pLogStream , this->m_flushLog);
	}
	else
#endif
	{
#ifndef HQ_OPENGLES //TO DO: add to GLES later
		if (GLEW_VERSION_3_0)
			this->vStreamMan = new HQVertexStreamManagerGL_VAO(pEnum->caps.maxVertexAttribs, this->m_pLogStream, this->m_flushLog);
		else
#endif
			this->vStreamMan = new HQVertexStreamManagerGL(pEnum->caps.maxVertexAttribs, this->m_pLogStream, this->m_flushLog);
	}
	//create render target manager
#ifndef HQ_IPHONE_PLATFORM
	if (GLEW_EXT_framebuffer_object || GLEW_VERSION_3_0)
#	ifdef HQ_ANDROID_PLATFORM
		this->renderTargetMan = new HQRenderTargetManagerFBO(1 , static_cast<HQTextureManagerGL*> (this->textureMan) , this->m_pLogStream , this->m_flushLog);
#	else
		this->renderTargetMan = new HQRenderTargetManagerFBO(pEnum->caps.maxDrawBuffers , static_cast<HQTextureManagerGL*> (this->textureMan) , this->m_pLogStream , this->m_flushLog);
#	endif
	else
		this->renderTargetMan = new DummyRenderTargetManager();
#else
	this->renderTargetMan = new HQRenderTargetManagerFBO([glc getFrameBuffer] , 1 , static_cast<HQTextureManagerGL*> (this->textureMan) , this->m_pLogStream , this->m_flushLog);
#endif

	//assign appropriate function pointers
#ifdef GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
	if (GLEW_VERSION_4_2)
		glMemoryBarrierWrapper = glMemoryBarrier;
	else
#endif
		glMemoryBarrierWrapper = glMemoryBarrierDummy;

#ifdef GL_MAX_COMPUTE_WORK_GROUP_COUNT
	if (GLEW_VERSION_4_3)
		glDispatchComputeWrapper = glDispatchCompute;
	else
#endif
		glDispatchComputeWrapper = glDispatchComputeDummy;

#ifdef GL_DISPATCH_INDIRECT_BUFFER
	if (GLEW_VERSION_4_3)
	{
		glDispatchComputeIndirectWrapper = glDispatchComputeIndirect;
		glDrawArraysIndirectWrapper = glDrawArraysIndirect;
		glDrawElementsIndirectWrapper = glDrawElementsIndirect;
	}
	else
#endif
	{
		glDispatchComputeIndirectWrapper = glDispatchComputeIndirectDummy;
		glDrawArraysIndirectWrapper = glDrawArraysIndirectDummy;
		glDrawElementsIndirectWrapper = glDrawElementsIndirectDummy;
	}

}

//***********************************
//enable / disable vsync
//***********************************
void HQDeviceGL::EnableVSync(bool enable)
{
	if (enable == this->IsVSyncEnabled())
		return;
	
	this->EnableVSyncNonSave(enable);

	pEnum->SaveSettingFile(this->settingFileDir);
}


void HQDeviceGL::EnableVSyncNonSave(bool enable)
{
	int interval;
	if(enable)
	{
		interval = 1;
		this->flags |= VSYNC_ENABLE;
		pEnum->vsync = 1;
	}
	else
	{
		interval = 0;
		this->flags &= ~VSYNC_ENABLE;
		pEnum->vsync = 0;
	}
#ifdef WIN32
	if(WGLEW_EXT_swap_control)
		wglSwapIntervalEXT(interval);
#elif defined HQ_LINUX_PLATFORM
	if(GLXEW_EXT_swap_control)
		glXSwapIntervalEXT(this->dpy, winfo.swapDrawable, interval);
	else if (GLXEW_MESA_swap_control)
		glXSwapIntervalMESA(interval);
#elif defined HQ_MAC_PLATFORM
	[glc setValues: &interval forParameter: NSOpenGLCPSwapInterval];
#endif
}

//***********************************
//begin render
//***********************************
HQReturnVal HQDeviceGL::BeginRender(HQBool clearPixel,HQBool clearDepth,HQBool clearStencil,HQBool clearWholeRenderTarget){
#ifdef WIN32
	if(hRC==NULL)
#elif defined (HQ_LINUX_PLATFORM) || defined HQ_ANDROID_PLATFORM
	if(glc==NULL)
#elif defined HQ_MAC_PLATFORM || defined HQ_IPHONE_PLATFORM
	if (glc == nil)
#endif
	{
		Log("not init before use!");
		return HQ_DEVICE_NOT_INIT;
	}

	if (this->flags & RENDER_BEGUN)
		return HQ_FAILED_RENDER_ALREADY_BEGUN;

#ifdef DEVICE_LOST_POSSIBLE
	if (this->flags & DEVICE_LOST)
		return HQ_FAILED_DEVICE_LOST;
#endif

#ifdef HQ_IPHONE_PLATFORM
	if ([EAGLContext currentContext] != glc)
		[EAGLContext setCurrentContext:glc];
#elif defined HQ_MAC_PLATFORM
	if ([NSOpenGLContext currentContext] != glc)
		[glc makeCurrentContext];
#endif
#if 0
	GLbitfield l_flags=0;

	if(clearPixel)
		l_flags|=GL_COLOR_BUFFER_BIT;//0x00004000
	if(clearDepth)
		l_flags|=GL_DEPTH_BUFFER_BIT;//0x00000100
	if(clearStencil)
		l_flags|=GL_STENCIL_BUFFER_BIT;//0x00000400
#else
	GLbitfield l_flags = (clearPixel << 14) | (clearDepth << 8) | (clearStencil << 10);
#endif
	
	if(!clearWholeRenderTarget)//ko clear toàn bộ buffer
	{
		glEnable(GL_SCISSOR_TEST);
		glClear(l_flags);
		glDisable(GL_SCISSOR_TEST);
	}
	else glClear(l_flags);

	this->flags |= RENDER_BEGUN;

	return HQ_OK;
}
//****************************************
//end render
//****************************************
HQReturnVal HQDeviceGL::EndRender(){

#if defined DEBUG || defined _DEBUG

#ifdef WIN32
	if(hRC==NULL)
#elif defined (HQ_LINUX_PLATFORM) || defined HQ_ANDROID_PLATFORM
	if(glc==NULL)
#elif defined HQ_MAC_PLATFORM  || defined HQ_IPHONE_PLATFORM
	if (glc == nil)
#endif
	{
		Log("not init before use!");
		return HQ_DEVICE_NOT_INIT;
	}
#endif //DEBUG

	if ((this->flags & RENDER_BEGUN)== 0)
		return HQ_FAILED_RENDER_NOT_BEGUN;

#ifdef DEVICE_LOST_POSSIBLE
	if (this->flags & DEVICE_LOST)
		return HQ_FAILED_DEVICE_LOST;
#endif

#if 0
#ifndef HQ_MAC_PLATFORM
	glFlush();
#endif
#endif

	this->flags &= ~RENDER_BEGUN;

	return HQ_OK;
}

/*----------------------------------
DisplayBackBuffer()
------------------------------*/
HQReturnVal HQDeviceGL::DisplayBackBuffer()
{

#if defined DEBUG || defined _DEBUG

#ifdef WIN32
	if(hRC==NULL)
#elif defined (HQ_LINUX_PLATFORM) || defined HQ_ANDROID_PLATFORM
	if(glc==NULL)
#elif defined HQ_MAC_PLATFORM  || defined HQ_IPHONE_PLATFORM
	if (glc == nil)
#endif
	{
		Log("not init before use!");
		return HQ_DEVICE_NOT_INIT;
	}
#endif //DEBUG

#ifdef DEVICE_LOST_POSSIBLE
	if (this->flags & DEVICE_LOST)
		return HQ_FAILED_DEVICE_LOST;
#endif

#ifdef WIN32
	SwapBuffers(hDC);
#elif defined (HQ_LINUX_PLATFORM)
	glXSwapBuffers(this->dpy, winfo.swapDrawable);
#elif defined HQ_IPHONE_PLATFORM
	[glc presentRenderbuffer];
#elif defined HQ_MAC_PLATFORM
	[glc flushBuffer];
#elif defined HQ_ANDROID_PLATFORM
	glc->SwapBuffers();
#endif

	return HQ_OK;
}

//***********************************
//clear
//***********************************
HQReturnVal HQDeviceGL::Clear(HQBool clearPixel,HQBool clearDepth,HQBool clearStencil,HQBool clearWholeRenderTarget)
{
#if defined DEBUG || defined _DEBUG
#ifdef WIN32
	if(hRC==NULL)
#elif defined (HQ_LINUX_PLATFORM) || defined HQ_ANDROID_PLATFORM
	if(glc==NULL)
#elif defined HQ_MAC_PLATFORM  || defined HQ_IPHONE_PLATFORM
	if (glc == nil)
#endif
	{
		Log("not init before use!");
		return HQ_DEVICE_NOT_INIT;
	}
#endif

#ifdef DEVICE_LOST_POSSIBLE
	if (this->flags & DEVICE_LOST)
		return HQ_FAILED_DEVICE_LOST;
#endif

#if 0
	GLbitfield l_flags=0;

	if(clearPixel)
		l_flags|=GL_COLOR_BUFFER_BIT;//0x00004000
	if(clearDepth)
		l_flags|=GL_DEPTH_BUFFER_BIT;//0x00000100
	if(clearStencil)
		l_flags|=GL_STENCIL_BUFFER_BIT;//0x00000400
#else
	GLbitfield l_flags = (clearPixel << 14) | (clearDepth << 8) | (clearStencil << 10);
#endif
	
	if(!clearWholeRenderTarget)//ko clear toàn bộ buffer
	{
		glEnable(GL_SCISSOR_TEST);
		glClear(l_flags);
		glDisable(GL_SCISSOR_TEST);
	}
	else glClear(l_flags);

	return HQ_OK;
}
//***********************************
//set clear values
//***********************************
void HQDeviceGL::SetClearColori(hq_uint32 red,hq_uint32 green,hq_uint32 blue,hq_uint32 alpha){

	clearColor = HQColorRGBAi(red, green, blue, alpha);

	glClearColor(clearColor.r, clearColor.g,
				clearColor.b, clearColor.a);
}
void HQDeviceGL::SetClearColorf(hq_float32 red, hq_float32 green, hq_float32 blue, hq_float32 alpha){

	clearColor = HQColorRGBA(red, green, blue, alpha);

	glClearColor(clearColor.r, clearColor.g,
				clearColor.b, clearColor.a);
}
void HQDeviceGL::SetClearDepthVal(hq_float32 val){

	clearDepth = val;

#ifdef HQ_OPENGLES
	glClearDepthf((GLclampf)val);
#else
	glClearDepth((GLclampd)val);
#endif
}
void HQDeviceGL::SetClearStencilVal(hq_uint32 val){
	clearStencil = val;

	glClearStencil((GLint)val);
}

//********************
//setup pixel
//********************
int HQDeviceGL::CreateContext( HQRenderDeviceInitInput input, const char* coreProfile)
{
	int version_major = 1, version_minor = 0;
	const char* minorStart = strchr(coreProfile, '.');
	if(minorStart != NULL)
		sscanf(minorStart + 1, "%d", &version_minor);
	sscanf(coreProfile, "%d", &version_major);
#ifdef WIN32
	PIXELFORMATDESCRIPTOR pixFmt;
	int ipixelFormat;

	memset(&pixFmt,0,sizeof(PIXELFORMATDESCRIPTOR));
	//get color bits
	helper::FormatInfo(pEnum->selectedBufferSetting->pixelFmt,
					   &pixFmt.cColorBits,&pixFmt.cRedBits,
					   &pixFmt.cGreenBits,&pixFmt.cBlueBits,
					   &pixFmt.cAlphaBits,NULL,NULL);
	//get depth ,stencil bits
	helper::FormatInfo(pEnum->selectedDepthStencilFmt,NULL,NULL,
					   NULL,NULL,NULL,&pixFmt.cDepthBits,&pixFmt.cStencilBits);

	if(!WGLEW_ARB_pixel_format)
	{
		pixFmt.nSize=sizeof(PIXELFORMATDESCRIPTOR);
		pixFmt.nVersion   = 1;
		pixFmt.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;

		pixFmt.iPixelType = PFD_TYPE_RGBA;
		pixFmt.iLayerType = PFD_MAIN_PLANE;

		ipixelFormat=ChoosePixelFormat(hDC,&pixFmt);//find best match
		if(ipixelFormat==0)
			return -1;
	}
	else{

		hq_float32 fAttributes[] = {0, 0};
		UINT  numFormats;
		int msampleEnable=GL_FALSE;
		if(pEnum->selectedMulSampleType > 0 && WGLEW_ARB_multisample)
			msampleEnable=GL_TRUE;

		int iAttributes[] = { WGL_DRAW_TO_WINDOW_ARB,GL_TRUE,
						WGL_SUPPORT_OPENGL_ARB, GL_TRUE,
						WGL_ACCELERATION_ARB, WGL_FULL_ACCELERATION_ARB,
						WGL_COLOR_BITS_ARB,(int) pixFmt.cColorBits,
						WGL_RED_BITS_ARB, (int )pixFmt.cRedBits,
						//WGL_RED_SHIFT_ARB, (int )pixFmt.cRedShift,
						WGL_GREEN_BITS_ARB, (int )pixFmt.cGreenBits,
						//WGL_GREEN_SHIFT_ARB, (int )pixFmt.cGreenShift,
						WGL_BLUE_BITS_ARB, (int )pixFmt.cBlueBits,
						//WGL_BLUE_SHIFT_ARB, (int )pixFmt.cBlueShift,
						WGL_ALPHA_BITS_ARB, (int )pixFmt.cAlphaBits,
						//WGL_ALPHA_SHIFT_ARB, (int )pixFmt.cAlphaShift,
						WGL_DEPTH_BITS_ARB, (int)pixFmt.cDepthBits,
						WGL_STENCIL_BITS_ARB, (int)pixFmt.cStencilBits,
						WGL_DOUBLE_BUFFER_ARB, GL_TRUE,
						WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
						WGL_SWAP_METHOD_ARB,WGL_SWAP_EXCHANGE_ARB,
						WGL_SAMPLE_BUFFERS_ARB,msampleEnable,
						WGL_SAMPLES_ARB, (int)pEnum->selectedMulSampleType ,
						0, 0 };
		if(!WGLEW_ARB_multisample)
		{
			iAttributes[26]=0;
			iAttributes[27]=0;
			iAttributes[28]=0;
			iAttributes[29]=0;
		}
		if(wglChoosePixelFormatARB(hDC, iAttributes, fAttributes,
			1, &ipixelFormat, &numFormats)==FALSE || numFormats<1)
		{
			//remove WGL_SWAP_METHOD_ARB
			iAttributes[24] = iAttributes[26];
			iAttributes[25] = iAttributes[27];
			iAttributes[26] = iAttributes[28];
			iAttributes[27] = iAttributes[29];
			iAttributes[28] = iAttributes[29] = 0;

			if(wglChoosePixelFormatARB(hDC, iAttributes, fAttributes,
				1, &ipixelFormat, &numFormats)==FALSE || numFormats<1)
			{
				return -2;//still failed
			}
		}
	}//else of if(!WGLEW_ARB_pixel_format)

	if(SetPixelFormat(hDC,ipixelFormat,&pixFmt)==FALSE)
		return 1;

	//now crete context
	if (!WGLEW_ARB_create_context_profile || version_major < 3)
	{
		version_major = 1; version_minor = 0;
		this->hRC = wglCreateContext(this->hDC);
	}
	else
	{
		int ctxtAttributes[] = {
			WGL_CONTEXT_MAJOR_VERSION_ARB, version_major,//request core profile version major
			WGL_CONTEXT_MINOR_VERSION_ARB, version_minor,//request core profile version minor
			WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
			0, 0,
		};

		this->hRC = wglCreateContextAttribsARB(this->hDC, NULL, ctxtAttributes);
		if (this->hRC == NULL)
		{
			//retry without core profile
			this->Log("Warning : Cannot create device using version %d.%d core profile!", version_major, version_minor);
			version_major = 1; version_minor = 0;

			this->hRC = wglCreateContext(this->hDC);
		}
	}

	if (this->hRC == NULL)
		return - 3;

	if (version_major >= 3)
	{
		this->Log("Using OpenGL version %d.%d core profile!", version_major, version_minor);
		usingCoreProfile = true;
	}

#elif defined (HQ_LINUX_PLATFORM)
	XVisualInfo *vi = NULL;
	GLXFBConfig *glxConfigs = NULL;
	hq_ubyte8 R,G,B,A,D,S;
	//get color bits
	helper::FormatInfo(pEnum->selectedBufferSetting->pixelFmt,NULL,&R,
                           &G,&B,
                           &A,NULL,NULL);
	//get depth ,stencil bits
	helper::FormatInfo(pEnum->selectedDepthStencilFmt,NULL,NULL,
                           NULL,NULL,NULL,&D,&S);

	if (!GLXEW_VERSION_1_3)
	{
		GLint iAttributes[] = { GLX_RED_SIZE, R,
                                GLX_GREEN_SIZE, G,
                                GLX_BLUE_SIZE, B,
                                GLX_ALPHA_SIZE, A,
                                GLX_DEPTH_SIZE,D,
                                GLX_STENCIL_SIZE,S,
                                GLX_DOUBLEBUFFER,
                                GLX_RGBA,
                                GLX_SAMPLE_BUFFERS_ARB,GL_TRUE,
                                GLX_SAMPLES_ARB,(int)pEnum->selectedMulSampleType,
                                None };

		if(pEnum->selectedMulSampleType==0 || !GLXEW_ARB_multisample)
		{
		        iAttributes[14]=iAttributes[15]=iAttributes[16]=iAttributes[17]=None;
		}
		vi=glXChooseVisual(this->dpy,XDefaultScreen(this->dpy),iAttributes);
	}//if (!GLXEW_VERSION_1_3)
	else
	{
		hqint32 numReturned = 0;
		GLint iAttributes[] = { GLX_RED_SIZE, R,
                                GLX_GREEN_SIZE, G,
                                GLX_BLUE_SIZE, B,
                                GLX_ALPHA_SIZE, A,
                                GLX_DEPTH_SIZE,D,
                                GLX_STENCIL_SIZE,S,
                                GLX_DOUBLEBUFFER, True,
                                GLX_SAMPLE_BUFFERS_ARB,GL_TRUE,
                                GLX_SAMPLES_ARB,(int)pEnum->selectedMulSampleType,
                                None };

		if(pEnum->selectedMulSampleType==0 || !GLXEW_ARB_multisample)
		{
		        iAttributes[14]=iAttributes[15]=iAttributes[16]=iAttributes[17]=None;
		}


		glxConfigs = glXChooseFBConfig( this->dpy, XDefaultScreen(this->dpy),iAttributes, &numReturned );

		if (glxConfigs == NULL || numReturned == 0)
			return -2;
		vi = glXGetVisualFromFBConfig( this->dpy, glxConfigs[0] );
	}//else of if (!GLXEW_VERSION_1_3)


	if(vi == NULL)
	{
		return -2;
	}


	//create window
        winfo.cmap=XCreateColormap(this->dpy, DefaultRootWindow(this->dpy),
                    vi->visual, AllocNone);
        XSetWindowAttributes swa;
        swa.colormap=winfo.cmap;
        unsigned long wFlags = CWColormap | CWEventMask;
        swa.event_mask=ExposureMask | FocusChangeMask | StructureNotifyMask |KeyPressMask | KeyReleaseMask | PointerMotionMask | ButtonPressMask | ButtonReleaseMask | ButtonMotionMask;

        if((flags & WINDOWED) == 0)//full screen
        {
            swa.override_redirect = True;
#	if defined HQ_USE_XFREE86_VIDMODE
            XF86VidModeSwitchToMode(this->dpy, XDefaultScreen(this->dpy), pEnum->selectedResolution->x11DisplayMode);
            XF86VidModeSetViewPort(this->dpy,XDefaultScreen(this->dpy), 0, 0);
#	else /*---------using randr----------*/
	    XRRSetScreenConfig(this->dpy, pEnum->screenConfig, RootWindow(this->dpy, 0), 
				  pEnum->selectedResolution->x11ScreenSizeIndex, 
				  pEnum->currentScreenRotation, 
				  CurrentTime);
#	endif
            wFlags |= CWOverrideRedirect;

            winfo.win=XCreateWindow(this->dpy, DefaultRootWindow(this->dpy),
                    0, 0, sWidth, sHeight, 0, vi->depth, InputOutput,
                    vi->visual, wFlags, &swa);

            //XMapRaised(this->dpy, winfo.win);//commented because we don't need to show the window just yet
        }
        else
        {
            winfo.win=XCreateWindow(this->dpy, winfo.parent,
                    winfo.x, winfo.y, sWidth, sHeight, 0, vi->depth, InputOutput,
                    vi->visual, wFlags, &swa);
            //XMapWindow(this->dpy, winfo.win);//commented because we don't need to show the window just yet
        }
	if (!GLXEW_VERSION_1_3)
	{
        	glc = glXCreateContext(this->dpy, vi, NULL, GL_TRUE);
		winfo.swapDrawable = winfo.win;
	}
	else
	{
		winfo.glxWin = glXCreateWindow( this->dpy, glxConfigs[0], winfo.win, NULL );
		winfo.swapDrawable = winfo.glxWin;

		if (!GLXEW_ARB_create_context_profile || version_major < 3)
			glc = glXCreateNewContext( this->dpy, glxConfigs[0], GLX_RGBA_TYPE,
				 NULL, True );
		else
		{
			int ctxAttributes[] = {
				GLX_CONTEXT_MAJOR_VERSION_ARB, version_major,//request core profile version major
				GLX_CONTEXT_MINOR_VERSION_ARB, version_minor,//request core profile version minor
				GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
				0, 0 };
			glc = glXCreateContextAttribsARB(this->dpy, glxConfigs[0], 
				 NULL, True,   ctxAttributes);
			if (glc == NULL)
			{
				//try again with no core profile version
				this->Log("Warning : Cannot create device using version %d.%d core profile!", version_major, version_minor);
				glc = glXCreateNewContext( this->dpy, glxConfigs[0], GLX_RGBA_TYPE,
				 	NULL, True );
			}
			else {
				this->Log("Using OpenGL version %d.%d core profile!", version_major, version_minor);
				usingCoreProfile = true;
			}
		}

		XFree(glxConfigs);

	}//else of if (!GLXEW_VERSION_1_3)
        if(glc==NULL)
            return -1;
//#elif defined (HQ_LINUX_PLATFORM)
	
#elif defined HQ_MAC_PLATFORM
	hq_ubyte8 R,G,B,A,D,S;
	//get color bits
	helper::FormatInfo(pEnum->selectedBufferSetting->pixelFmt,NULL,&R,
					   &G,&B,
					   &A,NULL,NULL);
	//get depth ,stencil bits
	helper::FormatInfo(pEnum->selectedDepthStencilFmt,NULL,NULL,
					   NULL,NULL,NULL,&D,&S);

	GLuint attributes[] = {
		NSOpenGLPFADoubleBuffer,
		NSOpenGLPFAColorSize, static_cast<GLuint>(R + G + B),
		NSOpenGLPFAAlphaSize, A,
		NSOpenGLPFADepthSize, D,
		NSOpenGLPFAStencilSize, S,
		NSOpenGLPFAClosestPolicy,
        NSOpenGLPFAOpenGLProfile, /*NSOpenGLProfileVersionLegacy,*/ NSOpenGLProfileVersion3_2Core,
		NSOpenGLPFAAccelerated,
		NSOpenGLPFANoRecovery,
		NSOpenGLPFAMultisample,
		NSOpenGLPFASampleBuffers ,1 ,
		NSOpenGLPFASamples , pEnum->selectedMulSampleType ,
		0
	};

	if(pEnum->selectedMulSampleType==0)
	{
		attributes[14]=0;
		attributes[15]=0;
		attributes[16]=0;
		attributes[17]=0;
		attributes[18]=0;
	}
	if(!pEnum->caps.hardwareAccel)
		attributes[12] = 0;

	
	NSOpenGLPixelFormat* pixelformat = [ [ NSOpenGLPixelFormat alloc ] initWithAttributes:
										(NSOpenGLPixelFormatAttribute*) attributes ];

	if ( pixelformat == nil )
	{
        //try again with legacy opengl
        attributes[11] = NSOpenGLProfileVersionLegacy;
        pixelformat = [ [ NSOpenGLPixelFormat alloc ] initWithAttributes:
                       (NSOpenGLPixelFormatAttribute*) attributes ];
        if (pixelformat == nil)
            return 1;
    }
    
	this->glc = [ [ HQAppleOpenGLContext alloc ]
           initWithFormat: pixelformat
           andView: input->nsView
           andViewWidthPointer : &this->sWidth
           andViewHeightPointer : &this->sHeight];
    
    [pixelformat release];
    
    if (this->glc == nil)
        return 2;
#elif defined HQ_IPHONE_PLATFORM
	//create main context
	this->glc = [[HQIOSOpenGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2
									withEAGLLayer: input->eaglLayer
								   andColorFormat: pEnum->selectedPixelFormat
							andDepthStencilFormat: pEnum->selectedDepthStencilFmt];
    if (this->glc == nil)
        return 2;
#elif defined HQ_ANDROID_PLATFORM
	this->jeglConfig = pEnum->GetJEGLConfig();
	if (this->jeglConfig == NULL)
		return 1;
    
	try{
		//create context
		this->glc = new HQAndroidOpenGLContext(*input , this->jeglConfig , pEnum->selectedApiLevel);
	}
	catch (std::bad_alloc e)
	{
		return 2;
	}
#endif
	return 0;
}

void HQDeviceGL::GetAllDisplayResolution(HQResolution *resolutionList , hq_uint32& numResolutions)
{
#ifdef HQ_OPENGLES
	numResolutions = 1;
	if (resolutionList != NULL)
	{	
		resolutionList[0].width = this->sWidth;
		resolutionList[0].height = this->sHeight;
	}
#else
	pEnum->GetAllDisplayResolution(resolutionList , numResolutions);
#endif
}

HQReturnVal HQDeviceGL::SetDisplayMode(hq_uint32 width,hq_uint32 height,bool windowed)
{
#ifdef HQ_OPENGLES
	return HQ_FAILED;
#else
	return ResizeBackBuffer(width, height, windowed, true);
#endif
}
HQReturnVal HQDeviceGL::OnWindowSizeChanged(hq_uint32 width,hq_uint32 height)
{
	if (!this->IsWindowed())
		return HQ_FAILED;
	return ResizeBackBuffer(width, height, this->IsWindowed(), false);
}

HQReturnVal HQDeviceGL::ResizeBackBuffer(hq_uint32 width,hq_uint32 height, bool windowed, bool resizeWindow)//thay đổi chế độ hiển thị màn hình
{
	//currently not support switching between windowed and fullscreen
	if (windowed != this->IsWindowed())
		Log("Warning : Switching between windowed and fullscreen mode is not supported.");
	
	
	if (width == this->sWidth && height == this->sHeight)
		return HQ_OK;

	windowed = this->IsWindowed();//get current mode
#ifndef HQ_OPENGLES
	bool found = pEnum->ChangeSelectedDisplay(width,height , windowed);
	if(!found)
		return HQ_FAILED;
#endif
#ifdef WIN32
	RECT Rect={0,0,width,height};
	if(!windowed)
	{

		if(ChangeDisplaySettings(&pEnum->selectedResolution->w32DisplayMode,
			CDS_FULLSCREEN)==DISP_CHANGE_SUCCESSFUL)//failed
		{
			if(resizeWindow)
			{
				AdjustWindowRect(&Rect,WS_POPUP,FALSE);
				SetWindowPos(winfo.hwind, HWND_TOP, 0, 0,
					Rect.right-Rect.left, Rect.bottom-Rect.top, SWP_NOZORDER);
			}
		}
	}
	else
	{
		if(resizeWindow)
		{
			AdjustWindowRect(&Rect,winfo.styles,FALSE);
			SetWindowPos(winfo.hwind, HWND_TOP, winfo.x, winfo.y,
				Rect.right-Rect.left, Rect.bottom-Rect.top, SWP_NOZORDER);
		}
	}

	ShowWindow(winfo.hwind, SW_SHOW);
#elif defined HQ_LINUX_PLATFORM
   if(!windowed)
    {
#	if defined HQ_USE_XFREE86_VIDMODE
        XF86VidModeSwitchToMode(this->dpy,XDefaultScreen(this->dpy),pEnum->selectedResolution->x11DisplayMode);
        XF86VidModeSetViewPort(this->dpy,XDefaultScreen(this->dpy),0,0);
#	else /*---------using randr----------*/
	XRRSetScreenConfig(this->dpy, pEnum->screenConfig, RootWindow(this->dpy, 0), 
				  pEnum->selectedResolution->x11ScreenSizeIndex, 
				  pEnum->currentScreenRotation, 
				  CurrentTime);
#	endif
	
        XMoveResizeWindow(this->dpy,winfo.win,0,0,width,height);
        XMapRaised(this->dpy,winfo.win);
        XGrabKeyboard(this->dpy, winfo.win, True, GrabModeAsync,
                    GrabModeAsync, CurrentTime);
        XGrabPointer(this->dpy, winfo.win, True, ButtonPressMask,
                    GrabModeAsync, GrabModeAsync, winfo.win, None, CurrentTime);
    }
    else
    {
        if(resizeWindow)
		{
			XResizeWindow(this->dpy,winfo.win,width,height);
			XMapWindow(this->dpy, winfo.win);
		}
        flags |= WINDOWED;
    }

    XFlush(this->dpy);

#elif defined HQ_MAC_PLATFORM
	if(resizeWindow)
	{
		if (windowed)
			return HQ_FAILED;//not allow resize windowed mode in mac os x
		/*
		if ([glc view ] != _view)
			[glc setView:_view];
		[glc update];
		[glc makeCurrentContext];
		 */
		NSView * view = [glc view];
		NSRect rect = [view frame];
		//resize view frame
		rect.size.width = width ;
		rect.size.height = height;
		[[glc view] setFrame: rect];
		//resize view bound
		rect = [view bounds];
		rect.size.width = width ;
		rect.size.height = height;
		[[glc view] setBounds: rect];
		//resize window
		rect = [[view window ] frame];
		rect.size.width = width ;
		rect.size.height = height;
		rect.origin.x = 0;
		rect.origin.y = [[NSScreen mainScreen] frame].size.height - height;
		[[view window ] setFrame : rect display : YES];
		[glc update];
		CGDisplaySetDisplayMode( CGMainDisplayID() , pEnum->selectedResolution->cgDisplayMode , NULL);//change display mode
		[[view window] makeKeyAndOrderFront:nil];
	}//if(resizeWindow)
#endif

	pEnum->SaveSettingFile(this->settingFileDir);


    sWidth = width;
    sHeight = height;

	static_cast<HQBaseRenderTargetManager*> (renderTargetMan)->OnBackBufferResized(sWidth, sHeight);

    this->SetViewPort(this->currentVP);

    return HQ_OK;
}


/*----------------render-------------------*/
void HQDeviceGL::SetPrimitiveMode(HQPrimitiveMode _primitiveMode)
{
	this->primitiveMode = this->primitiveLookupTable[_primitiveMode];
}
HQReturnVal HQDeviceGL::Draw(hq_uint32 vertexCount , hq_uint32 firstVertex)
{
	if ((this->flags & RENDER_BEGUN)== 0)
		return HQ_FAILED_RENDER_NOT_BEGUN;
	static_cast<HQVertexStreamManagerGL*> (vStreamMan) ->OnDraw();//tell vertex stream manager that we are about to draw
	static_cast<HQBaseShaderManagerGL*> (shaderMan)->OnDraw();//tell shader manager that we are about to draw
	
	glDrawArrays(this->primitiveMode , firstVertex , vertexCount);
	return HQ_OK;
}
HQReturnVal HQDeviceGL::DrawPrimitive(hq_uint32 primitiveCount , hq_uint32 firstVertex)
{
	if ((this->flags & RENDER_BEGUN)== 0)
		return HQ_FAILED_RENDER_NOT_BEGUN;
	hquint vertexCount;
	switch (primitiveMode)
	{
	case GL_TRIANGLES :
		vertexCount = primitiveCount * 3;
		break;
	case GL_TRIANGLE_STRIP:
		vertexCount = primitiveCount + 2;
		break;
	case GL_POINTS:
		vertexCount = primitiveCount ;
		break;
	case GL_LINES:
		vertexCount = primitiveCount << 1;//primitiveCount * 2
		break;
	case GL_LINE_STRIP:
		vertexCount = primitiveCount + 1;
		break;
	default:
		vertexCount = 0;
	}

	static_cast<HQVertexStreamManagerGL*> (vStreamMan) ->OnDraw();//tell vertex stream manager that we are about to draw
	static_cast<HQBaseShaderManagerGL*> (shaderMan)->OnDraw();//tell shader manager that we are about to draw

	glDrawArrays(this->primitiveMode , firstVertex , vertexCount);
	return HQ_OK;
}
HQReturnVal HQDeviceGL::DrawIndexed(hq_uint32 numVertices , hq_uint32 indexCount , hq_uint32 firstIndex )
{
	if ((this->flags & RENDER_BEGUN)== 0)
		return HQ_FAILED_RENDER_NOT_BEGUN;

	hq_uint32 offset = (firstIndex << static_cast<HQVertexStreamManagerGL*> (this->vStreamMan)->GetIndexShiftFactor());//should be 2 if data type is unsigned int and 1 if unsigned short
	
	static_cast<HQVertexStreamManagerGL*> (vStreamMan) ->OnDraw();//tell vertex stream manager that we are about to draw
	static_cast<HQBaseShaderManagerGL*> (shaderMan)->OnDraw();//tell shader manager that we are about to draw

	glDrawElements(this->primitiveMode , 
					indexCount , 
					static_cast<HQVertexStreamManagerGL*> (this->vStreamMan)->GetIndexDataType() , 
					(hqubyte8*)static_cast<HQVertexStreamManagerGL*> (this->vStreamMan)->GetIndexStartAddress() + offset);
	return HQ_OK;
}

HQReturnVal HQDeviceGL::DrawIndexedPrimitive(hq_uint32 numVertices , hq_uint32 primitiveCount , hq_uint32 firstIndex )
{
	if ((this->flags & RENDER_BEGUN)== 0)
		return HQ_FAILED_RENDER_NOT_BEGUN;
	

	hquint indexCount;
	switch (primitiveMode)
	{
	case GL_TRIANGLES :
		indexCount = primitiveCount * 3;
		break;
	case GL_TRIANGLE_STRIP:
		indexCount = primitiveCount + 2;
		break;
	case GL_POINTS:
		indexCount = primitiveCount ;
		break;
	case GL_LINES:
		indexCount = primitiveCount << 1;//primitiveCount * 2
		break;
	case GL_LINE_STRIP:
		indexCount = primitiveCount + 1;
		break;
	default:
		indexCount = 0;
	}

	hq_uint32 offset = (firstIndex << static_cast<HQVertexStreamManagerGL*> (this->vStreamMan)->GetIndexShiftFactor());//should be 2 if data type is unsigned int and 1 if unsigned short

	static_cast<HQVertexStreamManagerGL*> (vStreamMan) ->OnDraw();//tell vertex stream manager that we are about to draw
	static_cast<HQBaseShaderManagerGL*> (shaderMan)->OnDraw();//tell shader manager that we are about to draw

	glDrawElements(this->primitiveMode , 
					indexCount , 
					static_cast<HQVertexStreamManagerGL*> (this->vStreamMan)->GetIndexDataType() , 
					(hqubyte8*)static_cast<HQVertexStreamManagerGL*> (this->vStreamMan)->GetIndexStartAddress() + offset);
	return HQ_OK;
}

HQReturnVal HQDeviceGL::DrawInstancedIndirect(HQDrawIndirectArgsBuffer* buffer, hquint32 elementIndex)
{
	HQBufferGL* pGLBuffer = static_cast<HQBufferGL*> (buffer);

#ifdef GL_DISPATCH_INDIRECT_BUFFER
	this->BindDrawIndirectBuffer(pGLBuffer->bufferName);
#endif

	static_cast<HQVertexStreamManagerGL*> (vStreamMan)->OnDraw();//tell vertex stream manager that we are about to draw
	static_cast<HQBaseShaderManagerGL*> (shaderMan)->OnDraw();//tell shader manager that we are about to draw

	glDrawArraysIndirectWrapper(this->primitiveMode, (void*) (pGLBuffer->elementSize * elementIndex));

	return HQ_OK;
}

HQReturnVal HQDeviceGL::DrawIndexedInstancedIndirect(HQDrawIndexedIndirectArgsBuffer* buffer, hquint32 elementIndex)
{
	HQBufferGL* pGLBuffer = static_cast<HQBufferGL*> (buffer);

#ifdef GL_DISPATCH_INDIRECT_BUFFER
	this->BindDrawIndirectBuffer(pGLBuffer->bufferName);
#endif

	static_cast<HQVertexStreamManagerGL*> (vStreamMan)->OnDraw();//tell vertex stream manager that we are about to draw
	static_cast<HQBaseShaderManagerGL*> (shaderMan)->OnDraw();//tell shader manager that we are about to draw

	glDrawElementsIndirectWrapper(this->primitiveMode, 
		static_cast<HQVertexStreamManagerGL*> (this->vStreamMan)->GetIndexDataType(), 
		(void*)(pGLBuffer->elementSize * elementIndex));

	return HQ_OK;
}

HQReturnVal HQDeviceGL::SetViewPort(const HQViewPort &viewport)
{
	HQReturnVal re = HQ_OK;
	UINT width , height;
	
	width = static_cast<HQBaseRenderTargetManager*> (renderTargetMan)->GetRTWidth();
	height = static_cast<HQBaseRenderTargetManager*> (renderTargetMan)->GetRTHeight();

	if (viewport.x + viewport.width > width || viewport.y + viewport.height > height)//viewport area is invalid
	{
		this->currentVP.width = width;
		this->currentVP.height = height;
		this->currentVP.x = this->currentVP.y = 0;
		
		re = HQ_WARNING_VIEWPORT_IS_INVALID;
	}
	else
		this->currentVP = viewport;

	GLint Y= height - (this->currentVP.y + this->currentVP.height);
	glViewport(this->currentVP.x , Y,
				this->currentVP.width , this->currentVP.height);
	glScissor(this->currentVP.x , Y,
			   this->currentVP.width , this->currentVP.height);

	return re;
}


void * HQDeviceGL::GetRawHandle()
{ 
	return NULL; 
}


HQReturnVal HQDeviceGL::DispatchCompute(hquint32 numGroupX, hquint32 numGroupY, hquint32 numGroupZ)
{
	glDispatchComputeWrapper(numGroupX, numGroupY, numGroupZ);
	return HQ_OK;
}

HQReturnVal HQDeviceGL::DispatchComputeIndirect(HQComputeIndirectArgsBuffer* buffer, hquint32 elementIndex)
{
	HQBufferGL* pGLBuffer = static_cast<HQBufferGL*> (buffer);

#ifdef GL_DISPATCH_INDIRECT_BUFFER
	this->BindDispatchIndirectBuffer(pGLBuffer->bufferName);
#endif
	glDispatchComputeIndirectWrapper(pGLBuffer->elementSize * elementIndex);

	return HQ_OK;
}

void HQDeviceGL::TextureUAVBarrier()
{
#ifdef GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
	glMemoryBarrierWrapper(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
#endif
}

void HQDeviceGL::BufferUAVBarrier()
{
#ifdef GL_SHADER_STORAGE_BARRIER_BIT
	glMemoryBarrierWrapper(GL_SHADER_STORAGE_BARRIER_BIT);
#endif
}
