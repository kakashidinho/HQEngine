/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D9PCH.h"
#include "HQDeviceD3D9.h"
#include <string.h>




HQDeviceD3D9* g_pD3DDev=0;
/*---------------------------------*/
DWORD FtoDW(hq_float32 f) {return *((DWORD*)&f);}
//***********************************
//create device
//***********************************
extern "C" {
HQReturnVal CreateDevice(hModule pDll,LPHQRenderDevice *pDev,bool flushDebugLog , bool debugLayer){
	if(g_pD3DDev)
		return HQ_DEVICE_ALREADY_EXISTS;
	*pDev=new HQDeviceD3D9(pDll , flushDebugLog);

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
    if(g_pD3DDev!=NULL)
    {
        re = (*ppDev)->Release();
    }
    *ppDev = NULL;
    return re;
}
}
/*----------------------
HQDeviceD3D9
----------------------*/
HQDeviceD3D9::HQDeviceD3D9(HMODULE _pDll , bool flushLog) 
:HQBaseRenderDevice("Direct3D9" , "D3D9 Render Device :" ,flushLog)
{


	pDll=_pDll;

	sWidth=800;
	sHeight=600;

	flags = WINDOWED;

	pD3D=0;
	pDevice=0;
	clearColor=D3DCOLOR_ARGB(0,0,0,0);
	clearDepth=1.0f;
	clearStencil=0;

	
	
	pEnum=0;

	memset(&d3dp,0,sizeof(D3DPRESENT_PARAMETERS));
	
	this->textureMan=0;
	this->vStreamMan=0;
	this->shaderMan=0;
	this->stateMan = 0;
	this->renderTargetMan = 0;

	this->currentVPs = HQ_NEW HQViewPort[1];
	this->maxNumVPs = 1;
	
	this->primitiveMode = D3DPT_TRIANGLELIST;
	this->primitiveLookupTable[HQ_PRI_TRIANGLES] = D3DPT_TRIANGLELIST;
	this->primitiveLookupTable[HQ_PRI_TRIANGLE_STRIP] = D3DPT_TRIANGLESTRIP;
	this->primitiveLookupTable[HQ_PRI_POINT_SPRITES] = D3DPT_POINTLIST;
	this->primitiveLookupTable[HQ_PRI_LINES] = D3DPT_LINELIST;
	this->primitiveLookupTable[HQ_PRI_LINE_STRIP] = D3DPT_LINESTRIP;

	g_pD3DDev=this;
}
//***********************************
//destructor,release
//***********************************
HQDeviceD3D9::~HQDeviceD3D9(){
	
	SafeDelete(pEnum);

	SafeDeleteTypeCast(HQShaderManagerD3D9*, shaderMan);
	SafeDeleteTypeCast(HQTextureManagerD3D9*, textureMan);
	SafeDeleteTypeCast(HQVertexStreamManagerD3D9*, vStreamMan);
	SafeDeleteTypeCast(HQRenderTargetManagerD3D9*, renderTargetMan);
	SafeDeleteTypeCast(HQStateManagerD3D9*, stateMan);

	SafeRelease(pDevice);
	SafeRelease(pD3D);

	
	Log("Released!");
}
HQReturnVal HQDeviceD3D9::Release(){
	if(g_pD3DDev != NULL)
    {
        delete this;
        g_pD3DDev = NULL;
    }
	return HQ_OK;
}
//***********************************
//init
//***********************************
HQReturnVal HQDeviceD3D9::Init(HWND _hwnd,const char* settingFileDir,HQLogStream* logFileStream, const char *additionalSettings)
{
	if(this->IsRunning())
	{
		Log("Already init!");
		return HQ_FAILED;
	}
	if (_hwnd == NULL)
	{
		Log("Init() is called with invalid parameters!");
		return HQ_FAILED;
	}

	//window info
	winfo.hwind=_hwnd;
	winfo.styles = (GetWindowLongPtrA(_hwnd,GWL_STYLE) & (~WS_MAXIMIZEBOX));
	winfo.styles &= (~WS_THICKFRAME);
	winfo.hparent = (HWND)GetWindowLongPtrA(_hwnd,GWLP_HWNDPARENT);

	RECT rect;
	GetWindowRect(_hwnd,&rect);
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

	//already opened log file
	this->SetLogStream(logFileStream);
	
	//create direct3d object
	pD3D=Direct3DCreate9(D3D_SDK_VERSION);
	if(!pD3D)
	{
		Log("Direct3DCreate9() failed!");
		return HQ_FAILED_CREATE_DEVICE;
	}
	
	//create HQDeviceEnumD3D9 object
	pEnum=new HQDeviceEnumD3D9();
	//enum
	pEnum->Enum(pD3D);
	//parse setting file
	pEnum->ParseSettingFile(settingFileDir,pD3D);
	
	if (settingFileDir)
		this->CopySettingFileDir(settingFileDir);

	sWidth=pEnum->selectedMode->Width;
	sHeight=pEnum->selectedMode->Height;
	if(pEnum->selectedCombo->windowed)
		flags |=WINDOWED;
	else
		flags &=(~WINDOWED);
		

	//save setting to file
	pEnum->SaveSettingFile(settingFileDir);

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

	

	//create device
	this->d3dp.AutoDepthStencilFormat= pEnum->selectedDepthStencilFmt;
	this->d3dp.BackBufferCount=1;
	this->d3dp.BackBufferFormat=pEnum->selectedCombo->backFmt;
	this->d3dp.BackBufferWidth=sWidth;
	this->d3dp.BackBufferHeight=sHeight;
	this->d3dp.EnableAutoDepthStencil=pEnum->selectedDepthStencilFmt != D3DFMT_FORCE_DWORD;
	this->d3dp.hDeviceWindow=winfo.hwind;
	if(!this->IsWindowed())
		this->d3dp.FullScreen_RefreshRateInHz = pEnum->selectedMode->RefreshRate;

	DWORD quality[2];
	HRESULT hr=pD3D->CheckDeviceMultiSampleType(pEnum->selectedAdapter->adapter,pEnum->selectedDevice->devType,
		pEnum->selectedCombo->backFmt,pEnum->selectedCombo->windowed,pEnum->selectedMulSampleType,&quality[0]);//check multisample type on back buffer
	if(!FAILED(hr))
		hr=pD3D->CheckDeviceMultiSampleType(pEnum->selectedAdapter->adapter,pEnum->selectedDevice->devType,
			pEnum->selectedDepthStencilFmt,pEnum->selectedCombo->windowed,pEnum->selectedMulSampleType,&quality[1]);//check multisample type on depth stencil buffer

	if(FAILED(hr))
	{
		this->d3dp.MultiSampleQuality=0;
		this->d3dp.MultiSampleType=D3DMULTISAMPLE_NONE;
	}
	else{
		this->d3dp.MultiSampleQuality=min(quality[0] , quality[1])-1;
		this->d3dp.MultiSampleType=pEnum->selectedMulSampleType;
	}
	if (pEnum->vsync != 0)
	{
		this->flags |= VSYNC_ENABLE;
		this->d3dp.PresentationInterval= D3DPRESENT_INTERVAL_DEFAULT;
	}
	else
	{
		this->flags &= ~VSYNC_ENABLE;
		this->d3dp.PresentationInterval= D3DPRESENT_INTERVAL_IMMEDIATE;
	}
	this->d3dp.SwapEffect=D3DSWAPEFFECT_DISCARD;
	this->d3dp.Windowed=pEnum->selectedCombo->windowed;
	
	hr=pD3D->CreateDevice(pEnum->selectedAdapter->adapter,
								  pEnum->selectedDevice->devType,
								  winfo.hwind,
								  pEnum->selectedVertexProcBehavior,
								  &d3dp,
								  &pDevice);
	if(FAILED(hr)||!pDevice)
	{
		Log("CreateDevice() failed!Error Code %d",hr);
		pDevice=0;
		SafeRelease(pD3D);
		SafeDelete(pEnum);
		return HQ_FAILED_CREATE_DEVICE;
	}

	this->currentVPs[0].x = this->currentVPs[0].y = this->d3dViewPort.X = this->d3dViewPort.Y = 0;
	this->currentVPs[0].width = this->d3dViewPort.Width = sWidth;
	this->currentVPs[0].height = this->d3dViewPort.Height = sHeight;
	this->d3dViewPort.MinZ = 0.0f;
	this->d3dViewPort.MaxZ = 1.0f;
	
	this->textureMan=new HQTextureManagerD3D9(pDevice, pEnum->selectedDevice->dCaps.TextureCaps ,
											  pEnum->selectedDevice->dCaps.s3tc_dxtFlags,
											  pEnum->selectedDevice->numVertexShaderSamplers ,
											  pEnum->selectedDevice->numPixelShaderSamplers ,
											  logFileStream , this->m_flushLog);
	this->vStreamMan=new HQVertexStreamManagerD3D9(pDevice,logFileStream , this->m_flushLog);
	this->shaderMan=new HQShaderManagerD3D9(pDevice,logFileStream , this->m_flushLog);
	this->renderTargetMan = new HQRenderTargetManagerD3D9(pDevice ,
		pEnum->selectedDevice->dCaps.NumSimultaneousRTs,
		static_cast<HQTextureManagerD3D9*> (this->textureMan) , 
		logFileStream,
		this->m_flushLog);
	this->stateMan = new HQStateManagerD3D9(pDevice , 
											pEnum->selectedDevice->numVertexShaderSamplers ,
											pEnum->selectedDevice->numPixelShaderSamplers ,
											pEnum->selectedDevice->dCaps.MaxAnisotropy,
											logFileStream ,
											this->m_flushLog);


	this->flags |= RUNNING;
	Log("Successfully created! Renderer : %s" , this->pEnum->selectedAdapter->adapterID.Description);
	return HQ_OK;
}
void HQDeviceD3D9::EnableVSync(bool enable)
{
	if (enable == this->IsVSyncEnabled())
		return;
	DWORD interval;
	if(enable)
	{
		interval = D3DPRESENT_INTERVAL_DEFAULT;
		this->flags |= VSYNC_ENABLE;
		pEnum->vsync = 1;
	}
	else
	{
		interval = D3DPRESENT_INTERVAL_IMMEDIATE;
		this->flags &= ~VSYNC_ENABLE;
		pEnum->vsync = 0;
	}
	this->OnLostDevice();
	this->d3dp.PresentationInterval = interval;
	this->OnResetDevice();

	pEnum->SaveSettingFile(this->settingFileDir);
}

void HQDeviceD3D9::OnLostDevice()
{
	static_cast<HQVertexStreamManagerD3D9*> (this->vStreamMan)->OnLostDevice();
	static_cast<HQStateManagerD3D9*> (this->stateMan)->OnLostDevice();
	static_cast<HQRenderTargetManagerD3D9*> (this->renderTargetMan)->OnLostDevice();
}
void HQDeviceD3D9::OnResetDevice()
{
	pDevice->Reset(&this->d3dp);

	static_cast<HQVertexStreamManagerD3D9*> (this->vStreamMan)->OnResetDevice();
	static_cast<HQStateManagerD3D9*> (this->stateMan)->OnResetDevice();
	static_cast<HQRenderTargetManagerD3D9*> (this->renderTargetMan)->OnResetDevice();
	static_cast<HQTextureManagerD3D9*> (this->textureMan)->OnResetDevice();
	static_cast<HQShaderManagerD3D9*> (this->shaderMan)->OnResetDevice();

	this->SetViewport(this->currentVPs[0]);
}

bool HQDeviceD3D9::IsDeviceLost()
{
	if (!pDevice)
		return false;
	HRESULT hr;
	
	hr = pDevice->TestCooperativeLevel();

	if(hr == D3DERR_DEVICELOST)
	{
		if ((this->flags & DEVICE_LOST) == 0)
		{
			this->OnLostDevice();
			this->flags |= DEVICE_LOST;
		}
	}
	else if (hr == D3DERR_DEVICENOTRESET)
	{
		this->flags &= (~DEVICE_LOST);
		this->OnResetDevice();
	}

	return (this->flags & DEVICE_LOST) != 0;
}
//***********************************
//begin render
//***********************************
HQReturnVal HQDeviceD3D9::BeginRender(HQBool isClearPixel, HQBool isClearDepth, HQBool isClearStencil, hquint32 numRTsToClear){
#if defined DEBUG || defined _DEBUG
	if(!pDevice)
	{
		Log("not init before use!");
		return HQ_DEVICE_NOT_INIT;
	}
#endif
	if (this->flags & RENDER_BEGUN)
		return HQ_FAILED_RENDER_ALREADY_BEGUN;

	if (this->flags & DEVICE_LOST)
		return HQ_FAILED_DEVICE_LOST;
	
	HRESULT hr;
#if 0
	DWORD l_flags=0;
	if(isClearPixel)
		l_flags|=D3DCLEAR_TARGET;//0x1
	if(isClearDepth)
		l_flags|=D3DCLEAR_ZBUFFER;//0x2
	if(isClearStencil)
		l_flags|=D3DCLEAR_STENCIL;//0x4
#else
	DWORD l_flags = (isClearPixel) | (isClearDepth << 1) | (isClearStencil << 2);
#endif
	
	//clear toàn bộ màn hình hoặc toàn bộ bề mặt render target
	{
		D3DVIEWPORT9 VP={0,0,
			static_cast<HQRenderTargetManagerD3D9*> (this->renderTargetMan)->GetRTWidth(),
			static_cast<HQRenderTargetManagerD3D9*> (this->renderTargetMan)->GetRTHeight(),
			0.0f,1.0f};

		pDevice->SetViewport(&VP);//set viewport full kích thước render target
		
		if (isClearPixel && numRTsToClear > 0)
		{
			//clear first {numRTsToClear} render targets
			static_cast<HQRenderTargetManagerD3D9*> (this->renderTargetMan)->ClearRenderTargets(numRTsToClear);
			//clear depth stencil
			hr = pDevice->Clear(0, 0, l_flags & (~D3DCLEAR_TARGET), clearColor, clearDepth, clearStencil);
		}
		else //clear all render targets
			hr=pDevice->Clear(0,0,l_flags,clearColor,clearDepth,clearStencil);

		pDevice->SetViewport(&this->d3dViewPort);//chỉnh lại viewport đang active
	}
	
#if defined DEBUG || defined _DEBUG		
	if(FAILED(hr))
	{
		this->Log("Clear() failed!Error Code %d. Make sure you didn't clear depth/stencil when there is no depth/stencil buffer",hr);
		return HQ_FAILED;
	}
#endif

	hr=pDevice->BeginScene();
#if defined DEBUG || defined _DEBUG
	if(FAILED(hr))
	{
		this->Log("BeginScene() failed!Error Code %d",hr);
		return HQ_FAILED;
	}
#endif
	
	this->flags |= RENDER_BEGUN;

	return HQ_OK;
}
//****************************************
//end render
//****************************************
HQReturnVal HQDeviceD3D9::EndRender(){
#if defined DEBUG || defined _DEBUG
	if(!pDevice)
	{
		Log("not init before use!");
		return HQ_DEVICE_NOT_INIT;
	}
#endif
	if ((this->flags & RENDER_BEGUN) == 0)
		return HQ_FAILED_RENDER_NOT_BEGUN;
	if(this->flags & DEVICE_LOST)
		return HQ_FAILED_DEVICE_LOST;

	HRESULT hr;
	hr=pDevice->EndScene();
#if defined DEBUG || defined _DEBUG
	if(FAILED(hr)){
		Log("EndScene() failed!Error Code %d",hr);
		return HQ_FAILED;
	}
#endif
	
	this->flags &= ~RENDER_BEGUN;

	return HQ_OK;
}


/*----------------------------------
DisplayBackBuffer()
------------------------------*/
HQReturnVal HQDeviceD3D9::DisplayBackBuffer()
{
	HRESULT hr = pDevice->Present(0,0,0,0);
#if defined DEBUG || defined _DEBUG
	if(FAILED(hr)){
		Log("Present() failed!Error Code %d",hr);
		return HQ_FAILED;
	}
#endif
	

	return HQ_OK;
}

//***********************************
//clear
//***********************************
HQReturnVal HQDeviceD3D9::Clear(HQBool isClearPixel,HQBool isClearDepth,HQBool isClearStencil, hquint32 numRTsToClear)
{
#if defined DEBUG || defined _DEBUG
	if(!pDevice)
	{
		Log("not init before use!");
		return HQ_DEVICE_NOT_INIT;
	}
#endif

	if (this->flags & DEVICE_LOST)
		return HQ_FAILED_DEVICE_LOST;

#if 0
	DWORD l_flags=0;
	if(isClearPixel)
		l_flags|=D3DCLEAR_TARGET;//0x1
	if(isClearDepth)
		l_flags|=D3DCLEAR_ZBUFFER;//0x2
	if(isClearStencil)
		l_flags|=D3DCLEAR_STENCIL;//0x4
#else
	DWORD l_flags = (isClearPixel) | (isClearDepth << 1) | (isClearStencil << 2);
#endif

	HRESULT hr;

	//clear toàn bộ màn hình hoặc toàn bộ bề mặt render target
	{
		D3DVIEWPORT9 VP={0,0,
			static_cast<HQRenderTargetManagerD3D9*> (this->renderTargetMan)->GetRTWidth(),
			static_cast<HQRenderTargetManagerD3D9*> (this->renderTargetMan)->GetRTHeight(),
			0.0f,1.0f};

		pDevice->SetViewport(&VP);
		
		if (isClearPixel && numRTsToClear > 0)
		{
			//clear first {numRTsToClear} render targets
			static_cast<HQRenderTargetManagerD3D9*> (this->renderTargetMan)->ClearRenderTargets(numRTsToClear);
			//clear depth stencil
			hr = pDevice->Clear(0, 0, l_flags & (~D3DCLEAR_TARGET), clearColor, clearDepth, clearStencil);
		}
		else //clear all render targets
			hr = pDevice->Clear(0, 0, l_flags, clearColor, clearDepth, clearStencil);

		pDevice->SetViewport(&this->d3dViewPort);//chỉnh lại viewport đang active
	}
	
#if defined DEBUG || defined _DEBUG		
	if(FAILED(hr))
	{
		this->Log("Clear() failed!Error Code %d. Make sure you didn't clear depth/stencil when there is no depth/stencil buffer",hr);
		return HQ_FAILED;
	}
#endif
	
	return HQ_OK;
}
//***********************************
//set clear values
//***********************************
void HQDeviceD3D9::SetClearColori(hq_uint32 red,hq_uint32 green,hq_uint32 blue,hq_uint32 alpha){
	clearColor=D3DCOLOR_ARGB(alpha,red,green,blue);
}
void HQDeviceD3D9::SetClearColorf(hq_float32 red, hq_float32 green, hq_float32 blue, hq_float32 alpha){
	clearColor=D3DCOLOR_COLORVALUE(red,green,blue,alpha);
}
void HQDeviceD3D9::SetClearDepthVal(hq_float32 val){
	clearDepth=val;
}
void HQDeviceD3D9::SetClearStencilVal(hq_uint32 val){
	clearStencil=val;
}

void HQDeviceD3D9::GetClearColor(HQColor &clearColorOut) const 
{
	clearColorOut.r = ((clearColor >> 16) & 0xff) / 255.f;
	clearColorOut.g = ((clearColor >> 8) & 0xff) / 255.f;
	clearColorOut.b = ((clearColor >> 0) & 0xff) / 255.f;
	clearColorOut.a = ((clearColor >> 24) & 0xff) / 255.f;
}

/*--------------------------------------*/
HQReturnVal HQDeviceD3D9::SetDisplayMode(hq_uint32 width,hq_uint32 height,bool windowed)
{
	return ResizeBackBuffer(width, height, windowed, true);
}
HQReturnVal HQDeviceD3D9::OnWindowSizeChanged(hq_uint32 width,hq_uint32 height)
{
	if (!this->IsWindowed())
		return HQ_FAILED;
	return ResizeBackBuffer(width, height, this->IsWindowed(), false);
}
HQReturnVal HQDeviceD3D9::ResizeBackBuffer(hq_uint32 width,hq_uint32 height, bool windowed, bool resizeWindow)
{
	if (width == sWidth && height == sHeight && windowed == this->IsWindowed())
		return HQ_OK;
	bool found = pEnum->ChangeSelectedDisplay(width,height,windowed);
	if(!found)
		return HQ_FAILED;
	
	this->OnLostDevice();

	RECT Rect={0,0,width,height};
	if(!windowed)
	{
		SetParent(winfo.hwind,NULL);
		SetWindowLongPtrW(winfo.hwind,GWL_STYLE,WS_POPUP);
		if(resizeWindow)
		{
			AdjustWindowRect(&Rect,WS_POPUP,FALSE);
			SetWindowPos(winfo.hwind, HWND_TOP, 0, 0,
				Rect.right-Rect.left, Rect.bottom-Rect.top, SWP_NOZORDER);
		}

		this->d3dp.FullScreen_RefreshRateInHz = pEnum->selectedMode->RefreshRate;
		flags &= (~WINDOWED);
	}
	else
	{
		SetParent(winfo.hwind,winfo.hparent);
		SetWindowLongPtrW(winfo.hwind,GWL_STYLE,winfo.styles);
		if(resizeWindow)
		{
			AdjustWindowRect(&Rect,winfo.styles,FALSE);
			SetWindowPos(winfo.hwind, HWND_TOP, winfo.x, winfo.y,
				Rect.right-Rect.left, Rect.bottom-Rect.top, SWP_NOZORDER);
		}
		this->d3dp.FullScreen_RefreshRateInHz = 0;
		flags |=WINDOWED;
	}
	
	pEnum->SaveSettingFile(this->settingFileDir);

	
    
    sWidth = width;
    sHeight = height;

	
	this->d3dp.BackBufferWidth = width;
	this->d3dp.BackBufferHeight = height;
	this->d3dp.Windowed = windowed;

	DWORD quality[2];
	HRESULT hr=pD3D->CheckDeviceMultiSampleType(pEnum->selectedAdapter->adapter,pEnum->selectedDevice->devType,
		pEnum->selectedCombo->backFmt,this->d3dp.Windowed,pEnum->selectedMulSampleType,&quality[0]);//check multisample type on back buffer
	if(!FAILED(hr))
		hr=pD3D->CheckDeviceMultiSampleType(pEnum->selectedAdapter->adapter,pEnum->selectedDevice->devType,
			pEnum->selectedDepthStencilFmt,this->d3dp.Windowed,pEnum->selectedMulSampleType,&quality[1]);//check multisample type on depth stencil buffer

	if(FAILED(hr))
	{
		this->d3dp.MultiSampleQuality=0;
		this->d3dp.MultiSampleType=D3DMULTISAMPLE_NONE;
	}
	else{
		this->d3dp.MultiSampleQuality=min(quality[0] , quality[1])-1;
		this->d3dp.MultiSampleType=pEnum->selectedMulSampleType;
	}

	this->OnResetDevice();

	ShowWindow(winfo.hwind, SW_SHOW);

	return HQ_OK;
}


/*----------------render-------------------*/
void HQDeviceD3D9::SetPrimitiveMode(HQPrimitiveMode _primitiveMode) 
{
	this->primitiveMode = this->primitiveLookupTable[_primitiveMode];
}
HQReturnVal HQDeviceD3D9::Draw(hq_uint32 vertexCount , hq_uint32 firstVertex) 
{
	if ((this->flags & RENDER_BEGUN)== 0)
		return HQ_FAILED_RENDER_NOT_BEGUN;
	if (this->flags & DEVICE_LOST)
		return HQ_FAILED_DEVICE_LOST;
	UINT primitiveCount;
	switch(this->primitiveMode)
	{
	case D3DPT_TRIANGLELIST:
		primitiveCount = vertexCount / 3;
		break;
	case D3DPT_TRIANGLESTRIP:
#if defined DEBUG || defined _DEBUG
		if (vertexCount < 2)
			primitiveCount = 0;
		else
#endif
			primitiveCount = vertexCount - 2;
		break;
	case D3DPT_POINTLIST:
		primitiveCount = vertexCount;
		break;
	case D3DPT_LINELIST:
		primitiveCount = vertexCount >> 1;//vertexCount / 2
		break;
	case D3DPT_LINESTRIP:
#if defined DEBUG || defined _DEBUG
		if (vertexCount < 1)
			primitiveCount = 0;
		else
#endif
			primitiveCount = vertexCount - 1;
		break;
	default:
		primitiveCount = 0;
	}

	static_cast<HQShaderManagerD3D9*>(this-> shaderMan)->Commit();
	pDevice->DrawPrimitive(this->primitiveMode , firstVertex , primitiveCount);
	return HQ_OK;
}

HQReturnVal HQDeviceD3D9::DrawPrimitive(hq_uint32 primitiveCount , hq_uint32 firstVertex) 
{
	if ((this->flags & RENDER_BEGUN)== 0)
		return HQ_FAILED_RENDER_NOT_BEGUN;
	if (this->flags & DEVICE_LOST)
		return HQ_FAILED_DEVICE_LOST;
	
	static_cast<HQShaderManagerD3D9*>(this->shaderMan)->Commit();
	pDevice->DrawPrimitive(this->primitiveMode , firstVertex , primitiveCount);
	return HQ_OK;
}

HQReturnVal HQDeviceD3D9::DrawIndexed(hq_uint32 numVertices , hq_uint32 indexCount , hq_uint32 firstIndex )
{
	if ((this->flags & RENDER_BEGUN)== 0)
		return HQ_FAILED_RENDER_NOT_BEGUN;
	if (this->flags & DEVICE_LOST)
		return HQ_FAILED_DEVICE_LOST;
	UINT primitiveCount;
	switch(this->primitiveMode)
	{
	case D3DPT_TRIANGLELIST:
		primitiveCount = indexCount / 3;
		break;
	case D3DPT_TRIANGLESTRIP:
#if defined DEBUG || defined _DEBUG
		if (indexCount < 2)
			primitiveCount = 0;
		else
#endif
			primitiveCount = indexCount - 2;
		break;
	case D3DPT_POINTLIST:
		primitiveCount = indexCount;
		break;
	case D3DPT_LINELIST:
		primitiveCount = indexCount >> 1;//indexCount / 2
		break;
	case D3DPT_LINESTRIP:
#if defined DEBUG || defined _DEBUG
		if (indexCount < 1)
			primitiveCount = 0;
		else
#endif
			primitiveCount = indexCount - 1;
		break;
	default:
		primitiveCount = 0;
	}

	static_cast<HQShaderManagerD3D9*>(this->shaderMan)->Commit();
	pDevice->DrawIndexedPrimitive(this->primitiveMode , 0 , 0 , numVertices , firstIndex , primitiveCount);
	return HQ_OK;
}

HQReturnVal HQDeviceD3D9::DrawIndexedPrimitive(hq_uint32 numVertices , hq_uint32 primitiveCount , hq_uint32 firstIndex )
{
	if ((this->flags & RENDER_BEGUN)== 0)
		return HQ_FAILED_RENDER_NOT_BEGUN;
	if (this->flags & DEVICE_LOST)
		return HQ_FAILED_DEVICE_LOST;

	static_cast<HQShaderManagerD3D9*>(this->shaderMan)->Commit();
	pDevice->DrawIndexedPrimitive(this->primitiveMode , 0 , 0 , numVertices , firstIndex , primitiveCount);
	return HQ_OK;
}

HQReturnVal HQDeviceD3D9::SetViewport(const HQViewPort &viewport)
{
	HQReturnVal re = HQ_OK;
	UINT width = static_cast<HQRenderTargetManagerD3D9*> (this->renderTargetMan)->GetRTWidth();
	UINT height = static_cast<HQRenderTargetManagerD3D9*> (this->renderTargetMan)->GetRTHeight();
	
	if (viewport.x + viewport.width > width || viewport.y + viewport.height > height)//viewport area is invalid
	{
		this->currentVPs[0].width = width;
		this->currentVPs[0].height = height;
		this->currentVPs[0].x = this->currentVPs[0].y = 0;

		re = HQ_WARNING_VIEWPORT_IS_INVALID;
	}
	else
		this->currentVPs[0] = viewport;
	
	this->d3dViewPort.X = this->currentVPs[0].x;
	this->d3dViewPort.Y = this->currentVPs[0].y;
	this->d3dViewPort.Width = this->currentVPs[0].width;
	this->d3dViewPort.Height = this->currentVPs[0].height;

	pDevice->SetViewport(&this->d3dViewPort);

	return re;
}
