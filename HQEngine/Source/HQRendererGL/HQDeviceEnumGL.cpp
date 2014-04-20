/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"

#include "glHeaders.h"
#ifdef HQ_OPENGLES
#include "HQDeviceGL.h"
#else
#include "HQDeviceEnumGL.h"
#include "HQRenderTargetGL.h"
#endif

#include <stdio.h>
#include <math.h>
#include "../HQHashTable.h"

extern "C"
{
#if defined(_WIN32)
extern GLenum GLEWAPIENTRY wglewContextInit (void);
#elif !defined(__ANDROID__) && !defined(__native_client__) && (!defined(__APPLE__) || defined(GLEW_APPLE_GLX))
extern GLenum GLEWAPIENTRY glxewContextInit (void);
#endif /* _WIN32 */
}

#define REQUIRE_EXACT_RGB 1
#define REQUIRE_EXACT_DEPTH_STENCIL 0
#define REFRESHRATEDIFF 0.00001

const int numPixelFmt=9;//số lượng pixel format hỗ trợ
const FORMAT PixelFormat[]={
	SFMT_A8R8G8B8,   SFMT_X8R8G8B8,
	SFMT_R8G8B8, SFMT_R5G6B5,
	SFMT_A1R5G5B5, SFMT_X1R5G5B5,
	SFMT_X4R4G4B4, SFMT_A4R4G4B4,
	SFMT_A2B10G10R10
};

const int numMulSample=4;//số lượng kiểu siêu lấy mẫu hỗ trợ
const hquint32 MulSample[]=
{
	0,
	2,
	4,
	8
};



//helper functions
namespace helper{
#ifdef HQ_IPHONE_PLATFORM
	const NSString* GetEAGLColorFormat(FORMAT fmt)
	{
		switch(fmt)
		{
			case SFMT_R5G6B5:
			case SFMT_X1R5G5B5:
				return kEAGLColorFormatRGB565;
			default: return kEAGLColorFormatRGBA8;
		}
	}

	GLenum GetEAGLDepthStencilFormat(FORMAT depthStencilFmt)
	{
		switch(depthStencilFmt)
		{
			case SFMT_D24S8:
			case SFMT_D24X4S4:
			case SFMT_D15S1:
				return GL_DEPTH24_STENCIL8_OES;
			case SFMT_D32:
			case SFMT_D24X8:
				return GL_DEPTH_COMPONENT24_OES;
			case SFMT_D16:
				return GL_DEPTH_COMPONENT16;
			case SFMT_S8:
				return GL_STENCIL_INDEX8;
			case SFMT_NODEPTHSTENCIL:
				return 0;
			default: return 0;
		}
	}
	
#else

	void FormatInfo(FORMAT fmt, hq_ubyte8 *pRGBBitCount,
					hq_ubyte8 *pRBits,
					hq_ubyte8 *pGBits,
					hq_ubyte8 *pBBits,
					hq_ubyte8 *pABits,
					hq_ubyte8 *pDBits,hq_ubyte8 *pSBits)//depth,stencil
	{
		switch(fmt)
		{
		case SFMT_R8G8B8:
			if(pRGBBitCount)
				*pRGBBitCount=24;
			if(pRBits)
			{
				*pRBits=8;
			}
			if(pGBits)
			{
				*pGBits=8;
			}
			if(pBBits)
			{
				*pBBits=8;
			}
			if(pABits)
			{
				*pABits=0;
			}
			break;
		case SFMT_A8R8G8B8:
			if(pRGBBitCount)
				*pRGBBitCount=24;
			if(pRBits)
			{
				*pRBits=8;
			}
			if(pGBits)
			{
				*pGBits=8;
			}
			if(pBBits)
			{
				*pBBits=8;
			}
			if(pABits)
			{
				*pABits=8;
			}
			break;
		case SFMT_X8R8G8B8:
			if(pRGBBitCount)
				*pRGBBitCount=24;
			if(pRBits)
			{
				*pRBits=8;
			}
			if(pGBits)
			{
				*pGBits=8;
			}
			if(pBBits)
			{
				*pBBits=8;
			}
			if(pABits)
			{
				*pABits=0;
			}
			break;
		case SFMT_R5G6B5:
			if(pRGBBitCount)
				*pRGBBitCount=16;
			if(pRBits)
			{
				*pRBits=5;
			}
			if(pGBits)
			{
				*pGBits=6;
			}
			if(pBBits)
			{
				*pBBits=5;
			}
			if(pABits)
			{
				*pABits=0;
			}
			break;
		case SFMT_A1R5G5B5:
			if(pRGBBitCount)
				*pRGBBitCount=15;
			if(pRBits)
			{
				*pRBits=5;
			}
			if(pGBits)
			{
				*pGBits=5;
			}
			if(pBBits)
			{
				*pBBits=5;
			}
			if(pABits)
			{
				*pABits=1;
			}
			break;
		case SFMT_X1R5G5B5:
			if(pRGBBitCount)
				*pRGBBitCount=15;
			if(pRBits)
			{
				*pRBits=5;
			}
			if(pGBits)
			{
				*pGBits=5;
			}
			if(pBBits)
			{
				*pBBits=5;
			}
			if(pABits)
			{
				*pABits=0;
			}
			break;
		case SFMT_X4R4G4B4:
			if(pRGBBitCount)
				*pRGBBitCount=12;
			if(pRBits)
			{
				*pRBits=4;
			}
			if(pGBits)
			{
				*pGBits=4;
			}
			if(pBBits)
			{
				*pBBits=4;
			}
			if(pABits)
			{
				*pABits=0;
			}
			break;
		case SFMT_A4R4G4B4:
			if(pRGBBitCount)
				*pRGBBitCount=12;
			if(pRBits)
			{
				*pRBits=4;
			}
			if(pGBits)
			{
				*pGBits=4;
			}
			if(pBBits)
			{
				*pBBits=4;
			}
			if(pABits)
			{
				*pABits=4;
			}
			break;
		case SFMT_A2B10G10R10:
			if(pRGBBitCount)
				*pRGBBitCount=30;
			if(pRBits)
			{
				*pRBits=10;
			}
			if(pGBits)
			{
				*pGBits=10;
			}
			if(pBBits)
			{
				*pBBits=10;
			}
			if(pABits)
			{
				*pABits=2;
			}
			break;
		case SFMT_D24S8:
			if(pDBits)
				*pDBits=24;
			if(pSBits)
				*pSBits=8;
			break;
		case SFMT_D24X4S4:
			if(pDBits)
				*pDBits=24;
			if(pSBits)
				*pSBits=4;
			break;
		case SFMT_D15S1:
			if(pDBits)
				*pDBits=15;
			if(pSBits)
				*pSBits=1;
			break;
		case SFMT_D32:
			if(pDBits)
				*pDBits=32;
			if(pSBits)
				*pSBits=0;
			break;
		case SFMT_D24X8:
			if(pDBits)
				*pDBits=24;
			if(pSBits)
				*pSBits=0;
			break;
		case SFMT_D16:
			if(pDBits)
				*pDBits=16;
			if(pSBits)
				*pSBits=0;
			break;
		case SFMT_S8:
			if(pDBits)
				*pDBits=0;
			if(pSBits)
				*pSBits=8;
			break;
		case SFMT_NODEPTHSTENCIL:
			if(pDBits)
				*pDBits=0;
			if(pSBits)
				*pSBits=0;
			break;
		default: return;
		}
	}
#endif
#ifdef WIN32
	LRESULT CALLBACK DummyProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
	{
		return DefWindowProc(hwnd, message, wParam, lParam);
	}
#endif
};
//********************************************
//constructor
//********************************************
#ifdef WIN32
HQDeviceEnumGL::HQDeviceEnumGL(HMODULE pDll){
	this->pDll = pDll;
#elif defined HQ_LINUX_PLATFORM
HQDeviceEnumGL::HQDeviceEnumGL(Display *dpy){
	this->dpy = dpy;
#elif defined HQ_ANDROID_PLATFORM
HQDeviceEnumGL::HQDeviceEnumGL(jobject _jegl, jobject _jdisplay, jint apiLevel){
	if (apiLevel == 1 || apiLevel == 2)
		this->selectedApiLevel = apiLevel;
	else
		this->selectedApiLevel = 1;

	this->jegl = _jegl;
	this->jdisplay = _jdisplay;
#else
HQDeviceEnumGL::HQDeviceEnumGL(){
#	ifdef HQ_MAC_PLATFORM
	modeList = NULL;
	currentScreenDisplayMode = NULL;
#	endif
#endif

#ifndef HQ_OPENGLES
	windowed=true;
#else
	selectedPixelFormat = SFMT_R5G6B5;
#endif
	selectedDepthStencilFmt=SFMT_D16;
	selectedMulSampleType=0;
	vsync = 0;//default vsync is off
}
HQDeviceEnumGL::~HQDeviceEnumGL()
{
#ifdef HQ_LINUX_PLATFORM /*----------linux---------------*/
#	if defined HQ_USE_XFREE86_VIDMODE//xfree86 vidmode
    XFree(modes);
#	else //randr
    XRRFreeScreenConfigInfo(this->screenConfig);
#	endif
#elif defined HQ_MAC_PLATFORM /*------------mac osx-------------*/
	if (modeList != NULL)
		CFRelease(modeList);
	if (currentScreenDisplayMode != NULL)
		CGDisplayModeRelease(currentScreenDisplayMode);
#endif
}
//********************************************
//Truy vấn khả năng của card đồ họa
//********************************************
//helper functions
#ifdef WIN32
bool HQDeviceEnumGL::CheckPixelFmt(FORMAT format){
	int ipixelFormat;

	PIXELFORMATDESCRIPTOR pixFmt;
	memset(&pixFmt,0,sizeof(PIXELFORMATDESCRIPTOR));
	//get color bits
	helper::FormatInfo(format,&pixFmt.cColorBits,&pixFmt.cRedBits,
					   &pixFmt.cGreenBits,&pixFmt.cBlueBits,
					   &pixFmt.cAlphaBits,NULL,NULL);

	if(WGLEW_ARB_pixel_format)
		goto wgl;//will use wglChoosePixelFormatARB instead

	pixFmt.nSize=sizeof(PIXELFORMATDESCRIPTOR);
	pixFmt.nVersion   = 1;
    pixFmt.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pixFmt.iPixelType = PFD_TYPE_RGBA;
	pixFmt.iLayerType = PFD_MAIN_PLANE;

	ipixelFormat=ChoosePixelFormat(hDC,&pixFmt);//find best match
	if(ipixelFormat==0)
		return false;
	PIXELFORMATDESCRIPTOR bestFmt;
	DescribePixelFormat(hDC,ipixelFormat,sizeof(PIXELFORMATDESCRIPTOR),&bestFmt);

	if(pixFmt.cRedBits != bestFmt.cRedBits ||
	   pixFmt.cGreenBits != bestFmt.cGreenBits || pixFmt.cBlueBits != bestFmt.cBlueBits ||
	   pixFmt.cAlphaBits != bestFmt.cAlphaBits/* || pixFmt.cRedShift != bestFmt.cRedShift ||
	   pixFmt.cGreenShift != bestFmt.cGreenShift || pixFmt.cBlueShift != bestFmt.cBlueShift ||
	   pixFmt.cAlphaShift != bestFmt.cAlphaShift*/
	   )//không như mong muốn
		return false;
	return true;
wgl:
	hq_float32 fAttributes[] = {0, 0};
	UINT  numFormats;
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
					WGL_DOUBLE_BUFFER_ARB, GL_TRUE,
					WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
					0, 0 };
	if(wglChoosePixelFormatARB(hDC, iAttributes, fAttributes,
		1, &ipixelFormat, &numFormats)==FALSE)
		return false;
	if(numFormats<1)
		return false;
	bool match=false;

	for(UINT i=0;i<numFormats;++i)
	{
		//kiểm tra lại 1 lần :
		int iResults[5];
		iAttributes[0]=WGL_RED_BITS_ARB;
		iAttributes[1]=WGL_GREEN_BITS_ARB;
		iAttributes[2]=WGL_BLUE_BITS_ARB;
		iAttributes[3]=WGL_ALPHA_BITS_ARB;
		//iAttributes[4]=WGL_COLOR_BITS_ARB;
		//iAttributes[5]=WGL_RED_SHIFT_ARB;
		//iAttributes[6]=WGL_GREEN_SHIFT_ARB;
		//iAttributes[7]=WGL_BLUE_SHIFT_ARB;
		//iAttributes[8]=WGL_ALPHA_SHIFT_ARB;
		iAttributes[4]=0;
		if(wglGetPixelFormatAttribivARB(hDC, ipixelFormat+i, 0, 4,iAttributes, iResults)==FALSE)
			continue;
		if(pixFmt.cRedBits == iResults[0] &&
		   pixFmt.cGreenBits == iResults[1] && pixFmt.cBlueBits == iResults[2] &&
		   pixFmt.cAlphaBits == iResults[3] /*|| pixFmt.cRedShift != iResults[5] ||
		   pixFmt.cGreenShift != iResults[6] || pixFmt.cBlueShift != iResults[7] ||
		   pixFmt.cAlphaShift != iResults[8]*/
		   )//không như mong muốn
		   {
			   match=true;
			   break;
		   }
	}
	if(!match)
		return false;

	return true;
}

bool HQDeviceEnumGL::CheckDepthStencilFmt(BufferInfo& bufInfo){
	int ipixelFormat;

	PIXELFORMATDESCRIPTOR pixFmt;
	memset(&pixFmt,0,sizeof(PIXELFORMATDESCRIPTOR));
	//get color bits
	helper::FormatInfo(bufInfo.pixelFmt,&pixFmt.cColorBits,&pixFmt.cRedBits,
					   &pixFmt.cGreenBits,&pixFmt.cBlueBits,
					   &pixFmt.cAlphaBits,NULL,NULL);
	//get depth ,stencil bits
	helper::FormatInfo(bufInfo.depthStencilFmt,NULL,NULL,
					   NULL,NULL,NULL,&pixFmt.cDepthBits,&pixFmt.cStencilBits);

	if(WGLEW_ARB_pixel_format)
		goto wgl;//will use wglChoosePixelFormatARB instead


	pixFmt.nSize=sizeof(PIXELFORMATDESCRIPTOR);
	pixFmt.nVersion   = 1;
    pixFmt.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	if (pixFmt.cDepthBits = pixFmt.cStencilBits = 0)
		pixFmt.dwFlags |= PFD_DEPTH_DONTCARE;
	pixFmt.iPixelType = PFD_TYPE_RGBA;
	pixFmt.iLayerType = PFD_MAIN_PLANE;

	//lưu các giá trị mong muốn
	hq_ubyte8 depthBits=pixFmt.cDepthBits;
	hq_ubyte8 stencilBits=pixFmt.cStencilBits;

	ipixelFormat=ChoosePixelFormat(hDC,&pixFmt);//find best match
	if(ipixelFormat==0)
		return false;
	/*
	DescribePixelFormat(hDC,ipixelFormat,sizeof(PIXELFORMATDESCRIPTOR),&pixFmt);

	if(pixFmt.cDepthBits != depthBits || pixFmt.cStencilBits != stencilBits)//không như mong muốn
		return false;
	*/
	return true;
wgl:
	hq_float32 fAttributes[] = {0, 0};
	UINT  numFormats;
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
					0, 0 };
	if(wglChoosePixelFormatARB(hDC, iAttributes, fAttributes,
		1, &ipixelFormat, &numFormats)==FALSE)
		return false;
	if(numFormats<1)
		return false;
	/*
	//kiểm tra lại 1 lần :
	int iResults[3];
	iAttributes[0]=WGL_DEPTH_BITS_ARB;
	iAttributes[1]=WGL_STENCIL_BITS_ARB;
	iAttributes[2]=0;
	if(wglGetPixelFormatAttribivARB(hDC, ipixelFormat, 0, 2,iAttributes, iResults)==FALSE)
		return false;
	if(pixFmt.cDepthBits != iResults[0] || pixFmt.cStencilBits != iResults[1])//không như mong muốn
		return false;
	*/
	return true;
}

bool HQDeviceEnumGL::CheckMultisample(BufferInfo &bufInfo)
{
	if(bufInfo.maxMulSampleLevel==0)//không dùng multisample
		return true;
	if(!WGLEW_ARB_pixel_format || !WGLEW_ARB_multisample)
		return false;
	int ipixelFormat;

	PIXELFORMATDESCRIPTOR pixFmt;
	memset(&pixFmt,0,sizeof(PIXELFORMATDESCRIPTOR));
	//get color bits
	helper::FormatInfo(bufInfo.pixelFmt,&pixFmt.cColorBits,&pixFmt.cRedBits,
					   &pixFmt.cGreenBits,&pixFmt.cBlueBits,
					   &pixFmt.cAlphaBits,NULL,NULL);
	//get depth ,stencil bits
	helper::FormatInfo(bufInfo.depthStencilFmt,NULL,NULL,
					   NULL,NULL,NULL,&pixFmt.cDepthBits,&pixFmt.cStencilBits);

	hq_float32 fAttributes[] = {0, 0};
	UINT  numFormats;
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
					WGL_SAMPLE_BUFFERS_ARB,GL_TRUE,
					WGL_SAMPLES_ARB, (int)bufInfo.maxMulSampleLevel ,
					0, 0 };
	if(wglChoosePixelFormatARB(hDC, iAttributes, fAttributes,
		1, &ipixelFormat, &numFormats)==FALSE)
		return false;
	if(numFormats<1)
		return false;

	//kiểm tra lại 1 lần :
	int iResults[3];
	iAttributes[0]=WGL_SAMPLE_BUFFERS_ARB;
	iAttributes[1]=WGL_SAMPLES_ARB;
	iAttributes[2]=0;
	if(wglGetPixelFormatAttribivARB(hDC, ipixelFormat, 0, 2,iAttributes, iResults)==FALSE)
		return false;
	if(GL_TRUE != iResults[0] || bufInfo.maxMulSampleLevel != iResults[1])//không như mong muốn
		return false;
	return true;
}
#elif defined (HQ_LINUX_PLATFORM)
bool HQDeviceEnumGL::CheckPixelFmt(FORMAT format){
	XVisualInfo *vi=NULL;

        hq_ubyte8 R,G,B,A;
	//get color bits
	helper::FormatInfo(format,NULL,&R,
                           &G,&B,
                           &A,NULL,NULL);

	GLint iAttributes[] = { GLX_RED_SIZE, R,
                               GLX_GREEN_SIZE, G,
                                GLX_BLUE_SIZE, B,
                                GLX_ALPHA_SIZE, A,
                                GLX_DOUBLEBUFFER,
                                GLX_RGBA,
                                None };


        vi=glXChooseVisual(dpy,0,iAttributes);
	if(vi==NULL)
		return false;

        //kiểm tra lại 1 lần :
        int iResults[4];

        int iAttributes2[]={
                        GLX_RED_SIZE,
                        GLX_GREEN_SIZE,
                        GLX_BLUE_SIZE,
                        GLX_ALPHA_SIZE
                        };
        for(int i=0;i<4;++i)
        {
            if((glXGetConfig(dpy,vi,iAttributes2[i],&iResults[i]))!=0)
                return false;
        }
        if(iAttributes[1] != iResults[0]||iAttributes[3] != iResults[1]||
           iAttributes[5] != iResults[2]||iAttributes[7] != iResults[3]
           )//không như mong muốn
        {
            return false;
	}

	return true;
}

bool HQDeviceEnumGL::CheckDepthStencilFmt(BufferInfo& bufInfo){
	XVisualInfo *vi=NULL;
        hq_ubyte8 R,G,B,A,D,S;
        //get color bits
	helper::FormatInfo(bufInfo.pixelFmt,NULL,&R,
                           &G,&B,
                           &A,NULL,NULL);
	//get depth ,stencil bits
	helper::FormatInfo(bufInfo.depthStencilFmt,NULL,NULL,
                           NULL,NULL,NULL,&D,&S);

	GLint iAttributes[] = { GLX_RED_SIZE, R,
                               GLX_GREEN_SIZE, G,
                                GLX_BLUE_SIZE, B,
                                GLX_ALPHA_SIZE, A,
                                GLX_DEPTH_SIZE,D,
                                GLX_STENCIL_SIZE,S,
                                GLX_DOUBLEBUFFER,
                                GLX_RGBA,
                                None };


        vi=glXChooseVisual(dpy,0,iAttributes);
	if(vi==NULL)
		return false;
		/*
        //kiểm tra lại 1 lần :
        int iResults[2];

        int iAttributes2[]={
                        GLX_DEPTH_SIZE,
                        GLX_STENCIL_SIZE
                        };
       for(int i=0;i<2;++i)
        {
            if((glXGetConfig(dpy,vi,iAttributes2[i],&iResults[i]))!=0)
                return false;
        }
        if(iAttributes[9] != iResults[0]||iAttributes[11] != iResults[1]
           )//không như mong muốn
        {
            return false;
	}
	*/
	return true;
}

bool HQDeviceEnumGL::CheckMultisample(BufferInfo &bufInfo)
{
	if(bufInfo.maxMulSampleLevel==0)//không dùng multisample
		return true;
	if(!GLXEW_ARB_multisample)
		return false;
	XVisualInfo *vi=NULL;

         hq_ubyte8 R,G,B,A,D,S;
        //get color bits
	helper::FormatInfo(bufInfo.pixelFmt,NULL,&R,
                           &G,&B,
                           &A,NULL,NULL);
	//get depth ,stencil bits
	helper::FormatInfo(bufInfo.depthStencilFmt,NULL,NULL,
                           NULL,NULL,NULL,&D,&S);

	GLint iAttributes[] = { GLX_RED_SIZE, R,
                               GLX_GREEN_SIZE, G,
                                GLX_BLUE_SIZE, B,
                                GLX_ALPHA_SIZE, A,
                                GLX_DEPTH_SIZE,D,
                                GLX_STENCIL_SIZE,S,
                                GLX_DOUBLEBUFFER,
                                GLX_RGBA,
                                GLX_SAMPLE_BUFFERS_ARB,GL_TRUE,
                                GLX_SAMPLES_ARB,(int)bufInfo.maxMulSampleLevel,
                                None };

        vi=glXChooseVisual(dpy,0,iAttributes);
	if(vi==NULL)
		return false;

        //kiểm tra lại 1 lần :
        int iResults[2];

        int iAttributes2[]={
                        GLX_SAMPLE_BUFFERS_ARB,
                        GLX_SAMPLES_ARB
                        };
        for(int i=0;i<2;++i)
        {
            if((glXGetConfig(dpy,vi,iAttributes2[i],&iResults[i]))!=0)
                return false;
        }
        if(iAttributes[15] != iResults[0]||iAttributes[17] != iResults[1]
           )//không như mong muốn
        {
            return false;
	}

	return true;
}

#elif defined (HQ_MAC_PLATFORM)
bool HQDeviceEnumGL::CheckPixelFmt(FORMAT format){

	CGLPixelFormatObj pixelFormatObj = NULL;
	GLint numPixelFormats ;

	hq_ubyte8 R,G,B,A;
	//get color bits
	helper::FormatInfo(format,NULL,&R,
					   &G,&B,
					   &A,NULL,NULL);

	CGLPixelFormatAttribute attribs[11] = {
		kCGLPFADoubleBuffer,
		kCGLPFADisplayMask, (CGLPixelFormatAttribute) CGDisplayIDToOpenGLDisplayMask( kCGDirectMainDisplay ) ,
		kCGLPFAColorSize,	(CGLPixelFormatAttribute) (R + G + B),
		kCGLPFAAlphaSize , (CGLPixelFormatAttribute)A,
		kCGLPFAClosestPolicy,
		kCGLPFAAccelerated,
		kCGLPFANoRecovery,
		(CGLPixelFormatAttribute) 0
	};
	if (!caps.hardwareAccel) {
		attribs[8] = (CGLPixelFormatAttribute) 0;
	}

	CGLError cglError = CGLChoosePixelFormat( attribs, &pixelFormatObj, &numPixelFormats );

	if (cglError != kCGLNoError || pixelFormatObj == NULL) {
		return false;
	}

	CGLDestroyPixelFormat( pixelFormatObj );
	return true;
}

bool HQDeviceEnumGL::CheckDepthStencilFmt(BufferInfo& bufInfo){
	hq_ubyte8 R,G,B,A,D,S;
	//get color bits
	helper::FormatInfo(bufInfo.pixelFmt,NULL,&R,
					   &G,&B,
					   &A,NULL,NULL);
	//get depth ,stencil bits
	helper::FormatInfo(bufInfo.depthStencilFmt,NULL,NULL,
					   NULL,NULL,NULL,&D,&S);

	CGLPixelFormatObj pixelFormatObj = NULL;
	GLint numPixelFormats ;

	CGLPixelFormatAttribute attribs[15] = {
		kCGLPFADoubleBuffer,
		kCGLPFADisplayMask, (CGLPixelFormatAttribute)CGDisplayIDToOpenGLDisplayMask( kCGDirectMainDisplay ) ,
		kCGLPFAColorSize,	(CGLPixelFormatAttribute)(R + G + B),
		kCGLPFAAlphaSize , (CGLPixelFormatAttribute)A ,
		kCGLPFADepthSize , (CGLPixelFormatAttribute)D ,
		kCGLPFAStencilSize , (CGLPixelFormatAttribute)S ,
		kCGLPFAClosestPolicy,
		kCGLPFAAccelerated,
		kCGLPFANoRecovery,
		(CGLPixelFormatAttribute) 0
	};
	if (!caps.hardwareAccel) {
		attribs[12] = (CGLPixelFormatAttribute) 0;
	}


	CGLError cglError = CGLChoosePixelFormat( attribs, &pixelFormatObj, &numPixelFormats );

	if (cglError != kCGLNoError || pixelFormatObj == NULL) {
		return false;
	}

	CGLDestroyPixelFormat( pixelFormatObj );
	return true;
}

bool HQDeviceEnumGL::CheckMultisample(BufferInfo &bufInfo)
{
	if(bufInfo.maxMulSampleLevel==0)//không dùng multisample
		return true;

	if(!GLEW_ARB_multisample)
		return false;

	hq_ubyte8 R,G,B,A,D,S;
	//get color bits
	helper::FormatInfo(bufInfo.pixelFmt,NULL,&R,
					   &G,&B,
					   &A,NULL,NULL);
	//get depth ,stencil bits
	helper::FormatInfo(bufInfo.depthStencilFmt,NULL,NULL,
					   NULL,NULL,NULL,&D,&S);

	CGLPixelFormatObj pixelFormatObj = NULL;
	GLint numPixelFormats ;

	CGLPixelFormatAttribute attribs[20] = {
		kCGLPFADoubleBuffer,
		kCGLPFADisplayMask,(CGLPixelFormatAttribute) CGDisplayIDToOpenGLDisplayMask( kCGDirectMainDisplay ) ,
		kCGLPFAColorSize,(CGLPixelFormatAttribute)	(R + G + B),
		kCGLPFAAlphaSize ,(CGLPixelFormatAttribute) A ,
		kCGLPFADepthSize ,(CGLPixelFormatAttribute) D ,
		kCGLPFAStencilSize ,(CGLPixelFormatAttribute) S ,
		kCGLPFAClosestPolicy,
		kCGLPFAMultisample,
		kCGLPFASampleBuffers , (CGLPixelFormatAttribute) 1,
		kCGLPFASamples , (CGLPixelFormatAttribute) bufInfo.maxMulSampleLevel,
		kCGLPFAAccelerated,
		kCGLPFANoRecovery,
		(CGLPixelFormatAttribute) 0
	};
	if (!caps.hardwareAccel) {
		attribs[18] = (CGLPixelFormatAttribute) 0;
	}


	CGLError cglError = CGLChoosePixelFormat( attribs, &pixelFormatObj, &numPixelFormats );

	if (cglError != kCGLNoError || pixelFormatObj == NULL) {
		return false;
	}

	CGLDestroyPixelFormat( pixelFormatObj );
	return true;
}
#endif

//main capabilities
void HQDeviceEnumGL::CheckCapabilities()
{
	if(glewInit()!=GLEW_OK)
		return;


	GLint maxVal;
	
	//get max texture size
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxVal);
	caps.maxTextureSize = maxVal;
	glGetIntegerv(GL_MAX_CUBE_MAP_TEXTURE_SIZE, &maxVal);
	caps.maxCubeTextureSize = maxVal;
	//get max anosotropic support
	if(GLEW_EXT_texture_filter_anisotropic)
	{
		glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxVal);
		caps.maxAF=maxVal;
	}
	else
		caps.maxAF=0;

#ifndef HQ_OPENGLES
	caps.maxUniformBufferSlots = 0;
	if (GLEW_VERSION_3_1)
	{
		glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS , & maxVal);
		caps.maxUniformBufferSlots = maxVal;
	}

	caps.maxDrawBuffers = 0;
#endif

#ifndef HQ_IPHONE_PLATFORM
	if (GLEW_EXT_framebuffer_object || GLEW_VERSION_3_0)
	{
#endif
#ifndef HQ_OPENGLES
		glGetIntegerv(GL_MAX_DRAW_BUFFERS , & maxVal);
		caps.maxDrawBuffers = maxVal;

		glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS_EXT, &maxVal);
		if ((GLint)caps.maxDrawBuffers > maxVal)
			caps.maxDrawBuffers = maxVal;
#endif
		/*-----check supported render buffer format-------*/
		CheckRenderBufferFormatSupported();
#ifndef HQ_IPHONE_PLATFORM
	}
	else
	{
		for (int i = 0; i < NUM_RTT_FORMAT; ++i)
			this->caps.rttInternalFormat[i] = false;
		for (int i = 0; i < NUM_DS_FORMAT; ++i)
			this->caps.dsFormat[i] = false;
	}
#endif

	//get max fixed function texture unit
	glGetIntegerv(GL_MAX_TEXTURE_UNITS , & maxVal);
	caps.nFFTextureUnits = maxVal;

	/*-----------get max sampler units per shader-------------*/
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS , & maxVal);
	if (maxVal > 32)//openGL can't use more than 32 texture units , even if graphics card supports more than 32 texture image units
		caps.nShaderSamplerUnits = 32;
	else
		caps.nShaderSamplerUnits = maxVal;

	glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS , & maxVal);
	caps.nVertexShaderSamplers = maxVal;

	glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS , & maxVal);
	caps.nFragmentShaderSamplers = maxVal;


	if (GLEW_EXT_geometry_shader4 || GLEW_VERSION_3_2)
	{
		glGetIntegerv(GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS, &maxVal);
		caps.nGeometryShaderSamplers = maxVal;
	}
	else
		caps.nGeometryShaderSamplers = 0;

#ifdef GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS
	if (GLEW_VERSION_4_3)
	{
		glGetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS, &maxVal);
		caps.nComputeShaderSamplers = maxVal;
	}
	else
#endif//ifdef GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS
		caps.nComputeShaderSamplers = 0;

	//get max vertex attributes
	if(GLEW_VERSION_2_0)//shader support
	{
		glGetIntegerv(GL_MAX_VERTEX_ATTRIBS , &maxVal);
		caps.maxVertexAttribs = (maxVal < 16)? maxVal : 16;
	}
	else//fixed function
	{
		caps.maxVertexAttribs = 3 + 1;//caps.nFFTextureUnits; only support 1 fixed function texture stage for now
	}

	/*----------get max image units------------*/
#ifdef GL_MAX_IMAGE_UNITS
	if (GLEW_VERSION_4_2)
	{
		glGetIntegerv(GL_MAX_IMAGE_UNITS, &maxVal);
		caps.nImageUnits = maxVal;//image load store
		glGetIntegerv(GL_MAX_FRAGMENT_IMAGE_UNIFORMS, &maxVal);
		caps.nFragmentImageUnits = maxVal;//image load store

	}
	else
#endif//ifdef GL_MAX_IMAGE_UNITS
	{
		caps.nImageUnits//image load store
			= caps.nFragmentImageUnits//image load store
			= 0;
	}

#ifdef GL_MAX_COMPUTE_IMAGE_UNIFORMS
	if (GLEW_VERSION_4_3)
	{
		glGetIntegerv(GL_MAX_COMPUTE_IMAGE_UNIFORMS, &maxVal);
		caps.nComputeImageUnits = maxVal;
	}
	else
#endif//ifdef GL_MAX_COMPUTE_IMAGE_UNIFORMS
		caps.nComputeImageUnits = 0;//image load store

	//shader storage blocks
#ifdef GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS
	if (GLEW_VERSION_4_3)
	{
		glGetIntegerv(GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS, &maxVal);
		caps.nShaderStorageBlocks = maxVal;
		glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &maxVal);
		caps.nComputeShaderStorageBlocks = maxVal;
		glGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, &maxVal);
		caps.nFragmentShaderStorageBlocks = maxVal;
	}
	else
#endif//#ifdef GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS
	{
		caps.nShaderStorageBlocks = caps.nFragmentShaderStorageBlocks = caps.nComputeShaderStorageBlocks = 0;
	}

	//compute thread groups
#ifdef GL_MAX_COMPUTE_WORK_GROUP_COUNT
	if (GLEW_VERSION_4_3)
	{
		glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxVal);
		caps.nComputeGroupsX = maxVal;
		glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &maxVal);
		caps.nComputeGroupsY = maxVal;
		glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &maxVal);
		caps.nComputeGroupsZ = maxVal;
	}
	else
#endif//GL_MAX_COMPUTE_WORK_GROUP_COUNT
	{
		caps.nComputeGroupsX = 0;
		caps.nComputeGroupsY = 0;
		caps.nComputeGroupsZ = 0;
	}

	glGetError();//reset error flags
}

void HQDeviceEnumGL::CheckRenderBufferFormatSupported()
{
	GLuint fbo , texture , depthStencilBuffer[2];
	GLenum format, dataType;
	GLint internalFmt ;
	GLdepthStencilFormat depthStencilFmt;
	GLenum status;
	glGetError();//reset flags;
#ifndef HQ_OPENGLES
	if (GLEW_VERSION_3_0 || GLEW_ARB_framebuffer_object) {//core in openGL 3.0
#elif defined HQ_ANDROID_PLATFORM
	if (GLEW_VERSION_2_0){//core in openGL es 2.0
#endif //default ios support fbo
        GLuint currentActiveFBO;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, (GLint*)&currentActiveFBO);
		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		/*------------check render target texture format----------*/
		for (int i = 0; i < NUM_RTT_FORMAT; ++i) {
			glGenTextures(1, &texture);
			glBindTexture(GL_TEXTURE_2D, texture);
			HQRenderTargetManagerFBO::GetGLImageFormat((HQRenderTargetFormat) i, internalFmt , format , dataType );
			if (internalFmt == 0)
			{
				caps.rttInternalFormat[i] = false;
			}
			else {
				glTexImage2D(GL_TEXTURE_2D,
							 0, internalFmt,
							 1, 1,
							 0, format, dataType,
							 NULL);
				if (glGetError() != GL_NO_ERROR)
					caps.rttInternalFormat[i] = false;
				else {
					glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
					status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
					if (GL_FRAMEBUFFER_COMPLETE == status)
						caps.rttInternalFormat[i] = true;
					else
						caps.rttInternalFormat[i] = false;
				}
			}
			glBindTexture(GL_TEXTURE_2D, 0);
			glDeleteTextures(1, &texture);
		}
		/*------------check depth stencil buffer format-----------*/
		for (int i = 0; i < NUM_DS_FORMAT; ++i) {
			glGenRenderbuffers(2, depthStencilBuffer);
			depthStencilFmt = HQRenderTargetManagerFBO::GetGLFormat((HQDepthStencilFormat) i );
			if (depthStencilFmt.depthFormat != 0)
			{
				glBindRenderbuffer(GL_RENDERBUFFER, depthStencilBuffer[0]);
				glRenderbufferStorage(GL_RENDERBUFFER, depthStencilFmt.depthFormat, 1, 1);
				if (glGetError() != GL_NO_ERROR)
				{
					glDeleteRenderbuffers(2, depthStencilBuffer);
					caps.dsFormat[i] = false;
					continue;
				}
			}
			if (depthStencilFmt.stencilFormat != 0)
			{
				glBindRenderbuffer(GL_RENDERBUFFER, depthStencilBuffer[1]);
				glRenderbufferStorage(GL_RENDERBUFFER, depthStencilFmt.stencilFormat, 1, 1);
				if (glGetError() != GL_NO_ERROR)
				{
					glDeleteRenderbuffers(2, depthStencilBuffer);
					caps.dsFormat[i] = false;
					continue;
				}
			}
			caps.dsFormat[i] = true;
			glDeleteRenderbuffers(2, depthStencilBuffer);
		}
		glBindFramebuffer(GL_FRAMEBUFFER, currentActiveFBO);
		glDeleteFramebuffers(1, &fbo);
#ifndef HQ_IPHONE_PLATFORM
	}
	else //EXT/OES version
	{
        GLuint currentActiveFBO;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, (GLint*)&currentActiveFBO);
		glGenFramebuffersEXT(1, &fbo);
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
		/*------------check render target texture format----------*/
		for (int i = 0; i < NUM_RTT_FORMAT; ++i) {
			glGenTextures(1, &texture);
			glBindTexture(GL_TEXTURE_2D, texture);
			HQRenderTargetManagerFBO::GetGLImageFormat((HQRenderTargetFormat) i, internalFmt , format , dataType );
			if (internalFmt == 0)
			{
				caps.rttInternalFormat[i] = false;
			}
			else {

				glTexImage2D(GL_TEXTURE_2D,
							 0, internalFmt,
							 1, 1,
							 0, format, dataType,
							 NULL);
				if (glGetError() != GL_NO_ERROR)
					caps.rttInternalFormat[i] = false;
				else {
					glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, texture, 0);
					status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
					if (GL_FRAMEBUFFER_COMPLETE_EXT == status)
						caps.rttInternalFormat[i] = true;
					else
						caps.rttInternalFormat[i] = false;
				}
			}
			glBindTexture(GL_TEXTURE_2D, 0);
			glDeleteTextures(1, &texture);
		}
		/*------------check depth stencil buffer format-----------*/
		for (int i = 0; i < NUM_DS_FORMAT; ++i) {
			glGenRenderbuffersEXT(2, depthStencilBuffer);
			depthStencilFmt = HQRenderTargetManagerFBO::GetGLFormat((HQDepthStencilFormat) i );
			if (depthStencilFmt.depthFormat != 0)
			{
				glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthStencilBuffer[0]);
				glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, depthStencilFmt.depthFormat, 1, 1);
				if (glGetError() != GL_NO_ERROR)
				{
					glDeleteRenderbuffersEXT(2, depthStencilBuffer);
					caps.dsFormat[i] = false;
					continue;
				}
			}
			if (depthStencilFmt.stencilFormat != 0)
			{
				glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthStencilBuffer[1]);
				glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, depthStencilFmt.stencilFormat, 1, 1);
				if (glGetError() != GL_NO_ERROR)
				{
					glDeleteRenderbuffersEXT(2, depthStencilBuffer);
					caps.dsFormat[i] = false;
					continue;
				}
			}
			caps.dsFormat[i] = true;
			glDeleteRenderbuffersEXT(2, depthStencilBuffer);
		}
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, currentActiveFBO);
		glDeleteFramebuffersEXT(1, &fbo);
	}
#endif
}

#ifndef HQ_OPENGLES
//enumerate every display modes and pixel formats supported
void HQDeviceEnumGL::EnumAllDisplayModes()
{
	
#ifdef WIN32
	/*-------------------create dummy window for checking opengl capacities--------------------------*/
	WNDCLASSEX      wndclass;
	HWND            hwnd;
	// initialize the window
	wndclass.hIconSm       = LoadIcon(NULL,IDI_APPLICATION);
	wndclass.hIcon         = LoadIcon(NULL,IDI_APPLICATION);
	wndclass.cbSize        = sizeof(wndclass);
	wndclass.lpfnWndProc   = helper::DummyProc;
	wndclass.cbClsExtra    = 0;
	wndclass.cbWndExtra    = 0;
	wndclass.hInstance     = (HINSTANCE)pDll;
	wndclass.hCursor       = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)(COLOR_WINDOW);
	wndclass.lpszMenuName  = NULL;
	wndclass.lpszClassName = L"Dummy";
	wndclass.style         = CS_HREDRAW | CS_VREDRAW ;

	if (RegisterClassEx(&wndclass) == 0)
		return ;
	if (!(hwnd = CreateWindowEx( NULL, L"Dummy",
								L"Dummy WINDOW",
								WS_OVERLAPPED,
								CW_USEDEFAULT,
								CW_USEDEFAULT,
								640, 480, NULL, NULL, pDll, NULL)))
		return ;

	this->hDC = GetDC(hwnd);

	PIXELFORMATDESCRIPTOR pixFmt;
	memset(&pixFmt,0,sizeof(PIXELFORMATDESCRIPTOR));
	pixFmt.nSize=sizeof(PIXELFORMATDESCRIPTOR);
	pixFmt.nVersion   = 1;
    	pixFmt.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    	pixFmt.iPixelType = PFD_TYPE_RGBA;
	pixFmt.iLayerType = PFD_MAIN_PLANE;

	int ipixelfmt=ChoosePixelFormat(hDC,&pixFmt);
	SetPixelFormat(hDC,ipixelfmt,&pixFmt);

	HGLRC hRC=wglCreateContext(hDC);
	if(wglMakeCurrent(hDC,hRC)==FALSE)
		return;
	
	
	if(wglewContextInit()!=GLEW_OK)
		return;

#elif defined (HQ_LINUX_PLATFORM)
	if(dpy==NULL)
		return;

	/*------------create dummy window--------------*/
	GLint                   att[] = { GLX_RGBA, GLX_DOUBLEBUFFER, None };
	XVisualInfo             *vi;
	Colormap                cmap;
	XSetWindowAttributes    swa;
	Window                  win;
	GLXContext              glc;
	vi = glXChooseVisual(dpy, XDefaultScreen(dpy), att);
	cmap = XCreateColormap(dpy, DefaultRootWindow(dpy), vi->visual, AllocNone);
	swa.colormap = cmap;
	win = XCreateWindow(dpy, DefaultRootWindow(dpy), 0, 0, 640, 480, 0, vi->depth, InputOutput, vi->visual, CWColormap, &swa);

	//XMapWindow(dpy, win);
	XStoreName(dpy, win, "Dummy Window");
	glc = glXCreateContext(dpy, vi, NULL, GL_TRUE);
	if(glc==NULL)
		return;
	if(glXMakeCurrent(dpy, win, glc)==False)
		return;

	if(glxewContextInit()!=GLEW_OK)
		return;
#elif defined HQ_IPHONE_PLATFORM
	//create dummy context for checking capabilities
	EAGLContext* glc = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
	if (glc == nil) {
		return;
	}

	if(![EAGLContext setCurrentContext:glc])
	{
		[glc release];
		return;
	}
#elif defined (HQ_MAC_PLATFORM)
	//create dummy context for checking capabilities
	CGDirectDisplayID display = CGMainDisplayID ();
	CGOpenGLDisplayMask myDisplayMask = CGDisplayIDToOpenGLDisplayMask (display);

	CGLPixelFormatAttribute attribs[] = {
		kCGLPFAAccelerated,
		kCGLPFANoRecovery,
		kCGLPFADisplayMask,(CGLPixelFormatAttribute) myDisplayMask,
		(CGLPixelFormatAttribute) 0
	};

	CGLPixelFormatObj pixelFormat = NULL;
	GLint numPixelFormats = 0;
	CGLContextObj myCGLContext = 0;
	CGLContextObj curr_ctx = CGLGetCurrentContext ();

	caps.hardwareAccel = true;
	CGLChoosePixelFormat (attribs, &pixelFormat, &numPixelFormats);
	if (pixelFormat == NULL)
	{
		//try with software renderer
		CGLPixelFormatAttribute attribs[] = {
			kCGLPFADisplayMask,(CGLPixelFormatAttribute) myDisplayMask,
			(CGLPixelFormatAttribute) 0
		};

		caps.hardwareAccel = false;

		CGLChoosePixelFormat (attribs, &pixelFormat, &numPixelFormats);
		if (pixelFormat == NULL)
			return;
	}

	CGLCreateContext (pixelFormat, NULL, &myCGLContext);
	CGLDestroyPixelFormat (pixelFormat);
	CGLSetCurrentContext (myCGLContext);

	if (myCGLContext == 0)
		return;
#endif

	/*---------------------------------------------*/
	Resolution myRes;
	//screen resolution
	UINT maxW , maxH;
#ifdef WIN32
	maxW=GetDeviceCaps(hDC,HORZRES);
	maxH=GetDeviceCaps(hDC,VERTRES);

	/*----find all display mode-----------*/
	myRes.w32DisplayMode.dmSize=sizeof(DEVMODE);
	myRes.w32DisplayMode.dmFields=DM_PELSWIDTH | DM_PELSHEIGHT | DM_BITSPERPEL;
	//get current screen display mode
	EnumDisplaySettings(NULL , ENUM_CURRENT_SETTINGS , &this->currentScreenDisplayMode);

	hquint32 mode = 0;
	while(EnumDisplaySettings(NULL , mode , &myRes.w32DisplayMode))
	{
		if (myRes.w32DisplayMode.dmBitsPerPel == currentScreenDisplayMode.dmBitsPerPel &&
			myRes.w32DisplayMode.dmDisplayFlags == currentScreenDisplayMode.dmDisplayFlags &&
			myRes.w32DisplayMode.dmDisplayFrequency == currentScreenDisplayMode.dmDisplayFrequency &&
			myRes.w32DisplayMode.dmPelsWidth <= maxW &&
			myRes.w32DisplayMode.dmPelsHeight <= maxH
			)
		{
			myRes.width = myRes.w32DisplayMode.dmPelsWidth;
			myRes.height = myRes.w32DisplayMode.dmPelsHeight;
			this->reslist.PushBack(myRes);
		}
		++mode;
	}

#elif defined (HQ_LINUX_PLATFORM) /*-----------linux--------------*/

    //truy van cac do phan giai man hinh
    int screen = DefaultScreen(dpy);
	maxW=DisplayWidth(dpy,screen);
	maxH=DisplayHeight(dpy,screen);

#	if defined HQ_USE_XFREE86_VIDMODE
    XF86VidModeGetAllModeLines(dpy, screen, &this->modeNum, &this->modes);
    //first element is current video mode
    this->currentScreenDisplayMode = this->modes[0];

    myRes.width = this->modes[0]->hdisplay;
    myRes.height = this->modes[0]->vdisplay;
    myRes.x11DisplayMode = this->modes[0];
    myRes.x11RefreshRate = (double)this->modes[0]->dotclock * 1000/ (this->modes[0]->vtotal * this->modes[0]->htotal);
    this->reslist.PushBack(myRes);

    for (int i = 1; i < this->modeNum ; ++i)
    {
        if (this->modes[i]->hdisplay <= maxW &&
            this->modes[i]->vdisplay <= maxH)
            {

                myRes.width = this->modes[i]->hdisplay;
                myRes.height = this->modes[i]->vdisplay;
                myRes.x11DisplayMode = this->modes[i];
                myRes.x11RefreshRate = (double)this->modes[i]->dotclock * 1000/ (this->modes[i]->vtotal * this->modes[i]->htotal);

                this->reslist.PushBack(myRes);
            }

    }
#	else
    /*--------using Randr-----------*/
    Window root = RootWindow(dpy, 0);
    XRRScreenSize *modes = NULL;
    int numModes = 0;
    short currentScreenRefreshRate = 0;
    this->screenConfig = XRRGetScreenInfo (dpy, root);
    
    //get all possible screen size
    modes = XRRSizes(dpy, 0, &numModes);
  
    //get current screen size
    this->currentScreenSizeIndex = XRRConfigCurrentConfiguration (this->screenConfig, &this->currentScreenRotation);
    //get current screen refresh rate
    currentScreenRefreshRate = XRRConfigCurrentRate(this->screenConfig);
    //push current screen config to the beginning of the list
    myRes.width = modes[this->currentScreenSizeIndex].width;
    myRes.height = modes[this->currentScreenSizeIndex].height;
    myRes.x11ScreenSizeIndex = this->currentScreenSizeIndex;
    myRes.x11RefreshRate = (double)currentScreenRefreshRate;
    this->reslist.PushBack(myRes);
    
    //now push the rest to the list
    for (int i = 0; i < numModes ; ++i)
    {
        if (modes[i].width <= maxW &&
            modes[i].height <= maxH)
            {
                int numRates;
                short * rates = XRRRates(dpy, 0, i, &numRates);
                
                for (int r = 0; r < numRates; ++r)
                {
                    short refresh_rate = rates[r];
                    if (i != this->currentScreenSizeIndex || refresh_rate != currentScreenRefreshRate)//ignore current screen config
                    {
                    myRes.width = modes[i].width;
                    myRes.height = modes[i].height;
                    myRes.x11ScreenSizeIndex = i;
                    myRes.x11RefreshRate = (double)refresh_rate;
                    this->reslist.PushBack(myRes);
                    }
                }
            }

    }
    
#	endif//#	if defined HQ_USE_XFREE86_VIDMODE

#elif defined (HQ_MAC_PLATFORM) /*----------mac osx------------*/
	/*----find all display mode-----------*/
	this->currentScreenDisplayMode = CGDisplayCopyDisplayMode(CGMainDisplayID());
	modeList = CGDisplayCopyAllDisplayModes(CGMainDisplayID(), NULL);
	CFStringRef currentPixelEncoding = CGDisplayModeCopyPixelEncoding(currentScreenDisplayMode);
	double currentRefreshRate = CGDisplayModeGetRefreshRate(currentScreenDisplayMode);
	maxW=CGDisplayModeGetWidth(currentScreenDisplayMode);
	maxH=CGDisplayModeGetHeight(currentScreenDisplayMode);

	CFIndex count = CFArrayGetCount (modeList);
	CGDisplayModeRef displayMode;
	CFStringRef pixelEncoding;
	double refreshRate;


	for (CFIndex i = 0; i < count;  ++i) {
		displayMode = (CGDisplayModeRef)CFArrayGetValueAtIndex (modeList, i);
		pixelEncoding = CGDisplayModeCopyPixelEncoding(displayMode);
		refreshRate = CGDisplayModeGetRefreshRate(displayMode);

		myRes.width = CGDisplayModeGetWidth(displayMode);
		myRes.height = CGDisplayModeGetHeight(displayMode);
		myRes.cgDisplayMode = displayMode;

		if(myRes.width <= maxW && myRes.height <= maxH &&
		   fabs (refreshRate - currentRefreshRate) <= REFRESHRATEDIFF &&
		   CFStringCompare(pixelEncoding,currentPixelEncoding,0)==kCFCompareEqualTo)
			this->reslist.PushBack(myRes);

		CFRelease(pixelEncoding);
	}


	CFRelease(currentPixelEncoding);
#endif


	this->selectedResolution = &reslist.GetFront();

	for(int k=0;k<numPixelFmt;++k)//for all supported display format
	{
		BufferInfo binfo;
		if(!CheckPixelFmt(PixelFormat[k]))//not supported
			continue;
		binfo.pixelFmt=PixelFormat[k];

		//chọn depthStencil format tốt nhất
		bool success;

		binfo.depthStencilFmt=SFMT_D24S8;

		success=CheckDepthStencilFmt(binfo);

		if(!success){
			binfo.depthStencilFmt=SFMT_D24X4S4;
			success=CheckDepthStencilFmt(binfo);
		}
		if(!success){
			binfo.depthStencilFmt=SFMT_D15S1;
			success=CheckDepthStencilFmt(binfo);
		}
		if(!success){
			binfo.depthStencilFmt=SFMT_D24X8;
			success=CheckDepthStencilFmt(binfo);
		}
		if(!success){
			binfo.depthStencilFmt=SFMT_D16;
			success=CheckDepthStencilFmt(binfo);
		}
		if(!success)
			continue;

		//chọn kiểu siêu lấy mẫu tốt nhất
		for(int l=numMulSample-1; l>=0; --l)
		{
			binfo.maxMulSampleLevel=MulSample[l];
			if(CheckMultisample(binfo))
				break;
		}

		bufferInfoList.PushBack(binfo);
	}//for(k)

	this->selectedBufferSetting=&bufferInfoList.GetFront();

#ifdef WIN32

	//done!Destroy dummy window
	wglMakeCurrent(NULL,NULL);
	wglDeleteContext(hRC);
	ReleaseDC(hwnd,hDC);
	DestroyWindow(hwnd);
	UnregisterClass(L"Dummy",pDll);
#elif defined (HQ_LINUX_PLATFORM)

	//done!Destroy dummy window
	glXMakeCurrent(dpy, None, NULL);
	glXDestroyContext(dpy, glc);
	XDestroyWindow(dpy, win);

#elif defined HQ_IPHONE_PLATFORM

	//delete dummy context
	[EAGLContext setCurrentContext:nil];
	[glc release];

#elif defined HQ_MAC_PLATFORM

	//done! destroy dummy context
	CGLDestroyContext (myCGLContext);
	CGLSetCurrentContext (curr_ctx);
#endif
}
#endif
//****************************************
//parse setting file
//****************************************

#ifdef HQ_MAC_PLATFORM
void HQDeviceEnumGL::ParseSettingFile(const char *settingFile , hq_uint32 width , hquint32 height , bool windowed)
#else
void HQDeviceEnumGL::ParseSettingFile(const char *settingFile)
#endif
{

	FILE * save=0;
	if(!settingFile)
		return;
	save=fopen(settingFile,"r");
	if(!save)
		return;
	int d1;
	UINT sWidth,sHeight;//no use in openGL ES

	fscanf(save,"Basic Settings\n");
	fscanf(save,"Width=%u\n",&sWidth);
	fscanf(save,"Height=%u\n",&sHeight);
	fscanf(save,"Windowed=%d\n",&d1);
	fscanf(save,"RefreshRate=%d\n",&this->unUseValue[0]);//unuse value , it is for direct3d
	fscanf(save,"VSync=%d\n\n",&this->vsync);

#ifdef HQ_MAC_PLATFORM
	//mac osx version doesn't care about width & height stored in setting file if view currently in windowed mode
	//the windowed mode stored in setting file also is ignored
	if (windowed)
	{
		sWidth = width;
		sHeight = height;
	}
	
	d1 = windowed;

#endif

#ifndef HQ_OPENGLES

	this->windowed = (d1 !=0);//no use in openGL ES
	//find resoltion in standard resolutions list that matches requested resolution
#if defined HQ_LINUX_PLATFORM
    const Resolution & currentResolution = this->reslist.GetFront();
    double minRefeshRateDiff = 9999999.0;
#endif
	bool found = false;
	HQLinkedListNode<Resolution> *pRNode = reslist.GetRoot();
	for(hq_uint32 i = 0 ; i < reslist.GetSize() ; ++i)
	{
		if(pRNode->m_element.width==sWidth && pRNode->m_element.height==sHeight)
		{
#ifdef HQ_LINUX_PLATFORM
            //find display mode that has closest matching refresh rate
		    double refreshRateDiff = fabs(pRNode->m_element.x11RefreshRate - currentResolution.x11RefreshRate);
		    if (refreshRateDiff < minRefeshRateDiff)
		    {
		        minRefeshRateDiff = refreshRateDiff;
#endif
                this->selectedResolution=&pRNode->m_element;
                found = true;
#ifdef HQ_LINUX_PLATFORM
		    }
#else
			break;
#endif
		}

		pRNode= pRNode->m_pNext;
	}
	if(!found)//not found, this is custom resolution, it only is allowed in windowed mode
	{
		this->customWindowedRes.width = sWidth;
		this->customWindowedRes.height = sHeight;
		this->selectedResolution = &customWindowedRes;
#ifdef HQ_LINUX_PLATFORM
        this->customWindowedRes.x11RefreshRate = currentResolution.x11RefreshRate;
#endif
		windowed = 1;
	}

#endif
	fscanf(save,"Advanced Settings\n");
	fscanf(save,"Adapter=%d\n",&this->unUseValue[1]);//unuse


	fscanf(save,"Device Type=%d\n",&this->unUseValue[2]);//unuse
	fscanf(save,"BackBuffer Format=%d\n",&d1);//opengl display/pixel format lấy từ back buffer format theo chuẩn direct3D
#ifdef HQ_OPENGLES
	this->selectedPixelFormat = (FORMAT)d1;
#else
	//find binfo
	HQLinkedListNode<BufferInfo> *pBNode = bufferInfoList.GetRoot();
	for(hq_uint32 i = 0 ; i < bufferInfoList.GetSize() ; ++i)
	{
		if(pBNode->m_element.pixelFmt==d1)
		{
			this->selectedBufferSetting=&pBNode->m_element;
			break;
		}
		pBNode= pBNode->m_pNext;
	}
#endif
	fscanf(save,"DepthStencil Format=%d\n",&d1);

#ifdef HQ_OPENGLES
	this->selectedDepthStencilFmt = (FORMAT)d1;
#else
	//check valid value
	BufferInfo binfo= *selectedBufferSetting;
	binfo.depthStencilFmt=(FORMAT)d1;

	if(!CheckDepthStencilFmt(binfo))
		this->selectedDepthStencilFmt=selectedBufferSetting->depthStencilFmt;//fall back
	else
		this->selectedDepthStencilFmt=(FORMAT)d1;
#endif
	fscanf(save,"Vertex Processing=%d\n",&this->unUseValue[3]);
	fscanf(save,"Multisample Type=%d\n",(int*)&this->selectedMulSampleType);

#ifndef HQ_OPENGLES
	//check valid value
	if(selectedMulSampleType > selectedBufferSetting->maxMulSampleLevel)
		selectedMulSampleType = selectedBufferSetting->maxMulSampleLevel;//fallback to max value supported
#endif

	fclose(save);
}
//***********************
//save setting
//***********************
void HQDeviceEnumGL::SaveSettingFile(const char *settingFile){
	if(!settingFile)
		return;
	FILE *save=0;
	save=fopen(settingFile,"w");
	if(!save)
		return;
	fprintf(save,"Basic Settings\n");
#ifdef HQ_OPENGLES
	fprintf(save,"Width=%u\n",g_pOGLDev->GetWidth());
	fprintf(save,"Height=%u\n", g_pOGLDev->GetHeight());
	fprintf(save,"Windowed=%d\n",1);
	fprintf(save,"RefreshRate=%d\n",this->unUseValue[0]);
#else
	fprintf(save,"Width=%u\n",selectedResolution->width);
	fprintf(save,"Height=%u\n",selectedResolution->height);
	fprintf(save,"Windowed=%d\n",(int)this->windowed);
	fprintf(save,"RefreshRate=%d\n",this->unUseValue[0]);
#endif
	fprintf(save,"VSync=%d\n\n",this->vsync);

	fprintf(save,"Advanced Settings\n");
	fprintf(save,"Adapter=%d\n",this->unUseValue[1]);//giá trị cũ lấy trong file lúc parse setting file
	fprintf(save,"Device Type=%d\n",this->unUseValue[2]);
#ifdef HQ_OPENGLES
	fprintf(save,"BackBuffer Format=%d\n",(int)selectedPixelFormat);
#else
	fprintf(save,"BackBuffer Format=%d\n",(int)selectedBufferSetting->pixelFmt);//opengl display/pixel format là back buffer format theo chuẩn direct3D
#endif
	fprintf(save,"DepthStencil Format=%d\n",(int)selectedDepthStencilFmt);
	fprintf(save,"Vertex Processing=%d\n",this->unUseValue[3]);
	fprintf(save,"Multisample Type=%d\n",(int)this->selectedMulSampleType);

	fclose(save);
}

#ifdef HQ_ANDROID_PLATFORM
jobject HQDeviceEnumGL::GetJEGLConfig()
{
	JNIEnv *jenv = GetCurrentThreadJEnv();
	if (jenv == NULL)
		return NULL;
	jobject jconfig = NULL;
	jint red, green, blue, alpha, depth, stencil;
	jint lred, lgreen, lblue, lalpha, ldepth, lstencil;
	switch(this->selectedPixelFormat)
	{
		case SFMT_R5G6B5:
		case SFMT_X1R5G5B5:
			red = 5; green = 6; blue = 5;
			alpha = 0;
			break;
		default: red = 8; green = 8; blue = 8;
			alpha = 8;
	}

	switch(this->selectedDepthStencilFmt)
	{
		case SFMT_D24S8:
		case SFMT_D24X4S4:
		case SFMT_D15S1:
			depth = 24; stencil = 8;
			break;
		case SFMT_D32:
		case SFMT_D24X8:
			depth = 24 ;stencil = 0;
			break;
		case SFMT_D16:
			depth = 16; stencil = 0;
			break;
		case SFMT_S8:
			depth = 0; stencil = 8;
			break;
		case SFMT_NODEPTHSTENCIL:
		default: 
			depth = stencil = 0;
	}

	//try to find suitable frame buffer config
	jint num_configs = 0;
	do {
		jint attributes[] = {
			J_EGL_RENDERABLE_TYPE, J_EGL_OPENGL_ES2_BIT,
			J_EGL_RED_SIZE, red,
			J_EGL_GREEN_SIZE, green,
			J_EGL_BLUE_SIZE, blue,
			J_EGL_ALPHA_SIZE, alpha,
			J_EGL_DEPTH_SIZE , depth,
			J_EGL_STENCIL_SIZE , stencil,
			J_EGL_NONE
		};

		if (this->selectedApiLevel == 1)
			attributes[1] = J_EGL_OPENGL_ES_BIT;
		
		jintArray jattrib_list = jenv->NewIntArray(15);
		jenv->SetIntArrayRegion(jattrib_list, 0, 15, attributes);//copy to java type

		//get num configs
		jintArray valueArray = jenv->NewIntArray(1);

		jenv->CallBooleanMethod(this->jegl,
								ge_jeglChooseConfigMethodID,
								this->jdisplay,
								jattrib_list,
								NULL,
								0,
								valueArray);

		num_configs = GetValue(jenv, valueArray);
		
		if (num_configs > 0)
		{
			//create java EGLConfig array
			jclass jeglconfigClass = jenv->FindClass("javax/microedition/khronos/egl/EGLConfig");
			jobjectArray jconfigList = jenv->NewObjectArray(num_configs , jeglconfigClass , NULL);

			//get returned configs
			jenv->CallBooleanMethod(this->jegl,
									ge_jeglChooseConfigMethodID,
									this->jdisplay,
									jattrib_list,
									jconfigList,
									num_configs,
									valueArray);

			//choose the right config
			for (jint i = 0 ; i < num_configs ; ++i)
			{
				jobject ljconfig = jenv->GetObjectArrayElement(jconfigList , i);
				this->GetPixelFormat(jenv, ljconfig, lred, lgreen, lblue, lalpha, ldepth, lstencil);
				if (
	#if REQUIRE_EXACT_RGB
					red == lred && green == lgreen ,blue == lblue && alpha == lalpha 
	#else
					true
	#endif
	#if REQUIRE_EXACT_DEPTH_STENCIL
	&&	depth == ldepth && lstencil == stencil
	#endif
	)
				{
					__android_log_print(ANDROID_LOG_INFO, "GL Render Device :", "Selected pixel format= (R=%d, G=%d, B=%d, D=%d, S=%d)", lred, lgreen, lblue, ldepth, lstencil);
					jconfig = jenv->NewGlobalRef(ljconfig);
					break;
				}

				//release local ref
				jenv->DeleteLocalRef(ljconfig);
			}

			jenv->DeleteLocalRef(jeglconfigClass);
			jenv->DeleteLocalRef(jconfigList);
		}//if (num_configs > 0)
		else {
			//maybe due to depth and stencil size are too large.
			//reduce the size of depth & stencil and retry.
			if (depth > 1)
				depth = 1;
			else 
				depth = 0;
			if (stencil > 1)
				stencil = 1;
			else
				stencil = 0;
		}
		//release local ref
		jenv->DeleteLocalRef(valueArray);
		jenv->DeleteLocalRef(jattrib_list);

	} while (num_configs == 0);//will retry until we find a suitable config


	return jconfig;
}


void HQDeviceEnumGL::GetPixelFormat(JNIEnv *jenv, jobject jeglConfig , 
							jint &red, jint& green, jint& blue, jint& alpha,
							jint &depth, jint& stencil)
{
	jintArray valueArray = jenv->NewIntArray(1);
	//get alpha size
	jenv->CallBooleanMethod(
							this->jegl , 
							ge_jeglGetConfigAttribMethodID,
							this->jdisplay,
							jeglConfig,
							J_EGL_ALPHA_SIZE,
							valueArray);
	alpha = GetValue(jenv, valueArray);
	//get red size
	jenv->CallBooleanMethod(
							this->jegl , 
							ge_jeglGetConfigAttribMethodID,
							this->jdisplay,
							jeglConfig,
							J_EGL_RED_SIZE,
							valueArray);
	red = GetValue(jenv, valueArray);
	//get green size
	jenv->CallBooleanMethod(
							this->jegl , 
							ge_jeglGetConfigAttribMethodID,
							this->jdisplay,
							jeglConfig,
							J_EGL_GREEN_SIZE,
							valueArray);
	green = GetValue(jenv, valueArray);
	//get blue size
	jenv->CallBooleanMethod(
							this->jegl , 
							ge_jeglGetConfigAttribMethodID,
							this->jdisplay,
							jeglConfig,
							J_EGL_BLUE_SIZE,
							valueArray);
	blue = GetValue(jenv, valueArray);

	//depth size
	jenv->CallBooleanMethod(
							this->jegl , 
							ge_jeglGetConfigAttribMethodID,
							this->jdisplay,
							jeglConfig,
							J_EGL_DEPTH_SIZE,
							valueArray);
	depth = GetValue(jenv, valueArray);

	//stencil size
	jenv->CallBooleanMethod(
							this->jegl , 
							ge_jeglGetConfigAttribMethodID,
							this->jdisplay,
							jeglConfig,
							J_EGL_STENCIL_SIZE,
							valueArray);
	stencil = GetValue(jenv, valueArray);

	//release local ref
	jenv->DeleteLocalRef(valueArray);
}

#elif !defined HQ_OPENGLES

//************************************************
//get all available fullscreen display resolutions
//************************************************

struct ResolutionHashFunc
{
	hq_uint32 operator()(const HQResolution &val) const
	{
		return (val.width << 16) | val.height;
	}
};

class ResolutonEqual
{
public:
	bool operator() (const HQResolution & lhs , const HQResolution & rhs) const
	{
		if (lhs.width == rhs.width && lhs.height == rhs.height)
			return true;
		return false;
	}

};

void HQDeviceEnumGL::GetAllDisplayResolution(HQResolution *resolutionList , hq_uint32& numResolutions)
{

	HQResolution res;

	HQHashTable<HQResolution , bool  ,ResolutionHashFunc , ResolutonEqual> existTable(reslist.GetSize() * 2 + 1);

	HQLinkedListNode<Resolution> *pRNode = reslist.GetRoot();
	if (resolutionList != NULL)//save array
	{
		hquint32 j = 0;
		for(hq_uint32 i = 0 ; i < reslist.GetSize() && j < numResolutions ; ++i)
		{
			res.width = pRNode->m_element.width;
			res.height = pRNode->m_element.height;

			if (existTable.Add(res , true))//độ phân giải này chưa cho vào list
			{
				resolutionList[j++] = res;
			}

			pRNode = pRNode->m_pNext;
		}
	}
	else//count max number of resolutions
	{
		numResolutions = 0;
		for(hq_uint32 i = 0 ; i < reslist.GetSize() ; ++i)
		{
			res.width = pRNode->m_element.width;
			res.height = pRNode->m_element.height;

			if (existTable.Add(res , true))//độ phân giải này chưa cho vào list
			{
				numResolutions++;
			}

			pRNode = pRNode->m_pNext;
		}
	}
}
//**********************************************
//change selected display mode
//**********************************************
bool HQDeviceEnumGL::ChangeSelectedDisplay(hq_uint32 width,hq_uint32 height , bool windowed)
{
#ifdef HQ_LINUX_PLATFORM
    const Resolution & currentResolution = this->reslist.GetFront();
    double minRefeshRateDiff = 9999999.0;
#endif
	bool found = false;
	HQLinkedListNode<Resolution> *pRNode = reslist.GetRoot();
	for(hq_uint32 i = 0 ; i < reslist.GetSize() ; ++i)
	{
		if(pRNode->m_element.width==width && pRNode->m_element.height==height)
		{
#ifdef HQ_LINUX_PLATFORM
            //find display mode that has closest matching refresh rate
		    double refreshRateDiff = fabs(pRNode->m_element.x11RefreshRate - currentResolution.x11RefreshRate);
		    if (refreshRateDiff < minRefeshRateDiff)
		    {
		        minRefeshRateDiff = refreshRateDiff;
#endif
				this->selectedResolution=&pRNode->m_element;
                found = true;
#ifdef HQ_LINUX_PLATFORM
		    }
#else
			break;
#endif
		}

		pRNode = pRNode->m_pNext;
	}
	if(!found && windowed)
	{
		this->customWindowedRes.width = width;
		this->customWindowedRes.height = height;
#ifdef HQ_LINUX_PLATFORM
		this->customWindowedRes.x11RefreshRate = currentResolution.x11RefreshRate;
#endif
		this->selectedResolution = &customWindowedRes;
		return true;
	}
	return found;
}
#endif
