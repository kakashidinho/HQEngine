/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "Android.h"

#include <new>
#include <sched.h>
#include <android/log.h>

#define DEBUG_LOG 0

const jint J_EGL_CONTEXT_CLIENT_VERSION	= 0x3098;
static jfieldID g_EGL_NO_CONTEXT_ID = NULL;//java field id
static jfieldID g_EGL_NO_SURFACE_ID = NULL;//java field id
static jmethodID g_jeglCreateWindowSurfaceMethodID = NULL;//java method id
static jmethodID g_jeglCreateContextMethodID = NULL;//java method id
static jmethodID g_jeglDestroySurfaceMethodID = NULL;//java method id
static jmethodID g_jeglDestroyContextMethodID = NULL;//java method id
static jmethodID g_jeglMakeCurrentMethodID = NULL;//java method id
static jmethodID g_jeglSwapBuffersMethodID = NULL;//java method id
jmethodID ge_jeglGetConfigAttribMethodID = NULL;
jmethodID ge_jeglChooseConfigMethodID = NULL;

extern int HQEngineInternalIsJSurfaceCreated();
extern void HQEngineInternalRunOnUiThread(void (*runFunc)(void), bool wait);
extern void HQEngineInternalSetViewVisibility(jobject jview, bool visible);

struct ChangeSurfaceFormatArg_t
{
	jobject jview;
	jobject jsurface;
	jint jformat;
};

static ChangeSurfaceFormatArg_t * changeSurfaceFormatArg = NULL;

/*--------change native window's surface format on ui thread-------*/
static void ChangeSurfaceFormat()
{
	JNIEnv * jenv = GetCurrentThreadJEnv();
	
	//destroy surface
	HQEngineInternalSetViewVisibility(changeSurfaceFormatArg->jview, false);
	
	jclass JSurfaceHolder = jenv->FindClass("android/view/SurfaceHolder");
	
	if (JSurfaceHolder == NULL )
		throw std::bad_alloc();
	
	/*------------------get method and field ids------------*/
	
	//get SurfaceHolder.setFormat (int format) method ID
	jmethodID l_jsetFormatMethodID = jenv->GetMethodID(
			JSurfaceHolder,
			"setFormat",
			"(I)V"
		);
	
	/*-----------------change native window surface's pixel format----------*/
	jenv->CallVoidMethod(
			changeSurfaceFormatArg->jsurface , 
			l_jsetFormatMethodID , 
			changeSurfaceFormatArg->jformat);
	
	//delete local refs
	jenv->DeleteLocalRef(JSurfaceHolder);
	
	//recreate surface
	HQEngineInternalSetViewVisibility(changeSurfaceFormatArg->jview, true);
}

/*-----------JNI--------------*/
extern JavaVM *ge_jvm;

jint GetValue (JNIEnv * jenv,jintArray array)
{
	jint value;
	jboolean isCopy;
	jint *pValue = jenv->GetIntArrayElements(array , &isCopy );
	value = *pValue;
	jenv->ReleaseIntArrayElements(array , pValue , 0);
	return value;
}

void InitEGLMethodAndAttribsIDs()
{
	JNIEnv * jenv = GetCurrentThreadJEnv();
	jclass JEGLClass = jenv->FindClass("javax/microedition/khronos/egl/EGL10");

	if (g_EGL_NO_CONTEXT_ID == NULL)
	{
		//get EGL10.EGL_NO_CONTEXT field
		g_EGL_NO_CONTEXT_ID = jenv->GetStaticFieldID(
							JEGLClass,
							"EGL_NO_CONTEXT", 
							"Ljavax/microedition/khronos/egl/EGLContext;");
	}
	
	if (g_EGL_NO_SURFACE_ID == NULL)
	{
		//get EGL10.EGL_NO_SURFACE field
		g_EGL_NO_SURFACE_ID = jenv->GetStaticFieldID(
							JEGLClass,
							"EGL_NO_SURFACE", 
							"Ljavax/microedition/khronos/egl/EGLSurface;");
	}
	
	if (ge_jeglGetConfigAttribMethodID == NULL)
	{
		//get eglGetConfigAttrib(EGLDisplay display, EGLConfig config, int attribute, int[] value) method id
		ge_jeglGetConfigAttribMethodID = jenv->GetMethodID(
			JEGLClass, 
			"eglGetConfigAttrib",
			"(Ljavax/microedition/khronos/egl/EGLDisplay;Ljavax/microedition/khronos/egl/EGLConfig;I[I)Z");
	}
	
	if (ge_jeglChooseConfigMethodID == NULL)
	{
		//get eglChooseConfig(EGLDisplay display, int[] attrib_list, EGLConfig[] configs, int config_size, int[] num_config) methodID
		ge_jeglChooseConfigMethodID = jenv->GetMethodID(
			JEGLClass, 
			"eglChooseConfig",
			"(Ljavax/microedition/khronos/egl/EGLDisplay;[I[Ljavax/microedition/khronos/egl/EGLConfig;I[I)Z");
	}
	
	if (g_jeglMakeCurrentMethodID == NULL)
	{
		//get java eglMakeCurrent(EGLDisplay display, EGLSurface draw, EGLSurface read, EGLContext context) methodID
		g_jeglMakeCurrentMethodID = jenv->GetMethodID(
			JEGLClass, 
			"eglMakeCurrent",
			"(Ljavax/microedition/khronos/egl/EGLDisplay;Ljavax/microedition/khronos/egl/EGLSurface;Ljavax/microedition/khronos/egl/EGLSurface;Ljavax/microedition/khronos/egl/EGLContext;)Z");
	}
	
	if (g_jeglSwapBuffersMethodID == NULL)
	{
		//get java eglSwapBuffers(EGLDisplay display, EGLSurface surface) methodID
		g_jeglSwapBuffersMethodID = jenv->GetMethodID(
			JEGLClass, 
			"eglSwapBuffers",
			"(Ljavax/microedition/khronos/egl/EGLDisplay;Ljavax/microedition/khronos/egl/EGLSurface;)Z");
	}
	
	if (g_jeglCreateContextMethodID == NULL)
	{
		//get java eglCreateContext(EGLDisplay display, EGLConfig config, EGLContext share_context, int[] attrib_list) method ID
		g_jeglCreateContextMethodID = jenv->GetMethodID(
			JEGLClass, 
			"eglCreateContext",
			"(Ljavax/microedition/khronos/egl/EGLDisplay;Ljavax/microedition/khronos/egl/EGLConfig;Ljavax/microedition/khronos/egl/EGLContext;[I)Ljavax/microedition/khronos/egl/EGLContext;");
	}
	
	
	if (g_jeglCreateWindowSurfaceMethodID == NULL)
	{
		//get java eglCreateWindowSurface(EGLDisplay display, EGLConfig config, Object native_window, int[] attrib_list) methodID
		g_jeglCreateWindowSurfaceMethodID = jenv->GetMethodID(
			JEGLClass, 
			"eglCreateWindowSurface",
			"(Ljavax/microedition/khronos/egl/EGLDisplay;Ljavax/microedition/khronos/egl/EGLConfig;Ljava/lang/Object;[I)Ljavax/microedition/khronos/egl/EGLSurface;");
	}
	
	
	if (g_jeglDestroySurfaceMethodID == NULL)
	{
		//get eglDestroySurface(EGLDisplay display, EGLSurface surface) method id
		g_jeglDestroySurfaceMethodID = jenv->GetMethodID(
			JEGLClass, 
			"eglDestroySurface",
			"(Ljavax/microedition/khronos/egl/EGLDisplay;Ljavax/microedition/khronos/egl/EGLSurface;)Z");
	}
	
	if (g_jeglDestroyContextMethodID == NULL)
	{
		//get eglDestroyContext(EGLDisplay display, EGLContext context) method id
		g_jeglDestroyContextMethodID = jenv->GetMethodID(
			JEGLClass, 
			"eglDestroyContext",
			"(Ljavax/microedition/khronos/egl/EGLDisplay;Ljavax/microedition/khronos/egl/EGLContext;)Z");
	}

	//delete local ref
	jenv->DeleteLocalRef(JEGLClass);
}

/*-----------HQAndroidOpenGLContext-----------------*/

HQAndroidOpenGLContext::HQAndroidOpenGLContext(
						const HQAndroidRenderDeviceInitInput & input,
						jobject jeglConfig,
						jint apiLevel
					)
:m_input(input) , m_apiLevel(apiLevel), m_jenv(NULL)
{
	/*--------get jni env-----------*/
	m_jenv = GetCurrentThreadJEnv();
	if (m_jenv == NULL)
		throw std::bad_alloc();
	
	/*--------make sure this is an instance of HQEngineView class--------*/
	jclass JHQEngineViewClass = m_jenv->GetObjectClass(m_input.jengineView);
	if (JHQEngineViewClass == NULL)
		throw std::bad_alloc();
	
	m_jenv->GetMethodID(
			JHQEngineViewClass, 
			"isHQEngineViewClass",
			"()V");
			
	
	/*--------wait for surface creation so opengl context can be created-------*/
	while (HQEngineInternalIsJSurfaceCreated() == 0)
	{
#if DEBUG_LOG
		__android_log_print(ANDROID_LOG_INFO, "HQAndroidOpenGLContext", "waiting for surface creation so OpenGL context can be created");
#endif
		sched_yield();
	}
	
	/*------get surface holder-----------*/
	
	jmethodID l_jgetSurfaceHolderMethodID = m_jenv->GetMethodID(
			JHQEngineViewClass, 
			"getHolder",
			"()Landroid/view/SurfaceHolder;");
			
	jobject l_jsurfaceHolder = m_jenv->CallObjectMethod(
							m_input.jengineView ,
							l_jgetSurfaceHolderMethodID);
	
	
	/*---------create global ref------------*/
	m_jeglConfig = m_jenv->NewGlobalRef(jeglConfig);
	m_jsurfaceHolder = m_jenv->NewGlobalRef(l_jsurfaceHolder);
	
	/*---------------get surface's width and height-----------*/
	this->GetSurfaceSize();
	
	/*-----------------change native window surface's pixel format----------*/
	//create parameters
	changeSurfaceFormatArg = new ChangeSurfaceFormatArg_t();
	changeSurfaceFormatArg->jview = m_input.jengineView;
	changeSurfaceFormatArg->jsurface = m_jsurfaceHolder;
	changeSurfaceFormatArg->jformat = this->GetNativeWindowPixelFormat();
	
	HQEngineInternalRunOnUiThread(ChangeSurfaceFormat, true);
	
	//done, delete parameters
	delete changeSurfaceFormatArg;
	changeSurfaceFormatArg = NULL;
	
	
	//delete local references
	m_jenv->DeleteLocalRef(l_jsurfaceHolder);
	m_jenv->DeleteLocalRef(JHQEngineViewClass);
	
		
	//create context
	try{
		this->OnReset();
	}
	catch (std::bad_alloc e)
	{
	
		//delete global references	
		m_jenv->DeleteGlobalRef(m_jeglConfig);
		m_jenv->DeleteGlobalRef(m_jsurfaceHolder);
		throw e;
	}

}


HQAndroidOpenGLContext::~HQAndroidOpenGLContext()
{
	m_jenv = GetCurrentThreadJEnv();
	
	this->OnLost();
	
	//delete global references	
	m_jenv->DeleteGlobalRef(m_jdrawSurface);
	m_jenv->DeleteGlobalRef(m_jeglContext);
	m_jenv->DeleteGlobalRef(m_jeglConfig);
	m_jenv->DeleteGlobalRef(m_jsurfaceHolder);
}

void HQAndroidOpenGLContext::OnReset()
{
	/*--------get jni env-----------*/
	m_jenv = GetCurrentThreadJEnv();
	if (m_jenv == NULL)
		throw std::bad_alloc();
	
	jclass JEGLClass = m_jenv->FindClass("javax/microedition/khronos/egl/EGL10");
	
	if (JEGLClass == NULL )
		throw std::bad_alloc();
	
	
	/*----------------create window surface-------------*/
	jobject J_EGL_NO_SURFACE = m_jenv->GetStaticObjectField(JEGLClass, g_EGL_NO_SURFACE_ID);
	
	jobject l_jdrawSurface = m_jenv->CallObjectMethod(
							m_input.jegl ,
							g_jeglCreateWindowSurfaceMethodID,
							m_input.jdisplay,
							m_jeglConfig,
							m_jsurfaceHolder,
							NULL);
	if (l_jdrawSurface == NULL || 
		m_jenv->IsSameObject(l_jdrawSurface , J_EGL_NO_SURFACE) == JNI_TRUE)
	{
		//delete local ref
		m_jenv->DeleteLocalRef(JEGLClass);
		m_jenv->DeleteLocalRef(l_jdrawSurface);
		throw std::bad_alloc();
	}
	m_jdrawSurface = m_jenv->NewGlobalRef(l_jdrawSurface);
	if (m_jdrawSurface == NULL)//can't create global ref
	{
		//destroy surface
		m_jenv->CallBooleanMethod(
								m_input.jegl , 
								g_jeglDestroySurfaceMethodID,
								m_input.jdisplay,
								l_jdrawSurface);
		//delete local ref
		m_jenv->DeleteLocalRef(JEGLClass);
		m_jenv->DeleteLocalRef(l_jdrawSurface);
		throw std::bad_alloc();
	}
	m_jenv->DeleteLocalRef(l_jdrawSurface);
	
	/*----------------create context--------------------*/
	jobject J_EGL_NO_CONTEXT = m_jenv->GetStaticObjectField(JEGLClass, g_EGL_NO_CONTEXT_ID);
	jint attrib_list[] = {J_EGL_CONTEXT_CLIENT_VERSION , m_apiLevel , J_EGL_NONE};//attrib list
	jintArray jattrib_list = m_jenv->NewIntArray(3);
	m_jenv->SetIntArrayRegion(jattrib_list, 0, 3, attrib_list);//copy to java type
	
	jobject l_jeglContext = m_jenv->CallObjectMethod(
							m_input.jegl ,
							g_jeglCreateContextMethodID,
							m_input.jdisplay,
							m_jeglConfig,
							J_EGL_NO_CONTEXT,
							jattrib_list);
	if (l_jeglContext == NULL || 
		m_jenv->IsSameObject(l_jeglContext , J_EGL_NO_CONTEXT) == JNI_TRUE)	
	{
		//destroy surface
		m_jenv->CallBooleanMethod(
								m_input.jegl , 
								g_jeglDestroySurfaceMethodID,
								m_input.jdisplay,
								m_jdrawSurface);
		m_jenv->DeleteGlobalRef(m_jdrawSurface);
		//delete local ref
		m_jenv->DeleteLocalRef(jattrib_list);
		m_jenv->DeleteLocalRef(JEGLClass);
		m_jenv->DeleteLocalRef(l_jeglContext);
		throw std::bad_alloc();
	}
	m_jeglContext = m_jenv->NewGlobalRef(l_jeglContext);
	if (m_jeglContext == NULL)//can't create global ref
	{
		//destroy surface
		m_jenv->CallBooleanMethod(
								m_input.jegl , 
								g_jeglDestroySurfaceMethodID,
								m_input.jdisplay,
								m_jdrawSurface);
		//destroy context
		m_jenv->CallBooleanMethod(
								m_input.jegl , 
								g_jeglDestroyContextMethodID,
								m_input.jdisplay,
								l_jeglContext);
		
		//delete global references	
		m_jenv->DeleteGlobalRef(m_jdrawSurface);
		//delete local ref
		m_jenv->DeleteLocalRef(JEGLClass);
		m_jenv->DeleteLocalRef(jattrib_list);
		m_jenv->DeleteLocalRef(l_jeglContext);
		throw std::bad_alloc();
	}
	
	//delete local refs
	m_jenv->DeleteLocalRef(JEGLClass);
	m_jenv->DeleteLocalRef(jattrib_list);
	m_jenv->DeleteLocalRef(l_jeglContext);
	
	
	m_lostState = false;
}
void HQAndroidOpenGLContext::OnLost()
{
	m_jenv = GetCurrentThreadJEnv();
	
	jclass JEGLClass = m_jenv->FindClass("javax/microedition/khronos/egl/EGL10");
	jobject J_EGL_NO_SURFACE = m_jenv->GetStaticObjectField(JEGLClass, g_EGL_NO_SURFACE_ID);
	jobject J_EGL_NO_CONTEXT = m_jenv->GetStaticObjectField(JEGLClass, g_EGL_NO_CONTEXT_ID);
	
	m_jenv->CallBooleanMethod(
							m_input.jegl , 
							g_jeglMakeCurrentMethodID,
							m_input.jdisplay,
							J_EGL_NO_SURFACE,
							J_EGL_NO_SURFACE,
							J_EGL_NO_CONTEXT
								);
	
	if (!m_lostState)
	{
		//destroy surface
		m_jenv->CallBooleanMethod(
								m_input.jegl , 
								g_jeglDestroySurfaceMethodID,
								m_input.jdisplay,
								m_jdrawSurface);
		//destroy context
		m_jenv->CallBooleanMethod(
								m_input.jegl , 
								g_jeglDestroyContextMethodID,
								m_input.jdisplay,
								m_jeglContext);
							
		m_lostState = true;
	}
	//delete local reference
	m_jenv->DeleteLocalRef(JEGLClass);
}

void HQAndroidOpenGLContext::GetSurfaceSize()
{
	//get class
	jclass JSurfaceHolder = m_jenv->FindClass("android/view/SurfaceHolder");
	
	if (JSurfaceHolder == NULL )
		throw std::bad_alloc();

	static jmethodID jgetFrameRectMethodID = NULL;
	if (jgetFrameRectMethodID == NULL)
	{
		//get SurfaceHolder.getSurfaceFrame() method ID
		jgetFrameRectMethodID = m_jenv->GetMethodID(
					JSurfaceHolder,
					"getSurfaceFrame",
					"()Landroid/graphics/Rect;"
					);
	}
	//get surface's frame rect
	jobject jframeRect = m_jenv->CallObjectMethod(m_jsurfaceHolder , jgetFrameRectMethodID);
	
	//find Rect class
	jclass JRectClass = m_jenv->FindClass("android/graphics/Rect");
	static jmethodID jgetWidthMethodID = NULL;
	static jmethodID jgetHeightMethodID = NULL;
	if (jgetWidthMethodID == NULL)
	{
		//Rect.width()
		jgetWidthMethodID = m_jenv->GetMethodID(
					JRectClass,
					"width",
					"()I"
					);
	}
	if (jgetHeightMethodID == NULL)
	{
		//Rect.height()
		jgetHeightMethodID = m_jenv->GetMethodID(
					JRectClass,
					"height",
					"()I"
					);
	}
	
	m_surfaceWidth = m_jenv->CallIntMethod(jframeRect , jgetWidthMethodID);
	m_surfaceHeight = m_jenv->CallIntMethod(jframeRect , jgetHeightMethodID);
	
	//delete local refs
	m_jenv->DeleteLocalRef(JSurfaceHolder);
	m_jenv->DeleteLocalRef(JRectClass);
	m_jenv->DeleteLocalRef(jframeRect);
}

jint HQAndroidOpenGLContext::GetNativeWindowPixelFormat( )
{
	jintArray valueArray = m_jenv->NewIntArray(1);
	jint alpha, red, green, blue;

	//get alpha size
	m_jenv->CallBooleanMethod(
							m_input.jegl , 
							ge_jeglGetConfigAttribMethodID,
							m_input.jdisplay,
							m_jeglConfig,
							J_EGL_ALPHA_SIZE,
							valueArray);
	alpha = GetValue(m_jenv, valueArray);
	//get red size
	m_jenv->CallBooleanMethod(
							m_input.jegl , 
							ge_jeglGetConfigAttribMethodID,
							m_input.jdisplay,
							m_jeglConfig,
							J_EGL_RED_SIZE,
							valueArray);
	red = GetValue(m_jenv, valueArray);
	//get green size
	m_jenv->CallBooleanMethod(
							m_input.jegl , 
							ge_jeglGetConfigAttribMethodID,
							m_input.jdisplay,
							m_jeglConfig,
							J_EGL_GREEN_SIZE,
							valueArray);
	green = GetValue(m_jenv, valueArray);
	//get blue size
	m_jenv->CallBooleanMethod(
							m_input.jegl , 
							ge_jeglGetConfigAttribMethodID,
							m_input.jdisplay,
							m_jeglConfig,
							J_EGL_BLUE_SIZE,
							valueArray);
	blue = GetValue(m_jenv, valueArray);
	
	//get native window pixel format
	jclass JPixelFormat = m_jenv->FindClass("android/graphics/PixelFormat");
	static jfieldID TRANSLUCENT_ID = NULL;
	static jfieldID OPAQUE_ID = NULL;
	if (TRANSLUCENT_ID == NULL)
	{
		//get PixleFormat.TRANSLUCENT field id
		TRANSLUCENT_ID = m_jenv->GetStaticFieldID(
							JPixelFormat,
							"TRANSLUCENT", 
							"I");
	}
	
	if (OPAQUE_ID == NULL)
	{
		//get PixleFormat.OPAQUE field id
		OPAQUE_ID = m_jenv->GetStaticFieldID(
							JPixelFormat,
							"OPAQUE", 
							"I");
	}


	jint TRANSLUCENT = m_jenv->GetStaticIntField(JPixelFormat, TRANSLUCENT_ID);
	jint OPAQUE = m_jenv->GetStaticIntField(JPixelFormat, OPAQUE_ID);
	

	//release local ref
	m_jenv->DeleteLocalRef(valueArray);
	m_jenv->DeleteLocalRef(JPixelFormat);
	
	if (red == 8 && green == 8 && blue == 8 && alpha == 8)
		return TRANSLUCENT;
	return OPAQUE;
}

bool HQAndroidOpenGLContext::IsLost()
{
	return HQEngineInternalIsJSurfaceCreated() == 0;
}

void HQAndroidOpenGLContext::SwapBuffers()
{
	if (HQEngineInternalIsJSurfaceCreated() == 1)
		m_jenv->CallBooleanMethod(
							m_input.jegl , 
							g_jeglSwapBuffersMethodID,
							m_input.jdisplay,
							m_jdrawSurface);
}

bool HQAndroidOpenGLContext::MakeCurrent()
{
	m_jenv = GetCurrentThreadJEnv();
	if (m_jenv == NULL)
		return false;
		
	return m_jenv->CallBooleanMethod(
							m_input.jegl , 
							g_jeglMakeCurrentMethodID,
							m_input.jdisplay,
							m_jdrawSurface,
							m_jdrawSurface,
							m_jeglContext
								);
}
