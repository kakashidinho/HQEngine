/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_ANDROID_RENDERER_H
#define HQ_ANDROID_RENDERER_H
#include "../HQRenderDevice.h"
#include "AndroidGLES.h"



/*------------------------------------------------------
this file is used when EGL native API is not available
------------------------------------------------------*/
/*------config types----------*/
const jint J_EGL_ALPHA_SIZE = 0x3021;
const jint J_EGL_BLUE_SIZE = 0x3022;
const jint J_EGL_GREEN_SIZE = 0x3023;
const jint J_EGL_RED_SIZE = 0x3024;
const jint J_EGL_DEPTH_SIZE	= 0x3025;
const jint J_EGL_STENCIL_SIZE = 0x3026;
const jint J_EGL_NONE = 0x3038;
const jint J_EGL_RENDERABLE_TYPE = 0x3040;
/*-----config values---------*/
const jint J_EGL_OPENGL_ES_BIT = 0x0001;//for J_EGL_RENDERABLE_TYPE
const jint J_EGL_OPENGL_ES2_BIT = 0x0004;//for J_EGL_RENDERABLE_TYPE

extern JavaVM * ge_jvm;
extern jmethodID ge_jeglGetConfigAttribMethodID;//eglGetConfigAttrib(EGLDisplay display, EGLConfig config, int attribute, int[] value)
extern jmethodID ge_jeglChooseConfigMethodID;//eglChooseConfig(EGLDisplay display, int[] attrib_list, EGLConfig[] configs, int config_size, int[] num_config) methodID

//jni helper functions
extern JNIEnv * AttachCurrenThreadJEnv();

inline JNIEnv * GetCurrentThreadJEnv()
{
	JNIEnv *env;
	jint re = ge_jvm->GetEnv((void**)&env, JNI_VERSION_1_2);

	if(re == JNI_EDETACHED) 
	{
		return AttachCurrenThreadJEnv();
	}

	return env;
}


void InitEGLMethodAndAttribsIDs();//get java egl's methods and attributes' IDs

jint GetValue (JNIEnv * jenv, jintArray array);//get value from one element java array

/*------HQAndroidOpenGLContext------------*/
class HQAndroidOpenGLContext
{
	public:
		HQAndroidOpenGLContext(
				const HQAndroidRenderDeviceInitInput & input ,
				jobject jeglConfig,	//reference to javax.microedition.khronos.egl.EGLConfig object
				jint apiLevel//1 or 2
				);
		~HQAndroidOpenGLContext();
		
		bool IsLost();
		
		void SwapBuffers();
		bool MakeCurrent();
		jint GetSurfaceWidth() {return m_surfaceWidth;}
		jint GetSurfaceHeight() {return m_surfaceHeight;}
		
		void OnReset();
		void OnLost();
		
	private:
		jint GetNativeWindowPixelFormat();
		void GetSurfaceSize();
		HQAndroidRenderDeviceInitInput m_input;
		JNIEnv * m_jenv;
		jobject m_jsurfaceHolder;//global reference to android.view.SurfaceHolder object
		jobject m_jdrawSurface;//global reference to javax.microedition.khronos.egl.EGLSurface object
		jobject m_jeglContext;//global reference to javax.microedition.khronos.egl.EGLContext object
		jobject m_jeglConfig;//global reference to javax.microedition.khronos.egl.EGLConfig object
		jint m_surfaceWidth;
		jint m_surfaceHeight;
		jint m_apiLevel;
		
		bool m_lostState;
};

#endif
