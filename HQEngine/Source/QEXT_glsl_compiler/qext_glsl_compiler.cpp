/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#define _CRT_SECURE_NO_WARNINGS
#include "jnicompilerheader.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef __APPLE__

#define GL_GEOMETRY_SHADER 0x8DD9
#define GL_TESS_EVALUATION_SHADER 0x8E87
#define GL_TESS_CONTROL_SHADER 0x8E88

#import <QuartzCore/QuartzCore.h>
#import <OpenGL/gl.h>
#import <OpenGL/glu.h>
#import <OpenGL/glext.h>

CGLContextObj currentCGLContext;
CGLContextObj myCGLContext = 0;

#else

#include <GL/glew.h>
#include <GL/glut.h>

int glutWindow = 0;

#endif

#ifdef WIN32
#	pragma comment(lib, "glut32.lib")
#	pragma comment(lib,"opengl32.lib")
#endif

const char semanticKeywords[] =
	"\
#define VPOSITION\n\
#define VCOLOR\n\
#define VNORMAL\n\
#define VTEXCOORD0\n\
#define VTEXCOORD1\n\
#define VTEXCOORD2\n\
#define VTEXCOORD3\n\
#define VTEXCOORD4\n\
#define VTEXCOORD5\n\
#define VTEXCOORD6\n\
#define VTEXCOORD7\n\
#define VTANGENT\n\
#define VBINORMAL\n\
#define VBLENDWEIGHT\n\
#define VBLENDINDICES\n\
#define VPSIZE\n"
;

const char samplerKeywords[] =
"\
#define TEXUNIT0\n\
#define TEXUNIT1\n\
#define TEXUNIT2\n\
#define TEXUNIT3\n\
#define TEXUNIT4\n\
#define TEXUNIT5\n\
#define TEXUNIT6\n\
#define TEXUNIT7\n\
#define TEXUNIT8\n\
#define TEXUNIT9\n\
#define TEXUNIT10\n\
#define TEXUNIT11\n\
#define TEXUNIT12\n\
#define TEXUNIT13\n\
#define TEXUNIT14\n\
#define TEXUNIT15\n\
#define TEXUNIT16\n\
#define TEXUNIT17\n\
#define TEXUNIT18\n\
#define TEXUNIT19\n\
#define TEXUNIT20\n\
#define TEXUNIT21\n\
#define TEXUNIT22\n\
#define TEXUNIT23\n\
#define TEXUNIT24\n\
#define TEXUNIT25\n\
#define TEXUNIT26\n\
#define TEXUNIT27\n\
#define TEXUNIT28\n\
#define TEXUNIT29\n\
#define TEXUNIT30\n\
#define TEXUNIT31\n";



bool ready = false;
void NullDisplay() {};


#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     hqengineshadercompiler_HQEngineShaderCompilerView
 * Method:    InitGL
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_hqengineshadercompiler_HQEngineShaderCompilerView_InitGL
  (JNIEnv *env, jobject object)
{
#ifdef __APPLE__
	if (myCGLContext != 0)
		return 1;
	//create dummy context
	CGDirectDisplayID display = CGMainDisplayID ();
	CGOpenGLDisplayMask myDisplayMask = CGDisplayIDToOpenGLDisplayMask (display);

	CGLPixelFormatAttribute attribs[] = {
		kCGLPFADisplayMask,(CGLPixelFormatAttribute) myDisplayMask,
		(CGLPixelFormatAttribute) 0
	};

	CGLPixelFormatObj pixelFormat = NULL;
	GLint numPixelFormats = 0;
	currentCGLContext = CGLGetCurrentContext ();

	CGLChoosePixelFormat (attribs, &pixelFormat, &numPixelFormats);
	if (pixelFormat == NULL)
	{
		return 0;
	}

	CGLCreateContext (pixelFormat, NULL, &myCGLContext);
	CGLDestroyPixelFormat (pixelFormat);
	CGLSetCurrentContext (myCGLContext);
	if (myCGLContext == 0)
		return 0;
#else
	if (glutWindow != 0)
		return 1;
	char* argv[1];
	char   dummyString[8];
	argv[0] = dummyString;
	int      argc = 0;
	glutInit(&argc, argv); //initialize the tool kit
	glutInitDisplayMode(GLUT_SINGLE |GLUT_RGB);//set the display mode
	glutInitWindowSize(100, 100); //set window size
	glutInitWindowPosition(20, 20); // set window position on screen
	glutWindow = glutCreateWindow("null window"); // open the screen window
	glutDisplayFunc(NullDisplay);
	if (glutWindow == 0 || glewInit() != GLEW_OK)
		return 0;

#endif
	ready = true;
	return 1;
}

/*
 * Class:     hqengineshadercompiler_HQEngineShaderCompilerView
 * Method:    ReleaseGL
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_hqengineshadercompiler_HQEngineShaderCompilerView_ReleaseGL
  (JNIEnv * env, jobject obj)
{

#ifdef __APPLE__
	if (myCGLContext != 0)
	{
		CGLDestroyContext (myCGLContext);
		CGLSetCurrentContext (currentCGLContext);
		myCGLContext = 0;
	}
#else
	if (glutWindow)
	{
		glutDestroyWindow(glutWindow);
		glutWindow = 0;
	}
#endif

	ready = false;
}

/*
 * Class:     hqengineshadercompiler_HQEngineShaderCompilerView
 * Method:    CompileGLSL
 * Signature: (Ljava/lang/String;II)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_hqengineshadercompiler_HQEngineShaderCompilerView_CompileGLSL
  (JNIEnv *env, jobject obj, jstring fileName, jstring version, jstring macros , jint shaderType)
{
	if (!ready)
		return env->NewStringUTF("");
#ifndef __APPLE__
	if (!GLEW_VERSION_2_0)
		return env->NewStringUTF("openGL 2.0 is not supported!");
#endif
	GLenum shaderTypeGL;

	switch (shaderType)
	{
	case 0:
		shaderTypeGL = GL_VERTEX_SHADER;
		break;
	case 1:
		shaderTypeGL = GL_GEOMETRY_SHADER;
		break;
	case 2:
		shaderTypeGL = GL_FRAGMENT_SHADER;
		break;
	case 3:
		shaderTypeGL = GL_TESS_CONTROL_SHADER;
		break;
	case 4:
		shaderTypeGL = GL_TESS_EVALUATION_SHADER;
		break;
	}


	GLuint shader = glCreateShader(shaderTypeGL);

	if (glGetError() == GL_INVALID_ENUM)
		return env->NewStringUTF("this shader type is not supported!");

	const char *cfileName;
	jboolean isCopy;

	cfileName = env->GetStringUTFChars(fileName , &isCopy);

	FILE *f = fopen(cfileName,"rb");

	env->ReleaseStringUTFChars(fileName , cfileName);

	fseek(f, 0  , SEEK_END);
	long size = ftell(f);
	rewind(f);

	char *source = new char[size + 1];
	if(source == NULL)
	{
		fclose(f);

		glDeleteShader(shader);
		return env->NewStringUTF("memory allocation failed!");
	}

	fread(source,size ,1,f);
	source[size] = '\0';
	fclose(f);

	char *versionDeclare = NULL;
	const char *cVersion = env->GetStringUTFChars(version ,&isCopy);
	if (strcmp(cVersion , ""))
	{
		unsigned int len = strlen(cVersion);
		versionDeclare = new char[len + 11];
		sprintf(versionDeclare , "#version %s\n" , cVersion);
	}
	env->ReleaseStringUTFChars(version , cVersion);

	const char *cMacros = env->GetStringUTFChars(macros , &isCopy);

	const GLchar* sourceArray[] = {
		"",
		cMacros,
#ifdef GLES
		"#define HQEXT_GLSL_ES\n",
#else
		"#define HQEXT_GLSL\n",
#endif
		semanticKeywords,
		samplerKeywords,
		"#line 0 0\n",
		source
	};
	if (versionDeclare != NULL)//NULL if user doesn't specify version
		sourceArray[0] = versionDeclare;
	if (shaderTypeGL != GL_VERTEX_SHADER)
		sourceArray[3] = "";//only vertex shader need sematic definitions

	jstring result;

	glShaderSource(shader, 7, (const GLchar**)sourceArray, NULL);
	glCompileShader(shader);



	GLint compileOK;
	glGetShaderiv(shader , GL_COMPILE_STATUS , &compileOK);
	int infologLength = 0;
    int charsWritten  = 0;
    char *infoLog;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH,&infologLength);
    if (infologLength > 0)
    {
        infoLog = (char *)malloc(infologLength);
        glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);

		if (!strcmp(infoLog , ""))
			result = env->NewStringUTF("compilation succeeded!");
		else
			result = env->NewStringUTF(infoLog);

        free(infoLog);
    }
	else
		result = env->NewStringUTF("compilation succeeded!");

	if (versionDeclare != NULL)
		delete [] versionDeclare;
	delete[] source;
	glDeleteShader(shader);
	env->ReleaseStringUTFChars(macros , cMacros);

	return result;
}

#ifdef __cplusplus
}
#endif
