/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "jnicompilerheader.h"

#include "../../../../HQEngine/Source/HQEngine/HQDefaultFileManager.h"
#include "../../../../HQEngine/Source/HQRendererGL/HQShaderGL_GLSL_VarParser.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sstream>

#define PREPROCESS_FILE

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

#include "GL/glew.h"
#include "glut.h"

int glutWindow = 0;

#endif

#ifdef WIN32
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
	case 5: 
		shaderTypeGL = GL_COMPUTE_SHADER;
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

	std::string processed_src = source;
	std::string version_string_in_src = "";
	/*------ Remove #version in original source---------*/ 
	{
		size_t pos1 = processed_src.find("#");
		if (pos1 != std::string::npos)
		{
			size_t pos2 = processed_src.find("version", pos1);
			if (pos2 != std::string::npos)
			{
				bool found = true;
				for (size_t i = pos1 + 1; i < pos2; ++i)//make sure only white spaces are between "#" and "version"
				{
					char c = processed_src[i];
					if (c != ' ' && c != '\t' && c != '\r')
					{
						found = false;
						break;
					}
				}
				
				if (found)
				{
					size_t pos3 = processed_src.find("\n", pos2 + 7);
					version_string_in_src.assign(source + pos1, pos3 - pos1 + 1); 
					for (size_t i = pos1; i < pos3; ++i)
					{
						processed_src[i] = ' ';//remove from the source
					}
				}

			}//if (pos2 != std::string::npos)
		}//if (pos1 != std::string::npos)
	}
	
	int version_number = 0;
	if (versionDeclare != NULL)
		sscanf(versionDeclare, "#version %d", &version_number);
	else
		sscanf(version_string_in_src.c_str(), "#version %d", &version_number);
	std::string uniform_buffer_extension_line = "";
	const char prefDefExtVersion[] = "#define HQEXT_GLSL\n";
	if (GLEW_VERSION_3_1 || GLEW_ARB_uniform_buffer_object)
	{
		if (version_number < 140)
			uniform_buffer_extension_line = "#extension GL_ARB_uniform_buffer_object : enable\n";
	}

	std::stringstream message;

#ifdef PREPROCESS_FILE

	/*--------preprocess code-------------*/
	{
		HQDefaultFileManager includeFileManager;
		HQLoggableObject nullLogger("", false, 1);
		HQVarParserGL preprocessor(&nullLogger);

		//combine the code
		std::string codeToPreprocess;
		if (versionDeclare != NULL)//use version in source if user doesn't specify any in command arguments
			codeToPreprocess = versionDeclare;
		else
			codeToPreprocess = version_string_in_src;
		if (shaderTypeGL == GL_VERTEX_SHADER)//vertex attribute semantic keywords
			codeToPreprocess += semanticKeywords;

		codeToPreprocess += cMacros;
		codeToPreprocess += prefDefExtVersion;
		codeToPreprocess += samplerKeywords;
		codeToPreprocess += uniform_buffer_extension_line +
			processed_src;

		//find source file's containing directory
		std::string source_file_std_string = cfileName;
		std::string containingFoler;

		//find last '/' or '\\'
		size_t pos1 = source_file_std_string.find_last_of('/');
		size_t pos2 = source_file_std_string.find_last_of('\\');
		size_t last_slash;
		if (pos1 != std::string::npos)
		{
			last_slash = pos1;
			if (pos2 != std::string::npos && pos2 > pos1)
				last_slash = pos2;
		}
		else
			last_slash = pos2;

		if (last_slash != std::string::npos)
		{
			containingFoler = source_file_std_string.substr(0, last_slash);
			includeFileManager.AddFirstSearchPath(containingFoler.c_str());
		}

		//now preprocess
		preprocessor.Parse(codeToPreprocess.c_str(), &includeFileManager, "", NULL,
			processed_src, true, NULL, NULL, NULL);

		//print preprocess code
		hquint32 line = 1;
		char c;
		message << "preprocessed code:\n";
		message << line << ": ";
		for (hquint32 i = 0; i < processed_src.size(); ++i)
		{
			c = processed_src[i];
			message << c;
			if (c == '\n')
			{
				line ++;
				message << line << ": ";
			}
		}

		message << "\n";
	}


	//now send to opengl
	const GLchar* sourceArray[] = {
		processed_src.c_str()
	};
#else
	const GLchar* sourceArray[] = {
		version_string_in_src.c_str(),
		cMacros,
		prefDefExtVersion,
		semanticKeywords,
		samplerKeywords,
		uniform_buffer_extension_line.c_str(),
		"#line 0 0\n",
		processed_src.c_str()
	};
	if (versionDeclare != NULL)//use version in source if user doesn't specify version
		sourceArray[0] = versionDeclare;
	if (shaderTypeGL != GL_VERTEX_SHADER)
		sourceArray[3] = "";//only vertex shader need sematic definitions
#endif
	jstring result;

	int sourceSize = sizeof(sourceArray) / sizeof(GLchar*);

	glShaderSource(shader, sourceSize, (const GLchar**)sourceArray, NULL);
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
			message << "compilation succeeded!\n";
		else
			message << infoLog << "\n";

        free(infoLog);
    }
	else
		message << "compilation succeeded!\n";

	if (versionDeclare != NULL)
		delete [] versionDeclare;
	delete[] source;
	glDeleteShader(shader);
	env->ReleaseStringUTFChars(macros , cMacros);
	
	result = env->NewStringUTF(message.str().c_str());

	return result;
}

#ifdef __cplusplus
}
#endif
