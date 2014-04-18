/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "HQShaderGL_GLSLController_inline.h"
#include "HQFixedFunctionShaderManagerGL.h"
#include "HQDeviceGL.h"
#include <string.h>

HQBaseShaderManagerGL * HQCreateShaderManager(int shaderManagerType, HQLogStream *logFileStream , bool flushLog)
{
	HQBaseShaderManagerGL * shaderMan = NULL;
	/*---------create shader manager object based on capabilities and option---------*/
	typedef HQShaderManagerGL<HQGLSLShaderController, HQBaseShaderManagerGL_FakeUBO> GLSLShaderManager;
#ifdef HQ_GL_UNIFORM_BUFFER_DEFINED
	typedef HQShaderManagerGL<HQGLSLShaderController , HQBaseShaderManagerGL_UBO> GLSLShaderManagerUBO;
#	ifdef HQ_GLSL_SHADER_PIPELINE_DEFINED
	typedef HQShaderManagerGL<HQGLSLShaderPipelineController, HQBaseShaderManagerGL_UBO> GLSLShaderPipelineManagerUBO;
#	endif

	bool uniformBufferSupported = GLEW_VERSION_3_1 == GL_TRUE;
#endif

	switch (shaderManagerType)
	{
	case HQ_GLSL_SHADER_MANAGER:
#ifdef HQ_GL_UNIFORM_BUFFER_DEFINED
		if (uniformBufferSupported)
		{
#	ifdef HQ_GLSL_SHADER_PIPELINE_DEFINED
			if (GLEW_VERSION_4_1)
				shaderMan = new GLSLShaderPipelineManagerUBO(logFileStream, "GL Shader Pipeline Manager:", flushLog);
			else
#endif
				shaderMan = new GLSLShaderManagerUBO(logFileStream, "GL Shader Manager (UBO supported):", flushLog);

		}
		else
#endif//#ifdef HQ_GL_UNIFORM_BUFFER_DEFINED
		{
#ifdef HQ_ANDROID_PLATFORM
			if (!GLEW_VERSION_2_0)
				shaderMan = new HQFixedFunctionShaderManagerGL(logFileStream, flushLog);
			else
#endif
				shaderMan = new GLSLShaderManager(logFileStream , "GL Shader Manager:" , flushLog);
		}
			
		break;
	}

	return shaderMan;
}
