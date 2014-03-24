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
	typedef HQShaderManagerGL<HQGLSLShaderController , HQBaseCommonShaderManagerGL> GLSLShaderManager;
#ifndef HQ_OPENGLES
	typedef HQShaderManagerGL<HQGLSLShaderController , HQBaseShaderManagerGL_UBO> GLSLShaderManagerUBO;

	bool uniformBufferSupported = GLEW_VERSION_3_1 || GLEW_ARB_uniform_buffer_object;
#endif

#ifndef HQ_OPENGLES
	switch (shaderManagerType)
	{
	case GLSL_SHADER_MANAGER:
		if (uniformBufferSupported)
			shaderMan = new GLSLShaderManagerUBO(logFileStream , "GL Shader Manager (UBO supported):" , flushLog);
		else
#endif
		{
#ifdef HQ_ANDROID_PLATFORM
			if (!GLEW_VERSION_2_0)
				shaderMan = new HQFixedFunctionShaderManagerGL(logFileStream, flushLog);
			else
#endif
				shaderMan = new GLSLShaderManager(logFileStream , "GL Shader Manager:" , flushLog);
		}
		
#ifndef HQ_OPENGLES		
		break;
	}
#endif

	return shaderMan;
}
