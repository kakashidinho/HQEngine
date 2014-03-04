/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "HQShaderGL_CgController_inline.h"
#include "HQShaderGL_GLSLController_inline.h"
#include "HQFixedFunctionShaderManagerGL.h"
#include "HQDeviceGL.h"
#include <string.h>

HQBaseShaderManagerGL * HQCreateShaderManager(int shaderManagerType, HQLogStream *logFileStream , bool flushLog)
{
	HQBaseShaderManagerGL * shaderMan = NULL;
	/*---------create shader manager object based on capabilities and option---------*/
	typedef HQShaderManagerGL<HQGLSLShaderController , HQBaseCommonShaderManagerGL> GLSLShaderManager;
#ifndef GLES
	typedef HQShaderManagerGL<HQCombineShaderController , HQBaseCommonShaderManagerGL> CombineShaderManager;
	typedef HQShaderManagerGL<HQCombineShaderController , HQBaseShaderManagerGL_UBO> CombineShaderManagerUBO;
	typedef HQShaderManagerGL<HQGLSLShaderController , HQBaseShaderManagerGL_UBO> GLSLShaderManagerUBO;
	typedef HQShaderManagerGL<HQCgShaderController , HQBaseCommonShaderManagerGL> CgShaderManager;
	typedef HQShaderManagerGL<HQCgShaderController , HQBaseShaderManagerGL_UBO> CgShaderManagerUBO;

	bool uniformBufferSupported = GLEW_VERSION_3_1 || GLEW_ARB_uniform_buffer_object;
#endif

#ifndef GLES
	switch (shaderManagerType)
	{
	case COMBINE_SHADER_MANAGER:
		if (uniformBufferSupported)
			shaderMan = new CombineShaderManagerUBO(logFileStream , "GL Shader Manager :" , flushLog);
		else
			shaderMan = new CombineShaderManager(logFileStream , "GL Shader Manager :" , flushLog);
		break;
	case CG_SHADER_MANAGER:
		if (uniformBufferSupported)
			shaderMan = new CgShaderManagerUBO(logFileStream , "GL Shader Manager - Cg only :" , flushLog);
		else
			shaderMan = new CgShaderManager(logFileStream , "GL Shader Manager - Cg only:" , flushLog);
		break;
	case GLSL_SHADER_MANAGER:
		if (uniformBufferSupported)
			shaderMan = new GLSLShaderManagerUBO(logFileStream , "GL Shader Manager - GLSL only:" , flushLog);
		else
#endif
		{
#ifdef ANDROID
			if (!GLEW_VERSION_2_0)
				shaderMan = new HQFixedFunctionShaderManagerGL(logFileStream, flushLog);
			else
#endif
				shaderMan = new GLSLShaderManager(logFileStream , "GL Shader Manager - GLSL only:" , flushLog);
		}
		
#ifndef GLES		
		break;
	}
#endif

	return shaderMan;
}
