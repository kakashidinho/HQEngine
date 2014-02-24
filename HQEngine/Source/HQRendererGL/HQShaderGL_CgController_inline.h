/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_SHADER_GL_CG_INLINE_H
#define HQ_SHADER_GL_CG_INLINE_H

#include "HQShaderGL_CgController.h"
#include <string.h>

#ifndef GLES

#ifndef _CGENTRY
#	ifdef WIN32
#		define _CGENTRY __cdecl
#	else
#		define _CGENTRY
#	endif
#endif




inline HQReturnVal HQBaseCgShaderController::DeActiveProgramCg(HQSharedPtr<HQBaseShaderProgramGL>& pProgram)
{
	
	if(pProgram->vertexShaderID != HQ_NOT_USE_VSHADER)
	{
		cgGLUnbindProgram(this->cgVertexProfile);
		cgGLDisableProfile(this->cgVertexProfile);
	}
	if(pProgram->pixelShaderID != HQ_NOT_USE_PSHADER)
	{
		cgGLUnbindProgram(this->cgFragmentProfile);
		cgGLDisableProfile(this->cgFragmentProfile);
	}
	if(pProgram->geometryShaderID != HQ_NOT_USE_GSHADER)
	{
		cgGLUnbindProgram(this->cgGeometryProfile);
		cgGLDisableProfile(this->cgGeometryProfile);
	}
	
	return HQ_OK;
}
inline HQReturnVal HQBaseCgShaderController::ActiveProgramCg(HQSharedPtr<HQBaseShaderProgramGL>& pProgram)
{
	cgGLBindProgram(pProgram->program);
	return HQ_OK;
}
/*-----------------------*/
inline HQReturnVal HQBaseCgShaderController::SetUniformIntCg(CGparameter param, const hq_int32* pValues,
								hq_uint32 numElements)
{
	cgSetParameterValueir(param,(int)numElements,pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseCgShaderController::SetUniform2IntCg(CGparameter param, const hq_int32* pValues,
								hq_uint32 numElements)
{
	cgSetParameterValueir(param,(int)numElements * 2,pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseCgShaderController::SetUniform3IntCg(CGparameter param, const hq_int32* pValues,
								hq_uint32 numElements)
{
	cgSetParameterValueir(param,(int)numElements * 3,pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseCgShaderController::SetUniform4IntCg(CGparameter param, const hq_int32* pValues,
								hq_uint32 numElements)
{
	cgSetParameterValueir(param,(int)numElements * 4,pValues);

	return HQ_OK;
}


inline HQReturnVal HQBaseCgShaderController::SetUniformFloatCg(CGparameter param, const hq_float32* pValues,
									hq_uint32 numElements)
{
	cgSetParameterValuefr(param,(int)numElements,pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseCgShaderController::SetUniform2FloatCg(CGparameter param, const hq_float32* pValues,
									hq_uint32 numElements)
{
	cgSetParameterValuefr(param,(int)numElements * 2,pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseCgShaderController::SetUniform3FloatCg(CGparameter param, const hq_float32* pValues,
									hq_uint32 numElements)
{
	cgSetParameterValuefr(param,(int)numElements * 3,pValues);

	return HQ_OK;
}

inline HQReturnVal HQBaseCgShaderController::SetUniform4FloatCg(CGparameter param, const hq_float32* pValues,
									hq_uint32 numElements)
{
	cgSetParameterValuefr(param,(int)numElements * 4,pValues);

	return HQ_OK;
}



inline HQReturnVal HQBaseCgShaderController::SetUniformMatrixCg(CGparameter param, const HQBaseMatrix4* pMatrices,
									hq_uint32 numMatrices)
{
	cgSetParameterValuefr(param,(int)numMatrices * 16,(hq_float32*)pMatrices);

	//cgGLSetMatrixParameterArrayfr(param,0,numMatrices,(hq_float32*)pMatrices);

	return HQ_OK;
}

inline HQReturnVal HQBaseCgShaderController::SetUniformMatrixCg(CGparameter param, const HQBaseMatrix3x4* pMatrices,
									hq_uint32 numMatrices)
{
	cgSetParameterValuefr(param,(int)numMatrices * 12,(hq_float32*)pMatrices);

	//cgGLSetMatrixParameterArrayfr(param,0,numMatrices,(hq_float32*)pMatrices);

	return HQ_OK;
}

#endif//ifndef GLES


#endif
