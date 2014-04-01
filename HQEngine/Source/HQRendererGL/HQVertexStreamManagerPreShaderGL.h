/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_VSTREAM_MAN_PRE_SHADER_H
#define HQ_VSTREAM_MAN_PRE_SHADER_H

#include "HQVertexStreamManagerGL.h"

/*--------no shader + vbo----------*/
class HQVertexStreamManagerNoShaderGL : public HQVertexStreamManagerGL
{
protected:
	bool IsVertexAttribValid(const HQVertexAttribDesc &vAttribDesc);
public:
	HQVertexStreamManagerNoShaderGL(hq_uint32 maxVertexAttribs , HQLogStream *logFileStream , bool flushLog)
		:HQVertexStreamManagerGL(maxVertexAttribs , logFileStream , flushLog)
	{
	}

	HQReturnVal SetVertexBuffer(HQVertexBuffer* vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride ) ;

	HQReturnVal SetVertexInputLayout(HQVertexLayout* inputLayoutID) ;

	/*--------for HQVertexStreamManDelegateGL---------*/
	inline static void EnableVertexAttribArray(GLuint index);
	inline static void DisableVertexAttribArray(GLuint index);
	inline static void SetVertexAttribPointer(const HQVertexAttribInfoGL &vAttribInfo , hq_uint32 stride, const HQBufferGL *vbuffer);
};

#endif
