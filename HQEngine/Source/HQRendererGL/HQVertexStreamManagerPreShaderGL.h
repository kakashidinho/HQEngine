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

	HQReturnVal SetVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride ) ;

	HQReturnVal SetVertexInputLayout(hq_uint32 inputLayoutID) ;

	/*--------for HQVertexStreamManDelegateGL---------*/
	inline static void EnableVertexAttribArray(GLuint index);
	inline static void DisableVertexAttribArray(GLuint index);
	inline static void SetVertexAttribPointer(const HQVertexAttribInfoGL &vAttribInfo , hq_uint32 stride, const HQBufferGL *vbuffer);
};

/*--------no shader + vbo + no map buffer----------*/
class HQVertexStreamManagerNoShaderNoMapGL : public HQVertexStreamManagerNoMapGL
{
public:
	HQVertexStreamManagerNoShaderNoMapGL(hq_uint32 maxVertexAttribs , HQLogStream *logFileStream , bool flushLog)
		:HQVertexStreamManagerNoMapGL(maxVertexAttribs , logFileStream , flushLog)
	{
	}

	HQReturnVal SetVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride ) ;

	HQReturnVal SetVertexInputLayout(hq_uint32 inputLayoutID) ;

	/*--------for HQVertexStreamManDelegateGL---------*/
	inline static void EnableVertexAttribArray(GLuint index);
	inline static void DisableVertexAttribArray(GLuint index);
	inline static void SetVertexAttribPointer(const HQVertexAttribInfoGL &vAttribInfo , hq_uint32 stride, const HQBufferGL *vbuffer);
};

/*-------no shader + no vbo--------*/
class HQVertexStreamManagerArrayGL : public HQVertexStreamManagerNoShaderGL
{
public:
	HQVertexStreamManagerArrayGL(hq_uint32 maxVertexAttribs , HQLogStream *logFileStream , bool flushLog)
		:HQVertexStreamManagerNoShaderGL(maxVertexAttribs , logFileStream , flushLog)
	{
	}
	HQReturnVal CreateVertexBuffer(const void *initData , hq_uint32 size , 
									bool dynamic , bool isForPointSprites,
									hq_uint32 *pVertexBufferID);
	HQReturnVal CreateIndexBuffer(const void *initData , 
								  hq_uint32 size , bool dynamic  ,
								  HQIndexDataType indexDataType  , 
								  hq_uint32 *pIndexBufferID);

	HQReturnVal SetVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride ) ;

	HQReturnVal SetIndexBuffer(hq_uint32 indexBufferID );

	HQReturnVal SetVertexInputLayout(hq_uint32 inputLayoutID) ;
	
	HQReturnVal MapVertexBuffer(hq_uint32 vertexBufferID , HQMapType mapType , void **ppData) ;
	HQReturnVal UnmapVertexBuffer(hq_uint32 vertexBufferID) ;
	HQReturnVal MapIndexBuffer( HQMapType mapType , void **ppData) ;
	HQReturnVal UnmapIndexBuffer() ;

	HQReturnVal UpdateVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 offset , hq_uint32 size , const void * pData);
	HQReturnVal UpdateIndexBuffer(hq_uint32 offset , hq_uint32 size , const void * pData);
	HQReturnVal UpdateIndexBuffer(hq_uint32 indexBufferID, hq_uint32 offset , hq_uint32 size , const void * pData);

	/*--------for HQVertexStreamManDelegateGL---------*/
	inline void BindVertexBuffer(GLuint vertexBufferName) {}//do nothing, because we use vertex array instead of VBO
	inline static void SetVertexAttribPointer(const HQVertexAttribInfoGL &vAttribInfo , hq_uint32 stride, const HQBufferGL *vbuffer);
};

#endif
