/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _VERTEX_STREAM_MAN_GL_
#define _VERTEX_STREAM_MAN_GL_

#include "glHeaders.h"
#include "../HQVertexStreamManager.h"
#include "../HQLoggableObject.h"
#include "../HQItemManager.h"
#include "HQCommonVertexStreamManGL.h"



struct HQIndexBufferGL :public HQBufferGL
{
	HQIndexBufferGL(hq_uint32 size , GLenum usage , HQIndexDataType dataType)
		: HQBufferGL(size , usage )
	{
		switch (dataType)
		{
		case HQ_IDT_UINT:
			this->dataType = GL_UNSIGNED_INT;
			break;
		default:
			this->dataType = GL_UNSIGNED_SHORT;
		}
	}

	void OnCreated(HQVertexStreamManagerGL *manager, const void *initData);

	GLenum dataType;
};

/*--------
fully capable vertex manager
shader + vbo + oes map buffer
--------*/
class HQVertexStreamManagerGL: public HQVertexStreamManager , public HQLoggableObject
{
protected:
	HQItemManager<HQBufferGL> vertexBuffers;
	HQItemManager<HQIndexBufferGL> indexBuffers;
	HQItemManager<HQVertexInputLayoutGL> inputLayouts;
	HQVertexStreamGL *streams;

	GLuint currentBoundVBuffer;
	HQSharedPtr<HQIndexBufferGL> activeIndexBuffer;
	GLenum indexDataType;
	hq_uint32 indexShiftFactor;
	void *indexStartAddress;//for compatibility with vertex array
	HQSharedPtr<HQVertexInputLayoutGL> activeInputLayout;
	
	hq_uint32 maxVertexAttribs;
	HQVertexAttribInfoNodeGL *vAttribInfoNodeCache;

	void ConvertToVertexAttribInfo(const HQVertexAttribDesc &vAttribDesc ,HQVertexAttribInfoGL &vAttribInfo);
	virtual bool IsVertexAttribValid(const HQVertexAttribDesc &vAttribDesc) {return true;}////check if vertex attribute' desc is valid
	
	
	HQVertexStreamManagerGL(const char *logPrefix, hq_uint32 maxVertexAttribs , HQLogStream *logFileStream , bool flushLog);

public:
	HQVertexStreamManagerGL(hq_uint32 maxVertexAttribs , HQLogStream *logFileStream , bool flushLog);
	~HQVertexStreamManagerGL() ;
	
	
	GLuint GetCurrentBoundVBuffer() {return this->currentBoundVBuffer;}
	void InvalidateCurrentBoundVBuffer() {this->currentBoundVBuffer = 0;}
	inline GLenum GetIndexDataType() {return this->indexDataType;}
	inline hq_uint32 GetIndexShiftFactor() {return this->indexShiftFactor;}
	inline void * GetIndexStartAddress() {return this->indexStartAddress;}//for compatibility with vertex array

	HQReturnVal CreateVertexBuffer(const void *initData , hq_uint32 size , 
									bool dynamic , bool isForPointSprites,
									hq_uint32 *pVertexBufferID);
	HQReturnVal CreateIndexBuffer(const void *initData , 
								  hq_uint32 size , bool dynamic  ,
								  HQIndexDataType indexDataType  , 
								  hq_uint32 *pIndexBufferID);

	HQReturnVal CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDesc , 
												hq_uint32 numAttrib ,
												hq_uint32 vertexShaderID , 
												hq_uint32 *pInputLayoutID);

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

	HQReturnVal RemoveVertexBuffer(hq_uint32 vertexBufferID) ;
	HQReturnVal RemoveIndexBuffer(hq_uint32 indexBufferID) ;
	HQReturnVal RemoveVertexInputLayout(hq_uint32 inputLayoutID) ;
	void RemoveAllVertexBuffer() ;
	void RemoveAllIndexBuffer() ;
	void RemoveAllVertexInputLayout() ;

#ifdef DEVICE_LOST_POSSIBLE
	virtual void OnLost();
	virtual void OnReset();
#endif

	/*--------for HQVertexStreamManDelegateGL---------*/
	inline void BindVertexBuffer(GLuint vertexBufferName)
	{
		if(this->currentBoundVBuffer != vertexBufferName)
		{
			glBindBuffer(GL_ARRAY_BUFFER , vertexBufferName);
			this->currentBoundVBuffer = vertexBufferName;
		}
	}
	inline static void EnableVertexAttribArray(GLuint index);
	inline static void DisableVertexAttribArray(GLuint index);
	inline static void SetVertexAttribPointer(const HQVertexAttribInfoGL &vAttribInfo , hq_uint32 stride, const HQBufferGL *vbuffer);
};

#ifdef GLES
/*-------shader + vbo but no map buffer supported------*/
class HQVertexStreamManagerNoMapGL : public HQVertexStreamManagerGL
{
public:
	HQVertexStreamManagerNoMapGL(hq_uint32 maxVertexAttribs , HQLogStream *logFileStream , bool flushLog)
		:HQVertexStreamManagerGL("GL Vertex Stream Manager (no buffer mapping) :", maxVertexAttribs , logFileStream , flushLog)
	{
	}

	HQReturnVal MapVertexBuffer(hq_uint32 vertexBufferID , HQMapType mapType , void **ppData) ;
	HQReturnVal UnmapVertexBuffer(hq_uint32 vertexBufferID) ;
	HQReturnVal MapIndexBuffer( HQMapType mapType , void **ppData) ;
	HQReturnVal UnmapIndexBuffer() ;

	HQReturnVal UpdateVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 offset , hq_uint32 size , const void * pData);
	HQReturnVal UpdateIndexBuffer(hq_uint32 offset , hq_uint32 size , const void * pData);
	HQReturnVal UpdateIndexBuffer(hq_uint32 indexBufferID, hq_uint32 offset , hq_uint32 size , const void * pData);

protected:
	HQVertexStreamManagerNoMapGL(const char *logPrefix, hq_uint32 maxVertexAttribs , HQLogStream *logFileStream , bool flushLog)
		:HQVertexStreamManagerGL(logPrefix, maxVertexAttribs , logFileStream , flushLog)
	{
	}
};
#endif

#endif
