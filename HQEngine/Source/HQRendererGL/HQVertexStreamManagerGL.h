/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _VERTEX_STREAM_MAN_GL_
#define _VERTEX_STREAM_MAN_GL_

#include "glHeaders.h"
#include "../BaseImpl/HQBaseImplCommon.h"
#include "../HQVertexStreamManager.h"
#include "../HQLoggableObject.h"
#include "../HQItemManager.h"
#include "HQCommonVertexStreamManGL.h"

#ifdef WIN32
#	pragma warning( push )
#	pragma warning( disable : 4250 )//dominance inheritance of HQBufferGL
#endif

//vertex buffer
struct HQVertexBufferGL : public HQBufferGL, public HQVertexBuffer, public HQBaseIDObject
{
	HQVertexBufferGL(HQVertexStreamManagerGL *manager, hq_uint32 size, GLenum usage)
	: HQBufferGL(size, usage)
	{
		this->manager = manager;
	}
	~HQVertexBufferGL();

	void OnCreated(const void *initData);

	HQVertexStreamManagerGL *manager;
};

//mappable vertex buffer
struct HQMappableVertexBufferGL : public HQVertexBufferGL {
	HQMappableVertexBufferGL(HQVertexStreamManagerGL *manager, hq_uint32 size, GLenum usage)
	: HQVertexBufferGL(manager, size, usage) {}

	virtual hquint32 GetSize() const { return size; }///mappable size
	virtual HQReturnVal Update(hq_uint32 offset, hq_uint32 size, const void * pData);
	virtual HQReturnVal Unmap();
	virtual HQReturnVal GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size);
};


//unmappable vertex buffer
struct HQUnmappableVertexBufferGL : public HQVertexBufferGL, public HQSysMemBuffer{
	HQUnmappableVertexBufferGL(HQVertexStreamManagerGL *manager, hq_uint32 size, GLenum usage);

	void OnCreated(const void *initData);

	//implement HQSysMemBuffer
	virtual HQReturnVal Update(hq_uint32 offset, hq_uint32 size, const void * pData);
	virtual HQReturnVal Unmap();
	virtual HQReturnVal GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size);
};

//index buffer
struct HQIndexBufferGL :public HQBufferGL, public HQIndexBuffer, public HQBaseIDObject
{
	HQIndexBufferGL(HQVertexStreamManagerGL *manager, hq_uint32 size, GLenum usage, HQIndexDataType dataType)
		: HQBufferGL(size , usage )
	{
		this->manager = manager;

		switch (dataType)
		{
		case HQ_IDT_UINT:
			this->dataType = GL_UNSIGNED_INT;
			break;
		default:
			this->dataType = GL_UNSIGNED_SHORT;
		}
	}

	void OnCreated(const void *initData);

	GLenum dataType;

	HQVertexStreamManagerGL *manager;
};


//mappable index buffer
struct HQMappableIndexBufferGL : public HQIndexBufferGL {
	HQMappableIndexBufferGL(HQVertexStreamManagerGL *manager, hq_uint32 size, GLenum usage, HQIndexDataType dataType)
	: HQIndexBufferGL(manager, size, usage, dataType) {}

	virtual hquint32 GetSize() const { return size; }///mappable size
	virtual HQReturnVal Update(hq_uint32 offset, hq_uint32 size, const void * pData);
	virtual HQReturnVal Unmap();
	virtual HQReturnVal GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size);
};

//unmappable index buffer
struct HQUnmappableIndexBufferGL : public HQIndexBufferGL, public HQSysMemBuffer{
	HQUnmappableIndexBufferGL(HQVertexStreamManagerGL *manager, hq_uint32 size, GLenum usage, HQIndexDataType dataType);

	void OnCreated(const void *initData);

	//implement HQSysMemBuffer
	virtual HQReturnVal Update(hq_uint32 offset, hq_uint32 size, const void * pData);
	virtual HQReturnVal Unmap();
	virtual HQReturnVal GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size);
};

#ifdef WIN32
#	pragma warning( pop )
#endif


/*--------
fully capable vertex manager
shader + vbo
--------*/
class HQVertexStreamManagerGL: public HQVertexStreamManager , public HQLoggableObject
{

public:
	HQVertexStreamManagerGL(hq_uint32 maxVertexAttribs , HQLogStream *logFileStream , bool flushLog);
	~HQVertexStreamManagerGL() ;
	
	virtual void OnDraw() {}//this is called just before drawing
	
	HQSharedPtr<HQIndexBufferGL> GetActiveIndexBuffer() { return activeIndexBuffer; }
	GLuint GetCurrentBoundVBuffer() {return this->currentBoundVBuffer;}
	void InvalidateCurrentBoundVBuffer() {this->currentBoundVBuffer = 0;}
	inline GLenum GetIndexDataType() {return this->indexDataType;}
	inline hq_uint32 GetIndexShiftFactor() {return this->indexShiftFactor;}
	inline void * GetIndexStartAddress() {return this->indexStartAddress;}//for compatibility with vertex array
	inline bool IsVertexAttribActive(hquint32 attribIndex) const {return this->activeInputLayout != NULL? false: ((this->activeInputLayout->flags & (0x1 << attribIndex)) != 0);}

	HQReturnVal CreateVertexBuffer(const void *initData , hq_uint32 size , 
									bool dynamic , bool isForPointSprites,
									HQVertexBuffer **pVertexBufferID);
	HQReturnVal CreateIndexBuffer(const void *initData , 
								  hq_uint32 size , bool dynamic  ,
								  HQIndexDataType indexDataType  , 
								  HQIndexBuffer **pIndexBufferID);

	HQReturnVal CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDesc , 
												hq_uint32 numAttrib ,
												HQShaderObject* vertexShaderID , 
												HQVertexLayout **pInputLayoutID);

	HQReturnVal SetVertexBuffer(HQVertexBuffer* vertexBufferID, hq_uint32 streamIndex, hq_uint32 stride);

	HQReturnVal SetIndexBuffer(HQIndexBuffer* indexBufferID );

	HQReturnVal SetVertexInputLayout(HQVertexLayout* inputLayoutID) ;

	HQReturnVal RemoveVertexBuffer(HQVertexBuffer* vertexBufferID);
	HQReturnVal RemoveIndexBuffer(HQIndexBuffer* indexBufferID);
	HQReturnVal RemoveVertexInputLayout(HQVertexLayout* inputLayoutID);
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
	inline static void SetVertexAttribPointer(const HQVertexAttribInfoGL &vAttribInfo, hq_uint32 stride, const HQBufferGL *vbuffer);
protected:
	HQVertexStreamManagerGL(const char *logPrefix, hq_uint32 maxVertexAttribs, HQLogStream *logFileStream, bool flushLog);

	void ConvertToVertexAttribInfo(const HQVertexAttribDesc &vAttribDesc, HQVertexAttribInfoGL &vAttribInfo);
	virtual bool IsVertexAttribValid(const HQVertexAttribDesc &vAttribDesc) { return true; }////check if vertex attribute' desc is valid

	virtual HQVertexBufferGL * CreateNewVertexBufferObj(hq_uint32 size, GLenum usage);
	virtual HQIndexBufferGL * CreateNewIndexBufferObj(hq_uint32 size, GLenum usage, HQIndexDataType dataType);

	HQIDItemManager<HQVertexBufferGL> vertexBuffers;
	HQIDItemManager<HQIndexBufferGL> indexBuffers;
	HQIDItemManager<HQVertexInputLayoutGL> inputLayouts;
	HQVertexStreamGL *streams;

	GLuint currentBoundVBuffer;
	HQSharedPtr<HQIndexBufferGL> activeIndexBuffer;
	GLenum indexDataType;
	hq_uint32 indexShiftFactor;
	void *indexStartAddress;//for compatibility with vertex array
	HQSharedPtr<HQVertexInputLayoutGL> activeInputLayout;

	hq_uint32 maxVertexAttribs;
	HQVertexAttribInfoNodeGL *vAttribInfoNodeCache;


};

#endif
