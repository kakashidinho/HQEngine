/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _VERTEX_STREAM_MAN_
#define _VERTEX_STREAM_MAN_

#include "d3d9.h"
#include "../HQVertexStreamManager.h"
#include "../HQLoggableObject.h"
#include "../BaseImpl/HQBaseImplCommon.h"
#include "../HQItemManager.h"


struct HQBufferD3D9
{
	HQBufferD3D9(LPDIRECT3DDEVICE9 pD3DDevice , hq_uint32 size , bool isDynamic)
	{
		this->pD3DDevice = pD3DDevice;
		this->isDynamic = isDynamic;
		this->size = size;
	}

	LPDIRECT3DDEVICE9 pD3DDevice;
	bool isDynamic;
	hq_uint32 size;
};

struct HQVertexBufferD3D9 : public HQBufferD3D9, public HQVertexBuffer, public HQBaseIDObject
{
	HQVertexBufferD3D9(LPDIRECT3DDEVICE9 pD3DDevice , hq_uint32 size , bool isDynamic , bool isForPointSprites) 
		: HQBufferD3D9(pD3DDevice , size , isDynamic)
	{
		this->isForPointSprites = isForPointSprites;
		this->pD3DBuffer = NULL;
		OnResetDevice();
	}
	~HQVertexBufferD3D9() { OnLostDevice() ;}

	virtual hquint32 GetSize() const { return size; }

	virtual HQReturnVal Update(hq_uint32 offset, hq_uint32 size, const void * pData);
	virtual HQReturnVal Unmap();
	virtual HQReturnVal GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size);
	virtual HQReturnVal CopyContent(void * dest);

	//implement HQGraphicsResourceRawRetrievable
	virtual void * GetRawHandle() { return pD3DBuffer; }

	void OnResetDevice();
	void OnLostDevice() {SafeRelease(pD3DBuffer);}
	
	bool isForPointSprites;
	LPDIRECT3DVERTEXBUFFER9 pD3DBuffer;
};


struct HQIndexBufferD3D9 : public HQBufferD3D9, public HQIndexBuffer, public HQBaseIDObject
{
	HQIndexBufferD3D9(LPDIRECT3DDEVICE9 pD3DDevice , hq_uint32 size , 
		bool isDynamic , HQIndexDataType dataType) 
		: HQBufferD3D9(pD3DDevice , size , isDynamic)
	{
		this->dataType = dataType;
		this->pD3DBuffer = NULL;
		OnResetDevice();
	}
	~HQIndexBufferD3D9() {OnLostDevice() ;}

	virtual hquint32 GetSize() const { return size; }

	virtual HQReturnVal Update(hq_uint32 offset, hq_uint32 size, const void * pData);
	virtual HQReturnVal Unmap();
	virtual HQReturnVal GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size);
	virtual HQReturnVal CopyContent(void * dest);

	//implement HQGraphicsResourceRawRetrievable
	virtual void * GetRawHandle() { return pD3DBuffer; }

	void OnResetDevice();
	void OnLostDevice() {SafeRelease(pD3DBuffer);}

	LPDIRECT3DINDEXBUFFER9 pD3DBuffer;
	HQIndexDataType dataType;
};


struct HQVertexInputLayoutD3D9 : public HQVertexLayout, public HQBaseIDObject
{
	HQVertexInputLayoutD3D9(LPDIRECT3DDEVICE9 pD3DDevice , D3DVERTEXELEMENT9 *elements) {
		this->pD3DDevice = pD3DDevice;
		this->elements = elements;
		pD3DDecl = NULL;
		OnResetDevice();
	}
	~HQVertexInputLayoutD3D9() {
		SafeDeleteArray(this->elements);
		OnLostDevice();
	}
	void OnResetDevice();
	void OnLostDevice() {SafeRelease(pD3DDecl);}

	LPDIRECT3DDEVICE9 pD3DDevice;
	LPDIRECT3DVERTEXDECLARATION9 pD3DDecl;
	D3DVERTEXELEMENT9 *elements;
};

struct HQVertexStreamD3D9
{
	HQVertexStreamD3D9() {stride = 0;}
	HQSharedPtr<HQVertexBufferD3D9> vertexBuffer;
	hq_uint32 stride;
};

class HQVertexStreamManagerD3D9: public HQVertexStreamManager , public HQLoggableObject
{
private:
	LPDIRECT3DDEVICE9 pD3DDevice;

	HQIDItemManager<HQVertexBufferD3D9> vertexBuffers;
	HQIDItemManager<HQIndexBufferD3D9> indexBuffers;
	HQIDItemManager<HQVertexInputLayoutD3D9> inputLayouts;

	HQSharedPtr<HQIndexBufferD3D9> activeIndexBuffer;
	HQSharedPtr<HQVertexInputLayoutD3D9> activeInputLayout;

	HQVertexStreamD3D9 streams[16];
	
	void ConvertToVertexElement(const HQVertexAttribDesc &vAttribDesc ,D3DVERTEXELEMENT9 &vElement);

public:
	HQVertexStreamManagerD3D9(LPDIRECT3DDEVICE9 pD3DDevice , HQLogStream *logFileStream , bool flushLog);
	~HQVertexStreamManagerD3D9() ;
	

	HQReturnVal CreateVertexBuffer(const void *initData , hq_uint32 size , bool dynamic , bool isForPointSprites ,HQVertexBuffer** pVertexBufferID);
	HQReturnVal CreateIndexBuffer(const void *initData , hq_uint32 size , bool dynamic , HQIndexDataType dataType , HQIndexBuffer** pIndexBufferID);


	HQReturnVal CreateVertexBufferUAV(const void *initData,
		hq_uint32 elementSize,
		hq_uint32 numElements,
		HQVertexBufferUAV **ppVertexBufferOut)
	{
		return HQ_FAILED;
	}

	HQReturnVal CreateIndexBufferUAV(const void *initData,
		hq_uint32 numElements,
		HQIndexDataType indexDataType,
		HQVertexBufferUAV **ppIndexBufferOut)
	{
		return HQ_FAILED;
	}

	HQReturnVal CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDesc , 
												hq_uint32 numAttrib ,
												HQShaderObject* vertexShaderID , 
												HQVertexLayout **pInputLayoutID);

	HQReturnVal SetVertexBuffer(HQVertexBuffer* vertexBufferID, hq_uint32 streamIndex, hq_uint32 stride);

	HQReturnVal SetIndexBuffer(HQIndexBuffer* indexBufferID);

	HQReturnVal SetVertexInputLayout(HQVertexLayout* inputLayoutID);
	
	HQReturnVal RemoveVertexBuffer(HQVertexBuffer* vertexBufferID);
	HQReturnVal RemoveIndexBuffer(HQIndexBuffer* indexBufferID);
	HQReturnVal RemoveVertexInputLayout(HQVertexLayout* inputLayoutID);
	void RemoveAllVertexBuffer() ;
	void RemoveAllIndexBuffer() ;
	void RemoveAllVertexInputLayout() ;

	void OnLostDevice();
	void OnResetDevice();
};


#endif
