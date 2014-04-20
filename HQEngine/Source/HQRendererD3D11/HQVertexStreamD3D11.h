/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _VERTEX_STREAM_MAN_
#define _VERTEX_STREAM_MAN_

#include "../HQVertexStreamManager.h"
#include "../HQItemManager.h"
#include "HQCommonD3D11.h"
#include "HQShaderD3D11.h"

#define MAX_VERTEX_ATTRIBS 16
#ifndef D3D11_1_UAV_SLOT_COUNT
#	define D3D11_1_UAV_SLOT_COUNT 64
#endif


#ifdef WIN32
#	pragma warning( push )
#	pragma warning( disable : 4250 )//dominance inheritance of HQBufferD3D11
#endif

struct HQVertexBufferD3D11 : public HQGenericBufferD3D11
{
	HQVertexBufferD3D11(bool dynamic, hq_uint32 size)
	: HQGenericBufferD3D11(HQ_VERTEX_BUFFER_D3D11, dynamic, size)
	{
	}
};

struct HQIndexBufferD3D11 : public HQGenericBufferD3D11
{
	HQIndexBufferD3D11(bool dynamic   , hq_uint32 size, HQIndexDataType dataType) 
	: HQGenericBufferD3D11(HQ_INDEX_BUFFER_D3D11, dynamic, size)
	{
		switch (dataType)
		{
		case HQ_IDT_UINT:
			this->d3dDataType = DXGI_FORMAT_R32_UINT;
			break;
		default:
			this->d3dDataType = DXGI_FORMAT_R16_UINT;
		}
	}

	DXGI_FORMAT d3dDataType;
};

//unordered access supported vertex buffer
struct HQVertexBufferUAVD3D11 : public HQVertexBufferD3D11
{
	HQVertexBufferUAVD3D11(hquint32 size)
	: HQVertexBufferD3D11(false, size)
	{
	}
};

//unordered access supported index buffer
struct HQIndexBufferUAVD3D11 : public HQIndexBufferD3D11
{
	HQIndexBufferUAVD3D11(hq_uint32 size, HQIndexDataType dataType)
	: HQIndexBufferD3D11(false, size, dataType)
	{
	}
};

#ifdef WIN32
#	pragma warning( pop )
#endif

struct HQVertexInputLayoutD3D11: public HQVertexLayout, public HQBaseIDObject
{
	HQVertexInputLayoutD3D11() {
		this->pD3DLayout = NULL;
	}
	~HQVertexInputLayoutD3D11() {
		SafeRelease(pD3DLayout);
	}

	ID3D11InputLayout * pD3DLayout;
};

struct HQVertexStreamD3D11
{
	HQVertexStreamD3D11() {stride = 0;}
	HQSharedPtr<HQVertexBufferD3D11> vertexBuffer;
	hq_uint32 stride;
};

class HQVertexStreamManagerD3D11: public HQVertexStreamManager , public HQLoggableObject
{

public:
	HQVertexStreamManagerD3D11(ID3D11Device* pD3DDevice , 
								ID3D11DeviceContext *pD3DContext, 
								HQShaderManagerD3D11 *pShaderMan,
								HQLogStream* logFileStream , bool flushLog);
	~HQVertexStreamManagerD3D11() ;
	

	HQReturnVal CreateVertexBuffer(const void *initData , hq_uint32 size , bool dynamic , bool isForPointSprites ,HQVertexBuffer **pVertexBufferID);
	HQReturnVal CreateIndexBuffer(const void *initData , hq_uint32 size , bool dynamic , HQIndexDataType dataType , HQIndexBuffer **pIndexBufferID);

	HQReturnVal CreateVertexBufferUAV(const void *initData,
		hq_uint32 elementSize,
		hq_uint32 numElements,
		HQVertexBufferUAV **ppVertexBufferOut);

	HQReturnVal CreateIndexBufferUAV(const void *initData,
		hq_uint32 numElements,
		HQIndexDataType indexDataType,
		HQVertexBufferUAV **ppIndexBufferOut);

	HQReturnVal CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDesc , 
												hq_uint32 numAttrib ,
												HQShaderObject* vertexShaderID , 
												HQVertexLayout **pInputLayoutID);

	//convert from raw pointer to shared pointer
	HQSharedPtr<HQVertexBufferD3D11> GetVertexBufferSharedPtr(HQGenericBufferD3D11* buffer)
	{
		return this->vertexBuffers.GetItemPointer(buffer);
	}
	HQSharedPtr<HQIndexBufferD3D11> GetIndexBufferSharedPtr(HQGenericBufferD3D11* buffer)
	{
		return this->indexBuffers.GetItemPointer(buffer);
	}

	HQReturnVal SetVertexBuffer(HQVertexBuffer* vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride ) ;

	HQReturnVal SetIndexBuffer(HQIndexBuffer* indexBufferID );

	HQReturnVal SetVertexInputLayout(HQVertexLayout* inputLayoutID) ;

	HQReturnVal RemoveVertexBuffer(HQVertexBuffer* vertexBufferID);
	HQReturnVal RemoveIndexBuffer(HQIndexBuffer* indexBufferID);
	HQReturnVal RemoveVertexInputLayout(HQVertexLayout* inputLayoutID);
	void RemoveAllVertexBuffer() ;
	void RemoveAllIndexBuffer() ;
	void RemoveAllVertexInputLayout() ;
	
#if HQ_D3D_CLEAR_VP_USE_GS
	void ChangeClearVBuffer(HQColorui color , hq_float32 depth);
#endif
	void BeginClearViewport();
	void EndClearViewport();

private:
	void ConvertToElementDesc(const HQVertexAttribDesc &vAttribDesc, D3D11_INPUT_ELEMENT_DESC &vElement);

	ID3D11Device * pD3DDevice;
	ID3D11DeviceContext *pD3DContext;
	ID3D11Buffer * pCleaVBuffer;//vertex buffer for clearing viewport
	ID3D11InputLayout *pClearInputLayout;//input layout for clearing viewport

	HQShaderManagerD3D11 *pShaderMan;

	HQIDItemManager<HQVertexBufferD3D11> vertexBuffers;
	HQIDItemManager<HQIndexBufferD3D11> indexBuffers;
	HQIDItemManager<HQVertexInputLayoutD3D11> inputLayouts;

	HQSharedPtr<HQIndexBufferD3D11> activeIndexBuffer;
	HQSharedPtr<HQVertexInputLayoutD3D11> activeInputLayout;

	HQVertexStreamD3D11 streams[MAX_VERTEX_ATTRIBS];

};


#endif
