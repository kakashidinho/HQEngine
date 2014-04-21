/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _HQ_VERTEX_MAN_
#define _HQ_VERTEX_MAN_
#include "HQVertexAttribute.h"
#include "HQReturnVal.h"
#include "HQRendererCoreType.h"

class HQVertexStreamManager
{
protected :
	virtual ~HQVertexStreamManager() {}
public:
	HQVertexStreamManager(){}
	///
	///{isForPointSprites} - vertex buffer này có phải dùng để render point sprites không 
	///
	virtual HQReturnVal CreateVertexBuffer(const void *initData , hq_uint32 size , 
											bool dynamic , bool isForPointSprites, 
											HQVertexBuffer **pVertexBufferID) = 0;
	///
	///{indexDataType} - kiểu dữ liệu của index
	///
	virtual HQReturnVal CreateIndexBuffer(const void *initData , 
										  hq_uint32 size , bool dynamic  ,
										  HQIndexDataType indexDataType  , 
										  HQIndexBuffer **pIndexBufferID) = 0;

	///
	///create unordered access buffer, supports read and write via shader
	///
	virtual HQReturnVal CreateVertexBufferUAV(const void *initData, 
												hq_uint32 elementSize,
												hq_uint32 numElements,
												HQVertexBufferUAV **ppVertexBufferOut) = 0;

	///
	///create unordered access buffer, supports read and write via shader
	///
	virtual HQReturnVal CreateIndexBufferUAV(const void *initData,
		hq_uint32 numElements,
		HQIndexDataType indexDataType,
		HQVertexBufferUAV **ppIndexBufferOut) = 0;

	///
	///{vertexShaderID} is ignored in D3D9 device. if {vertexShaderID} = NULL, this method will create 
	///input layout for fixed function shader. D3D11 & GL only accepts the following layout: 
	///position (x,y,z); color (r,g,b,a); normal (x,y,z); texcoords (u,v)
	///
	virtual HQReturnVal CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDescs , 
												hq_uint32 numAttrib ,
												HQShaderObject* vertexShaderID , 
												HQVertexLayout **pInputLayoutID) = 0;

	///
	///gắn vertex buffer {vertexBufferID} vào stream slot {streamIndex}. 
	///{stride} - khoảng cách giữa 2 dữ liệu vertex trong vertex buffer. 
	///Lưu ý : {streamIndex} phải nằm trong khoảng từ 0 đến giá trị mà method GetMaxVertexStream() của HQRenderDevice trả về trừ đi 1. 
	///			Direct3D11 : sẽ unset  buffer ra khỏi mọi UAV slots
	///
	virtual HQReturnVal SetVertexBuffer(HQVertexBuffer* vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride) =0;

	///
	///active index buffer {indexBufferID}. 
	///			Lưu ý Direct3D11 : sẽ unset  buffer ra khỏi mọi UAV slots
	///
	virtual HQReturnVal SetIndexBuffer(HQIndexBuffer* indexBufferID ) =0;

	///
	///active vertex input layout. 
	///Khi dùng direct3d 10 & 11 device , input layout phải ánh xạ hoàn toàn được vào các biến thành phần vertex trong vertex shader. 
	///
	virtual HQReturnVal SetVertexInputLayout(HQVertexLayout* inputLayoutID) = 0;
	
	virtual HQReturnVal RemoveVertexBuffer(HQVertexBuffer* vertexBufferID) = 0;
	virtual HQReturnVal RemoveIndexBuffer(HQIndexBuffer* indexBufferID) = 0;
	virtual HQReturnVal RemoveVertexInputLayout(HQVertexLayout* inputLayoutID) = 0;
	virtual void RemoveAllVertexBuffer() = 0;
	virtual void RemoveAllIndexBuffer() = 0;
	virtual void RemoveAllVertexInputLayout() = 0;
};

#endif
