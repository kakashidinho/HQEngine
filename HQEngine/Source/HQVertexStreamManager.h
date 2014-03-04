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
											hq_uint32 *pVertexBufferID) = 0;
	///
	///{indexDataType} - kiểu dữ liệu của index
	///
	virtual HQReturnVal CreateIndexBuffer(const void *initData , 
										  hq_uint32 size , bool dynamic  ,
										  HQIndexDataType indexDataType  , 
										  hq_uint32 *pIndexBufferID) = 0;

	///
	///{vertexShaderID} is ignored in D3D9 device. if {vertexShaderID} = HQ_NOT_USE_VSHADER, this method will create 
	///input layout for fixed function shader. D3D11 & GL only accepts the following layout: 
	///position (x,y,z); color (r,g,b,a); normal (x,y,z); texcoords (u,v)
	///
	virtual HQReturnVal CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDescs , 
												hq_uint32 numAttrib ,
												hq_uint32 vertexShaderID , 
												hq_uint32 *pInputLayoutID) = 0;

	///
	///gắn vertex buffer {vertexBufferID} vào stream slot {streamIndex}. 
	///{stride} - khoảng cách giữa 2 dữ liệu vertex trong vertex buffer. 
	///Lưu ý : {streamIndex} phải nằm trong khoảng từ 0 đến giá trị mà method GetMaxVertexStream() của HQRenderDevice trả về trừ đi 1
	///
	virtual HQReturnVal SetVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 streamIndex , hq_uint32 stride) =0;

	///
	///active index buffer {indexBufferID}
	///
	virtual HQReturnVal SetIndexBuffer(hq_uint32 indexBufferID ) =0;

	///
	///active vertex input layout. 
	///Khi dùng direct3d 10 & 11 device , input layout phải ánh xạ hoàn toàn được vào các biến thành phần vertex trong vertex shader. 
	///
	virtual HQReturnVal SetVertexInputLayout(hq_uint32 inputLayoutID) = 0;
	
	///
	///Lưu ý : không được map vertex / index buffer khi nó đã map , hoặc đang dùng để render
	///không nên map static vertex / index buffer
	///
	virtual HQReturnVal MapVertexBuffer(hq_uint32 vertexBufferID , HQMapType mapType , void **ppData) = 0;
	virtual HQReturnVal UnmapVertexBuffer(hq_uint32 vertexBufferID) = 0;
	///
	///ánh xạ địa chỉ của dữ liệu index buffer đang active (qua method SetIndexBuffer()) vào con trỏ {ppData} trỏ đến
	///
	virtual HQReturnVal MapIndexBuffer(HQMapType mapType , void **ppData) = 0;
	///
	///ngừng ánh xạ địa chỉ của dữ liệu index buffer đang active (qua method SetIndexBuffer())
	///
	virtual HQReturnVal UnmapIndexBuffer() = 0;
	
	///
	///Copy {pData} vào vertex buffer . {offset} & {size} tính theo byte.Nếu {offset} & {size} đều là 0 ,toàn bộ buffer sẽ được update .Lưu ý không nên update trên static buffer
	///
	virtual HQReturnVal UpdateVertexBuffer(hq_uint32 vertexBufferID , hq_uint32 offset , hq_uint32 size , const void * pData)= 0;
	///
	///Copy {pData} vào index buffer đang active. {offset} & {size} tính theo byte.Nếu {offset} & {size} đều là 0 ,toàn bộ buffer sẽ được update .Lưu ý không nên update trên static buffer
	///
	virtual HQReturnVal UpdateIndexBuffer(hq_uint32 offset , hq_uint32 size , const void * pData)= 0;

	///
	///Copy {pData} vào index buffer. {offset} & {size} tính theo byte.Nếu {offset} & {size} đều là 0 ,toàn bộ buffer sẽ được update .Lưu ý không nên update trên static buffer
	///
	virtual HQReturnVal UpdateIndexBuffer(hq_uint32 indexBufferID, hq_uint32 offset , hq_uint32 size , const void * pData)= 0;

	virtual HQReturnVal RemoveVertexBuffer(hq_uint32 vertexBufferID) = 0;
	virtual HQReturnVal RemoveIndexBuffer(hq_uint32 indexBufferID) = 0;
	virtual HQReturnVal RemoveVertexInputLayout(hq_uint32 inputLayoutID) = 0;
	virtual void RemoveAllVertexBuffer() = 0;
	virtual void RemoveAllIndexBuffer() = 0;
	virtual void RemoveAllVertexInputLayout() = 0;
};

#endif
