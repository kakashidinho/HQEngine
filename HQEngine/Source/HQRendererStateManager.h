/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _HQ_RENDER_STATE_MAN_H_
#define _HQ_RENDER_STATE_MAN_H_

#include "HQRendererCoreType.h"
#include "HQReturnVal.h"

class HQRendererStateManager 
{
protected:
	virtual ~HQRendererStateManager(){};
public:
	HQRendererStateManager() {}
	
	///default is HQ_FILL_SOLID
	virtual void SetFillMode(HQFillMode fillMode) = 0;
	///default is HQ_CULL_NONE
	virtual void SetFaceCulling(HQCullMode cullMode) = 0;

	/*-----depth stencil state------------------*/
	virtual HQReturnVal CreateDepthStencilState(const HQDepthStencilStateDesc &desc , hq_uint32 *pDSStateID) = 0;
	///
	///can cause unexpected behaviors if device not support two sided stencil
	///
	virtual HQReturnVal CreateDepthStencilStateTwoSide(const HQDepthStencilStateTwoSideDesc &desc , hq_uint32 *pDSStateID) = 0;
	///
	///{depthStencilStateID} = 0 => disable depth and stencil operations
	///
	virtual HQReturnVal ActiveDepthStencilState(hq_uint32 depthStencilStateID) = 0;
	///
	///fail if  {depthStencilStateID} = 0
	///
	virtual HQReturnVal RemoveDepthStencilState(hq_uint32 depthStencilStateID) = 0;
	


	/*------blend state-------------------------*/
	virtual HQReturnVal CreateBlendState( const HQBlendStateDesc &blendState , hq_uint32 *pBStateID) = 0;
	///
	///can cause unexpected behaviors if device not support extended blend state. 
	///Note : extended blend state is state that has separate alpha channel blend mode and blend operation other than HQ_BO_ADD
	///
	virtual HQReturnVal CreateBlendStateEx( const HQBlendStateExDesc &blendState , hq_uint32 *pBStateID) = 0;
	///
	///{blendStateID} = 0 => disable blend operation
	///
	virtual HQReturnVal ActiveBlendState(hq_uint32 blendStateID) = 0;
	///
	///fail if {blendStateID} = 0
	///
	virtual HQReturnVal RemoveBlendState(hq_uint32 blendStateID) = 0;



	/*------sampler state-------------------------*/
	virtual HQReturnVal CreateSamplerState(const HQSamplerStateDesc &desc , hq_uint32 *pSStateID) = 0;
	
	///
	///Direct3d : -{index} = {sampler slot} bitwise OR với enum HQShaderType để chỉ  {sampler slot} thuộc shader stage nào. 
	///			-Ví dụ muốn gắn sampler state vào sampler slot 3 của vertex shader , ta truyền tham số {index} = (3 | HQ_VERTEX_SHADER). 
	///			-Số sampler slot mỗi shader stage có tùy vào giá trị method GetMaxShaderStageSamplers() của render device trả về. 
	///			-pixel shader dùng chung sampler slot với fixed function sampler. 
	///OpenGL : -{index} là ID của texture. 
	///truyền {samplerStateID} = 0 nếu muốn set sampler state mặc định.Sampler state mặc định có filter mode là linear
	///
	virtual HQReturnVal SetSamplerState(hq_uint32 index , hq_uint32 samplerStateID) = 0;
	///
	///fail if {samplerStateID} = 0
	///
	virtual HQReturnVal RemoveSamplerState(hq_uint32 samplerStateID) = 0;
};

#endif
