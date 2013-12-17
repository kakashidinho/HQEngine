/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D9PCH.h"
#include "HQStateManagerD3D9.h"


//**********************************
//helper functions
//**********************************
namespace helper{
	void GetD3DDepthMode(HQDepthMode mode , HQDepthStencilStateD3D9 *dsState)
	{
		switch (mode)
		{
		case HQ_DEPTH_FULL:
			dsState->depthEnable = D3DZB_TRUE;
			dsState->depthWriteEnable = TRUE;
			dsState->depthFunc = D3DCMP_LESSEQUAL;
			break;
		case HQ_DEPTH_READONLY :
			dsState->depthEnable = D3DZB_TRUE;
			dsState->depthWriteEnable = FALSE;
			dsState->depthFunc = D3DCMP_LESSEQUAL;
			break;
		case HQ_DEPTH_WRITEONLY:
			dsState->depthEnable = D3DZB_TRUE;
			dsState->depthWriteEnable = TRUE;
			dsState->depthFunc = D3DCMP_ALWAYS;
			break;
		case HQ_DEPTH_NONE:
			dsState->depthEnable = D3DZB_FALSE;
			dsState->depthWriteEnable = TRUE;
			dsState->depthFunc = D3DCMP_LESSEQUAL;
			break;
		}
	}
	inline D3DSTENCILOP GetD3DStencilOp(HQStencilOp op)
	{
		switch(op)
		{
		case HQ_SOP_KEEP://giữ giá trị trên stencil buffer
			return D3DSTENCILOP_KEEP;
		case HQ_SOP_ZERO://set giá trị trên stencil buffer thành 0
			return D3DSTENCILOP_ZERO;
		case HQ_SOP_REPLACE://thay giá trị trên buffer thành giá rị tham khảo
			return D3DSTENCILOP_REPLACE;
		case HQ_SOP_INCR://tăng giá trị trên buffer lên 1 , nếu giá trị thành vượt giá trị lớn nhất có thể => giá trị trên buffer thành giá trị lớn nhất
			return D3DSTENCILOP_INCRSAT;
		case HQ_SOP_DECR://giảm giá trị trên buffer xuống 1 , nếu giá trị thành nhỏ hơn giá trị nhỏ nhất có thể => giá trị trên buffer thành giá trị nhỏ nhất
			return D3DSTENCILOP_DECRSAT;
		case HQ_SOP_INCR_WRAP://tăng giá trị trên buffer lên 1 , nếu giá trị thành vượt giá trị lớn nhất có thể => wrap giá trị
			return D3DSTENCILOP_INCR;
		case HQ_SOP_DECR_WRAP://giảm giá trị trên buffer xuống 1 , nếu giá trị thành nhỏ hơn giá trị nhỏ nhất có thể => wrap giá trị
			return D3DSTENCILOP_DECR;
		case HQ_SOP_INVERT://đảo các bit của giá trị
			return D3DSTENCILOP_INVERT;
		}
		return D3DSTENCILOP_FORCE_DWORD;
	}
	inline D3DCMPFUNC GetD3DStencilFunc(HQStencilFunc func)
	{
		switch(func)
		{
		case HQ_SF_NEVER://stencil test luôn fail
			return D3DCMP_NEVER;
		case HQ_SF_LESS://stencil pass khi op1 < op2 (tức là (ref value & readMask) < (buffer value & readMask) . ref value là giá trị tham khảo, buffer value là giá trị đang có trên buffer) 
			return D3DCMP_LESS;
		case HQ_SF_EQUAL://pass khi op1 = op2
			return D3DCMP_EQUAL;
		case HQ_SF_LESS_EQUAL://pass khi op1 <= op2
			return D3DCMP_LESSEQUAL;
		case HQ_SF_GREATER://pass khi op 1 > op2
			return D3DCMP_GREATER;
		case HQ_SF_NOT_EQUAL://pass khi op1 != op2
			return D3DCMP_NOTEQUAL;
		case HQ_SF_GREATER_EQUAL://pass khi op1 >= op2
			return D3DCMP_GREATEREQUAL;
		case HQ_SF_ALWAYS:// luôn luôn pass
			return D3DCMP_ALWAYS;
		}
		return D3DCMP_FORCE_DWORD;
	}

	
	D3DBLEND GetD3DBlendFactor(HQBlendFactor factor)
	{
		switch(factor)
		{
		case HQ_BF_ONE :
			return D3DBLEND_ONE;
			break;
		case HQ_BF_ZERO :
			return D3DBLEND_ZERO;
			break;
		case HQ_BF_SRC_COLOR :
			return D3DBLEND_SRCCOLOR;
			break;
		case HQ_BF_ONE_MINUS_SRC_COLOR:
			return D3DBLEND_INVSRCCOLOR;
			break;
		case HQ_BF_SRC_ALPHA:
			return D3DBLEND_SRCALPHA;
			break;
		case HQ_BF_ONE_MINUS_SRC_ALPHA:
			return D3DBLEND_INVSRCALPHA;
			break;
		}

		return D3DBLEND_FORCE_DWORD;
	}

	D3DBLENDOP GetD3DBlendOp(HQBlendOp op)
	{
		switch (op)
		{
		case HQ_BO_ADD :
			return D3DBLENDOP_ADD;
			break;
		case HQ_BO_SUBTRACT :
			return D3DBLENDOP_SUBTRACT;
			break;
		case HQ_BO_REVSUBTRACT  :
			return D3DBLENDOP_REVSUBTRACT;
			break;
		}
		return D3DBLENDOP_FORCE_DWORD;
	}

	
	inline DWORD GetD3DTAddressMode(HQTexAddressMode mode)
	{
		switch(mode)
		{
		case HQ_TAM_WRAP:
			return D3DTADDRESS_WRAP;
		case HQ_TAM_CLAMP:
			return D3DTADDRESS_CLAMP;
		case HQ_TAM_BORDER:
			return D3DTADDRESS_BORDER;
		case HQ_TAM_MIRROR:
			return D3DTADDRESS_MIRROR;
		}
		return 0xffffffff;
	}
	inline void GetD3DFilter(HQFilterMode mode , HQSamplerStateD3D9 *state)
	{
		switch(mode)
		{
		case HQ_FM_MIN_MAG_POINT:
			state->minFilter = D3DTEXF_POINT;
			state->magFilter = D3DTEXF_POINT;
			state->mipFilter = D3DTEXF_NONE;
			break;
		case HQ_FM_MIN_POINT_MAG_LINEAR:
			state->minFilter = D3DTEXF_POINT;
			state->magFilter = D3DTEXF_LINEAR;
			state->mipFilter = D3DTEXF_NONE;
			break;
		case HQ_FM_MIN_LINEAR_MAG_POINT:
			state->minFilter = D3DTEXF_LINEAR;
			state->magFilter = D3DTEXF_POINT;
			state->mipFilter = D3DTEXF_NONE;
			break;
		case HQ_FM_MIN_MAG_LINEAR:
			state->minFilter = D3DTEXF_LINEAR;
			state->magFilter = D3DTEXF_LINEAR;
			state->mipFilter = D3DTEXF_NONE;
			break;
		case HQ_FM_MIN_MAG_ANISOTROPIC:
			state->minFilter = D3DTEXF_ANISOTROPIC;
			state->magFilter = D3DTEXF_ANISOTROPIC;
			state->mipFilter = D3DTEXF_NONE;
			break;
		case HQ_FM_MIN_MAG_MIP_POINT:
			state->minFilter = D3DTEXF_POINT;
			state->magFilter = D3DTEXF_POINT;
			state->mipFilter = D3DTEXF_POINT;
			break;
		case HQ_FM_MIN_MAG_POINT_MIP_LINEAR:
			state->minFilter = D3DTEXF_POINT;
			state->magFilter = D3DTEXF_POINT;
			state->mipFilter = D3DTEXF_LINEAR;
			break;
		case HQ_FM_MIN_POINT_MAG_LINEAR_MIP_POINT:
			state->minFilter = D3DTEXF_POINT;
			state->magFilter = D3DTEXF_LINEAR;
			state->mipFilter = D3DTEXF_POINT;
			break;
		case HQ_FM_MIN_POINT_MAG_MIP_LINEAR:
			state->minFilter = D3DTEXF_POINT;
			state->magFilter = D3DTEXF_LINEAR;
			state->mipFilter = D3DTEXF_LINEAR;
			break;
		case HQ_FM_MIN_LINEAR_MAG_MIP_POINT:
			state->minFilter = D3DTEXF_LINEAR;
			state->magFilter = D3DTEXF_POINT;
			state->mipFilter = D3DTEXF_POINT;
			break;
		case HQ_FM_MIN_LINEAR_MAG_POINT_MIP_LINEAR:
			state->minFilter = D3DTEXF_LINEAR;
			state->magFilter = D3DTEXF_POINT;
			state->mipFilter = D3DTEXF_LINEAR;
			break;
		case HQ_FM_MIN_MAG_LINEAR_MIP_POINT:
			state->minFilter = D3DTEXF_LINEAR;
			state->magFilter = D3DTEXF_LINEAR;
			state->mipFilter = D3DTEXF_POINT;
			break;
		case HQ_FM_MIN_MAG_MIP_LINEAR:
			state->minFilter = D3DTEXF_LINEAR;
			state->magFilter = D3DTEXF_LINEAR;
			state->mipFilter = D3DTEXF_LINEAR;
			break;
		case HQ_FM_MIN_MAG_MIP_ANISOTROPIC:
			state->minFilter = D3DTEXF_ANISOTROPIC;
			state->magFilter = D3DTEXF_ANISOTROPIC;
			state->mipFilter = D3DTEXF_LINEAR;
			break;
		}
	}
};
/*------------------------------------------*/
HQDepthStencilStateD3D9::HQDepthStencilStateD3D9()
{
	this->depthEnable = D3DZB_FALSE;
	this->depthFunc = D3DCMP_LESSEQUAL;
	this->stencilEnable = FALSE;
}

void HQDepthStencilStateD3D9::OnResetDevice(LPDIRECT3DDEVICE9 pD3DDevice, bool retainCurrentStates)
{
	LPDIRECT3DSTATEBLOCK9 pD3DCurrentStates = NULL;
	if (retainCurrentStates)
		pD3DDevice->CreateStateBlock(D3DSBT_ALL , &pD3DCurrentStates);

	pD3DDevice->BeginStateBlock();

	pD3DDevice->SetRenderState(D3DRS_ZENABLE , this->depthEnable);
	pD3DDevice->SetRenderState(D3DRS_ZFUNC , this->depthFunc);
	pD3DDevice->SetRenderState(D3DRS_ZWRITEENABLE , this->depthWriteEnable);

	pD3DDevice->SetRenderState(D3DRS_STENCILENABLE , this->stencilEnable);
	
	pD3DDevice->SetRenderState(D3DRS_STENCILMASK , this->sreadMask);
	pD3DDevice->SetRenderState(D3DRS_STENCILWRITEMASK , this->swriteMask);
	pD3DDevice->SetRenderState(D3DRS_STENCILREF , this->sref);
	pD3DDevice->SetRenderState(D3DRS_STENCILFAIL , this->scw.stencilFailOp);
	pD3DDevice->SetRenderState(D3DRS_STENCILZFAIL , this->scw.depthFailOp);
	pD3DDevice->SetRenderState(D3DRS_STENCILPASS , this->scw.passOp);
	pD3DDevice->SetRenderState(D3DRS_STENCILFUNC , this->scw.stencilFunc);
	//enable twoside stencil or not
	pD3DDevice->SetRenderState(D3DRS_TWOSIDEDSTENCILMODE , this->stwoSide);

	pD3DDevice->SetRenderState(D3DRS_CCW_STENCILFAIL , this->sccw.stencilFailOp);
	pD3DDevice->SetRenderState(D3DRS_CCW_STENCILZFAIL , this->sccw.depthFailOp);
	pD3DDevice->SetRenderState(D3DRS_CCW_STENCILPASS , this->sccw.passOp);
	pD3DDevice->SetRenderState(D3DRS_CCW_STENCILFUNC , this->sccw.stencilFunc);

	pD3DDevice->EndStateBlock(&this->pD3DStateBlock);
	
	if (retainCurrentStates)
		pD3DCurrentStates->Apply();
	SafeRelease(pD3DCurrentStates);
}

/*------------------------------------------*/
HQBlendStateD3D9::HQBlendStateD3D9()
{
}

void HQBlendStateD3D9::OnResetDevice(LPDIRECT3DDEVICE9 pD3DDevice, bool retainCurrentStates)
{
	LPDIRECT3DSTATEBLOCK9 pD3DCurrentStates = NULL;
	if (retainCurrentStates)
		pD3DDevice->CreateStateBlock(D3DSBT_ALL , &pD3DCurrentStates);

	pD3DDevice->BeginStateBlock();

	pD3DDevice->SetRenderState(D3DRS_BLENDOP , this->operation);
	pD3DDevice->SetRenderState(D3DRS_BLENDOPALPHA , this->alphaOperation);

	pD3DDevice->SetRenderState(D3DRS_SRCBLEND , this->srcFactor);
	pD3DDevice->SetRenderState(D3DRS_DESTBLEND , this->destFactor);
	pD3DDevice->SetRenderState(D3DRS_SRCBLENDALPHA , this->srcAlphaFactor);
	pD3DDevice->SetRenderState(D3DRS_DESTBLENDALPHA , this->destAlphaFactor);
	
	pD3DDevice->SetRenderState(D3DRS_SEPARATEALPHABLENDENABLE , this->extState);

	pD3DDevice->EndStateBlock(&this->pD3DStateBlock);

	if (retainCurrentStates)
		pD3DCurrentStates->Apply();
	SafeRelease(pD3DCurrentStates);
}
/*------------------------------------------*/
HQSamplerStateD3D9::HQSamplerStateD3D9()
{
	this->addressU = this->addressV = D3DTADDRESS_WRAP;
	this->minFilter = D3DTEXF_LINEAR;
	this->magFilter = D3DTEXF_LINEAR;
	this->mipFilter = D3DTEXF_LINEAR;
	this->maxAnisotropy = 1;
	this->borderColor = 0x00000000;
}

/*------------------------------------------*/

HQStateManagerD3D9::HQStateManagerD3D9(LPDIRECT3DDEVICE9 pD3DDevice,
									DWORD maxVertexShaderSamplers , 
									DWORD maxPixelShaderSamplers ,
									hq_uint32 maxAFLevel ,
									HQLogStream *logFileStream , bool flushLog)
:HQLoggableObject(logFileStream , "D3D9 Render State Manager :" , flushLog)
{
	this->pD3DDevice = pD3DDevice;
	
	this->maxVertexShaderSamplers = maxVertexShaderSamplers;
	this->maxPixelShaderSamplers = maxPixelShaderSamplers;
	this->maxAFLevel = maxAFLevel;
	
	this->sState[0] = new HQSharedPtr<HQSamplerStateD3D9> [maxPixelShaderSamplers];
	if (maxVertexShaderSamplers > 0)
		this->sState[1] = new HQSharedPtr<HQSamplerStateD3D9> [maxVertexShaderSamplers];
	else
		this->sState[1] = NULL;
	/*--------default states---------*/
	this->currentFillMode = HQ_FILL_SOLID;

	this->currentCullMode = HQ_CULL_NONE;
	
	this->CreateDepthStencilState(HQDepthStencilStateDesc() , NULL);
	//create default blend state - alpha blend is disabled
	this->bStates.AddItem(new HQBlendStateD3D9() , NULL);
	this->CreateSamplerState(HQSamplerStateDesc() , NULL);

	this->dsState = this->dsStates.GetItemPointerNonCheck(0);//default depth stencil state
	this->bState = this->bStates.GetItemPointerNonCheck(0);//default blend state
	
	for (hq_uint32 i = 0 ;i < this->maxPixelShaderSamplers ; ++i)
	{
		this->sState[0][i] = this->sStates.GetItemPointerNonCheck(0);//default sampler state
	}

	for (hq_uint32 i = 0 ;i < this->maxVertexShaderSamplers ; ++i)
	{
		this->sState[1][i] = this->sStates.GetItemPointerNonCheck(0);//default sampler state
	}

	this->OnResetDeviceResetStates();

	Log("Init done!");
}
HQStateManagerD3D9::~HQStateManagerD3D9()
{
	SafeDeleteArray(this->sState[0]);
	SafeDeleteArray(this->sState[1]);
	Log ("Released!");
}

void HQStateManagerD3D9::OnResetDeviceResetStates()
{
	/*--------------enable point sprite--------------------------*/
	pD3DDevice->SetRenderState(D3DRS_POINTSPRITEENABLE , TRUE);
	pD3DDevice->SetRenderState(D3DRS_POINTSCALEENABLE , TRUE);
	/*--------fill mode-----------------------*/
	if(currentFillMode == HQ_FILL_WIREFRAME)
	{
		pD3DDevice->SetRenderState(D3DRS_FILLMODE,D3DFILL_WIREFRAME);
	}
	else
	{
		pD3DDevice->SetRenderState(D3DRS_FILLMODE,D3DFILL_SOLID);
	}
	/*--------cull mode-----------------------*/
	switch(currentCullMode)
	{
	case HQ_CULL_CW:
		pD3DDevice->SetRenderState(D3DRS_CULLMODE,D3DCULL_CW);
		break;
	case HQ_CULL_CCW:
		pD3DDevice->SetRenderState(D3DRS_CULLMODE,D3DCULL_CCW);
		break;
	case HQ_CULL_NONE:
		pD3DDevice->SetRenderState(D3DRS_CULLMODE,D3DCULL_NONE);
		break;
	}
	/*-------depth stencil state--------------*/
	this->dsState->pD3DStateBlock->Apply();

	/*-------blend state----------------------*/
	if (this->bState != this->bStates.GetItemPointerNonCheck(0))
	{
		pD3DDevice->SetRenderState(D3DRS_ALPHABLENDENABLE , TRUE);
		this->bState->pD3DStateBlock->Apply();
	}
		
	

	/*-------sampler state---------------------------*/
	for (hq_uint32 samplerIndex = 0 ;samplerIndex < this->maxPixelShaderSamplers ; ++samplerIndex)
	{
		HQSharedPtr<HQSamplerStateD3D9> samplerState = this->sState[0][samplerIndex];
		pD3DDevice->SetSamplerState(samplerIndex , D3DSAMP_ADDRESSU , samplerState->addressU);
		pD3DDevice->SetSamplerState(samplerIndex , D3DSAMP_ADDRESSV , samplerState->addressV);
		pD3DDevice->SetSamplerState(samplerIndex , D3DSAMP_BORDERCOLOR , samplerState->borderColor);
		pD3DDevice->SetSamplerState(samplerIndex,D3DSAMP_MAGFILTER, samplerState->magFilter);
		pD3DDevice->SetSamplerState(samplerIndex,D3DSAMP_MINFILTER, samplerState->minFilter);
		pD3DDevice->SetSamplerState(samplerIndex,D3DSAMP_MIPFILTER, samplerState->mipFilter);
		pD3DDevice->SetSamplerState(samplerIndex,D3DSAMP_MAXANISOTROPY,samplerState->maxAnisotropy);
	}

	for (hq_uint32 i = 0 ;i < this->maxVertexShaderSamplers ; ++i)
	{
		HQSharedPtr<HQSamplerStateD3D9> samplerState = this->sState[1][i];
		UINT samplerIndex = D3DVERTEXTEXTURESAMPLER0 + i;
		pD3DDevice->SetSamplerState(samplerIndex , D3DSAMP_ADDRESSU , samplerState->addressU);
		pD3DDevice->SetSamplerState(samplerIndex , D3DSAMP_ADDRESSV , samplerState->addressV);
		pD3DDevice->SetSamplerState(samplerIndex , D3DSAMP_BORDERCOLOR , samplerState->borderColor);
		pD3DDevice->SetSamplerState(samplerIndex,D3DSAMP_MAGFILTER, samplerState->magFilter);
		pD3DDevice->SetSamplerState(samplerIndex,D3DSAMP_MINFILTER, samplerState->minFilter);
		pD3DDevice->SetSamplerState(samplerIndex,D3DSAMP_MIPFILTER, samplerState->mipFilter);
		pD3DDevice->SetSamplerState(samplerIndex,D3DSAMP_MAXANISOTROPY,samplerState->maxAnisotropy);
	}
	/*-------fixed function state------*/
	pD3DDevice->SetRenderState(D3DRS_LIGHTING , FALSE);//disable lighting
}

void HQStateManagerD3D9::OnResetDevice()
{
	/*---------------recreate state block objects----------------*/

	HQItemManager<HQDepthStencilStateD3D9>::Iterator dsIte;
	this->dsStates.GetIterator(dsIte);
	while(!dsIte.IsAtEnd())
	{
		dsIte->OnResetDevice(pD3DDevice , false);
		++dsIte;
	}

	HQItemManager<HQBlendStateD3D9>::Iterator bIte;
	this->bStates.GetIterator(bIte);
	++bIte;//first blend state is default blend state , its state block doesn't need to recreate 
	while(!bIte.IsAtEnd())
	{
		bIte->OnResetDevice(pD3DDevice , false);
		++bIte;
	}

	/*--------------reset states---------*/
	this->OnResetDeviceResetStates();
}

void HQStateManagerD3D9::SetFillMode(HQFillMode fillMode) 
{
	if (this->currentFillMode != fillMode)
	{
		switch (fillMode)
		{
		case HQ_FILL_WIREFRAME:
			pD3DDevice->SetRenderState(D3DRS_FILLMODE,D3DFILL_WIREFRAME);
			break;
		case HQ_FILL_SOLID:
			pD3DDevice->SetRenderState(D3DRS_FILLMODE,D3DFILL_SOLID);
			break;
		}
		this->currentFillMode = fillMode;
	}
}
void HQStateManagerD3D9::SetFaceCulling(HQCullMode cullMode)
{
	if (this->currentCullMode != cullMode)
	{
		switch (cullMode)
		{
		case HQ_CULL_CW:
			pD3DDevice->SetRenderState(D3DRS_CULLMODE,D3DCULL_CW);
			break;
		case HQ_CULL_CCW:
			pD3DDevice->SetRenderState(D3DRS_CULLMODE,D3DCULL_CCW);
			break;
		case HQ_CULL_NONE:
			pD3DDevice->SetRenderState(D3DRS_CULLMODE,D3DCULL_NONE);
			break;
		}
		this->currentCullMode = cullMode;
	}
}

/*-----depth stencil state------------------*/
HQReturnVal HQStateManagerD3D9::CreateDepthStencilState(const HQDepthStencilStateDesc &desc , hq_uint32 *pDSStateID) 
{
	HQDepthStencilStateD3D9 *newState = new HQDepthStencilStateD3D9();
	if (newState == NULL || !this->dsStates.AddItem(newState , pDSStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}
	
	/*-----get depth mode---------*/
	helper::GetD3DDepthMode(desc.depthMode , newState);

	/*-----get stencil mode-------*/
	newState->stwoSide = FALSE;
	newState->stencilEnable = desc.stencilEnable ? TRUE : FALSE;
	newState->sref = desc.refVal;
	newState->sreadMask = desc.readMask;
	newState->swriteMask = desc.writeMask;

	newState->scw.stencilFailOp = newState->sccw.stencilFailOp = helper::GetD3DStencilOp(desc.stencilMode.failOp);
	newState->scw.depthFailOp = newState->sccw.depthFailOp = helper::GetD3DStencilOp(desc.stencilMode.depthFailOp);
	newState->scw.passOp = newState->sccw.passOp = helper::GetD3DStencilOp(desc.stencilMode.passOp);
	newState->scw.stencilFunc = newState->sccw.stencilFunc =helper::GetD3DStencilFunc(desc.stencilMode.compareFunc);
	
	newState->OnResetDevice(pD3DDevice , true);

	return HQ_OK;
}
HQReturnVal HQStateManagerD3D9::CreateDepthStencilStateTwoSide(const HQDepthStencilStateTwoSideDesc &desc , hq_uint32 *pDSStateID) 
{
	HQDepthStencilStateD3D9 *newState = new HQDepthStencilStateD3D9();
	if (newState == NULL || !this->dsStates.AddItem(newState , pDSStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}
	
	/*-----get depth mode---------*/
	helper::GetD3DDepthMode(desc.depthMode , newState);

	/*-----get stencil mode-------*/
	newState->stwoSide = TRUE;
	newState->stencilEnable = desc.stencilEnable?TRUE : FALSE;
	newState->sref = desc.refVal;
	newState->sreadMask = desc.readMask;
	newState->swriteMask = desc.writeMask;
	//front
	newState->scw.stencilFailOp = helper::GetD3DStencilOp(desc.cwFaceMode.failOp);
	newState->scw.depthFailOp = helper::GetD3DStencilOp(desc.cwFaceMode.depthFailOp);
	newState->scw.passOp = helper::GetD3DStencilOp(desc.cwFaceMode.passOp);
	newState->scw.stencilFunc = helper::GetD3DStencilFunc(desc.cwFaceMode.compareFunc);
	//back
	newState->sccw.stencilFailOp = helper::GetD3DStencilOp(desc.ccwFaceMode.failOp);
	newState->sccw.depthFailOp = helper::GetD3DStencilOp(desc.ccwFaceMode.depthFailOp);
	newState->sccw.passOp = helper::GetD3DStencilOp(desc.ccwFaceMode.passOp);
	newState->sccw.stencilFunc = helper::GetD3DStencilFunc(desc.ccwFaceMode.compareFunc);
	
	newState->OnResetDevice(pD3DDevice , true);
	return HQ_OK;
}
HQReturnVal HQStateManagerD3D9::ActiveDepthStencilState(hq_uint32 depthStencilStateID) 
{
	HQSharedPtr<HQDepthStencilStateD3D9> state = this->dsStates.GetItemPointer(depthStencilStateID);
#if defined DEBUG || defined _DEBUG
	if (state == NULL)
		return HQ_FAILED_INVALID_ID;
#endif
	
	if (state != this->dsState)
	{
		state->pD3DStateBlock->Apply();
		this->dsState = state;
	}

	return HQ_OK;
}
HQReturnVal HQStateManagerD3D9::RemoveDepthStencilState(hq_uint32 depthStencilStateID) 
{
	if (depthStencilStateID == 0)
		return HQ_FAILED;
	return (HQReturnVal) this->dsStates.Remove(depthStencilStateID);
}

/*------blend state-------------------------*/
HQReturnVal HQStateManagerD3D9::CreateBlendState( const HQBlendStateDesc &blendStateDesc , hq_uint32 *pBStateID) 
{
	HQBlendStateD3D9 *newState = new HQBlendStateD3D9();
	if (newState == NULL || !this->bStates.AddItem(newState , pBStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}
	
	newState->extState = FALSE;
	newState->operation = newState->alphaOperation = D3DBLENDOP_ADD;
	newState->srcFactor = newState->srcAlphaFactor = helper::GetD3DBlendFactor(blendStateDesc.srcFactor);
	newState->destFactor = newState->destAlphaFactor = helper::GetD3DBlendFactor(blendStateDesc.destFactor);
	
	newState->OnResetDevice(pD3DDevice , true);

	return HQ_OK;
}
HQReturnVal HQStateManagerD3D9::CreateBlendStateEx( const HQBlendStateExDesc &blendStateDesc , hq_uint32 *pBStateID) 
{
	HQBlendStateD3D9 *newState = new HQBlendStateD3D9();
	if (newState == NULL || !this->bStates.AddItem(newState , pBStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}
	newState->extState = TRUE;

	newState->operation = helper::GetD3DBlendOp(blendStateDesc.blendOp);
	newState->srcFactor = helper::GetD3DBlendFactor(blendStateDesc.srcFactor);
	newState->destFactor = helper::GetD3DBlendFactor(blendStateDesc.destFactor);

	newState->alphaOperation = helper::GetD3DBlendOp(blendStateDesc.alphaBlendOp);
	newState->srcAlphaFactor = helper::GetD3DBlendFactor(blendStateDesc.srcAlphaFactor);
	newState->destAlphaFactor = helper::GetD3DBlendFactor(blendStateDesc.destAlphaFactor);
	
	newState->OnResetDevice(pD3DDevice , true);

	return HQ_OK;
}
HQReturnVal HQStateManagerD3D9::ActiveBlendState(hq_uint32 blendStateID)
{
	HQSharedPtr<HQBlendStateD3D9> state = this->bStates.GetItemPointer(blendStateID);
#if defined DEBUG || defined _DEBUG
	if (state == NULL)
		return HQ_FAILED_INVALID_ID;
#endif
	if (state == this->bState)//no change
		return HQ_OK;
	if (blendStateID == 0 )
	{
		pD3DDevice->SetRenderState(D3DRS_ALPHABLENDENABLE , FALSE);
	}
	else
	{
		if (this->bState == this->bStates.GetItemPointerNonCheck(0))//current state is default blend state
		{
			pD3DDevice->SetRenderState(D3DRS_ALPHABLENDENABLE , TRUE);
		}
		
		state->pD3DStateBlock->Apply();

	}
	this->bState = state;
	return HQ_OK;
}
HQReturnVal HQStateManagerD3D9::RemoveBlendState(hq_uint32 blendStateID)  
{
	if (blendStateID == 0)
		return HQ_FAILED;
	return (HQReturnVal) this->bStates.Remove(blendStateID);
}

/*------sampler state-------------------------*/
HQReturnVal HQStateManagerD3D9::CreateSamplerState(const HQSamplerStateDesc &desc , hq_uint32 *pSStateID)
{
	HQSamplerStateD3D9 *newState = new HQSamplerStateD3D9();
	if (newState == NULL || !this->sStates.AddItem(newState , pSStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}

	newState->addressU = helper::GetD3DTAddressMode(desc.addressU);
	newState->addressV = helper::GetD3DTAddressMode(desc.addressV);
	newState->borderColor = D3DCOLOR_COLORVALUE(desc.borderColor.r , desc.borderColor.g , desc.borderColor.b , desc.borderColor.a) ;

	helper::GetD3DFilter(desc.filterMode , newState);
	
	if (desc.maxAnisotropy < 1)
		newState->maxAnisotropy = 1;
	else if (desc.maxAnisotropy > this->maxAFLevel)
		newState->maxAnisotropy = this->maxAFLevel;
	else
		newState->maxAnisotropy = desc.maxAnisotropy;

	return HQ_OK;
}

HQReturnVal HQStateManagerD3D9::SetSamplerState(hq_uint32 index , hq_uint32 samplerStateID)
{
	HQSharedPtr<HQSamplerStateD3D9> pState = this->sStates.GetItemPointer(samplerStateID);
#if defined DEBUG || defined _DEBUG
	if (pState == NULL)
		return HQ_FAILED_INVALID_ID;
#endif

	hq_uint32 samplerIndex = index & 0x0fffffff;
	hq_uint32 shaderStage = index & 0xf0000000;
	HQSharedPtr<HQSamplerStateD3D9> *ppCurrentState ;
	HQSamplerStateD3D9 *pCurrentStateRaw;//raw pointer

	switch (shaderStage)
	{
		case HQ_VERTEX_SHADER:
#if defined DEBUG || defined _DEBUG
		if (samplerIndex >= this->maxVertexShaderSamplers)
		{
			Log("SetSamplerState() Error : sampler slot is out of range!");
			return HQ_FAILED;
		}
#endif
		ppCurrentState = &this->sState[1][samplerIndex];
		samplerIndex += D3DVERTEXTEXTURESAMPLER0;
		break;
	case HQ_PIXEL_SHADER:
#if defined DEBUG || defined _DEBUG
		if (samplerIndex >= this->maxPixelShaderSamplers)
		{
			Log("SetSamplerState() Error : sampler slot is out of range!");
			return HQ_FAILED;
		}
#endif
		ppCurrentState = &this->sState[0][samplerIndex];
		break;
	default:
#if defined _DEBUG || defined DEBUG
		Log("Error : {index} parameter passing to SetSamplerState() method didn't bitwise OR with HQ_VERTEX_SHADER/HQ_PIXEL_SHADER!");
#endif
		return HQ_FAILED;
	}

	if (*ppCurrentState != pState)
	{
		pCurrentStateRaw = ppCurrentState->GetRawPointer();

		if (pCurrentStateRaw->addressU != pState->addressU)
			pD3DDevice->SetSamplerState(samplerIndex , D3DSAMP_ADDRESSU , pState->addressU);
		if (pCurrentStateRaw->addressV != pState->addressV)
			pD3DDevice->SetSamplerState(samplerIndex , D3DSAMP_ADDRESSV , pState->addressV);
		if (pCurrentStateRaw->borderColor != pState->borderColor)
			pD3DDevice->SetSamplerState(samplerIndex , D3DSAMP_BORDERCOLOR , pState->borderColor);
		if (pCurrentStateRaw->magFilter != pState->magFilter)
			pD3DDevice->SetSamplerState(samplerIndex,D3DSAMP_MAGFILTER, pState->magFilter);
		if (pCurrentStateRaw->minFilter != pState->minFilter)
			pD3DDevice->SetSamplerState(samplerIndex,D3DSAMP_MINFILTER, pState->minFilter);
		if (pCurrentStateRaw->mipFilter != pState->mipFilter)
			pD3DDevice->SetSamplerState(samplerIndex,D3DSAMP_MIPFILTER, pState->mipFilter);
		if (pCurrentStateRaw->maxAnisotropy != pState->maxAnisotropy)
			pD3DDevice->SetSamplerState(samplerIndex,D3DSAMP_MAXANISOTROPY,pState->maxAnisotropy);
		

		*ppCurrentState = pState;
	}


	return HQ_OK;
}
HQReturnVal HQStateManagerD3D9::RemoveSamplerState(hq_uint32 samplerStateID) 
{
	if (samplerStateID == 0)
		return HQ_FAILED;
	return (HQReturnVal) this->sStates.Remove(samplerStateID);
}

void HQStateManagerD3D9::OnLostDevice()
{
	/*---------------release state block objects----------------*/

	HQItemManager<HQDepthStencilStateD3D9>::Iterator dsIte;
	this->dsStates.GetIterator(dsIte);
	while(!dsIte.IsAtEnd())
	{
		dsIte->OnLostDevice();
		++dsIte;
	}

	HQItemManager<HQBlendStateD3D9>::Iterator bIte;
	this->bStates.GetIterator(bIte);
	++bIte;//first blend state is default blend state , it has no state block object
	while(!bIte.IsAtEnd())
	{
		bIte->OnLostDevice();
		++bIte;
	}
}
