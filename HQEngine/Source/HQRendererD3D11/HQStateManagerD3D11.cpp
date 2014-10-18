/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D11PCH.h"
#include "HQStateManagerD3D11.h"

#define NON_DEPTH_NON_STENCIL_CLEAR 0
#define HQ_DEPTH_NON_STENCIL_CLEAR 1
#define NON_DEPTH_STENCIL_CLEAR 2
#define HQ_DEPTH_STENCIL_CLEAR 3

const FLOAT g_blendFactor[4] = {0.0f , 0.0f , 0.0f , 0.0f}; 
ID3D11SamplerState * g_nullSampler = NULL;

//**********************************
//helper functions
//**********************************
namespace helper{
	void GetD3DDepthMode(HQDepthMode mode , D3D11_DEPTH_STENCIL_DESC &desc)
	{
		switch (mode)
		{
		case HQ_DEPTH_FULL:
			desc.DepthEnable = TRUE;
			desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
			desc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
			break;
		case HQ_DEPTH_READONLY :
			desc.DepthEnable = TRUE;
			desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
			desc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
			break;
		case HQ_DEPTH_WRITEONLY:
			desc.DepthEnable = TRUE;
			desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
			desc.DepthFunc = D3D11_COMPARISON_ALWAYS;
			break;
		case HQ_DEPTH_NONE:
			desc.DepthEnable = FALSE;
			desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
			desc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
			break;
		case HQ_DEPTH_READONLY_GREATEREQUAL:
			desc.DepthEnable = TRUE;
			desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
			desc.DepthFunc = D3D11_COMPARISON_GREATER_EQUAL;
			break;
		}
	}
	inline D3D11_STENCIL_OP GetD3DStencilOp(HQStencilOp op)
	{
		switch(op)
		{
		case HQ_SOP_KEEP://giữ giá trị trên stencil buffer
			return D3D11_STENCIL_OP_KEEP;
		case HQ_SOP_ZERO://set giá trị trên stencil buffer thành 0
			return D3D11_STENCIL_OP_ZERO;
		case HQ_SOP_REPLACE://thay giá trị trên buffer thành giá rị tham khảo
			return D3D11_STENCIL_OP_REPLACE;
		case HQ_SOP_INCR://tăng giá trị trên buffer lên 1 , nếu giá trị thành vượt giá trị lớn nhất có thể => giá trị trên buffer thành giá trị lớn nhất
			return D3D11_STENCIL_OP_INCR_SAT;
		case HQ_SOP_DECR://giảm giá trị trên buffer xuống 1 , nếu giá trị thành nhỏ hơn giá trị nhỏ nhất có thể => giá trị trên buffer thành giá trị nhỏ nhất
			return D3D11_STENCIL_OP_DECR_SAT;
		case HQ_SOP_INCR_WRAP://tăng giá trị trên buffer lên 1 , nếu giá trị thành vượt giá trị lớn nhất có thể => wrap giá trị
			return D3D11_STENCIL_OP_INCR;
		case HQ_SOP_DECR_WRAP://giảm giá trị trên buffer xuống 1 , nếu giá trị thành nhỏ hơn giá trị nhỏ nhất có thể => wrap giá trị
			return D3D11_STENCIL_OP_DECR;
		case HQ_SOP_INVERT://đảo các bit của giá trị
			return D3D11_STENCIL_OP_INVERT;
		}
		return D3D11_STENCIL_OP_KEEP;
	}
	inline D3D11_COMPARISON_FUNC GetD3DStencilFunc(HQStencilFunc func)
	{
		switch(func)
		{
		case HQ_SF_NEVER://stencil test luôn fail
			return D3D11_COMPARISON_NEVER;
		case HQ_SF_LESS://stencil pass khi op1 < op2 (tức là (ref value & readMask) < (buffer value & readMask) . ref value là giá trị tham khảo, buffer value là giá trị đang có trên buffer) 
			return D3D11_COMPARISON_LESS;
		case HQ_SF_EQUAL://pass khi op1 = op2
			return D3D11_COMPARISON_EQUAL;
		case HQ_SF_LESS_EQUAL://pass khi op1 <= op2
			return D3D11_COMPARISON_LESS_EQUAL;
		case HQ_SF_GREATER://pass khi op 1 > op2
			return D3D11_COMPARISON_GREATER;
		case HQ_SF_NOT_EQUAL://pass khi op1 != op2
			return D3D11_COMPARISON_NOT_EQUAL;
		case HQ_SF_GREATER_EQUAL://pass khi op1 >= op2
			return D3D11_COMPARISON_GREATER_EQUAL;
		case HQ_SF_ALWAYS:// luôn luôn pass
			return D3D11_COMPARISON_ALWAYS;
		}
		return D3D11_COMPARISON_ALWAYS;
	}

	D3D11_BLEND GetD3DBlendFactor(HQBlendFactor factor)
	{
		switch(factor)
		{
		case HQ_BF_ONE :
			return D3D11_BLEND_ONE;
			break;
		case HQ_BF_ZERO :
			return D3D11_BLEND_ZERO;
			break;
		case HQ_BF_SRC_COLOR :
			return D3D11_BLEND_SRC_COLOR;
			break;
		case HQ_BF_ONE_MINUS_SRC_COLOR:
			return D3D11_BLEND_INV_SRC_COLOR;
			break;
		case HQ_BF_SRC_ALPHA:
			return D3D11_BLEND_SRC_ALPHA;
			break;
		case HQ_BF_ONE_MINUS_SRC_ALPHA:
			return D3D11_BLEND_INV_SRC_ALPHA;
			break;
		}

		return (D3D11_BLEND) 0xffffffff;
	}

	D3D11_BLEND_OP GetD3DBlendOp(HQBlendOp op)
	{
		switch (op)
		{
		case HQ_BO_ADD :
			return D3D11_BLEND_OP_ADD;
			break;
		case HQ_BO_SUBTRACT :
			return D3D11_BLEND_OP_SUBTRACT;
			break;
		case HQ_BO_REVSUBTRACT  :
			return D3D11_BLEND_OP_REV_SUBTRACT;
			break;
		case HQ_BO_MIN:
			return D3D11_BLEND_OP_MIN;
			break;
		case HQ_BO_MAX:
			return D3D11_BLEND_OP_MAX;
			break;
		}
		return (D3D11_BLEND_OP)0xffffffff;
	}

	inline D3D11_TEXTURE_ADDRESS_MODE GetD3DTAddressMode(HQTexAddressMode mode)
	{
		switch(mode)
		{
		case HQ_TAM_WRAP:
			return D3D11_TEXTURE_ADDRESS_WRAP;
		case HQ_TAM_CLAMP:
			return D3D11_TEXTURE_ADDRESS_CLAMP;
		case HQ_TAM_BORDER:
			return D3D11_TEXTURE_ADDRESS_BORDER;
		case HQ_TAM_MIRROR:
			return D3D11_TEXTURE_ADDRESS_MIRROR;
		}
		return (D3D11_TEXTURE_ADDRESS_MODE) 0xffffffff;
	}

	inline void GetD3DFilter(HQFilterMode mode , D3D11_SAMPLER_DESC &stateDesc)
	{
		stateDesc.MinLOD = 0.0f;
		stateDesc.MaxLOD = D3D11_FLOAT32_MAX;
		switch(mode)
		{
		case HQ_FM_MIN_MAG_POINT:
			stateDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
			stateDesc.MaxLOD = 0.0f;
			break;
		case HQ_FM_MIN_POINT_MAG_LINEAR:
			stateDesc.Filter = D3D11_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT;
			stateDesc.MaxLOD = 0.0f;
			break;
		case HQ_FM_MIN_LINEAR_MAG_POINT:
			stateDesc.Filter = D3D11_FILTER_MIN_LINEAR_MAG_MIP_POINT;
			stateDesc.MaxLOD = 0.0f;
			break;
		case HQ_FM_MIN_MAG_LINEAR:
			stateDesc.Filter = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
			stateDesc.MaxLOD = 0.0f;
			break;
		case HQ_FM_MIN_MAG_ANISOTROPIC:
			stateDesc.Filter = D3D11_FILTER_ANISOTROPIC;
			stateDesc.MaxLOD = 0.0f;
			break;
		case HQ_FM_MIN_MAG_MIP_POINT:
			stateDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
			break;
		case HQ_FM_MIN_MAG_POINT_MIP_LINEAR:
			stateDesc.Filter = D3D11_FILTER_MIN_MAG_POINT_MIP_LINEAR;
			break;
		case HQ_FM_MIN_POINT_MAG_LINEAR_MIP_POINT:
			stateDesc.Filter = D3D11_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT;
			break;
		case HQ_FM_MIN_POINT_MAG_MIP_LINEAR:
			stateDesc.Filter = D3D11_FILTER_MIN_POINT_MAG_MIP_LINEAR;
			break;
		case HQ_FM_MIN_LINEAR_MAG_MIP_POINT:
			stateDesc.Filter = D3D11_FILTER_MIN_LINEAR_MAG_MIP_POINT;
			break;
		case HQ_FM_MIN_LINEAR_MAG_POINT_MIP_LINEAR:
			stateDesc.Filter = D3D11_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR;
			break;
		case HQ_FM_MIN_MAG_LINEAR_MIP_POINT:
			stateDesc.Filter = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
			break;
		case HQ_FM_MIN_MAG_MIP_LINEAR:
			stateDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
			break;
		case HQ_FM_MIN_MAG_MIP_ANISOTROPIC:
			stateDesc.Filter = D3D11_FILTER_ANISOTROPIC;
			break;
		}
	}
};

HQStateManagerD3D11::HQStateManagerD3D11(ID3D11Device * pD3DDevice ,
									ID3D11DeviceContext * pD3DContext ,
									BOOL multisampleEnable,
									HQLogStream* logFileStream , bool flushLog)
:HQLoggableObject(logFileStream , "D3D11 Render State Manager :" , flushLog)
{
	this->pD3DDevice = pD3DDevice;
	this->pD3DContext = pD3DContext;
	
	this->CreateClearViewportStates();
	/*--------default states---------*/
	//create default rasterizer states
	this->CreateRasterizerStates(multisampleEnable);
	this->currentFillMode = HQ_FILL_SOLID;
	this->currentCullMode = HQ_CULL_NONE;

	pD3DContext->RSSetState(this->rStates[currentFillMode][currentCullMode]);
	
	//create default depth stencil state
	this->CreateDepthStencilState(HQDepthStencilStateDesc() , NULL);
	this->dsState = this->dsStates.GetItemPointerNonCheck(0);//default depth stencil state
	pD3DContext->OMSetDepthStencilState(this->dsState->pD3DState , this->dsState->stencilRef);
	
	
	//create default blend state
	HQBlendStateD3D11 *defaultBState = HQ_NEW HQBlendStateD3D11();
	D3D11_BLEND_DESC blendDesc;
	blendDesc.AlphaToCoverageEnable = FALSE; blendDesc.IndependentBlendEnable = FALSE;
	blendDesc.RenderTarget[0].BlendEnable = FALSE;
	for (int i  = 0 ; i < 8 ; ++i)
		blendDesc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

	pD3DDevice->CreateBlendState(&blendDesc , &defaultBState->pD3DState);
	this->bStates.AddItem(defaultBState , NULL);

	this->bState = this->bStates.GetItemPointerNonCheck(0);//set default blend state to active state
	
	pD3DContext->OMSetBlendState(defaultBState->pD3DState , g_blendFactor , 0xffffffff);

	//create default sampler state
	this->CreateSamplerState(HQSamplerStateDesc() , NULL);
	//set default sampler state
	for(int i = 0 ; i < SHADER_STAGES ; ++i)
	{
		for (unsigned int j = 0 ; j < D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT ; ++j)
		{
			this->sState[i][j] = this->sStates.GetItemPointerNonCheck(0);
			switch(i)
			{
			case 0:
				if (j < g_pD3DDev->GetCaps().maxVertexSamplers)
					pD3DContext->VSSetSamplers(j  , 1 , &this->sState[i][j]->pD3DState);
				break;
			case 1:
				if (j < g_pD3DDev->GetCaps().maxGeometrySamplers)
					pD3DContext->GSSetSamplers(j  , 1 , &this->sState[i][j]->pD3DState);
				break;
			case 2:
				if (j < g_pD3DDev->GetCaps().maxPixelSamplers)
					pD3DContext->PSSetSamplers(j  , 1 , &this->sState[i][j]->pD3DState);
				break;
			case 3:
				if (j < g_pD3DDev->GetCaps().maxComputeSamplers)
					pD3DContext->CSSetSamplers(j, 1, &this->sState[i][j]->pD3DState);
				break;
			}
		}
	}

	Log("Init done!");
}
HQStateManagerD3D11::~HQStateManagerD3D11()
{
	for(int i = 0 ; i < SHADER_STAGES ; ++i)
	{
		for (int j = 0 ; j < D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT ; ++j)
		{
			this->sState[i][j] = this->sStates.GetItemPointerNonCheck(0);
			switch(i)
			{
			case 0:
				pD3DContext->VSSetSamplers(j  , 1 , &g_nullSampler);
				break;
			case 1:
				pD3DContext->GSSetSamplers(j  , 1 , &g_nullSampler);
				break;
			case 2:
				pD3DContext->PSSetSamplers(j  , 1 , &g_nullSampler);
				break;
			case 3:
				pD3DContext->CSSetSamplers(j, 1, &g_nullSampler);
				break;
			}
		}
	}
	pD3DContext->OMSetBlendState(NULL , g_blendFactor , 0xffffffff);
	pD3DContext->RSSetState(NULL);
	pD3DContext->OMSetDepthStencilState(NULL , 0x0);

	this->dsStates.RemoveAll();
	this->bStates.RemoveAll();
	this->sStates.RemoveAll();

	for(int i = 0 ; i < SHADER_STAGES ; ++i)
	{
		for (int j = 0 ; j < D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT ; ++j)
			this->sState[i][j] = HQSharedPtr<HQSamplerStateD3D11>::null;
	}
	//release rasterizer state
	for(int i = 0 ; i < HQ_FILL_MODES ; ++i)
	{
		for (int j = 0 ; j < HQ_CULL_MODES ; ++j)
			SafeRelease(this->rStates[i][j]);
	}

	//release clear viewport states
	SafeRelease(this->clearRState);

	SafeRelease(this->nonClearBState);

	SafeRelease(this->clearDSState[0]);
	SafeRelease(this->clearDSState[1]);
	SafeRelease(this->clearDSState[2]);
	SafeRelease(this->clearDSState[3]);

	Log ("Released!");
}

void HQStateManagerD3D11::CreateRasterizerStates(BOOL multisampleEnable)
{
	for(int i = 0 ; i < HQ_FILL_MODES ; ++i)
	{
		for (int j = 0 ; j < HQ_CULL_MODES ; ++j)
			this->rStates[i][j] = NULL;
	}

	//create raster states
	D3D11_RASTERIZER_DESC rasDesc;
	rasDesc.CullMode = D3D11_CULL_BACK;
	rasDesc.FrontCounterClockwise = FALSE;
	rasDesc.FillMode = D3D11_FILL_SOLID;
	rasDesc.AntialiasedLineEnable = TRUE;
	rasDesc.MultisampleEnable = multisampleEnable;
	rasDesc.DepthBias = 0;
	rasDesc.DepthBiasClamp = 0.0f;
	rasDesc.SlopeScaledDepthBias = 0.0f;
	rasDesc.ScissorEnable = FALSE;
	rasDesc.DepthClipEnable = TRUE;

	//default raster state : fill solid + cull CCW
	pD3DDevice->CreateRasterizerState(&rasDesc ,&this->rStates[HQ_FILL_SOLID][HQ_CULL_CCW]);
	//fill solid + cull CW
	rasDesc.CullMode = D3D11_CULL_FRONT;
	pD3DDevice->CreateRasterizerState(&rasDesc ,&this->rStates[HQ_FILL_SOLID][HQ_CULL_CW]);
	//fill solid + cull None
	rasDesc.CullMode = D3D11_CULL_NONE;
	pD3DDevice->CreateRasterizerState(&rasDesc ,&this->rStates[HQ_FILL_SOLID][HQ_CULL_NONE]);
	//fill wire frame + cull None
	rasDesc.FillMode = D3D11_FILL_WIREFRAME;
	pD3DDevice->CreateRasterizerState(&rasDesc ,&this->rStates[HQ_FILL_WIREFRAME][HQ_CULL_NONE]);
	//file wire frame + cull CW
	rasDesc.CullMode = D3D11_CULL_FRONT;
	pD3DDevice->CreateRasterizerState(&rasDesc ,&this->rStates[HQ_FILL_WIREFRAME][HQ_CULL_CW]);
	//fill wire frame + cull CCW
	rasDesc.CullMode = D3D11_CULL_BACK;
	pD3DDevice->CreateRasterizerState(&rasDesc ,&this->rStates[HQ_FILL_WIREFRAME][HQ_CULL_CCW]);
}


void HQStateManagerD3D11::CreateClearViewportStates()
{
	//rasterizer state
	D3D11_RASTERIZER_DESC rasDesc;
	rasDesc.CullMode = D3D11_CULL_NONE;
	rasDesc.FrontCounterClockwise = FALSE;
	rasDesc.FillMode = D3D11_FILL_SOLID;
	rasDesc.AntialiasedLineEnable = FALSE;
	rasDesc.MultisampleEnable = FALSE;
	rasDesc.DepthBias = 0;
	rasDesc.DepthBiasClamp = 0.0f;
	rasDesc.SlopeScaledDepthBias = 0.0f;
	rasDesc.ScissorEnable = FALSE;
	rasDesc.DepthClipEnable = TRUE;

	pD3DDevice->CreateRasterizerState(&rasDesc , &this->clearRState);

	//blend state
	D3D11_BLEND_DESC blendDesc;
	blendDesc.AlphaToCoverageEnable = FALSE;
	blendDesc.IndependentBlendEnable = FALSE;
	blendDesc.RenderTarget[0].BlendEnable = TRUE;
	blendDesc.RenderTarget[0].BlendOp = blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
	blendDesc.RenderTarget[0].SrcBlend = blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ZERO;
	blendDesc.RenderTarget[0].DestBlend = blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
	for (int i  = 0 ; i < 8 ; ++i)
		blendDesc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
	

	pD3DDevice->CreateBlendState(&blendDesc , &this->nonClearBState);

	//depth stencil states
	D3D11_DEPTH_STENCIL_DESC dsDesc;
	dsDesc.DepthEnable = FALSE;
	dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
	dsDesc.DepthFunc = D3D11_COMPARISON_ALWAYS;
	dsDesc.StencilEnable = FALSE;
	dsDesc.StencilReadMask = D3D11_DEFAULT_STENCIL_READ_MASK;
	dsDesc.StencilWriteMask = D3D11_DEFAULT_STENCIL_WRITE_MASK;
	dsDesc.FrontFace.StencilFunc = dsDesc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
	dsDesc.FrontFace.StencilFailOp = dsDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_REPLACE;
	dsDesc.FrontFace.StencilDepthFailOp = dsDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_REPLACE;
	dsDesc.FrontFace.StencilPassOp = dsDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_REPLACE;
	
	//non depth & stencil clearing state
	pD3DDevice->CreateDepthStencilState(&dsDesc , &this->clearDSState[NON_DEPTH_NON_STENCIL_CLEAR]);
	
	//depth clearing & non stencil clearing state
	dsDesc.DepthEnable = TRUE;
	dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	pD3DDevice->CreateDepthStencilState(&dsDesc , &this->clearDSState[HQ_DEPTH_NON_STENCIL_CLEAR]);

	//non depth clearing & stencil clearing state
	dsDesc.DepthEnable = FALSE;
	dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
	dsDesc.StencilEnable = TRUE;
	pD3DDevice->CreateDepthStencilState(&dsDesc , &this->clearDSState[NON_DEPTH_STENCIL_CLEAR]);

	//depth clearing & stencil clearing state
	dsDesc.DepthEnable = TRUE;
	dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	dsDesc.StencilEnable = TRUE;
	pD3DDevice->CreateDepthStencilState(&dsDesc , &this->clearDSState[HQ_DEPTH_STENCIL_CLEAR]);

}

void HQStateManagerD3D11::SetFillMode(HQFillMode fillMode) 
{
	if (this->currentFillMode != fillMode)
	{
		this->currentFillMode = fillMode;
		pD3DContext->RSSetState(this->rStates[currentFillMode][currentCullMode]);
	}
}
void HQStateManagerD3D11::SetFaceCulling(HQCullMode cullMode)
{
	if (this->currentCullMode != cullMode)
	{
		this->currentCullMode = cullMode;
		pD3DContext->RSSetState(this->rStates[currentFillMode][currentCullMode]);
	}
}

/*-----depth stencil state------------------*/
HQReturnVal HQStateManagerD3D11::CreateDepthStencilState(const HQDepthStencilStateDesc &desc , hq_uint32 *pDSStateID) 
{
	HQDepthStencilStateD3D11 *newState = HQ_NEW HQDepthStencilStateD3D11();
	
	D3D11_DEPTH_STENCIL_DESC dsDesc;
	/*-----get depth mode---------*/
	helper::GetD3DDepthMode(desc.depthMode , dsDesc);

	/*-----get stencil mode-------*/
	dsDesc.StencilEnable = desc.stencilEnable ? TRUE : FALSE;
	newState->stencilRef = desc.refVal;
	dsDesc.StencilReadMask = desc.readMask & 0xff;
	dsDesc.StencilWriteMask = desc.writeMask & 0xff;

	dsDesc.FrontFace.StencilFailOp = dsDesc.BackFace.StencilFailOp = helper::GetD3DStencilOp(desc.stencilMode.failOp);
	dsDesc.FrontFace.StencilDepthFailOp = dsDesc.BackFace.StencilDepthFailOp = helper::GetD3DStencilOp(desc.stencilMode.depthFailOp);
	dsDesc.FrontFace.StencilPassOp = dsDesc.BackFace.StencilPassOp = helper::GetD3DStencilOp(desc.stencilMode.passOp);
	dsDesc.FrontFace.StencilFunc = dsDesc.BackFace.StencilFunc =helper::GetD3DStencilFunc(desc.stencilMode.compareFunc);

	
	if (FAILED(pD3DDevice->CreateDepthStencilState(&dsDesc , &newState->pD3DState)))
	{		
		SafeDelete(newState);
		return HQ_FAILED;
	}
	if(!this->dsStates.AddItem(newState , pDSStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}
	
	return HQ_OK;
}
HQReturnVal HQStateManagerD3D11::CreateDepthStencilStateTwoSide(const HQDepthStencilStateTwoSideDesc &desc , hq_uint32 *pDSStateID) 
{
	HQDepthStencilStateD3D11 *newState = HQ_NEW HQDepthStencilStateD3D11();
	D3D11_DEPTH_STENCIL_DESC dsDesc;
	/*-----get depth mode---------*/
	helper::GetD3DDepthMode(desc.depthMode , dsDesc);

	/*-----get stencil mode-------*/
	dsDesc.StencilEnable = desc.stencilEnable ? TRUE : FALSE;
	newState->stencilRef = desc.refVal;
	dsDesc.StencilReadMask = desc.readMask & 0xff;
	dsDesc.StencilWriteMask = desc.writeMask & 0xff;

	dsDesc.FrontFace.StencilFailOp = helper::GetD3DStencilOp(desc.cwFaceMode.failOp);
	dsDesc.FrontFace.StencilDepthFailOp = helper::GetD3DStencilOp(desc.cwFaceMode.depthFailOp);
	dsDesc.FrontFace.StencilPassOp = helper::GetD3DStencilOp(desc.cwFaceMode.passOp);
	dsDesc.FrontFace.StencilFunc = helper::GetD3DStencilFunc(desc.cwFaceMode.compareFunc);

	dsDesc.BackFace.StencilFailOp = helper::GetD3DStencilOp(desc.ccwFaceMode.failOp);
	dsDesc.BackFace.StencilDepthFailOp = helper::GetD3DStencilOp(desc.ccwFaceMode.depthFailOp);
	dsDesc.BackFace.StencilPassOp = helper::GetD3DStencilOp(desc.ccwFaceMode.passOp);
	dsDesc.BackFace.StencilFunc = helper::GetD3DStencilFunc(desc.ccwFaceMode.compareFunc);

	
	if (FAILED(pD3DDevice->CreateDepthStencilState(&dsDesc , &newState->pD3DState)))
	{		
		SafeDelete(newState);
		return HQ_FAILED;
	}
	if(!this->dsStates.AddItem(newState , pDSStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}
HQReturnVal HQStateManagerD3D11::ActiveDepthStencilState(hq_uint32 depthStencilStateID) 
{
	HQSharedPtr<HQDepthStencilStateD3D11> state = this->dsStates.GetItemPointer(depthStencilStateID);
#if defined DEBUG || defined _DEBUG	
	if (state == NULL)
		return HQ_FAILED_INVALID_ID;
#endif
	
	if (state != this->dsState)
	{
		pD3DContext->OMSetDepthStencilState(state->pD3DState , state->stencilRef);

		this->dsState = state;
	}

	return HQ_OK;
}
HQReturnVal HQStateManagerD3D11::RemoveDepthStencilState(hq_uint32 depthStencilStateID) 
{
	if (depthStencilStateID == 0)
		return HQ_FAILED;
	return (HQReturnVal) this->dsStates.Remove(depthStencilStateID);
}

/*------blend state-------------------------*/
HQReturnVal HQStateManagerD3D11::CreateBlendState( const HQBlendStateDesc &blendStateDesc , hq_uint32 *pBStateID) 
{
	HQBlendStateD3D11 *newState = HQ_NEW HQBlendStateD3D11();
	
	D3D11_BLEND_DESC desc;
	
	desc.AlphaToCoverageEnable = FALSE;
	desc.IndependentBlendEnable = FALSE ;
	desc.RenderTarget[0].BlendOp = desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
	desc.RenderTarget[0].SrcBlend = desc.RenderTarget[0].SrcBlendAlpha= helper::GetD3DBlendFactor(blendStateDesc.srcFactor);
	desc.RenderTarget[0].DestBlend = desc.RenderTarget[0].DestBlendAlpha = helper::GetD3DBlendFactor(blendStateDesc.destFactor);
	
	desc.RenderTarget[0].BlendEnable = TRUE;
	for (int i  = 0 ; i < 8 ; ++i)
		desc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

	if (FAILED(pD3DDevice->CreateBlendState(&desc , &newState->pD3DState)))
	{
		SafeDelete(newState);
		return HQ_FAILED;
	}
	if(!this->bStates.AddItem(newState , pBStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}
HQReturnVal HQStateManagerD3D11::CreateBlendStateEx( const HQBlendStateExDesc &blendStateDesc , hq_uint32 *pBStateID) 
{
	HQBlendStateD3D11 *newState = HQ_NEW HQBlendStateD3D11();
	D3D11_BLEND_DESC desc;
	
	desc.AlphaToCoverageEnable = FALSE;
	desc.IndependentBlendEnable = FALSE ;
	desc.RenderTarget[0].BlendOp = helper::GetD3DBlendOp(blendStateDesc.blendOp);
	desc.RenderTarget[0].SrcBlend = helper::GetD3DBlendFactor(blendStateDesc.srcFactor);
	desc.RenderTarget[0].DestBlend = helper::GetD3DBlendFactor(blendStateDesc.destFactor);

	desc.RenderTarget[0].BlendOpAlpha = helper::GetD3DBlendOp(blendStateDesc.alphaBlendOp);
	desc.RenderTarget[0].SrcBlendAlpha = helper::GetD3DBlendFactor(blendStateDesc.srcAlphaFactor);
	desc.RenderTarget[0].DestBlendAlpha = helper::GetD3DBlendFactor(blendStateDesc.destAlphaFactor);
	
	desc.RenderTarget[0].BlendEnable = TRUE;
	for (int i  = 0 ; i < 8 ; ++i)
		desc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

	if (FAILED(pD3DDevice->CreateBlendState(&desc , &newState->pD3DState)))
	{
		SafeDelete(newState);
		return HQ_FAILED;
	}
	if(!this->bStates.AddItem(newState , pBStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}


HQReturnVal HQStateManagerD3D11::CreateIndependentBlendState(const HQIndieBlendStateDesc *blendStateDescs, hq_uint32 numStateDescs, hq_uint32 *pBStateID)
{
	HQBlendStateD3D11 *newState = HQ_NEW HQBlendStateD3D11();
	D3D11_BLEND_DESC desc;

	desc.AlphaToCoverageEnable = FALSE;
	desc.IndependentBlendEnable = TRUE;

	for (int i = 0; i < 8; ++i)//default values
	{
		desc.RenderTarget[i].BlendEnable = FALSE;
		desc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

		desc.RenderTarget[i].BlendOpAlpha = desc.RenderTarget[i].BlendOp = D3D11_BLEND_OP_ADD;
		desc.RenderTarget[i].SrcBlendAlpha = desc.RenderTarget[i].SrcBlend = D3D11_BLEND_ONE;
		desc.RenderTarget[i].DestBlendAlpha = desc.RenderTarget[i].DestBlend = D3D11_BLEND_ZERO;
	}

	for (hquint32 i = 0; i < numStateDescs; ++i)
	{
		const HQIndieBlendStateDesc &blendStateDesc = blendStateDescs[i];
		hquint32 rtIndex = blendStateDesc.renderTargetIndex;

		desc.RenderTarget[rtIndex].SrcBlendAlpha = desc.RenderTarget[rtIndex].SrcBlend = helper::GetD3DBlendFactor(blendStateDesc.srcFactor);
		desc.RenderTarget[rtIndex].DestBlendAlpha = desc.RenderTarget[rtIndex].DestBlend = helper::GetD3DBlendFactor(blendStateDesc.destFactor);

		desc.RenderTarget[rtIndex].BlendEnable = TRUE;
	}


	if (FAILED(pD3DDevice->CreateBlendState(&desc, &newState->pD3DState)))
	{
		SafeDelete(newState);
		return HQ_FAILED;
	}
	if (!this->bStates.AddItem(newState, pBStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

HQReturnVal HQStateManagerD3D11::CreateIndependentBlendStateEx(const HQIndieBlendStateExDesc *blendStateDescs, hq_uint32 numStateDescs, hq_uint32 *pBStateID)
{
	HQBlendStateD3D11 *newState = HQ_NEW HQBlendStateD3D11();
	D3D11_BLEND_DESC desc;

	desc.AlphaToCoverageEnable = FALSE;
	desc.IndependentBlendEnable = TRUE;

	for (int i = 0; i < 8; ++i)//default values
	{
		desc.RenderTarget[i].BlendEnable = FALSE;
		desc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

		desc.RenderTarget[i].BlendOpAlpha = desc.RenderTarget[i].BlendOp = D3D11_BLEND_OP_ADD;
		desc.RenderTarget[i].SrcBlendAlpha = desc.RenderTarget[i].SrcBlend = D3D11_BLEND_ONE;
		desc.RenderTarget[i].DestBlendAlpha = desc.RenderTarget[i].DestBlend = D3D11_BLEND_ZERO;
	}

	for (hquint32 i = 0; i < numStateDescs; ++i)
	{
		const HQIndieBlendStateExDesc &blendStateDesc = blendStateDescs[i];
		hquint32 rtIndex = blendStateDesc.renderTargetIndex;

		desc.RenderTarget[rtIndex].BlendOp = helper::GetD3DBlendOp(blendStateDesc.blendOp);
		desc.RenderTarget[rtIndex].SrcBlend = helper::GetD3DBlendFactor(blendStateDesc.srcFactor);
		desc.RenderTarget[rtIndex].DestBlend = helper::GetD3DBlendFactor(blendStateDesc.destFactor);

		desc.RenderTarget[rtIndex].BlendOpAlpha = helper::GetD3DBlendOp(blendStateDesc.alphaBlendOp);
		desc.RenderTarget[rtIndex].SrcBlendAlpha = helper::GetD3DBlendFactor(blendStateDesc.srcAlphaFactor);
		desc.RenderTarget[rtIndex].DestBlendAlpha = helper::GetD3DBlendFactor(blendStateDesc.destAlphaFactor);

		desc.RenderTarget[rtIndex].BlendEnable = TRUE;
	}


	if (FAILED(pD3DDevice->CreateBlendState(&desc, &newState->pD3DState)))
	{
		SafeDelete(newState);
		return HQ_FAILED;
	}
	if (!this->bStates.AddItem(newState, pBStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

HQReturnVal HQStateManagerD3D11::ActiveBlendState(hq_uint32 blendStateID)
{
	HQSharedPtr<HQBlendStateD3D11> state = this->bStates.GetItemPointer(blendStateID);
#if defined DEBUG || defined _DEBUG	
	if (state == NULL)
		return HQ_FAILED_INVALID_ID;
#endif
	if (state != this->bState)
	{
		pD3DContext->OMSetBlendState(state->pD3DState , g_blendFactor , 0xffffffff);
		this->bState = state;
	}
	return HQ_OK;
}
HQReturnVal HQStateManagerD3D11::RemoveBlendState(hq_uint32 blendStateID)  
{
	if (blendStateID == 0)
		return HQ_FAILED;
	return (HQReturnVal) this->bStates.Remove(blendStateID);
}

/*------sampler state-------------------------*/
HQReturnVal HQStateManagerD3D11::CreateSamplerState(const HQSamplerStateDesc &desc , hq_uint32 *pSStateID)
{
	HQSamplerStateD3D11 *newState = HQ_NEW HQSamplerStateD3D11();
	
	D3D11_SAMPLER_DESC sdesc;
	sdesc.MipLODBias = 0.0f;
	sdesc.ComparisonFunc = D3D11_COMPARISON_NEVER;

	sdesc.AddressU = helper::GetD3DTAddressMode(desc.addressU);
	sdesc.AddressV = helper::GetD3DTAddressMode(desc.addressV);
	sdesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;

	memcpy(sdesc.BorderColor , desc.borderColor.c , 4 * sizeof(hq_float32));

	helper::GetD3DFilter(desc.filterMode ,sdesc);

	sdesc.MaxAnisotropy = desc.maxAnisotropy;

	if (g_pD3DDev->GetFeatureLevel() < D3D_FEATURE_LEVEL_10_0 && sdesc.MaxLOD != D3D11_FLOAT32_MAX)
	{
		sdesc.MaxLOD = D3D11_FLOAT32_MAX;
		this->Log("CreateSamplerState() warning : mipmap cannot be disabled.");
	}
	
	if (FAILED(pD3DDevice->CreateSamplerState(&sdesc , &newState->pD3DState)))
	{
		SafeDelete(newState);
		return HQ_FAILED;
	}

	if (!this->sStates.AddItem(newState , pSStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

HQReturnVal HQStateManagerD3D11::SetSamplerState(hq_uint32 index , hq_uint32 samplerStateID)
{
	HQSharedPtr<HQSamplerStateD3D11> pState = this->sStates.GetItemPointer(samplerStateID);

	hq_uint32 samplerIndex = index & 0x0fffffff;

#if defined DEBUG || defined _DEBUG
	if (pState == NULL)
		return HQ_FAILED_INVALID_ID;
#endif


	hq_uint32 shaderStage = index & 0xf0000000;
	HQSharedPtr<HQSamplerStateD3D11> *ppCurrentState ;

	switch (shaderStage)
	{
	case HQ_VERTEX_SHADER:
#if defined DEBUG || defined _DEBUG
		if (samplerIndex >= g_pD3DDev->GetCaps().maxVertexSamplers)
		{
			Log("SetSamplerState() Error : sampler slot=%u is out of range!", samplerIndex);
			return HQ_FAILED;
		}
#endif
		ppCurrentState = &this->sState[0][samplerIndex];
		if (*ppCurrentState != pState)
		{
			pD3DContext->VSSetSamplers(samplerIndex , 1 , &pState->pD3DState);
			*ppCurrentState = pState;
		}
		break;
	case HQ_GEOMETRY_SHADER:
#if defined DEBUG || defined _DEBUG
		if (samplerIndex >= g_pD3DDev->GetCaps().maxGeometrySamplers)
		{
			Log("SetSamplerState() Error : sampler slot=%u is out of range!", samplerIndex);
			return HQ_FAILED;
		}
#endif
		ppCurrentState = &this->sState[1][samplerIndex];
		if (*ppCurrentState != pState)
		{
			pD3DContext->GSSetSamplers(samplerIndex , 1 , &pState->pD3DState);
			*ppCurrentState = pState;
		}
		break;
	case HQ_PIXEL_SHADER:
#if defined DEBUG || defined _DEBUG
		if (samplerIndex >= g_pD3DDev->GetCaps().maxPixelSamplers)
		{
			Log("SetSamplerState() Error : sampler slot=%u is out of range!", samplerIndex);
			return HQ_FAILED;
		}
#endif
		ppCurrentState = &this->sState[2][samplerIndex];
		if (*ppCurrentState != pState)
		{
			pD3DContext->PSSetSamplers(samplerIndex , 1 , &pState->pD3DState);
			*ppCurrentState = pState;
		}
		break;
	case HQ_COMPUTE_SHADER:
#if defined DEBUG || defined _DEBUG
		if (samplerIndex >= g_pD3DDev->GetCaps().maxComputeSamplers)
		{
			Log("SetSamplerState() Error : sampler slot=%u is out of range!", samplerIndex);
			return HQ_FAILED;
		}
#endif
		ppCurrentState = &this->sState[3][samplerIndex];
		if (*ppCurrentState != pState)
		{
			pD3DContext->CSSetSamplers(samplerIndex, 1, &pState->pD3DState);
			*ppCurrentState = pState;
		}
		break;
	default:
#if defined _DEBUG || defined DEBUG
		Log("Error : {index} parameter passing to SetSamplerState() method didn't bitwise OR with HQShaderType enum value!");
#endif
		return HQ_FAILED;
	}


	return HQ_OK;
}
HQReturnVal HQStateManagerD3D11::RemoveSamplerState(hq_uint32 samplerStateID) 
{
	if (samplerStateID == 0)
		return HQ_FAILED;
	return (HQReturnVal) this->sStates.Remove(samplerStateID);
}

void HQStateManagerD3D11::BeginClearViewport(HQBool clearColor , HQBool clearDepth, HQBool clearStencil,UINT8 stencilValue)
{
	pD3DContext->RSSetState(this->clearRState);
	if (clearColor)
		pD3DContext->OMSetBlendState(this->bStates.GetItemPointerNonCheck(0)->pD3DState , g_blendFactor , 0xffffffff);
	else
		pD3DContext->OMSetBlendState(this->nonClearBState , g_blendFactor , 0xffffffff);

	if (clearDepth && clearStencil)
		pD3DContext->OMSetDepthStencilState(this->clearDSState[HQ_DEPTH_STENCIL_CLEAR] , 0xffffffff & stencilValue);
	else if (clearDepth)
		pD3DContext->OMSetDepthStencilState(this->clearDSState[HQ_DEPTH_NON_STENCIL_CLEAR] , 0xffffffff & stencilValue);
	else if (clearStencil)
		pD3DContext->OMSetDepthStencilState(this->clearDSState[NON_DEPTH_STENCIL_CLEAR] , 0xffffffff & stencilValue);
	else
		pD3DContext->OMSetDepthStencilState(this->clearDSState[NON_DEPTH_NON_STENCIL_CLEAR] , 0xffffffff & stencilValue);
}
void HQStateManagerD3D11::EndClearViewport()
{
	pD3DContext->RSSetState(this->rStates[currentFillMode][currentCullMode]);
	pD3DContext->OMSetBlendState(this->bState->pD3DState , g_blendFactor , 0xffffffff);
	pD3DContext->OMSetDepthStencilState(this->dsState->pD3DState, this->dsState->stencilRef);
}
