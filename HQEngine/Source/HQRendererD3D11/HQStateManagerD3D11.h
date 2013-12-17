/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef STATE_MAN_D3D
#define STATE_MAN_D3D
#include "../HQRendererStateManager.h"
#include "../HQLoggableObject.h"
#include "../BaseImpl/HQBaseImplCommon.h"
#include "d3d11.h"
#include "../HQItemManager.h"

#define HQ_FILL_MODES 2
#define HQ_CULL_MODES 3
#define SHADER_STAGES 3

struct HQDepthStencilStateD3D11
{
	HQDepthStencilStateD3D11()
	{
		stencilRef = 0;
		pD3DState = NULL;
	}
	~HQDepthStencilStateD3D11()
	{
		SafeRelease(pD3DState);
	}
	
	UINT stencilRef;
	ID3D11DepthStencilState *pD3DState;

};


struct HQBlendStateD3D11
{
	HQBlendStateD3D11()
	{
		pD3DState = NULL;
	}
	~HQBlendStateD3D11()
	{
		SafeRelease(pD3DState);
	}


	ID3D11BlendState *pD3DState;
};



struct HQSamplerStateD3D11
{
	HQSamplerStateD3D11()
	{
		pD3DState = NULL;
	}
	~HQSamplerStateD3D11()
	{
		SafeRelease(pD3DState);
	}
	
	ID3D11SamplerState *pD3DState;
};

/*-------------------------------*/
class HQStateManagerD3D11 : public HQRendererStateManager , public HQLoggableObject
{
private :
	ID3D11Device * pD3DDevice;
	ID3D11DeviceContext * pD3DContext;

	HQFillMode  currentFillMode ;
	HQCullMode  currentCullMode;
	
	ID3D11RasterizerState * rStates[HQ_FILL_MODES][HQ_CULL_MODES];

	HQItemManager<HQDepthStencilStateD3D11> dsStates;
	HQItemManager<HQBlendStateD3D11> bStates;
	HQItemManager<HQSamplerStateD3D11> sStates;

	//current active state
	HQSharedPtr<HQDepthStencilStateD3D11> dsState;
	HQSharedPtr<HQBlendStateD3D11> bState;
	HQSharedPtr<HQSamplerStateD3D11> sState[SHADER_STAGES][D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT];
	
	//states for clearing viewport
	ID3D11RasterizerState * clearRState;
	ID3D11BlendState * nonClearBState;//for non color clearing
	ID3D11DepthStencilState *clearDSState[4];//for non depth & stencil clearing , depth clearing and non stencil clearing , non depth clearing and stencil clearing , depth and stencil clearing 

	void CreateRasterizerStates(BOOL multisampleEnable);
	void CreateClearViewportStates();
public:
	HQStateManagerD3D11(ID3D11Device * pD3DDevice ,
					ID3D11DeviceContext * pD3DContext ,
					BOOL multisampleEnable,
					HQLogStream* logFileStream , bool flushLog);
	~HQStateManagerD3D11();

	void SetFillMode(HQFillMode fillMode) ;
	void SetFaceCulling(HQCullMode cullMode);

	/*-----depth stencil state------------------*/
	HQReturnVal CreateDepthStencilState(const HQDepthStencilStateDesc &desc , hq_uint32 *pDSStateID) ;
	HQReturnVal CreateDepthStencilStateTwoSide(const HQDepthStencilStateTwoSideDesc &desc , hq_uint32 *pDSStateID) ;
	HQReturnVal ActiveDepthStencilState(hq_uint32 depthStencilStateID) ;
	HQReturnVal RemoveDepthStencilState(hq_uint32 depthStencilStateID) ;
	
	/*------blend state-------------------------*/
	HQReturnVal CreateBlendState( const HQBlendStateDesc &blendState , hq_uint32 *pBStateID) ;
	HQReturnVal CreateBlendStateEx( const HQBlendStateExDesc &blendState , hq_uint32 *pBStateID) ;
	HQReturnVal ActiveBlendState(hq_uint32 blendStateID);
	HQReturnVal RemoveBlendState(hq_uint32 blendStateID) ; 

	/*------sampler state-------------------------*/
	HQReturnVal CreateSamplerState(const HQSamplerStateDesc &desc , hq_uint32 *pSStateID);
	HQSharedPtr<HQSamplerStateD3D11> GetSamplerState(hq_uint32 samplerStateID) {return this->sStates.GetItemPointer(samplerStateID);}
	
	HQReturnVal SetSamplerState(hq_uint32 index , hq_uint32 samplerStateID);
	HQReturnVal RemoveSamplerState(hq_uint32 samplerStateID);

	/*----------------*/
	void BeginClearViewport(HQBool clearColor , HQBool clearDepth, HQBool clearStencil,UINT8 stencilValue);
	void EndClearViewport();
};

#endif
