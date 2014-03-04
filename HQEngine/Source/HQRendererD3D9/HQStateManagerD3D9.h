/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef STATE_MAN_D3D
#define STATE_MAN_D3D
#include "../HQRendererStateManager.h"
#include "../HQLoggableObject.h"
#include "../BaseImpl/HQBaseImplCommon.h"
#include "d3d9.h"
#include "../HQItemManager.h"

struct HQStencilModeD3D9
{
	DWORD stencilFunc;
	DWORD stencilFailOp;
	DWORD depthFailOp;
	DWORD passOp;
};

struct HQBaseStateBlockD3D9
{
	HQBaseStateBlockD3D9() :pD3DStateBlock(NULL) {}
	virtual ~HQBaseStateBlockD3D9() {OnLostDevice();}
	
	void OnLostDevice() {SafeRelease(pD3DStateBlock);}

	LPDIRECT3DSTATEBLOCK9 pD3DStateBlock;
};

struct HQDepthStencilStateD3D9 : public HQBaseStateBlockD3D9
{
	HQDepthStencilStateD3D9();
//affect current states , be careful to save current states before call this method
	void OnResetDevice(LPDIRECT3DDEVICE9 pD3DDevice, bool retainCurrentStates = false);

	/*---depth buffer state-----------*/
	DWORD depthEnable;
	DWORD depthFunc;
	DWORD depthWriteEnable;
	
	/*---stencil buffer state---------*/
	DWORD stwoSide;
	DWORD stencilEnable;
	DWORD sref;
	DWORD sreadMask;
	DWORD swriteMask;
	
	HQStencilModeD3D9 scw;//clockwise
	HQStencilModeD3D9 sccw;//counter clockwise
};


struct HQBlendStateD3D9: public HQBaseStateBlockD3D9
{
	HQBlendStateD3D9();
//affect current states , be careful to save current states before call this method
	void OnResetDevice(LPDIRECT3DDEVICE9 pD3DDevice , bool retainCurrentStates = false);

	DWORD extState;
	
	DWORD operation;
	DWORD alphaOperation;
	DWORD srcFactor , destFactor;
	DWORD srcAlphaFactor , destAlphaFactor;
};



struct HQSamplerStateD3D9
{
	HQSamplerStateD3D9();

	DWORD addressU;
	DWORD addressV;
	DWORD minFilter;
	DWORD magFilter;
	DWORD mipFilter;
	DWORD maxAnisotropy;
	DWORD borderColor;
};

/*-------------------------------*/
class HQStateManagerD3D9 : public HQRendererStateManager , public HQLoggableObject
{
private :

	LPDIRECT3DDEVICE9 pD3DDevice;
	HQFillMode  currentFillMode ;
	HQCullMode  currentCullMode;

	HQItemManager<HQDepthStencilStateD3D9> dsStates;
	HQItemManager<HQBlendStateD3D9> bStates;
	HQItemManager<HQSamplerStateD3D9> sStates;

	//current active state
	HQSharedPtr<HQDepthStencilStateD3D9> dsState;
	HQSharedPtr<HQBlendStateD3D9> bState;
	HQSharedPtr<HQSamplerStateD3D9> * sState[2];//for vertex & pixel shader
	
	DWORD maxPixelShaderSamplers;
	DWORD maxVertexShaderSamplers;

	hq_uint32 maxAFLevel;

	void OnResetDeviceResetStates();
public:
	HQStateManagerD3D9(LPDIRECT3DDEVICE9 pD3DDevice,
					DWORD maxVertexShaderSamplers , 
					DWORD maxPixelShaderSamplers ,
					hq_uint32 maxAFLevel ,
					HQLogStream *logFileStream , bool flushLog);
	~HQStateManagerD3D9();

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
	HQSharedPtr<HQSamplerStateD3D9> GetSamplerState(hq_uint32 samplerStateID) {return this->sStates.GetItemPointer(samplerStateID);}
	
	HQReturnVal SetSamplerState(hq_uint32 index , hq_uint32 samplerStateID);
	HQReturnVal RemoveSamplerState(hq_uint32 samplerStateID);


	/*--------------------*/
	void OnLostDevice();
	void OnResetDevice();
};

#endif
