/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef STATE_MAN_GL
#define STATE_MAN_GL
#include "../HQRendererStateManager.h"
#include "../HQLoggableObject.h"
#include "glHeaders.h"
#include "HQTextureManagerGL.h"
#include "../HQItemManager.h"

struct HQStencilModeGL
{
	GLenum stencilFunc;
	GLenum stencilFailOp;
	GLenum depthFailOp;
	GLenum passOp;
};

struct HQDepthStencilStateGL
{
	HQDepthStencilStateGL();

	/*---depth buffer state-----------*/
	bool depthEnable;
	GLenum depthFunc;
	GLboolean depthWriteEnable;
	
	/*---stencil buffer state---------*/
	bool stwoSide;
	bool stencilEnable;
	GLint sref;
	GLuint sreadMask;
	GLuint swriteMask;
	
	HQStencilModeGL sfront;
	HQStencilModeGL sback;
};


struct HQBlendStateGL
{
	HQBlendStateGL();

	bool extState;

	GLenum operation;
	GLenum alphaOperation;
	GLenum srcFactor , destFactor;
	GLenum srcAlphaFactor , destAlphaFactor;
};


/*-------------------------------*/
class HQStateManagerGL : public HQRendererStateManager , public HQLoggableObject
{
private :
	HQFillMode  currentFillMode ;
	HQCullMode  currentCullMode;

	HQItemManager<HQDepthStencilStateGL> dsStates;
	HQItemManager<HQBlendStateGL> bStates;
	HQItemManager<HQSamplerStateGL> sStates;

	//current active state
	HQSharedPtr<HQDepthStencilStateGL> dsState;
	HQSharedPtr<HQBlendStateGL> bState;

	HQTextureManagerGL *pTextureMan;
	hq_uint32 maxAFLevel;

	void ActiveDepthStencilState(const HQSharedPtr<HQDepthStencilStateGL> &state);
	void ActiveBlendState(const HQSharedPtr<HQBlendStateGL> &state);
public:
	HQStateManagerGL(HQTextureManagerGL *pTextureMan , hq_uint32 maxAFLevel ,
					HQLogStream *logFileStream , bool flushLog);
	~HQStateManagerGL();

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
	HQReturnVal CreateIndependentBlendState(const HQIndieBlendStateDesc *blendStateDescs, hq_uint32 numStateDescs, hq_uint32 *pBStateID);
	HQReturnVal CreateIndependentBlendStateEx(const HQIndieBlendStateExDesc *blendStateDescs, hq_uint32 numStateDescs, hq_uint32 *pBStateID);
	HQReturnVal ActiveBlendState(hq_uint32 blendStateID);
	HQReturnVal RemoveBlendState(hq_uint32 blendStateID) ; 

	/*------sampler state-------------------------*/
	HQReturnVal CreateSamplerState(const HQSamplerStateDesc &desc , hq_uint32 *pSStateID);
	HQSharedPtr<HQSamplerStateGL> GetSamplerState(hq_uint32 samplerStateID) {return this->sStates.GetItemPointer(samplerStateID);}
	
	HQReturnVal SetSamplerState(hq_uint32 index , hq_uint32 samplerStateID);
	HQReturnVal RemoveSamplerState(hq_uint32 samplerStateID); 

#if defined DEVICE_LOST_POSSIBLE
	/*-----device lost--------*/
	void OnReset();
#endif
};

#endif
