/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "HQStateManagerGL.h"


//**********************************
//helper functions
//**********************************
namespace helper{
	void GetGLDepthMode(HQDepthMode mode , HQDepthStencilStateGL *dsState)
	{
		switch (mode)
		{
		case HQ_DEPTH_FULL:
			dsState->depthEnable = true;
			dsState->depthWriteEnable = GL_TRUE;
			break;
		case HQ_DEPTH_READONLY :
			dsState->depthEnable = true;
			dsState->depthWriteEnable = GL_FALSE;
			break;
		case HQ_DEPTH_WRITEONLY:
			dsState->depthEnable = true;
			dsState->depthWriteEnable = GL_TRUE;
			dsState->depthFunc = GL_ALWAYS;
			break;
		case HQ_DEPTH_NONE:
			dsState->depthEnable = false;
			dsState->depthWriteEnable = GL_TRUE;
			dsState->depthFunc = GL_LEQUAL;
			break;
		case HQ_DEPTH_READONLY_GREATEREQUAL:
			dsState->depthEnable = true;
			dsState->depthWriteEnable = GL_FALSE;
			dsState->depthFunc = GL_GEQUAL;
			break;
		}
	}
	GLenum GetGLStencilOp(HQStencilOp op)
	{
		switch(op)
		{
		case HQ_SOP_KEEP://giữ giá trị trên stencil buffer
			return GL_KEEP;
		case HQ_SOP_ZERO://set giá trị trên stencil buffer thành 0
			return GL_ZERO;
		case HQ_SOP_REPLACE://thay giá trị trên buffer thành giá rị tham khảo
			return GL_REPLACE;
		case HQ_SOP_INCR://tăng giá trị trên buffer lên 1 , nếu giá trị thành vượt giá trị lớn nhất có thể => giá trị trên buffer thành giá trị lớn nhất
			return GL_INCR;
		case HQ_SOP_DECR://giảm giá trị trên buffer xuống 1 , nếu giá trị thành nhỏ hơn giá trị nhỏ nhất có thể => giá trị trên buffer thành giá trị nhỏ nhất
			return GL_DECR;
		case HQ_SOP_INCR_WRAP://tăng giá trị trên buffer lên 1 , nếu giá trị thành vượt giá trị lớn nhất có thể => wrap giá trị
			return GL_INCR_WRAP;
		case HQ_SOP_DECR_WRAP://giảm giá trị trên buffer xuống 1 , nếu giá trị thành nhỏ hơn giá trị nhỏ nhất có thể => wrap giá trị
			return GL_DECR_WRAP;
		case HQ_SOP_INVERT://đảo các bit của giá trị
			return GL_INVERT;
		}
		return 0xffffffff;
	}
	GLenum GetGLStencilFunc(HQStencilFunc func)
	{
		switch(func)
		{
		case HQ_SF_NEVER://stencil test luôn fail
			return GL_NEVER;
		case HQ_SF_LESS://stencil pass khi op1 < op2 (tức là (ref value & readMask) < (buffer value & readMask) . ref value là giá trị tham khảo, buffer value là giá trị đang có trên buffer)
			return GL_LESS;
		case HQ_SF_EQUAL://pass khi op1 = op2
			return GL_EQUAL;
		case HQ_SF_LESS_EQUAL://pass khi op1 <= op2
			return GL_LEQUAL;
		case HQ_SF_GREATER://pass khi op 1 > op2
			return GL_GREATER;
		case HQ_SF_NOT_EQUAL://pass khi op1 != op2
			return GL_NOTEQUAL;
		case HQ_SF_GREATER_EQUAL://pass khi op1 >= op2
			return GL_GEQUAL;
		case HQ_SF_ALWAYS:// luôn luôn pass
			return GL_ALWAYS;
		}
		return 0xffffffff;
	}

	GLenum GetGLBlendFactor(HQBlendFactor factor)
	{
		switch(factor)
		{
		case HQ_BF_ONE :
			return GL_ONE;
			break;
		case HQ_BF_ZERO :
			return GL_ZERO;
			break;
		case HQ_BF_SRC_COLOR :
			return GL_SRC_COLOR;
			break;
		case HQ_BF_ONE_MINUS_SRC_COLOR:
			return GL_ONE_MINUS_SRC_COLOR;
			break;
		case HQ_BF_SRC_ALPHA:
			return GL_SRC_ALPHA;
			break;
		case HQ_BF_ONE_MINUS_SRC_ALPHA:
			return GL_ONE_MINUS_SRC_ALPHA;
			break;
		}

		return 0;
	}

	GLenum GetGLBlendOp(HQBlendOp op)
	{
		switch (op)
		{
		case HQ_BO_ADD :
			return GL_FUNC_ADD;
			break;
		case HQ_BO_SUBTRACT :
			return GL_FUNC_SUBTRACT;
			break;
		case HQ_BO_REVSUBTRACT  :
			return GL_FUNC_REVERSE_SUBTRACT;
			break;
#ifndef HQ_OPENGLES //TO DO
		case HQ_BO_MIN:
			return GL_MIN;
			break;
		case HQ_BO_MAX:
			return GL_MAX;
			break;
#endif
		}
		return 0;
	}

	GLint GetGLTAddressMode(HQTexAddressMode mode)
	{
		switch(mode)
		{
		case HQ_TAM_WRAP:
			return GL_REPEAT;
		case HQ_TAM_CLAMP:
			return GL_CLAMP_TO_EDGE;
		case HQ_TAM_BORDER:
			return GL_CLAMP_TO_BORDER;
		case HQ_TAM_MIRROR:
			return GL_MIRRORED_REPEAT;
		}
		return 0xffffffff;
	}
	void GetGLFilter(HQFilterMode mode , HQSamplerStateGL *state)
	{
		switch(mode)
		{
		case HQ_FM_MIN_MAG_POINT:
			state->minMipFilter = GL_NEAREST;
			state->magFilter = GL_NEAREST;
			break;
		case HQ_FM_MIN_POINT_MAG_LINEAR:
			state->minMipFilter = GL_NEAREST;
			state->magFilter = GL_LINEAR;
			break;
		case HQ_FM_MIN_LINEAR_MAG_POINT:
			state->minMipFilter =  GL_LINEAR;
			state->magFilter = GL_NEAREST;
			break;
		case HQ_FM_MIN_MAG_LINEAR:
			state->minMipFilter =  GL_LINEAR;
			state->magFilter = GL_LINEAR;
			break;
		case HQ_FM_MIN_MAG_ANISOTROPIC:
			state->minMipFilter =  GL_LINEAR;
			state->magFilter = GL_LINEAR;
			break;
		case HQ_FM_MIN_MAG_MIP_POINT:
			state->minMipFilter = GL_NEAREST_MIPMAP_NEAREST;
			state->magFilter = GL_NEAREST;
			break;
		case HQ_FM_MIN_MAG_POINT_MIP_LINEAR:
			state->minMipFilter = GL_NEAREST_MIPMAP_LINEAR;
			state->magFilter = GL_NEAREST;
			break;
		case HQ_FM_MIN_POINT_MAG_LINEAR_MIP_POINT:
			state->minMipFilter = GL_NEAREST_MIPMAP_NEAREST;
			state->magFilter = GL_LINEAR;
			break;
		case HQ_FM_MIN_POINT_MAG_MIP_LINEAR:
			state->minMipFilter = GL_NEAREST_MIPMAP_LINEAR;
			state->magFilter = GL_LINEAR;
			break;
		case HQ_FM_MIN_LINEAR_MAG_MIP_POINT:
			state->minMipFilter = GL_LINEAR_MIPMAP_NEAREST;
			state->magFilter = GL_NEAREST;
			break;
		case HQ_FM_MIN_LINEAR_MAG_POINT_MIP_LINEAR:
			state->minMipFilter = GL_LINEAR_MIPMAP_LINEAR;
			state->magFilter = GL_NEAREST;
			break;
		case HQ_FM_MIN_MAG_LINEAR_MIP_POINT:
			state->minMipFilter = GL_LINEAR_MIPMAP_NEAREST;
			state->magFilter = GL_LINEAR;
			break;
		case HQ_FM_MIN_MAG_MIP_LINEAR:
			state->minMipFilter = GL_LINEAR_MIPMAP_LINEAR;
			state->magFilter = GL_LINEAR;
			break;
		case HQ_FM_MIN_MAG_MIP_ANISOTROPIC:
			state->minMipFilter = GL_LINEAR_MIPMAP_LINEAR;
			state->magFilter = GL_LINEAR;
			break;
		}
	}
};
/*------------------------------------------*/
HQDepthStencilStateGL::HQDepthStencilStateGL()
{
	this->depthEnable = false;
	this->depthFunc = GL_LEQUAL;
	this->stencilEnable = false;
}
/*------------------------------------------*/
HQBlendStateGL::HQBlendStateGL()
{
}
/*------------------------------------------*/
HQSamplerStateGL::HQSamplerStateGL()
{
	this->addressU = this->addressV = GL_REPEAT;
	this->magFilter = GL_LINEAR;
	this->minMipFilter = GL_LINEAR_MIPMAP_LINEAR;
	this->maxAnisotropy = 1.0f;
	this->borderColor.r = this->borderColor.g = this->borderColor.b =
	this->borderColor.a = 0.0f;
}
/*------------------------------------------*/

HQStateManagerGL::HQStateManagerGL(HQTextureManagerGL *pTextureMan ,hq_uint32 maxAFLevel ,
								   HQLogStream *logFileStream , bool flushLog)
:HQLoggableObject(logFileStream , "GL Render State Manager :" , flushLog)
{
	this->pTextureMan = pTextureMan;

	this->maxAFLevel = maxAFLevel;

	/*--------default states---------*/
	this->currentFillMode = HQ_FILL_SOLID;

	this->currentCullMode = HQ_CULL_NONE;

	this->CreateDepthStencilState(HQDepthStencilStateDesc() , NULL);
	this->CreateBlendState(HQBlendStateDesc() , NULL);
	this->CreateSamplerState(HQSamplerStateDesc() , NULL);

	this->dsState = this->dsStates.GetItemPointerNonCheck(0);
	this->bState = this->bStates.GetItemPointerNonCheck(0);


	glFrontFace(GL_CW);
	glDisable(GL_CULL_FACE);

	glDisable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glDisable(GL_STENCIL_TEST);

	glDisable(GL_BLEND);

	Log("Init done!");
}
HQStateManagerGL::~HQStateManagerGL()
{
	Log ("Released!");
}

void HQStateManagerGL::SetFillMode(HQFillMode fillMode)
{
#ifndef HQ_OPENGLES
	if (this->currentFillMode != fillMode)
	{
		switch (fillMode)
		{
		case HQ_FILL_WIREFRAME:
			glPolygonMode(GL_FRONT_AND_BACK , GL_LINE);
			break;
		case HQ_FILL_SOLID:
			glPolygonMode(GL_FRONT_AND_BACK , GL_FILL);
			break;
		}
		this->currentFillMode = fillMode;
	}
#endif
}
void HQStateManagerGL::SetFaceCulling(HQCullMode cullMode)
{
	if (this->currentCullMode != cullMode)
	{
		switch (cullMode)
		{
		case HQ_CULL_CW:
			glEnable(GL_CULL_FACE);
			glCullFace(GL_FRONT);
			break;
		case HQ_CULL_CCW:
			glEnable(GL_CULL_FACE);
			glCullFace(GL_BACK);
			break;
		case HQ_CULL_NONE:
			glDisable(GL_CULL_FACE);
			break;
		}
		this->currentCullMode = cullMode;
	}
}

/*-----depth stencil state------------------*/
HQReturnVal HQStateManagerGL::CreateDepthStencilState(const HQDepthStencilStateDesc &desc , hq_uint32 *pDSStateID)
{
	HQDepthStencilStateGL *newState = new HQDepthStencilStateGL();
	if (newState == NULL || !this->dsStates.AddItem(newState , pDSStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}

	/*-----get depth mode---------*/
	helper::GetGLDepthMode(desc.depthMode , newState);

	/*-----get stencil mode-------*/
	newState->stwoSide = false;
	newState->stencilEnable = desc.stencilEnable;
	newState->sref = (GLint)desc.refVal;
	newState->sreadMask = desc.readMask;
	newState->swriteMask = desc.writeMask;

	newState->sfront.stencilFailOp = newState->sback.stencilFailOp = helper::GetGLStencilOp(desc.stencilMode.failOp);
	newState->sfront.depthFailOp = newState->sback.depthFailOp = helper::GetGLStencilOp(desc.stencilMode.depthFailOp);
	newState->sfront.passOp = newState->sback.passOp = helper::GetGLStencilOp(desc.stencilMode.passOp);
	newState->sfront.stencilFunc = newState->sback.stencilFunc =helper::GetGLStencilFunc(desc.stencilMode.compareFunc);

	return HQ_OK;
}
HQReturnVal HQStateManagerGL::CreateDepthStencilStateTwoSide(const HQDepthStencilStateTwoSideDesc &desc , hq_uint32 *pDSStateID)
{
	HQDepthStencilStateGL *newState = new HQDepthStencilStateGL();
	if (newState == NULL || !this->dsStates.AddItem(newState , pDSStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}

	/*-----get depth mode---------*/
	helper::GetGLDepthMode(desc.depthMode , newState);

	/*-----get stencil mode-------*/
	newState->stwoSide = true;
	newState->stencilEnable = desc.stencilEnable;
	newState->sref = (GLint)desc.refVal;
	newState->sreadMask = desc.readMask;
	newState->swriteMask = desc.writeMask;
	//front
	newState->sfront.stencilFailOp = helper::GetGLStencilOp(desc.cwFaceMode.failOp);
	newState->sfront.depthFailOp = helper::GetGLStencilOp(desc.cwFaceMode.depthFailOp);
	newState->sfront.passOp = helper::GetGLStencilOp(desc.cwFaceMode.passOp);
	newState->sfront.stencilFunc = helper::GetGLStencilFunc(desc.cwFaceMode.compareFunc);
	//back
	newState->sback.stencilFailOp = helper::GetGLStencilOp(desc.ccwFaceMode.failOp);
	newState->sback.depthFailOp = helper::GetGLStencilOp(desc.ccwFaceMode.depthFailOp);
	newState->sback.passOp = helper::GetGLStencilOp(desc.ccwFaceMode.passOp);
	newState->sback.stencilFunc = helper::GetGLStencilFunc(desc.ccwFaceMode.compareFunc);

	return HQ_OK;
}


HQ_FORCE_INLINE void HQStateManagerGL::ActiveDepthStencilState(const HQSharedPtr<HQDepthStencilStateGL> &state)
{
	if (state->depthEnable != this->dsState->depthEnable)
	{
		if (state->depthEnable)
			glEnable(GL_DEPTH_TEST);
		else
			glDisable(GL_DEPTH_TEST);
	}
	if (state->depthFunc != this->dsState->depthFunc)
		glDepthFunc(state->depthFunc);
	if (state->depthWriteEnable != this->dsState->depthWriteEnable)
		glDepthMask(state->depthWriteEnable);

	if (state->stencilEnable != this->dsState->stencilEnable)
	{
		if (state->stencilEnable)
			glEnable(GL_STENCIL_TEST);
		else
			glDisable(GL_STENCIL_TEST);
	}
	if (state->stencilEnable)
	{
		if (state->stwoSide)//two sided stencil
		{
			glStencilFuncSeparate(GL_FRONT ,
				state->sfront.stencilFunc ,
				state->sref,
				state->sreadMask);

			glStencilFuncSeparate(GL_BACK ,
				state->sback.stencilFunc ,
				state->sref,
				state->sreadMask);

			glStencilOpSeparate(GL_FRONT ,
				state->sfront.stencilFailOp,
				state->sfront.depthFailOp,
				state->sfront.passOp);


			glStencilOpSeparate(GL_BACK ,
				state->sback.stencilFailOp,
				state->sback.depthFailOp,
				state->sback.passOp);
		}//if (state->stwoSide)
		else
		{
			glStencilFunc(state->sfront.stencilFunc ,
				state->sref,
				state->sreadMask);

			glStencilOp(state->sfront.stencilFailOp,
				state->sfront.depthFailOp,
				state->sfront.passOp);
		}
	}//if (state->stencilEnable)
	if (state->swriteMask != this->dsState->swriteMask)
		glStencilMask(state->swriteMask);
}

HQReturnVal HQStateManagerGL::ActiveDepthStencilState(hq_uint32 depthStencilStateID)
{
	HQSharedPtr<HQDepthStencilStateGL> state = this->dsStates.GetItemPointer(depthStencilStateID);
#if defined _DEBUG || defined DEBUG	
	if (state == NULL)
		return HQ_FAILED_INVALID_ID;
#endif

	if (state != this->dsState)
	{
		this->ActiveDepthStencilState(state);

		this->dsState = state;
	}

	return HQ_OK;
}
HQReturnVal HQStateManagerGL::RemoveDepthStencilState(hq_uint32 depthStencilStateID)
{
	if (depthStencilStateID == 0)
		return HQ_FAILED;
	return (HQReturnVal) this->dsStates.Remove(depthStencilStateID);
}

/*------blend state-------------------------*/
HQReturnVal HQStateManagerGL::CreateBlendState( const HQBlendStateDesc &blendStateDesc , hq_uint32 *pBStateID)
{
	HQBlendStateGL *newState = new HQBlendStateGL();
	if (newState == NULL || !this->bStates.AddItem(newState , pBStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}

	newState->extState = false;
	newState->operation = newState->alphaOperation = GL_FUNC_ADD;
	newState->srcFactor = newState->srcAlphaFactor = helper::GetGLBlendFactor(blendStateDesc.srcFactor);
	newState->destFactor = newState->destAlphaFactor = helper::GetGLBlendFactor(blendStateDesc.destFactor);

	return HQ_OK;
}
HQReturnVal HQStateManagerGL::CreateBlendStateEx( const HQBlendStateExDesc &blendStateDesc , hq_uint32 *pBStateID)
{
	HQBlendStateGL *newState = new HQBlendStateGL();
	if (newState == NULL || !this->bStates.AddItem(newState , pBStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}
	newState->extState = true;

	newState->operation = helper::GetGLBlendOp(blendStateDesc.blendOp);
	newState->srcFactor = helper::GetGLBlendFactor(blendStateDesc.srcFactor);
	newState->destFactor = helper::GetGLBlendFactor(blendStateDesc.destFactor);

	newState->alphaOperation = helper::GetGLBlendOp(blendStateDesc.alphaBlendOp);
	newState->srcAlphaFactor = helper::GetGLBlendFactor(blendStateDesc.srcAlphaFactor);
	newState->destAlphaFactor = helper::GetGLBlendFactor(blendStateDesc.destAlphaFactor);

	return HQ_OK;
}

HQReturnVal HQStateManagerGL::CreateIndependentBlendState(const HQIndieBlendStateDesc *blendStateDescs, hq_uint32 numStateDescs, hq_uint32 *pBStateID)
{
	//TO DO
	return HQ_FAILED;
}

HQReturnVal HQStateManagerGL::CreateIndependentBlendStateEx(const HQIndieBlendStateExDesc *blendStateDescs, hq_uint32 numStateDescs, hq_uint32 *pBStateID)
{
	//TO DO
	return HQ_FAILED;
}


HQ_FORCE_INLINE void HQStateManagerGL::ActiveBlendState(const HQSharedPtr<HQBlendStateGL> &state)
{
	if (this->bState == this->bStates.GetItemPointerNonCheck(0))//current state is default state
		glEnable(GL_BLEND);

	if (state->operation != this->bState->operation||
		state->alphaOperation != this->bState->alphaOperation)
	{
		glBlendEquationSeparate(state->operation , state->alphaOperation);
	}
	if (state->srcFactor != this->bState->srcFactor ||
		state->destFactor != this->bState->destFactor ||
		state->srcAlphaFactor != this->bState->srcAlphaFactor ||
		state->destAlphaFactor != this->bState->destAlphaFactor)
	{
		if (state->extState)
			glBlendFuncSeparate(state->srcFactor ,
								state->destFactor,
								state->srcAlphaFactor,
								state->destAlphaFactor);
		else
			glBlendFunc(state->srcFactor ,
						state->destFactor);
	}
}

HQReturnVal HQStateManagerGL::ActiveBlendState(hq_uint32 blendStateID)
{
	HQSharedPtr<HQBlendStateGL> state = this->bStates.GetItemPointer(blendStateID);
#if defined _DEBUG || defined DEBUG	
	if (state == NULL)
		return HQ_FAILED_INVALID_ID;
#endif
	if (state == this->bState)//no change
		return HQ_OK;
	if (blendStateID == 0)
	{
		glDisable(GL_BLEND);
		*state = *this->bState;//copy prev active blend state
	}
	else
	{
		this->ActiveBlendState(state);

	}
	this->bState = state;

	return HQ_OK;
}
HQReturnVal HQStateManagerGL::RemoveBlendState(hq_uint32 blendStateID)
{
	if (blendStateID == 0)
		return HQ_FAILED;
	return (HQReturnVal) this->bStates.Remove(blendStateID);
}

/*------sampler state-------------------------*/
HQReturnVal HQStateManagerGL::CreateSamplerState(const HQSamplerStateDesc &desc , hq_uint32 *pSStateID)
{
	HQSamplerStateGL *newState = new HQSamplerStateGL();
	if (newState == NULL || !this->sStates.AddItem(newState , pSStateID))
	{
		SafeDelete(newState);
		return HQ_FAILED_MEM_ALLOC;
	}

#ifdef HQ_OPENGLES
	
	if (desc.addressU == HQ_TAM_BORDER || desc.addressV == HQ_TAM_BORDER)
	{
		Log("Warning : Texture Adress Mode HQ_TAM_BORDER is not supported!");
	}
	
#endif
	newState->addressU = helper::GetGLTAddressMode(desc.addressU);
	newState->addressV = helper::GetGLTAddressMode(desc.addressV);
	newState->borderColor = desc.borderColor;

	helper::GetGLFilter(desc.filterMode , newState);

	if(desc.filterMode == HQ_FM_MIN_MAG_MIP_ANISOTROPIC || desc.filterMode == HQ_FM_MIN_MAG_ANISOTROPIC)
	{
		if (desc.maxAnisotropy < 1)
			newState->maxAnisotropy = 1.0f;
		else
			newState->maxAnisotropy = (GLfloat)desc.maxAnisotropy;

		if(newState->maxAnisotropy > this->maxAFLevel)
			newState->maxAnisotropy =(this->maxAFLevel > 0)? (GLfloat)this->maxAFLevel : 1.0f;
	}
	else
		newState->maxAnisotropy = 1.0f;

	return HQ_OK;
}

HQReturnVal HQStateManagerGL::SetSamplerState(hq_uint32 index , hq_uint32 samplerStateID)
{
	HQSharedPtr<HQBaseTexture> pTexture = this->pTextureMan->GetTextureSharedPtrAt(index);

	HQSharedPtr<HQSamplerStateGL> pState = this->sStates.GetItemPointer(samplerStateID);
#if defined _DEBUG || defined DEBUG
	if (pState == NULL || pTexture == NULL || pTexture->type == HQ_TEXTURE_BUFFER)
		return HQ_FAILED_INVALID_ID;
#endif

	HQTextureGL* pTextureGL = (HQTextureGL*)pTexture.GetRawPointer();

	if (pTextureGL->pSamplerState != pState)
	{
		GLenum target = pTextureGL->textureTarget;
		GLuint currentBoundTex;

		HQSharedPtr<HQBaseTexture> currentTexture = this->pTextureMan->GetActiveTextureUnitInfo().GetTexture(pTextureGL->type);
		if (currentTexture == NULL)
			currentBoundTex = 0;
		else
			currentBoundTex = *((GLuint*)currentTexture->pData);


		GLuint *pTextureName = (GLuint *) pTextureGL->pData;
		if (currentBoundTex != *pTextureName)
			glBindTexture(target , *pTextureName);

#if 0 && defined _DEBUG
		//check the correctness
		GLuint realBoundTex;
		glGetIntegerv(GL_TEXTURE_BINDING_2D, (GLint*)&realBoundTex);
		if (realBoundTex != *pTextureName)
			assert(false);
#endif

		/*-----------set sampler state------------*/

		if (pState->addressU != pTextureGL->pSamplerState->addressU )
			glTexParameteri(target,GL_TEXTURE_WRAP_S , pState->addressU);

		if (pState->addressV != pTextureGL->pSamplerState->addressV )
			glTexParameteri(target,GL_TEXTURE_WRAP_T , pState->addressV);

		if (pState->magFilter != pTextureGL->pSamplerState->magFilter)
			glTexParameteri(target, GL_TEXTURE_MAG_FILTER, pState->magFilter);

		if (pState->minMipFilter != pTextureGL->pSamplerState->minMipFilter)
			glTexParameteri(target, GL_TEXTURE_MIN_FILTER, pState->minMipFilter);

		if (GLEW_EXT_texture_filter_anisotropic)
		{
			if (pState->maxAnisotropy != pTextureGL->pSamplerState->maxAnisotropy)
				glTexParameterf(target, GL_TEXTURE_MAX_ANISOTROPY_EXT , pState->maxAnisotropy);
		}

	#ifndef HQ_OPENGLES
		if (pState->addressU == GL_CLAMP_TO_BORDER || pState->addressV == GL_CLAMP_TO_BORDER)
			glTexParameterfv(target , GL_TEXTURE_BORDER_COLOR , pState->borderColor.c);
	#endif

		if (currentBoundTex != *pTextureName)
			glBindTexture(target , currentBoundTex);

		pTextureGL->pSamplerState = pState;
	}


	return HQ_OK;
}
HQReturnVal HQStateManagerGL::RemoveSamplerState(hq_uint32 samplerStateID)
{
	if (samplerStateID == 0)
		return HQ_FAILED;
	return (HQReturnVal) this->sStates.Remove(samplerStateID);
}

#if defined DEVICE_LOST_POSSIBLE
void HQStateManagerGL::OnReset()
{
	//reset states
	glFrontFace(GL_CW);
	glDisable(GL_CULL_FACE);

	glDisable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glDisable(GL_STENCIL_TEST);

	glDisable(GL_BLEND);

	HQFillMode activeFillMode = this->currentFillMode;
	this->currentFillMode = HQ_FILL_SOLID;
	this->SetFillMode(activeFillMode);

	HQCullMode activeCullMode = this->currentCullMode;
	this->currentCullMode = HQ_CULL_NONE;
	this->SetFaceCulling(activeCullMode);

	if (this->dsState != this->dsStates.GetItemPointerNonCheck(0))
	{
		HQSharedPtr<HQDepthStencilStateGL> activeDState = this->dsState;
		this->dsState = this->dsStates.GetItemPointerNonCheck(0);
		this->ActiveDepthStencilState(activeDState);

		this->dsState = activeDState;
	}

	if (this->bState != this->bStates.GetItemPointerNonCheck(0))
	{
		HQSharedPtr<HQBlendStateGL> activeBState = this->bState;
		this->bState = this->bStates.GetItemPointerNonCheck(0);
		this->ActiveBlendState(activeBState);

		this->bState = activeBState;
	}

}
#endif
