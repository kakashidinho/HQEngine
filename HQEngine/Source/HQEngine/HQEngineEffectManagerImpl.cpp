/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/
#include "HQEnginePCH.h"
#include "../HQEngineApp.h"
#include "HQEngineEffectManagerImpl.h"

#include <sstream>

//TO DO: reload resources when device lost

//parser's functions and variables
extern HQDataReaderStream * hqengine_effect_parser_input_stream ;
extern std::stringstream	* hqengine_effect_parser_log_stream ;
extern HQEngineEffectParserNode * hqengine_effect_parser_root_result;

extern int hqengine_effect_parser_scan();
extern void hqengine_effect_parser_recover_from_error();
extern void hqengine_effect_parser_clean_up();

/*--------------common hash code------------------*/
template <class T>
static inline hquint32 SPtrHashCode(const HQSharedPtr<T>& ptr){
#if defined (__LP64__) || defined(_M_X64) || defined(__amd64__)
	return ((hquint64)ptr.GetRawPointer() * 2654435761) & 0xffffffff;
#else
	return (hquint32)ptr.GetRawPointer() * 2654435761;
#endif
}

template <class T>
static inline hquint32 UIntCastedHashCode(const T& key)
{
	return (hquint32) key *  2654435761;
}

/*---------------shader program------------*/
HQEngineShaderProgramWrapper::HQEngineShaderProgramWrapper()
: m_programID(NULL)
{
}
HQEngineShaderProgramWrapper::~HQEngineShaderProgramWrapper()
{
	if (m_programID != NULL)
		m_renderDevice->GetShaderManager()->RemoveProgram(m_programID);
}

HQReturnVal HQEngineShaderProgramWrapper::Init(const CreationParams& params)
{
	HQShaderObject* vID, *gID, *pID;
	vID = params.vertexShader != NULL ? (params.vertexShader->GetShader() != NULL ? params.vertexShader->GetShader() : NULL) : NULL;
	gID = params.geometryShader != NULL ? (params.geometryShader->GetShader() != NULL ? params.geometryShader->GetShader() : NULL) : NULL;
	pID = params.pixelShader != NULL ? (params.pixelShader->GetShader() != NULL ? params.pixelShader->GetShader() : NULL) : NULL;

	HQReturnVal re = m_renderDevice->GetShaderManager()->CreateProgram(vID, pID, gID, &m_programID);

	if (!HQFailed(re))
	{
		m_creationParams = params;
	}

	return re;
}


HQReturnVal HQEngineShaderProgramWrapper::Active(){
	return m_renderDevice->GetShaderManager()->ActiveProgram(m_programID);
}


hquint32 HQEngineShaderProgramWrapper::CreationParams::HashCode() const{
	hquint32 hash = SPtrHashCode(vertexShader);
	hash = 29 * hash + SPtrHashCode(geometryShader);
	hash = 29 * hash + SPtrHashCode(pixelShader);

	return hash;
}

bool HQEngineShaderProgramWrapper::CreationParams::Equal(const CreationParams* params2) const
{
	if (this->vertexShader != params2->vertexShader)
		return false;
	if (this->geometryShader != params2->geometryShader)
		return false;
	if (this->pixelShader != params2->pixelShader)
		return false;
	return true;
}

HQEngineShaderProgramWrapper::CreationParams& HQEngineShaderProgramWrapper::CreationParams::operator = (const CreationParams& params2)
{
	this->vertexShader = params2.vertexShader;
	this->geometryShader = params2.geometryShader;
	this->pixelShader = params2.pixelShader;

	return *this;
}

/*-------------blend state------------------*/
HQEngineBlendStateWrapper::HQEngineBlendStateWrapper()
:stateID(0)
{
}
HQEngineBlendStateWrapper::~HQEngineBlendStateWrapper()
{
	if (stateID != 0)
		m_renderDevice->GetStateManager()->RemoveBlendState(stateID);
}

HQReturnVal HQEngineBlendStateWrapper::Init(const CreationParams& params)
{
	HQReturnVal re;
	if (params.isExState)
		re = m_renderDevice->GetStateManager()->CreateBlendStateEx(params.descEx, &stateID);
	else
		re = m_renderDevice->GetStateManager()->CreateBlendState(params.descEx, &stateID);
	if (re == HQ_OK)
	{
		this->creationParams = params;
	}
	return re;
}

HQReturnVal HQEngineBlendStateWrapper::Active()
{
	return m_renderDevice->GetStateManager()->ActiveBlendState(stateID);
}

hquint32 HQEngineBlendStateWrapper::CreationParams::HashCode() const{
	hquint32 hash = UIntCastedHashCode(isExState);
	hash = hash * 29 + UIntCastedHashCode(descEx.srcFactor);
	hash = hash * 29 + UIntCastedHashCode(descEx.destFactor);

	if (this->isExState)
	{
		hash = hash * 29 + UIntCastedHashCode(descEx.blendOp);
		hash = hash * 29 + UIntCastedHashCode(descEx.srcAlphaFactor);
		hash = hash * 29 + UIntCastedHashCode(descEx.destAlphaFactor);
		hash = hash * 29 + UIntCastedHashCode(descEx.alphaBlendOp);
	}

	return hash;
}

bool HQEngineBlendStateWrapper::CreationParams::Equal(const CreationParams* params2) const
{
	bool equal = true;
	equal = equal && this->isExState == params2->isExState; 
	equal = equal && this->descEx.srcFactor == params2->descEx.srcFactor;
	equal = equal && this->descEx.destFactor == params2->descEx.destFactor;

	if (this->isExState)
	{
		equal = equal && descEx.blendOp == params2->descEx.blendOp;
		equal = equal && descEx.srcAlphaFactor == params2->descEx.srcAlphaFactor;
		equal = equal && descEx.destAlphaFactor == params2->descEx.destAlphaFactor;
		equal = equal && descEx.alphaBlendOp == params2->descEx.alphaBlendOp;
	}

	return equal;
}

HQEngineBlendStateWrapper::CreationParams& HQEngineBlendStateWrapper::CreationParams::operator = (const CreationParams& params2)
{
	this->isExState = params2.isExState;
	this->descEx = params2.descEx;

	return *this;
}


/*----------depth stencil state------------*/
HQEngineDSStateWrapper::HQEngineDSStateWrapper()
:stateID(0)
{
}
HQEngineDSStateWrapper::~HQEngineDSStateWrapper()
{
	if (stateID != 0)
		m_renderDevice->GetStateManager()->RemoveDepthStencilState(stateID);
}

HQReturnVal HQEngineDSStateWrapper::Init(const CreationParams& params)
{
	HQReturnVal re;
	if (params.isTwoSideState)
		re = m_renderDevice->GetStateManager()->CreateDepthStencilStateTwoSide(params.twoSideDesc, &stateID);
	else
		re = m_renderDevice->GetStateManager()->CreateDepthStencilState(params.desc, &stateID);
	
	if (re == HQ_OK)
	{
		this->creationParams = params;
	}
	return re;
}

HQReturnVal HQEngineDSStateWrapper::Active()
{
	return m_renderDevice->GetStateManager()->ActiveDepthStencilState(stateID);
}

hquint32 HQEngineDSStateWrapper::CreationParams::HashCode() const{
	hquint32 hash = UIntCastedHashCode(this->isTwoSideState);

	if (!this->isTwoSideState)
	{
		hash = hash * 29 + UIntCastedHashCode(desc.depthMode);
		hash = hash * 29 + UIntCastedHashCode(desc.stencilEnable);

		if (desc.stencilEnable)
		{
			hash = hash * 29 + UIntCastedHashCode(desc.readMask);
			hash = hash * 29 + UIntCastedHashCode(desc.writeMask);
			hash = hash * 29 + UIntCastedHashCode(desc.refVal);

			hash = hash * 29 + UIntCastedHashCode(desc.stencilMode.failOp);
			hash = hash * 29 + UIntCastedHashCode(desc.stencilMode.depthFailOp);
			hash = hash * 29 + UIntCastedHashCode(desc.stencilMode.passOp);
			hash = hash * 29 + UIntCastedHashCode(desc.stencilMode.compareFunc);
		}//if (desc.stencilEnable)
	}
	else
	{
		hash = hash * 29 + UIntCastedHashCode(twoSideDesc.depthMode);
		hash = hash * 29 + UIntCastedHashCode(twoSideDesc.stencilEnable);

		if (twoSideDesc.stencilEnable)
		{
			hash = hash * 29 + UIntCastedHashCode(twoSideDesc.readMask);
			hash = hash * 29 + UIntCastedHashCode(twoSideDesc.writeMask);
			hash = hash * 29 + UIntCastedHashCode(twoSideDesc.refVal);

			hash = hash * 29 + UIntCastedHashCode(twoSideDesc.cwFaceMode.failOp);
			hash = hash * 29 + UIntCastedHashCode(twoSideDesc.cwFaceMode.depthFailOp);
			hash = hash * 29 + UIntCastedHashCode(twoSideDesc.cwFaceMode.passOp);
			hash = hash * 29 + UIntCastedHashCode(twoSideDesc.cwFaceMode.compareFunc);

			hash = hash * 29 + UIntCastedHashCode(twoSideDesc.ccwFaceMode.failOp);
			hash = hash * 29 + UIntCastedHashCode(twoSideDesc.ccwFaceMode.depthFailOp);
			hash = hash * 29 + UIntCastedHashCode(twoSideDesc.ccwFaceMode.passOp);
			hash = hash * 29 + UIntCastedHashCode(twoSideDesc.ccwFaceMode.compareFunc);
		}//if (twoSideDesc.stencilEnable)
	}

	return hash;
}

bool HQEngineDSStateWrapper::CreationParams::Equal(const CreationParams* params2) const
{
	bool equal = true;
	equal = equal && this->isTwoSideState == params2->isTwoSideState;

	if (!this->isTwoSideState)
	{
		if (desc.stencilEnable)
		{
			equal = equal && memcmp(&this->desc, &params2->desc, sizeof(HQDepthStencilStateDesc)) == 0;
		}
	}
	else
	{
		if (twoSideDesc.stencilEnable)
		{
			equal = equal && memcmp(&this->twoSideDesc, &params2->twoSideDesc, sizeof(HQDepthStencilStateTwoSideDesc)) == 0;
		}
	}

	return equal;
}

HQEngineDSStateWrapper::CreationParams& HQEngineDSStateWrapper::CreationParams::operator = (const CreationParams& params2)
{
	this->isTwoSideState = params2.isTwoSideState;
	this->desc = params2.desc;
	this->twoSideDesc = params2.twoSideDesc;

	return *this;
}

/* --------------sampler state-----------------*/
HQEngineSamplerStateWrapper::HQEngineSamplerStateWrapper()
: stateID(0)
{
}
HQEngineSamplerStateWrapper::~HQEngineSamplerStateWrapper()
{
	if (stateID != 0)
		m_renderDevice->GetStateManager()->RemoveSamplerState(stateID);
}

HQReturnVal HQEngineSamplerStateWrapper::Init(const CreationParams& _desc)
{
	HQReturnVal re = m_renderDevice->GetStateManager()->CreateSamplerState(_desc, &stateID);
	if (re == HQ_OK)
		this->desc = _desc;

	return re;
}

HQReturnVal HQEngineSamplerStateWrapper::Apply(hquint32 index)
{
	return m_renderDevice->GetStateManager()->SetSamplerState(index, this->stateID);
}

hquint32 HQEngineSamplerStateWrapper::CreationParams::HashCode() const {
	hquint32 hash =  UIntCastedHashCode( this->filterMode);

	hash = hash * 29 + UIntCastedHashCode(this->addressU);
	hash = hash * 29 + UIntCastedHashCode(this->addressV);
	hash = hash * 29 + UIntCastedHashCode(this->maxAnisotropy);
	hash = hash * 29 + UIntCastedHashCode(this->borderColor);

	return hash;
}

bool HQEngineSamplerStateWrapper::CreationParams::Equal(const CreationParams* params2) const
{
	return !memcmp((HQSamplerStateDesc*)this, (HQSamplerStateDesc*)params2, sizeof(HQSamplerStateDesc));
}

HQEngineSamplerStateWrapper::CreationParams& HQEngineSamplerStateWrapper::CreationParams::operator = (const CreationParams& params2)
{
	memcpy((HQSamplerStateDesc*)this, (HQSamplerStateDesc*)&params2, sizeof(HQSamplerStateDesc));

	return *this;
}


/*--------depth stencil buffer---------------*/
HQEngineDSBufferWrapper::HQEngineDSBufferWrapper()
: bufferID(NULL)
{
}
HQEngineDSBufferWrapper::~HQEngineDSBufferWrapper()
{
	if (bufferID != NULL)
		m_renderDevice->GetRenderTargetManager()->RemoveDepthStencilBuffer(bufferID);
}

HQReturnVal HQEngineDSBufferWrapper::Init(const CreationParams &params)
{
	HQReturnVal re = m_renderDevice->GetRenderTargetManager()->CreateDepthStencilBuffer(
		params.width, params.height,
		params.format, HQ_MST_NONE,
		&bufferID);

	if (re == HQ_OK)
	{
		this->creationParams = params;
	}
	return re;
}

hquint32 HQEngineDSBufferWrapper::CreationParams::HashCode() const {
	hquint32 hash =  UIntCastedHashCode( this->format);

	hash = hash * 29 + UIntCastedHashCode(this->width);
	hash = hash * 29 + UIntCastedHashCode(this->height);

	return hash;
}

bool HQEngineDSBufferWrapper::CreationParams::Equal(const CreationParams* params2) const
{
	return format == params2->format 
		&& width == params2->width
		&& height == params2->height;
}

HQEngineDSBufferWrapper::CreationParams& HQEngineDSBufferWrapper::CreationParams::operator = (const CreationParams& params2)
{
	this->format = params2.format;
	this->width = params2.width;
	this->height = params2.height;

	return *this;
}

/*-----------------render target-----*/
hquint32 HQEngineRenderTargetWrapper::HashCode() const
{
	hquint32 hash =  SPtrHashCode( this->outputTexture);

	hash = hash * 29 + UIntCastedHashCode(this->cubeFace);

	return hash;
}

bool HQEngineRenderTargetWrapper::Equal(const HQEngineRenderTargetWrapper& rt2) const
{
	return this->outputTexture == rt2.outputTexture && this->cubeFace == rt2.cubeFace;
}

/*----------render targets group----------------*/
HQEngineRTGroupWrapper::HQEngineRTGroupWrapper()
:groupID(NULL)
{
}
HQEngineRTGroupWrapper::~HQEngineRTGroupWrapper()
{
	if (groupID != NULL)
	{
		m_renderDevice->GetRenderTargetManager()->RemoveRenderTargetGroup(groupID);
	}
}

HQReturnVal HQEngineRTGroupWrapper::Init(const CreationParams &params)
{
	HQRenderTargetDesc *outputDescs = HQ_NEW HQRenderTargetDesc[params.numOutputs];
	//init parameters for passing to render device
	for (hquint32 i = 0; i < params.numOutputs; ++i)
	{
		if (params.outputs[i].outputTexture == NULL)
			outputDescs[i].renderTargetID = NULL;
		else
			outputDescs[i].renderTargetID = params.outputs[i].outputTexture->GetRenderTargetView();
		outputDescs[i].cubeFace = (HQCubeTextureFace) params.outputs[i].cubeFace;
	}

	HQDepthStencilBufferView* depthstencilBufID = params.dsBuffer != NULL ? params.dsBuffer->bufferID : NULL;

	HQReturnVal re = m_renderDevice->GetRenderTargetManager()->CreateRenderTargetGroup(outputDescs, depthstencilBufID, params.numOutputs, &groupID);

	delete[] outputDescs;

	if (re == HQ_OK)
	{
		this->creationParams.numOutputs = params.numOutputs;
		this->creationParams.dsBuffer = params.dsBuffer;

		this->creationParams.outputs = HQ_NEW HQEngineRenderTargetWrapper[params.numOutputs];
		for (hquint32 i = 0; i < params.numOutputs; ++i)
			this->creationParams.outputs[i] = params.outputs[i];
	}

	return re;
}

HQReturnVal HQEngineRTGroupWrapper::Active()
{
	return m_renderDevice->GetRenderTargetManager()->ActiveRenderTargets(groupID);
}

HQEngineRTGroupWrapper::CreationParams::CreationParams()
: outputs(NULL) , numOutputs(0)
{
}

HQEngineRTGroupWrapper::CreationParams::~CreationParams()
{
}

hquint32 HQEngineRTGroupWrapper::CreationParams::HashCode() const{
	hquint32 hash =  SPtrHashCode( this->dsBuffer);
	
	for (hquint32 i = 0; i < this->numOutputs; ++i)
	{
		hash = hash * 29 + this->outputs[i].HashCode();
	}

	return hash;
}

bool HQEngineRTGroupWrapper::CreationParams::Equal(const CreationParams* params2) const
{
	bool equal = true;

	equal = equal && this->dsBuffer == params2->dsBuffer;
	equal = equal && this->numOutputs == params2->numOutputs;

	for (hquint32 i = 0; i < this->numOutputs; ++i)
	{
		equal = equal && this->outputs[i].Equal(params2->outputs[i]);
	}

	return equal;
}

/*------------rendering pass-----------------*/

//platform specific controller
class HQEngineRenderPassPlatformController {
public:
	virtual ~HQEngineRenderPassPlatformController(){}

	virtual void ApplyTextureStates(HQLinkedList<HQEngineTextureUnit >& textureUnits) = 0;
	virtual bool Validate(HQEngineBaseRenderPassImpl &pass, HQLoggableObject *logger) { return true; }
};

/*-----------base rendering pass class---------------*/
HQEngineBaseRenderPassImpl::HQEngineBaseRenderPassImpl(const char *name, HQEngineRenderPassPlatformController* _platformController)
:HQNamedGraphicsRelatedObj(name), platformController(_platformController)
{

}

HQEngineBaseRenderPassImpl::~HQEngineBaseRenderPassImpl(){

}

void HQEngineBaseRenderPassImpl::AddTextureUnit(const HQEngineTextureUnit& texunit)
{
	this->textureUnits.PushBack(texunit);
}

void HQEngineBaseRenderPassImpl::AddTextureUAVUnit(const HQEngineTextureUAVUnit& texunit)
{
	this->textureUAVUnits.PushBack(texunit);
}
void HQEngineBaseRenderPassImpl::AddBufferUAVSlot(const HQEngineShaderBufferUAVSlot& bufferSlot)
{
	this->bufferUAVSlots.PushBack(bufferSlot);
}

bool HQEngineBaseRenderPassImpl::Validate(HQLoggableObject *logger){
	return this->platformController->Validate(*this, logger);
}

void HQEngineBaseRenderPassImpl::ApplyTextureStates(){
	platformController->ApplyTextureStates(this->textureUnits);//delegating
}

HQReturnVal HQEngineBaseRenderPassImpl::Apply()
{
	//apply texture sampler states
	this->ApplyTextureStates();

	return HQ_OK;
}


#ifdef WIN32
#	pragma warning( push )
#	pragma warning( disable : 4250 )//dominance inheritance of GetName()
#endif

//normal rendering pipeline pass
struct HQEngineGraphicsPassImpl : public HQEngineBaseRenderPassImpl{
	HQEngineGraphicsPassImpl(const char* name, HQEngineRenderPassPlatformController* _platformController);
	virtual ~HQEngineGraphicsPassImpl();

	virtual HQReturnVal Apply();

	bool Validate(HQLoggableObject* logger);

	HQSharedPtr<HQEngineShaderProgramWrapper> shaderProgram;//shader program
	HQSharedPtr<HQEngineRTGroupWrapper> renderTargetGroup;//render targets. 
	HQSharedPtr<HQEngineBlendStateWrapper> blendState;//blend state
	HQSharedPtr<HQEngineDSStateWrapper> dsState;//depth stencil state

	HQCullMode faceCullingMode;

};

/*--------------compute pass----------------*/
struct HQEngineComputePassImpl : public HQEngineBaseRenderPassImpl{
	HQEngineComputePassImpl(const char* name, HQEngineRenderPassPlatformController* _platformController)
		: HQEngineBaseRenderPassImpl(name, _platformController)
	{
	}
	virtual ~HQEngineComputePassImpl()
	{
	}
	virtual HQReturnVal Apply();

	bool Validate(HQLoggableObject *logger);

	void SetShader(const HQSharedPtr<HQEngineShaderResImpl>& computeShader) { this->computeShader = computeShader; }

	HQSharedPtr<HQEngineShaderResImpl> computeShader;
};

#ifdef WIN32
#	pragma warning( pop )//dominance inheritance of GetName() warning
#endif

/*-------------HQEngineGraphicsPassImpl implementation ---------------*/
HQEngineGraphicsPassImpl::HQEngineGraphicsPassImpl(const char* name, HQEngineRenderPassPlatformController* _platformController)
:HQEngineBaseRenderPassImpl(name, _platformController)
{
}

HQEngineGraphicsPassImpl::~HQEngineGraphicsPassImpl()
{
}

bool HQEngineGraphicsPassImpl::Validate(HQLoggableObject* logger)
{
	//set default values
	if (this->shaderProgram == NULL) this->shaderProgram = HQ_NEW HQEngineShaderProgramWrapper();
	if (this->renderTargetGroup == NULL) this->renderTargetGroup = HQ_NEW HQEngineRTGroupWrapper();
	if (this->blendState == NULL) this->blendState = HQ_NEW HQEngineBlendStateWrapper();
	if (this->dsState == NULL) this->dsState = HQ_NEW HQEngineDSStateWrapper();

	//superclass's method
	return HQEngineBaseRenderPassImpl:: Validate(logger);
}


HQReturnVal HQEngineGraphicsPassImpl::Apply()
{
	/*-----------apply states, controlled texture units and more-------------*/
	this->m_renderDevice->GetStateManager()->SetFaceCulling(this->faceCullingMode);

	this->shaderProgram->Active();
	this->renderTargetGroup->Active();
	this->dsState->Active();
	this->blendState->Active();


	//apply texture UAV states
	HQLinkedList<HQEngineTextureUAVUnit >::Iterator ite;
	this->textureUAVUnits.GetIterator(ite);
	for (; !ite.IsAtEnd(); ++ite) //for each used texture unit
	{
		HQEngineTextureUAVUnit & unit = *ite;
		m_renderDevice->GetTextureManager()->SetTextureUAVForGraphicsShader(unit.unitIndex, unit.texture->GetTexture(), unit.mipLevel, unit.readable);
	}

	//apply buffer UAV states
	HQLinkedList<HQEngineShaderBufferUAVSlot >::Iterator ite2;
	this->bufferUAVSlots.GetIterator(ite2);
	for (; !ite2.IsAtEnd(); ++ite2) //for each used texture unit
	{
		HQEngineShaderBufferUAVSlot& slot = (*ite2);
		m_renderDevice->GetShaderManager()->SetBufferUAVForGraphicsShader(
			slot.slotIndex, slot.buffer->GetBuffer(), slot.firstElement, slot.numElements);
	}

	//super class's methd
	return HQEngineBaseRenderPassImpl::Apply();
}

/*-----------HQEngineComputePassImpl implementation-----------------*/
HQReturnVal HQEngineComputePassImpl::Apply()
{
	//activate shader
	m_renderDevice->GetShaderManager()->ActiveComputeShader(this->computeShader->GetShader());

	//apply texture UAV states
	HQLinkedList<HQEngineTextureUAVUnit >::Iterator ite;
	this->textureUAVUnits.GetIterator(ite);
	for (; !ite.IsAtEnd(); ++ite) //for each used texture unit
	{
		HQEngineTextureUAVUnit & unit = *ite;
		m_renderDevice->GetTextureManager()->SetTextureUAVForComputeShader(unit.unitIndex, unit.texture->GetTexture(), unit.mipLevel, unit.readable);
	}

	//apply buffer UAV states
	HQLinkedList<HQEngineShaderBufferUAVSlot >::Iterator ite2;
	this->bufferUAVSlots.GetIterator(ite2);
	for (; !ite2.IsAtEnd(); ++ite2) //for each used texture unit
	{
		HQEngineShaderBufferUAVSlot& slot = (*ite2);
		m_renderDevice->GetShaderManager()->SetBufferUAVForComputeShader(
			slot.slotIndex, slot.buffer->GetBuffer(), slot.firstElement, slot.numElements);
	}

	//super class's method
	return HQEngineBaseRenderPassImpl::Apply();
}

bool HQEngineComputePassImpl::Validate(HQLoggableObject *logger){
	if (this->computeShader == NULL)
	{
		logger->Log("Error: compute_pass '%s' has no compute shader!", m_name);
		return false;
	}

	//super class's method
	return HQEngineBaseRenderPassImpl::Validate(logger);
}

/*-------------platform specific controller-------------------*/
//D3D version of platform specific controller
class HQPlatformControllerD3D : public HQEngineRenderPassPlatformController, public HQGraphicsRelatedObj {
public:
	void ApplyTextureStates(HQLinkedList<HQEngineTextureUnit >& textureUnits)
	{
		HQLinkedList<HQEngineTextureUnit >::Iterator ite;
		textureUnits.GetIterator(ite);
		for (; !ite.IsAtEnd(); ++ite) //for each controlled texture unit
		{
			HQEngineTextureUnit & unit = *ite;
			//set sampler state
			unit.samplerState->Apply(unit.unitIndex);
			//set texture
			m_renderDevice->GetTextureManager()->SetTexture(unit.unitIndex, unit.texture->GetTexture());
		}
	}
};


//D3D11 version of platform specific controller
class HQPlatformControllerD3D11 : public HQPlatformControllerD3D {
public:

	//validate rendering/compute pass
	bool Validate(HQEngineBaseRenderPassImpl &pass, HQLoggableObject *logger)
	{
		bool success = true;

		HQClosedPrimeHashTable<std::string, HQSharedPtr<HQEngineTextureResImpl> > inputTextures;
		HQClosedPrimeHashTable<hquint32, std::string > usedUavSlots;
		//check for illegal use of buffer and texture as input and output

		/*------input textures-----------*/
		HQLinkedList<HQEngineTextureUnit >::Iterator input_tex_ite;
		pass.textureUnits.GetIterator(input_tex_ite);
		for (; !input_tex_ite.IsAtEnd(); ++input_tex_ite) //for each used texture unit
		{
			inputTextures.Add(input_tex_ite->texture->GetName(), input_tex_ite->texture);
		}

		/*----------output textures-------------*/
		HQLinkedList<HQEngineTextureUAVUnit >::Iterator output_tex_ite;
		pass.textureUAVUnits.GetIterator(output_tex_ite);
		for (; !output_tex_ite.IsAtEnd(); ++output_tex_ite) //for each used texture unit
		{
			usedUavSlots.Add(output_tex_ite->unitIndex, output_tex_ite->texture->GetName());
			if (inputTextures.GetItemPointer(output_tex_ite->texture->GetName()) != NULL)
			{
				logger->Log("Error: pass '%s' uses texture '%s' both in texture unit and UAV slot!", pass.GetName(), output_tex_ite->texture->GetName());
				success = false;
			}
		}

		//check for illegal binding buffer and texture to same UAV slot
		HQLinkedList<HQEngineShaderBufferUAVSlot >::Iterator buffer_ite;
		pass.bufferUAVSlots.GetIterator(buffer_ite);
		for (; !buffer_ite.IsAtEnd(); ++buffer_ite) //for each used buffer slot
		{
			std::string * boundTextureNamePtr = usedUavSlots.GetItemPointer(buffer_ite->slotIndex);
			if (boundTextureNamePtr != NULL)
			{
				logger->Log("Error: pass '%s' binds both buffer '%s' and texture '%s' to the same slot %u!"
					, pass.GetName(), buffer_ite->buffer->GetName(), boundTextureNamePtr->c_str(), buffer_ite->slotIndex);
				success = false;
			}
		}

		return success;
	}
};

//GL version of platform specific controller
class HQPlatformControllerGL : public HQEngineRenderPassPlatformController, public HQGraphicsRelatedObj {
public:
	void ApplyTextureStates(HQLinkedList<HQEngineTextureUnit >& textureUnits)
	{
		HQLinkedList<HQEngineTextureUnit >::Iterator ite;
		textureUnits.GetIterator(ite);
		for (; !ite.IsAtEnd(); ++ite) //for each controlled texture unit
		{
			HQEngineTextureUnit & unit = *ite;
			//set texture's sampler state
			unit.samplerState->Apply(unit.texture->GetTexture()->GetResourceIndex());
			//set texture
			m_renderDevice->GetTextureManager()->SetTexture(unit.unitIndex, unit.texture->GetTexture());
		}
	}
};

/*-------------rendering effect--------------------*/
HQEngineRenderEffectImpl::HQEngineRenderEffectImpl(const char* name, 
	HQClosedStringPrimeHashTable<HQSharedPtr<HQEngineBaseRenderPassImpl> >& passes)
: HQNamedGraphicsRelatedObj(name), m_passes(NULL), m_numPasses(0)
{
	m_numPasses = passes.GetNumItems();
	m_passes = HQ_NEW HQSharedPtr<HQEngineBaseRenderPassImpl>[m_numPasses];

	HQClosedStringPrimeHashTable<HQSharedPtr<HQEngineBaseRenderPassImpl> >::Iterator ite;
	passes.GetIterator(ite);
	
	//copy passes' info
	for (int i = 0; !ite.IsAtEnd(); ++ite, ++i){
		HQSharedPtr<HQEngineBaseRenderPassImpl> & pPass = *ite;
		m_passes[i] = pPass;
		m_passIdxMap.Add(pPass->GetName(), i);
	}
}
HQEngineRenderEffectImpl::~HQEngineRenderEffectImpl() 
{
	if (m_passes != NULL)
		delete[] m_passes;
}

HQEngineRenderPass* HQEngineRenderEffectImpl::GetPassByName(const char* name)
{
	bool found;
	hquint32 index = m_passIdxMap.GetItem(name, found);
	return this->GetPass(index);
}

hquint32 HQEngineRenderEffectImpl::GetPassIndexByName(const char* name)
{
	bool found;
	return m_passIdxMap.GetItem(name, found);
}

/*----------------effect loading session----------------*/
HQEngineEffectLoadSessionImpl::HQEngineEffectLoadSessionImpl(HQEngineEffectParserNode* root)
: m_root(root), m_type(HQ_EELT_STANDARD)
{
	if (m_root != NULL)
	{
		m_effectItem = NULL;
		m_effectGroup = m_root->GetFirstChild("techniques");
		while (m_effectItem == NULL && m_effectGroup != NULL)
		{
			m_effectItem = m_effectGroup->GetFirstChild("technique");
			if (m_effectItem == NULL)//try to find in next effect group
				m_effectGroup = m_effectGroup->GetNextSibling("techniques");
		} 
	}
	else
	{	
		m_effectGroup = m_effectItem = NULL;
	}
}

HQEngineEffectLoadSessionImpl::~HQEngineEffectLoadSessionImpl()
{
	HQEngineEffectParserNode::DeleteTree(m_root);
}

bool HQEngineEffectLoadSessionImpl::HasMoreEffects() const
{
	switch(m_type)
	{
	case HQ_EELT_STANDARD:
		if (m_root == NULL)
			return false;
		return (m_effectItem != NULL);

	default:
		return false;
	}
}

const HQEngineEffectParserNode * HQEngineEffectLoadSessionImpl::CurrentEffect()
{
	return m_type == HQ_EELT_STANDARD ? m_effectItem: NULL;
}

const HQEngineEffectParserNode * HQEngineEffectLoadSessionImpl::NextEffect() {
	if (m_type == HQ_EELT_STANDARD)
	{
		if (m_effectItem == NULL)
			return NULL;
		const HQEngineEffectParserNode * re = m_effectItem;

		m_effectItem = m_effectItem->GetNextSibling("technique");//advance to next item

		while (m_effectItem == NULL && m_effectGroup != NULL)//try to find in next group
		{
			m_effectGroup = m_effectGroup->GetNextSibling("techniques");
			if (m_effectGroup != NULL)
				m_effectItem = m_effectGroup->GetFirstChild("technique");
		}

		return re;
	}
	return NULL;
}

/*------------effect manager---------------*/

HQEngineEffectManagerImpl::HQEngineEffectManagerImpl(HQLogStream *stream, bool flushLog)
: HQLoggableObject(stream, "Engine's Effect Manager :", flushLog)
{
	m_pRDevice = HQEngineApp::GetInstance()->GetRenderDevice();
	//check if we are using OpenGL
	const char *renderer = m_pRDevice->GetDeviceDesc();
	if (strncmp(renderer, "OpenGL", 6) == 0)
	{
		m_platformController = HQ_NEW HQPlatformControllerGL();
		m_isGL = true;
	}
	else
	{
		if (strcmp(renderer, "Direct3D11") == 0)
			m_platformController = HQ_NEW HQPlatformControllerD3D11();
		else
			m_platformController = HQ_NEW HQPlatformControllerD3D();
		m_isGL = false;
	}
	/*------------init string to value mapping table--------------*/
	//none/cw/ccw
	m_cullModeMap.Add("none", HQ_CULL_NONE);
	m_cullModeMap.Add("cw", HQ_CULL_CW);
	m_cullModeMap.Add("ccw", HQ_CULL_CCW);

	//full/read only/write only/none
	m_depthModeMap.Add("full", HQ_DEPTH_FULL);
	m_depthModeMap.Add("read only", HQ_DEPTH_READONLY);
	m_depthModeMap.Add("write only", HQ_DEPTH_WRITEONLY);
	m_depthModeMap.Add("none", HQ_DEPTH_NONE);

	//keep/zero/replace/incre/decre/incre wrap/decre wrap
	m_stencilOpMap.Add("keep", HQ_SOP_KEEP);
	m_stencilOpMap.Add("zero", HQ_SOP_ZERO);
	m_stencilOpMap.Add("replace", HQ_SOP_REPLACE);
	m_stencilOpMap.Add("incre", HQ_SOP_INCR);
	m_stencilOpMap.Add("decre", HQ_SOP_DECR);
	m_stencilOpMap.Add("incre wrap", HQ_SOP_INCR_WRAP);
	m_stencilOpMap.Add("decre wrap", HQ_SOP_DECR_WRAP);

	//always/never/less/equal/less or equal/greater/greater or equal/not equal
	m_stencilFuncMap.Add("always", HQ_SF_ALWAYS);
	m_stencilFuncMap.Add("never", HQ_SF_NEVER);
	m_stencilFuncMap.Add("less", HQ_SF_LESS);
	m_stencilFuncMap.Add("equal", HQ_SF_EQUAL);
	m_stencilFuncMap.Add("less or equal", HQ_SF_LESS_EQUAL);
	m_stencilFuncMap.Add("greater", HQ_SF_GREATER);
	m_stencilFuncMap.Add("greater or equal", HQ_SF_GREATER_EQUAL);
	m_stencilFuncMap.Add("not equal", HQ_SF_NOT_EQUAL);

	//one/zero/src color/one minus src color/src alpha/one minus src alpha
	m_blendFactorMap.Add("one", HQ_BF_ONE);
	m_blendFactorMap.Add("zero", HQ_BF_ZERO);
	m_blendFactorMap.Add("src color", HQ_BF_SRC_COLOR);
	m_blendFactorMap.Add("one minus src color", HQ_BF_ONE_MINUS_SRC_COLOR);
	m_blendFactorMap.Add("src alpha", HQ_BF_SRC_ALPHA);
	m_blendFactorMap.Add("one minus src alpha", HQ_BF_ONE_MINUS_SRC_ALPHA);

	//add/sub/rev sub
	m_blendOpMap.Add("add", HQ_BO_ADD);
	m_blendOpMap.Add("sub", HQ_BO_SUBTRACT);
	m_blendOpMap.Add("rev sub", HQ_BO_REVSUBTRACT);

	//wrap/mirror/clamp/border
	m_taddrModeMap.Add("wrap", HQ_TAM_WRAP);
	m_taddrModeMap.Add("mirror", HQ_TAM_MIRROR);
	m_taddrModeMap.Add("clamp", HQ_TAM_CLAMP);
	m_taddrModeMap.Add("border", HQ_TAM_BORDER);

	//none/d24s8/d16/s8/d32
	m_dsFmtMap.Add("none", HQ_DSFMT_FORCE_DWORD);
	m_dsFmtMap.Add("d24s8", HQ_DSFMT_DEPTH_24_STENCIL_8);
	m_dsFmtMap.Add("d16", HQ_DSFMT_DEPTH_16);
	m_dsFmtMap.Add("s8", HQ_DSFMT_STENCIL_8);
	m_dsFmtMap.Add("d32", HQ_DSFMT_DEPTH_32);

	//texture filtering mode
	m_filterModeMap.Add("point_point_none", HQ_FM_MIN_MAG_POINT);
	m_filterModeMap.Add("point_linear_none", HQ_FM_MIN_POINT_MAG_LINEAR);
	m_filterModeMap.Add("linear_point_none", HQ_FM_MIN_LINEAR_MAG_POINT);
	m_filterModeMap.Add("linear_linear_none",HQ_FM_MIN_MAG_LINEAR);
	m_filterModeMap.Add("anisotropic_anisotropic_none", HQ_FM_MIN_MAG_ANISOTROPIC);
	m_filterModeMap.Add("point_point_point", HQ_FM_MIN_MAG_MIP_POINT);
	m_filterModeMap.Add("point_point_linear", HQ_FM_MIN_MAG_POINT_MIP_LINEAR);
	m_filterModeMap.Add("point_linear_point", HQ_FM_MIN_POINT_MAG_LINEAR_MIP_POINT);
	m_filterModeMap.Add("point_linear_linear", HQ_FM_MIN_POINT_MAG_MIP_LINEAR);
	m_filterModeMap.Add("linear_point_point", HQ_FM_MIN_LINEAR_MAG_MIP_POINT);
	m_filterModeMap.Add("linear_point_linear", HQ_FM_MIN_LINEAR_MAG_POINT_MIP_LINEAR);
	m_filterModeMap.Add("linear_linear_point", HQ_FM_MIN_MAG_LINEAR_MIP_POINT);
	m_filterModeMap.Add("linear_linear_linear", HQ_FM_MIN_MAG_MIP_LINEAR);
	m_filterModeMap.Add("anisotropic_anisotropic_anisotropic", HQ_FM_MIN_MAG_MIP_ANISOTROPIC);

	this->Log("Init done !");

}

HQEngineEffectManagerImpl::~HQEngineEffectManagerImpl()
{
	delete m_platformController;
	Log("Released!");
}

void HQEngineEffectManagerImpl::SetSuffix(const char* suffix)
{
	if (suffix)
		m_suffix = suffix;
}

HQReturnVal HQEngineEffectManagerImpl::AddEffectsFromFile(const char* fileName)
{
	HQEngineEffectLoadSession * session = this->BeginAddEffectsFromFile(fileName);
	if (session == NULL)
		return HQ_FAILED;
	
	HQReturnVal re = HQ_OK;

	while (this->HasMoreEffects(session)){
		if (this->AddNextEffect(session) != HQ_OK)
			re = HQ_FAILED;
	}

	this->EndAddEffects(session);
	

	return re;
}

HQEngineEffectLoadSession* HQEngineEffectManagerImpl::BeginAddEffectsFromFile(const char* raw_fileName)
{
	HQDataReaderStream* data_stream = HQEngineApp::GetInstance()->OpenFileForRead(raw_fileName);
	if (data_stream == NULL)
	{
		//try again with suffix
		std::string fileName = raw_fileName;
		HQEngineHelper::InsertFileNameSuffix(fileName, m_suffix);
		data_stream = HQEngineApp::GetInstance()->OpenFileForRead(fileName.c_str());
		if (data_stream == NULL)//still failed
		{
			this->Log("Error : Could not load effects from file %s! Could not open the file!", raw_fileName);
			return NULL;
		}
	}

	//prepare the parser
	std::stringstream log_stream;
	hqengine_effect_parser_input_stream = data_stream;
	hqengine_effect_parser_log_stream = &log_stream;

	//now parse the script
	if (hqengine_effect_parser_scan())
	{
		this->Log("Error : Could not load effect from file %s! %s", raw_fileName, hqengine_effect_parser_log_stream->str().c_str());
		HQEngineHelper::GlobalPoolReleaseAll();
		data_stream->Release();
		return NULL;
	}
	data_stream->Release();

	this->Log("Effects loading session from file '%s' started!", raw_fileName);

	HQEngineEffectParserNode *result = hqengine_effect_parser_root_result;
	hqengine_effect_parser_root_result = NULL;

	return HQ_NEW HQEngineEffectLoadSessionImpl(result);
}

bool HQEngineEffectManagerImpl::HasMoreEffects(HQEngineEffectLoadSession* session)
{
	HQEngineEffectLoadSessionImpl * effectLoadSession = static_cast <HQEngineEffectLoadSessionImpl*> (session);
	if (effectLoadSession == NULL)
		return false;
	return effectLoadSession->HasMoreEffects();
}

HQReturnVal HQEngineEffectManagerImpl::AddNextEffect(HQEngineEffectLoadSession* session)
{
	HQEngineEffectLoadSessionImpl * resLoadSession = static_cast <HQEngineEffectLoadSessionImpl*> (session);

	if (resLoadSession->HasMoreEffects() == false)
		return HQ_FAILED_NO_MORE_RESOURCE;
	switch (resLoadSession-> m_type)
	{
	case HQ_EELT_STANDARD:
		return this->LoadEffect(resLoadSession->NextEffect());
	break;
	} //switch (resLoadSession-> m_type)

	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::EndAddEffects(HQEngineEffectLoadSession* session)///for releasing loading session
{
	HQEngineEffectLoadSessionImpl * resLoadSession = static_cast <HQEngineEffectLoadSessionImpl*> (session);

	if (resLoadSession != NULL)
		delete resLoadSession;

	//release all memory blocks allocated for parser
	HQEngineHelper::GlobalPoolReleaseAll();

	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::LoadEffect(const HQEngineEffectParserNode * effectItem)
{
	int item_line = effectItem->GetSourceLine();
	const char *effect_name = effectItem->GetStrAttribute("name");
	if (effect_name == NULL)
	{
		Log("Error : %d : Could not load effect without any name!", item_line);
		return HQ_FAILED;
	}
	if (m_effects.GetItemPointer(effect_name) != NULL)
	{
		Log("Error : %d : Effect '%s' already exists!", effect_name, item_line);
		return HQ_FAILED_RESOURCE_EXISTS;
	}

	HQClosedStringPrimeHashTable<HQSharedPtr<HQEngineBaseRenderPassImpl> > passesTable;//rendering pass mapping table. TO DO: find a more efficient way to initialize the effect's passes

	const HQEngineEffectParserNode * pass = effectItem->GetFirstChild();
	for (; pass != NULL; pass = pass->GetNextSibling())
	{
		HQEngineBaseRenderPassImpl * newPass = NULL;
		const char *node_type = pass->GetType();
		const char *pass_name = pass->GetStrAttribute("name");
		int pass_line = pass->GetSourceLine();

		if (!strcmp(node_type, "pass")){

			//now start loading the pass
			HQEngineGraphicsPassImpl* newPassImpl = HQ_NEW HQEngineGraphicsPassImpl(pass_name, m_platformController);

			if (HQFailed(this->LoadPass(pass, newPassImpl)))
			{
				delete newPassImpl;
				return HQ_FAILED;
			}

			newPass = newPassImpl;

		}//if (!strcmp(pass->GetType(), "pass"))
		else if (!strcmp(node_type, "compute_pass")){
			//now start loading the pass
			HQEngineComputePassImpl* newPassImpl = HQ_NEW HQEngineComputePassImpl(pass_name, m_platformController);

			if (HQFailed(this->LoadComputePass(pass, newPassImpl)))
			{
				delete newPassImpl;
				return HQ_FAILED;
			}

			newPass = newPassImpl;
		}
		else {
			Log("Error : %d : Effect '%s' has invalid node '%s'!", pass_line, effect_name, node_type);
			return HQ_FAILED;
		}
		//insert to table
		if (newPass != NULL)
		{
			if (pass_name == NULL)
			{
				delete newPass;
				Log("Error : %d : Could not load a rendering pass from '%s' effect without any name!", pass_line, effect_name);
				return HQ_FAILED;
			}
			if (passesTable.GetItemPointer(pass_name) != NULL)
			{
				delete newPass;
				Log("Error : %d : Effect '%s' already has a pass named '%s'!", pass_line, effect_name, pass_name);
				return HQ_FAILED;
			}

			passesTable.Add(pass_name, newPass);
		}
	}//for (; attribute != NULL; attribute = attribute->GetNextSibling("pass"))

	//add new effect to table
	HQSharedPtr<HQEngineRenderEffectImpl> newEffect = HQ_NEW HQEngineRenderEffectImpl(effect_name, passesTable);
	
	m_effects.Add(effect_name, newEffect);


	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::LoadPass(const HQEngineEffectParserNode* passItem, HQEngineGraphicsPassImpl* newPass)
{
	const char emptyName[] = "";
	HQEngineResManagerImpl* resManager = static_cast<HQEngineResManagerImpl*> (HQEngineApp::GetInstance()->GetResourceManager());
	bool mappedValueFound = false;
	//default values
	HQCullMode faceCullMode = HQ_CULL_NONE;
	HQEngineShaderProgramWrapper::CreationParams shaderProgramParams;
	HQEngineDSStateWrapper::CreationParams dsStateParams;
	HQEngineBlendStateWrapper::CreationParams blendStateParams;
	HQEngineRTGroupWrapper::CreationParams rtGroupParams;
	bool useDefaultRT = true;
	bool useBlendState = false;

	const HQEngineEffectParserNode * elem = passItem->GetFirstChild();
	for (; elem != NULL; elem = elem->GetNextSibling())
	{
		const char *elemName = elem->GetType();
		int elem_line = elem->GetSourceLine();
		if (!strncmp(elemName, "texture_uav", 11))//texture UAV slot
		{
			hquint32 textureIdx;
			if (sscanf(elemName, "texture_uav%u", &textureIdx) == 0)
			{
				Log("Error : %d : invalid token %s!", elem_line, elemName);
				return HQ_FAILED;
			}

			HQEngineTextureUAVUnit texunit;
			texunit.unitIndex = textureIdx;

			if (HQFailed(this->ParseTextureUAVUnit(elem, texunit)))
				return HQ_FAILED;

			newPass->AddTextureUAVUnit(texunit);
		}//if (!strncmp(elemName, "texture_uav", 11))
		else if (!strncmp(elemName, "texture", 7))
		{
			hquint32 textureIdx;
			if (sscanf(elemName, "texture%u", &textureIdx) == 0)
			{
				Log("Error : %d : invalid token %s!", elem_line, elemName);
				return HQ_FAILED;
			}

			HQEngineTextureUnit texunit;
			if (m_isGL)
				texunit.unitIndex = textureIdx;
			else
				texunit.unitIndex = HQ_PIXEL_SHADER | textureIdx;

			if (HQFailed(this->ParseTextureUnit(elem, texunit)))
				return HQ_FAILED;

			newPass->AddTextureUnit(texunit);
		}//if (!strncmp(elemName, "texture", 7))
		else if (!strcmp(elemName, "blend"))
		{
			if (HQFailed(this->ParseBlendState(elem, blendStateParams)) )
				return HQ_FAILED;
			useBlendState = true;
		}
		else if (!strcmp(elemName, "stencil"))
		{
			if (HQFailed(this->ParseStencilState(elem, dsStateParams)) )
				return HQ_FAILED;
		}
		else if (!strcmp(elemName, "depth"))
		{
			const char *depthModeStr = elem->GetStrAttribute("value");
			dsStateParams.desc.depthMode = m_depthModeMap.GetItem(depthModeStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : %d : invalid depth=%s", elem_line, depthModeStr != NULL? depthModeStr: "");
				return HQ_FAILED;
			}
		}
		else if (!strcmp(elemName, "cull"))
		{
			const char *cullModeStr = elem->GetStrAttribute("value");
			faceCullMode = m_cullModeMap.GetItem(cullModeStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : %d : invalid cull=%s", elem_line, cullModeStr != NULL? cullModeStr : "");
				return HQ_FAILED;
			}
		}
		else if (!strcmp(elemName, "custom_targets"))
		{
			if (HQFailed(this->ParseRTGroup(elem, rtGroupParams)))
				return HQ_FAILED;
			useDefaultRT = false;
		}
		else if (!strcmp(elemName, "vertex_shader"))
		{
			const char *shader_res_name = elem->GetStrAttribute("value");
			if (shader_res_name == NULL)
				shader_res_name = emptyName;
			shaderProgramParams.vertexShader = resManager->GetShaderResourceSharedPtr(shader_res_name);
			if (shaderProgramParams.vertexShader == NULL)
			{
				Log("Error : %d : invalid vertex_shader=%s!", elem_line, shader_res_name);
				return HQ_FAILED;
			}
		}
		else if (!strcmp(elemName, "pixel_shader"))
		{
			const char *shader_res_name = elem->GetStrAttribute("value");
			if (shader_res_name == NULL)
				shader_res_name = emptyName;
			shaderProgramParams.pixelShader = resManager->GetShaderResourceSharedPtr(shader_res_name);
			if (shaderProgramParams.pixelShader == NULL)
			{
				Log("Error : %d : invalid pixel_shader=%s!", elem_line, shader_res_name);
				return HQ_FAILED;
			}
		}
		else if (!strcmp(elemName, "geometry_shader"))
		{
			const char *shader_res_name = elem->GetStrAttribute("value");
			if (shader_res_name == NULL)
				shader_res_name = emptyName;
			shaderProgramParams.geometryShader = resManager->GetShaderResourceSharedPtr(shader_res_name);
			if (shaderProgramParams.geometryShader == NULL)
			{
				Log("Error : %d : invalid geometry_shader=%s!", elem_line, shader_res_name);
				return HQ_FAILED;
			}
		}
		else if (!strncmp(elemName, "buffer_uav", 10))//bufer UAV slot
		{
			hquint32 bufferIdx;
			if (sscanf(elemName, "buffer_uav%u", &bufferIdx) == 0)
			{
				Log("Error : %d : invalid token %s!", elem_line, elemName);
				return HQ_FAILED;
			}

			HQEngineShaderBufferUAVSlot bufferSlot;
			bufferSlot.slotIndex = bufferIdx;

			if (HQFailed(this->ParseShaderBufferUAVUnit(elem, bufferSlot)))
				return HQ_FAILED;

			newPass->AddBufferUAVSlot(bufferSlot);
		}//else if (!strncmp(elemName, "buffer_uav", 10))
	}//for (; elem != NULL; elem = elem->GetNextSibling())

	//now create the program
	newPass->shaderProgram = this->CreateOrGetShaderProgram(shaderProgramParams);
	if (newPass->shaderProgram == NULL)
		return HQ_FAILED;
	
	//create render target group
	if (!useDefaultRT)
	{
		newPass->renderTargetGroup = this->CreateOrGetRTGroup(rtGroupParams);
		if (newPass->renderTargetGroup == NULL)
			return HQ_FAILED;
	}

	if (useBlendState)
	{
		//create blend state
		newPass->blendState = this->CreateOrGetBlendState(blendStateParams);
		if (newPass->blendState == NULL)
			return HQ_FAILED;
	}
	//create depth stencil state
	newPass->dsState = this->CreateOrGetDSState(dsStateParams);
	if (newPass->dsState == NULL)
		return HQ_FAILED;

	//face culling mode
	newPass->faceCullingMode = faceCullMode;

	//finalize
	if (!newPass->Validate(this))
		return HQ_FAILED;

	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::LoadComputePass(const HQEngineEffectParserNode* passItem, HQEngineComputePassImpl *newPass)
{
	HQEngineResManagerImpl* resManager = static_cast<HQEngineResManagerImpl*> (HQEngineApp::GetInstance()->GetResourceManager());
	const char emptyName[] = "";
	const HQEngineEffectParserNode * elem = passItem->GetFirstChild();
	for (; elem != NULL; elem = elem->GetNextSibling())
	{
		const char *elemName = elem->GetType();
		int elem_line = elem->GetSourceLine();
		if (!strncmp(elemName, "texture_uav", 11))//texture UAV slot
		{
			hquint32 textureIdx;
			if (sscanf(elemName, "texture_uav%u", &textureIdx) == 0)
			{
				Log("Error : %d : invalid token %s!", elem_line, elemName);
				return HQ_FAILED;
			}

			HQEngineTextureUAVUnit texunit;
			texunit.unitIndex = textureIdx;

			if (HQFailed(this->ParseTextureUAVUnit(elem, texunit)))
				return HQ_FAILED;

			newPass->AddTextureUAVUnit(texunit);
		}//if (!strncmp(elemName, "texture_uav", 11))
		else if (!strncmp(elemName, "texture", 7))//texture unit
		{
			hquint32 textureIdx;
			if (sscanf(elemName, "texture%u", &textureIdx) == 0)
			{
				Log("Error : %d : invalid token %s!", elem_line, elemName);
				return HQ_FAILED;
			}

			HQEngineTextureUnit texunit;
			if (m_isGL)
				texunit.unitIndex = textureIdx;
			else
				texunit.unitIndex = HQ_COMPUTE_SHADER | textureIdx;

			if (HQFailed(this->ParseTextureUnit(elem, texunit)))
				return HQ_FAILED;

			newPass->AddTextureUnit(texunit);
		}//else if (!strncmp(elemName, "texture", 7))
		else if (!strncmp(elemName, "buffer_uav", 10))//bufer UAV slot
		{
			hquint32 bufferIdx;
			if (sscanf(elemName, "buffer_uav%u", &bufferIdx) == 0)
			{
				Log("Error : %d : invalid token %s!", elem_line, elemName);
				return HQ_FAILED;
			}

			HQEngineShaderBufferUAVSlot bufferSlot;
			bufferSlot.slotIndex = bufferIdx;

			if (HQFailed(this->ParseShaderBufferUAVUnit(elem, bufferSlot)))
				return HQ_FAILED;

			newPass->AddBufferUAVSlot(bufferSlot);
		}//else if (!strncmp(elemName, "buffer_uav", 10))
		else if (!strcmp(elemName, "shader"))//compute shader
		{
			const char *shader_res_name = elem->GetStrAttribute("value");
			if (shader_res_name == NULL)
				shader_res_name = emptyName;
			HQSharedPtr<HQEngineShaderResImpl> shader = resManager->GetShaderResourceSharedPtr(shader_res_name);
			if (shader == NULL)
			{
				Log("Error : %d : invalid shader=%s!", elem_line, shader_res_name);
				return HQ_FAILED;
			}
			else if (shader->GetShaderType() != HQ_COMPUTE_SHADER){
				Log("Error : %d : shader=%s is not compute shader!", elem_line, shader_res_name);
				return HQ_FAILED;
			}

			newPass->SetShader(shader);
		}//else if (!strcmp(elemName, "shader"))
	}//for (; elem != NULL; elem = elem->GetNextSibling())

	if (!newPass->Validate(this))
		return HQ_FAILED;
	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::ParseTextureUnit(const HQEngineEffectParserNode* textureUnitElem, HQEngineTextureUnit& texunit)
{
	HQEngineResManagerImpl* resManager = static_cast<HQEngineResManagerImpl*> (HQEngineApp::GetInstance()->GetResourceManager());
	const char emptyName[] = "";
	int elem_line = textureUnitElem->GetSourceLine();

	//get sampler state
	HQEngineSamplerStateWrapper::CreationParams samplerParams;
	if (HQFailed(this->ParseSamplerState(textureUnitElem, samplerParams)))
		return HQ_FAILED;

	HQSharedPtr<HQEngineSamplerStateWrapper> samplerState = this->CreateOrGetSamplerState(samplerParams);
	if (samplerState == NULL)
		return HQ_FAILED;


	//get texture resource
	const HQEngineEffectParserNode* textureResNameInfo = textureUnitElem->GetFirstChild("source");
	if (textureResNameInfo == NULL)
	{
		Log("Error: %d : texture unit missing texture resource!", elem_line);
		return HQ_FAILED;
	}

	const char * textureResName = textureResNameInfo->GetStrAttribute("value");
	if (textureResName == NULL)
		textureResName = emptyName;

	const HQSharedPtr<HQEngineTextureResImpl> texture = resManager->GetTextureResourceSharedPtr(textureResName);
	if (texture == NULL)
	{
		Log("Error: %d : invalid texture resource named %s!", elem_line, textureResName);
		return HQ_FAILED;
	}
	//controlled texture unit
	texunit.texture = texture;
	texunit.samplerState = samplerState;

	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::ParseTextureUAVUnit(const HQEngineEffectParserNode* textureUnitElem, HQEngineTextureUAVUnit& texUAVUnit)
{
	HQEngineResManagerImpl* resManager = static_cast<HQEngineResManagerImpl*> (HQEngineApp::GetInstance()->GetResourceManager());
	int elem_line = textureUnitElem->GetSourceLine();
	hquint32 mipLevel = 0;
	bool readable = false;
	const char * textureResName = NULL;

	const HQEngineEffectParserNode * attribute = textureUnitElem->GetFirstChild();
	for (; attribute != NULL; attribute = attribute->GetNextSibling())
	{
		int attriLine = attribute->GetSourceLine();
		const char* attriName = attribute->GetType();
		const HQEngineEffectParserNode::ValueType& attriVal = attribute->GetAttribute("value");
		const char* attriValStr = attriVal.GetAsStringEmptyIfNone();

		if (!strcmp(attriName, "source"))
		{
			textureResName = attriValStr;
		}
		else if (!strcmp(attriName, "mip_level"))
		{
			const hqint32* valPtr = attriVal.GetAsIntPt();
			if (valPtr == NULL)
			{
				Log("Error: %d : invalid mip_level='%s'!", attriValStr);
				return HQ_FAILED;
			}

			mipLevel = *valPtr;
		}//else if (!strcmp(attriName, "mip_level"))
		else if (!strcmp(attriName, "readable"))
		{
			readable = (!strcmp(attriValStr, "true"));
		}
	}//for (; attribute != NULL; attribute = attribute->GetNextSibling())

	//get texture resource
	if (textureResName == NULL)
	{
		Log("Error: %d : texture UAV unit missing texture resource!", elem_line);
		return HQ_FAILED;
	}
	const HQSharedPtr<HQEngineTextureResImpl> texture = resManager->GetTextureResourceSharedPtr(textureResName);
	if (texture == NULL)
	{
		Log("Error: %d : invalid texture resource named %s!", elem_line, textureResName);
		return HQ_FAILED;
	}

	//store texture UAV unit
	texUAVUnit.texture = texture;
	texUAVUnit.mipLevel = mipLevel;
	texUAVUnit.readable = readable;

	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::ParseShaderBufferUAVUnit(const HQEngineEffectParserNode* bufferSlotElem, HQEngineShaderBufferUAVSlot& bufferUAVUnit)
{
	HQEngineResManagerImpl* resManager = static_cast<HQEngineResManagerImpl*> (HQEngineApp::GetInstance()->GetResourceManager());
	int elem_line = bufferSlotElem->GetSourceLine();
	hquint32 firstElement = 0;
	hquint32 numElements = 1;
	const char * bufferResName = NULL;

	const HQEngineEffectParserNode * attribute = bufferSlotElem->GetFirstChild();
	for (; attribute != NULL; attribute = attribute->GetNextSibling())
	{
		int attriLine = attribute->GetSourceLine();
		const char* attriName = attribute->GetType();
		const HQEngineEffectParserNode::ValueType& attriVal = attribute->GetAttribute("value");
		const char* attriValStr = attriVal.GetAsStringEmptyIfNone();

		if (!strcmp(attriName, "source"))
		{
			bufferResName = attriValStr;
		}
		else if (!strcmp(attriName, "first_element"))
		{
			const hqint32* valPtr = attriVal.GetAsIntPt();
			if (valPtr == NULL)
			{
				Log("Error: %d : invalid first element index='%s'!", attriValStr);
				return HQ_FAILED;
			}

			firstElement = *valPtr;
		}//else if (!strcmp(attriName, "first_element"))
		else if (!strcmp(attriName, "num_elements"))
		{
			const hqint32* valPtr = attriVal.GetAsIntPt();
			if (valPtr == NULL)
			{
				Log("Error: %d : invalid number of elements='%s'!", attriValStr);
				return HQ_FAILED;
			}

			numElements = *valPtr;
		}
	}//for (; attribute != NULL; attribute = attribute->GetNextSibling())

	//get buffer resource
	if (bufferResName == NULL)
	{
		Log("Error: %d : buffer UAV slot missing buffer resource!", elem_line);
		return HQ_FAILED;
	}
	const HQSharedPtr<HQEngineShaderBufferResImpl> buffer = resManager->GetShaderBufferResourceSharedPtr(bufferResName);
	if (buffer == NULL)
	{
		Log("Error: %d : invalid buffer resource named %s!", elem_line, bufferResName);
		return HQ_FAILED;
	}

	//store texture UAV unit
	bufferUAVUnit.buffer = buffer;
	bufferUAVUnit.firstElement = firstElement;
	bufferUAVUnit.numElements = numElements;

	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::ParseStencilState(const HQEngineEffectParserNode *stencilElem, HQEngineDSStateWrapper::CreationParams &params)
{
	bool mappedValueFound = false;
	HQStencilOp op;
	HQStencilFunc func;
	bool has_ccw_fail_op = stencilElem->GetFirstChild("ccw_fail_op") != NULL;
	bool has_ccw_depth_fail_op = stencilElem->GetFirstChild("ccw_depth_fail_op") != NULL;
	bool has_ccw_pass_op = stencilElem->GetFirstChild("ccw_pass_op") != NULL;
	bool has_ccw_compare_func = stencilElem->GetFirstChild("ccw_compare_func") != NULL;
	//find if there is "ccw*" elements
	if (has_ccw_fail_op
		|| has_ccw_depth_fail_op
		|| has_ccw_pass_op
		|| has_ccw_compare_func)
		params.isTwoSideState = true;
	else
		params.isTwoSideState = false;

	if (!params.isTwoSideState)
	{
		//set default values
		params.desc.stencilEnable = true;
		params.desc.readMask = 0xffffffff;
		params.desc.writeMask = 0xffffffff;
		params.desc.refVal = 0;
		params.desc.stencilMode.failOp = params.desc.stencilMode.depthFailOp = params.desc.stencilMode.passOp = HQ_SOP_KEEP;
		params.desc.stencilMode.compareFunc = HQ_SF_ALWAYS;

		const HQEngineEffectParserNode * attribute = stencilElem->GetFirstChild();
		for (; attribute != NULL; attribute = attribute->GetNextSibling())
		{
			int attriLine = attribute->GetSourceLine();
			const char* attriName = attribute->GetType();
			const HQEngineEffectParserNode::ValueType& attriVal = attribute->GetAttribute("value");
			const char* attriValStr = attriVal.GetAsStringEmptyIfNone();

			if (!strcmp(attriName, "read_mask"))
			{
				if (attriVal.GetAsIntPt() != NULL)
					params.desc.readMask = (hquint32)*attriVal.GetAsIntPt(); 
				else
				{
					Log("Error : %d : invalid read_mask!", attriLine);
					return HQ_FAILED;
				}
			}
			else if (!strcmp(attriName, "write_mask"))
			{
				if (attriVal.GetAsIntPt() != NULL)
					params.desc.writeMask = (hquint32)*attriVal.GetAsIntPt(); 
				else
				{
					Log("Error : %d : invalid write_mask!", attriLine);
					return HQ_FAILED;
				}
			}
			else if (!strcmp(attriName, "reference_value"))
			{
				if (attriVal.GetAsIntPt() != NULL)
					params.desc.refVal = (hquint32)*attriVal.GetAsIntPt(); 
				else
				{
					Log("Error : %d : invalid reference_value!", attriLine);
					return HQ_FAILED;
				}
			}
			else if (!strcmp(attriName, "fail_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : %d : invalid fail_op=%s!", attriLine, attriValStr);
					return HQ_FAILED;
				}
				params.desc.stencilMode.failOp = op;
			}//else if (!strcmp(attriName, "fail_op"))
			else if (!strcmp(attriName, "depth_fail_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : %d : invalid depth_fail_op=%s!", attriLine, attriValStr);
					return HQ_FAILED;
				}
				params.desc.stencilMode.depthFailOp = op;
			}
			else if (!strcmp(attriName, "pass_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : %d : invalid pass_op=%s!", attriLine, attriValStr);
					return HQ_FAILED;
				}
				params.desc.stencilMode.passOp = op;
			}
			else if (!strcmp(attriName, "compare_func"))
			{
				func = m_stencilFuncMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : %d : invalid compare_func=%s!", attriLine, attriValStr);
					return HQ_FAILED;
				}
				params.desc.stencilMode.compareFunc = func;
			}
		}//for (; attribute != NULL; attribute = attribute->GetNextSibling())
	}//if (!params.isTwoSideState)
	else
	{
		//set default values
		params.twoSideDesc.stencilEnable = true;
		params.twoSideDesc.readMask = 0xffffffff;
		params.twoSideDesc.writeMask = 0xffffffff;
		params.twoSideDesc.refVal = 0;
		params.twoSideDesc.cwFaceMode.failOp = params.twoSideDesc.cwFaceMode.depthFailOp 
			= params.twoSideDesc.cwFaceMode.passOp = HQ_SOP_KEEP;
		params.twoSideDesc.cwFaceMode.compareFunc = HQ_SF_ALWAYS;

		const HQEngineEffectParserNode * attribute = stencilElem->GetFirstChild();
		for (; attribute != NULL; attribute = attribute->GetNextSibling())
		{
			int attriLine = attribute->GetSourceLine();
			const char* attriName = attribute->GetType();
			const HQEngineEffectParserNode::ValueType& attriVal = attribute->GetAttribute("value");
			const char* attriValStr = attriVal.GetAsStringEmptyIfNone();

			if (!strcmp(attriName, "read_mask"))
			{
				if (attriVal.GetAsIntPt() != NULL)
					params.twoSideDesc.readMask = (hquint32)*attriVal.GetAsIntPt(); 
				else
				{
					Log("Error : %d : invalid read_mask!", attriLine);
					return HQ_FAILED;
				}
			}
			else if (!strcmp(attriName, "write_mask"))
			{
				if (attriVal.GetAsIntPt() != NULL)
					params.twoSideDesc.writeMask = (hquint32)*attriVal.GetAsIntPt(); 
				else
				{
					Log("Error : %d : invalid write_mask!", attriLine);
					return HQ_FAILED;
				}
			}
			else if (!strcmp(attriName, "reference_value"))
			{
				if (attriVal.GetAsIntPt() != NULL)
					params.twoSideDesc.refVal = (hquint32)*attriVal.GetAsIntPt(); 
				else
				{
					Log("Error : %d : invalid reference_value!", attriLine);
					return HQ_FAILED;
				}
			}
			else if (!strcmp(attriName, "fail_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : %d : invalid fail_op=%s!", attriLine, attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.cwFaceMode.failOp = op;
			}//else if (!strcmp(attriName, "fail_op"))
			else if (!strcmp(attriName, "depth_fail_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : %d : invalid depth_fail_op=%s!", attriLine, attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.cwFaceMode.depthFailOp = op;
			}
			else if (!strcmp(attriName, "pass_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : %d : invalid pass_op=%s!", attriLine, attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.cwFaceMode.passOp = op;
			}
			else if (!strcmp(attriName, "compare_func"))
			{
				func = m_stencilFuncMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : %d : invalid compare_func=%s!", attriLine, attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.cwFaceMode.compareFunc = func;
			}
			else if (!strcmp(attriName, "ccw_fail_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : %d : invalid ccw_fail_op=%s!", attriLine, attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.ccwFaceMode.failOp = op;
			}//else if (!strcmp(attriName, "fail_op"))
			else if (!strcmp(attriName, "ccw_depth_fail_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : %d : invalid ccw_depth_fail_op=%s!", attriLine, attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.ccwFaceMode.depthFailOp = op;
			}
			else if (!strcmp(attriName, "ccw_pass_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : %d : invalid ccw_pass_op=%s!", attriLine, attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.ccwFaceMode.passOp = op;
			}
			else if (!strcmp(attriName, "ccw_compare_func"))
			{
				func = m_stencilFuncMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : %d : invalid ccw_compare_func=%s!", attriLine, attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.ccwFaceMode.compareFunc = func;
			}
		}//for (; attribute != NULL; attribute = attribute->GetNextSibling())

		//set default ccwFaceMode
		if (!has_ccw_fail_op)
			params.twoSideDesc.ccwFaceMode.failOp = params.twoSideDesc.cwFaceMode.failOp;
		if (!has_ccw_depth_fail_op)
			params.twoSideDesc.ccwFaceMode.depthFailOp = params.twoSideDesc.cwFaceMode.depthFailOp;
		if (!has_ccw_pass_op)
			params.twoSideDesc.ccwFaceMode.passOp = params.twoSideDesc.cwFaceMode.passOp;
		if (!has_ccw_compare_func)
			params.twoSideDesc.ccwFaceMode.compareFunc = params.twoSideDesc.cwFaceMode.compareFunc;
	}//else of if (!params.isTwoSideState)

	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::ParseBlendState(const HQEngineEffectParserNode* blendElem, HQEngineBlendStateWrapper::CreationParams &params)
{
	HQBlendOp op;
	HQBlendFactor factor;
	bool mappedValueFound = false;
	bool alphaOpOverride = false;
	bool alphaSrcFactorOverride = false;
	bool alphaDstFactorOverride = false;

	//initialize default values
	params.isExState = false;
	params.descEx = HQBlendStateExDesc();

	const HQEngineEffectParserNode * attribute = blendElem->GetFirstChild();
	for (; attribute != NULL; attribute = attribute->GetNextSibling())
	{
		int attriLine = attribute->GetSourceLine();
		const char* attriName = attribute->GetType();
		const HQEngineEffectParserNode::ValueType& attriVal = attribute->GetAttribute("value");
		const char* attriValStr = attriVal.GetAsStringEmptyIfNone();

		if (!strcmp(attriName, "src_factor"))
		{
			factor = m_blendFactorMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : %d : invalid src_factor=%s", attriLine, attriValStr);
				return HQ_FAILED;
			}
			params.descEx.srcFactor = factor;
		}
		else if (!strcmp(attriName, "dest_factor"))
		{
			factor = m_blendFactorMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : %d : invalid dest_factor=%s", attriLine, attriValStr);
				return HQ_FAILED;
			}
			params.descEx.destFactor = factor;
		}
		else if (!strcmp(attriName, "operator"))
		{
			op = m_blendOpMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : %d : invalid operator=%s", attriLine, attriValStr);
				return HQ_FAILED;
			}
			params.isExState = true; params.descEx.blendOp = op;
		}
		else if (!strcmp(attriName, "alpha_operator"))
		{
			op = m_blendOpMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : %d: invalid alpha_operator=%s", attriLine, attriValStr);
				return HQ_FAILED;
			}
			params.isExState = true; params.descEx.alphaBlendOp = op; 
			alphaOpOverride = true;
		}
		else if (!strcmp(attriName, "alpha_src_factor"))
		{
			factor = m_blendFactorMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : %d : invalid alpha_src_factor=%s", attriLine, attriValStr);
				return HQ_FAILED;
			}
			params.isExState = true; params.descEx.srcAlphaFactor = factor;
			alphaSrcFactorOverride = true;
		}
		else if (!strcmp(attriName, "alpha_dest_factor"))
		{
			factor = m_blendFactorMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : %d : invalid alpha_dest_factor=%s", attriLine, attriValStr);
				return HQ_FAILED;
			}
			params.isExState = true; params.descEx.destAlphaFactor = factor;
			alphaDstFactorOverride = true;
		}
	}//for (; attribute != NULL; attribute = attribute->GetNextSibling())

	if (params.isExState)
	{
		//check if we need to use default values
		if (!alphaOpOverride)
			params.descEx.alphaBlendOp = params.descEx.blendOp;
		if (!alphaSrcFactorOverride)
			params.descEx.srcAlphaFactor = params.descEx.srcFactor;
		if (!alphaDstFactorOverride)
			params.descEx.destAlphaFactor = params.descEx.destFactor;
	}

	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::ParseSamplerState(const HQEngineEffectParserNode* textureElem, HQEngineSamplerStateWrapper::CreationParams &params)
{
	HQTexAddressMode tAddrMode;
	const char defaultfilter[] = "linear";
	const char defaultMipfilter[] = "none";
	const char anisotropicStr[] = "anisotropic";
	const char *min_filter_str = defaultfilter;
	const char *mag_filter_str = defaultfilter;
	const char *mip_filter_str = defaultMipfilter;
	bool mappedValueFound = false;

	//initialize default values
	params = HQEngineSamplerStateWrapper::CreationParams();

	const HQEngineEffectParserNode * attribute = textureElem->GetFirstChild();
	for (; attribute != NULL; attribute = attribute->GetNextSibling())
	{
		int attriLine = attribute->GetSourceLine();
		const char* attriName = attribute->GetType();
		const HQEngineEffectParserNode::ValueType& attriVal = attribute->GetAttribute("value");
		const char* attriValStr = attriVal.GetAsStringEmptyIfNone();

		if (!strcmp(attriName, "address_u"))
		{
			tAddrMode = m_taddrModeMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : %d : invalid address_u=%s", attriLine, attriValStr);
				return HQ_FAILED;
			}
			params.addressU = tAddrMode;
		}
		else if (!strcmp(attriName, "address_v"))
		{
			tAddrMode = m_taddrModeMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : %d : invalid address_v=%s", attriLine, attriValStr);
				return HQ_FAILED;
			}
			params.addressV = tAddrMode;
		}
		else if (!strcmp(attriName, "max_anisotropy"))
		{
			if (attriVal.GetAsIntPt() != NULL)
				params.maxAnisotropy = (hquint32)*attriVal.GetAsIntPt(); 
			else
			{
				Log("Error : %d : invalid max_anisotropy!", attriLine);
				return HQ_FAILED;
			}
		}
		else if (!strcmp(attriName, "border_color")) {
			hquint32 color;

			if (attriVal.GetAsIntPt() != NULL)
				color = (hquint32)*attriVal.GetAsIntPt(); 
			else
			{
				Log("Error : %d : invalid border_color!", attriLine);
				return HQ_FAILED;
			}

			hqubyte8 r = (color >> 24) & 0xff;
			hqubyte8 g = (color >> 16) & 0xff;
			hqubyte8 b = (color >> 8) & 0xff;
			hqubyte8 a = (color) & 0xff;
			params.borderColor = HQColorRGBAi(r, g, b, a);
		}
		else if (!strcmp(attriName, "min_filter")){
			min_filter_str = attriValStr;
		}
		else if (!strcmp(attriName, "mag_filter")){
			mag_filter_str = attriValStr;
		}
		else if (!strcmp(attriName, "mipmap_filter"))
		{
			mip_filter_str = attriValStr;
		}
	}//for (; attribute != NULL; attribute = attribute->GetNextSibling())

#if defined HQ_WIN_PHONE_PLATFORM//require mipmap in phone devices
	if (strcmp(mip_filter_str, "none") == 0)
	{
		Log("Warning: mipmap_filter='none' is not supported on this platform!");
		mip_filter_str = defaultfilter;
	}
#endif

	//if one mode is "anisotropic", the others must be "anisotropic" as well
	if (strcmp(min_filter_str, "anisotropic") == 0 ||
		strcmp(mag_filter_str, "anisotropic") == 0 ||
		strcmp(mip_filter_str, "anisotropic") == 0)
	{
		min_filter_str = mag_filter_str = anisotropicStr;
		if (strcmp(mip_filter_str, "none") != 0)
			mip_filter_str = anisotropicStr;
	}

	std::string filterModeStr = min_filter_str;
	filterModeStr += "_";
	filterModeStr += mag_filter_str;
	filterModeStr += "_";
	filterModeStr += mip_filter_str;
	
	HQFilterMode filterMode = m_filterModeMap.GetItem(filterModeStr, mappedValueFound);
	if (mappedValueFound)
		params.filterMode = filterMode;
	else
	{
		Log("Error: %d : unrecognized min-filer/mag-filter/mipnap-filter!", textureElem->GetSourceLine());
		return HQ_FAILED;
	}
	
	
	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::ParseDepthStencilBuffer(const HQEngineEffectParserNode* dsBufElem, HQEngineDSBufferWrapper::CreationParams &params)
{
	HQDepthStencilFormat format;
	bool mappedValueFound = false;

	format = m_dsFmtMap.GetItem(dsBufElem->GetStrAttributeEmptyIfNone("value"), mappedValueFound);
	if (!mappedValueFound)
	{
		Log("Error : %d : invalid depth_stencil_buffer_format!", dsBufElem->GetSourceLine());
		return HQ_FAILED;
	}

	params.format = format;

	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::ParseRTGroup(const HQEngineEffectParserNode* rtGroupElem, HQEngineRTGroupWrapper::CreationParams &params)
{
	HQEngineResManagerImpl* resManager = static_cast<HQEngineResManagerImpl*> (HQEngineApp::GetInstance()->GetResourceManager());
	HQEngineDSBufferWrapper::CreationParams dsBufferParams;
	bool useDSBuffer = false;//default is not using depth stencil buffer
	hquint32 maxNumRTs = m_pRDevice->GetMaxActiveRenderTargets();
	HQSharedArrayPtr<HQEngineRenderTargetWrapper> outputArray = HQ_NEW HQEngineRenderTargetWrapper[maxNumRTs];//auto release array
	hquint32 numOutputs = 0;
	hquint32 global_width = 0xffffffff;
	hquint32 global_height = 0xffffffff;
	hquint32 width, height;

	const HQEngineEffectParserNode * attribute = rtGroupElem->GetFirstChild();
	for (; attribute != NULL; attribute = attribute->GetNextSibling())
	{
		int attriLine = attribute->GetSourceLine();
		const char* attriName = attribute->GetType();
		const HQEngineEffectParserNode::ValueType& attriVal = attribute->GetAttribute("value");
		const char* attriValStr = attriVal.GetAsStringEmptyIfNone();

		if (!strcmp(attriName, "depth_stencil_buffer_format"))
		{
			useDSBuffer = true;
			if (HQFailed(this->ParseDepthStencilBuffer(attribute, dsBufferParams)))
				return HQ_FAILED;
		}
		else if (!strncmp(attriName, "output", 6))
		{
			hquint32 targetIdx = 0;
			HQEngineRenderTargetWrapper rtOutput;
			const char defaultCubeFaceStr[] = "+x";
			const char* cubeFaceStr = defaultCubeFaceStr;
			if (sscanf(attriName, "output%u", &targetIdx) != 1)
			{
				Log("Error : %d : invald token %s!", attriLine, attriName);
				return HQ_FAILED;
			}

			//now get the texture resource
			rtOutput.outputTexture = resManager->GetTextureResourceSharedPtr(attriValStr);
			if (rtOutput.outputTexture == NULL || rtOutput.outputTexture->IsRenderTarget() == false)
			{
				Log("Error : %d : invald render target resource=%s!", attriLine, attriValStr);
				return HQ_FAILED;
			}

			if (rtOutput.outputTexture->GetTexture()->GetType() == HQ_TEXTURE_CUBE)
			{
				//get cube face
				const char *scriptCubeFaceAttrStr = attribute->GetStrAttribute("cube_face");
				if (scriptCubeFaceAttrStr != NULL) cubeFaceStr = scriptCubeFaceAttrStr;

				if (!strcmp(cubeFaceStr, "+x"))
					rtOutput.cubeFace = HQ_CTF_POS_X;
				else if (!strcmp(cubeFaceStr, "-x"))
					rtOutput.cubeFace = HQ_CTF_NEG_X;
				else if (!strcmp(cubeFaceStr, "+y"))
					rtOutput.cubeFace = HQ_CTF_POS_Y;
				else if (!strcmp(cubeFaceStr, "-y"))
					rtOutput.cubeFace = HQ_CTF_NEG_Y;
				else if (!strcmp(cubeFaceStr, "+z"))
					rtOutput.cubeFace = HQ_CTF_POS_Z;
				else if (!strcmp(cubeFaceStr, "-z"))
					rtOutput.cubeFace = HQ_CTF_NEG_Z;
				else {
					Log("Error : %d : invald cube_face=%s!", attriLine, cubeFaceStr);
					return HQ_FAILED;
				}
			}//if (rtOutput.outputTexture->GetTexture()->GetType() == HQ_TEXTURE_CUBE)
			else if (rtOutput.outputTexture->GetTexture()->GetType() == HQ_TEXTURE_2D_ARRAY ||
				rtOutput.outputTexture->GetTexture()->GetType() == HQ_TEXTURE_2D_ARRAY_UAV)
			{
				//get array slice
				rtOutput.arraySlice = 0;
				const hqint32 *scriptArraySlicePtr = attribute->GetIntAttributePtr("array_slice");
				if (scriptArraySlicePtr != NULL)
				{
					rtOutput.arraySlice = (hquint32)*scriptArraySlicePtr;
				}

			}//else if (if (rtOutput.outputTexture->GetTexture()->GetType() == HQ_TEXTURE_2D_ARRAY))

			//get resource size
			rtOutput.outputTexture->GetTexture2DSize(width, height);
			if (global_width == 0xffffffff)
			{
				//store the global size
				dsBufferParams.width = global_width = width;
				dsBufferParams.height = global_height = height;
			}
			else
			{
				//check if the size is valid
				if (global_width != width || global_height != height)
				{
					Log("Error : %d : invald render target resource=%s! This resource has different size from the rest of the group!", attriLine, attriValStr);
					return HQ_FAILED;
				}

			}

			//everything is ok, now add to the array
			outputArray[targetIdx] = rtOutput;
			if (numOutputs <= targetIdx)
				numOutputs = targetIdx + 1;

		}//else if (!strncmp(attriName, "target", 6))

	}//for (; attribute != NULL; attribute = attribute->GetNextSibling())

	params.numOutputs = numOutputs;
	params.outputs = outputArray;
	//now create depth stencil buffer
	if (useDSBuffer)
		params.dsBuffer = this->CreateOrGetDSBuffer(dsBufferParams);

	return HQ_OK;
}

HQSharedPtr<HQEngineRTGroupWrapper> HQEngineEffectManagerImpl::CreateOrGetRTGroup(const HQEngineRTGroupWrapper::CreationParams& params)
{
	bool found;
	HQSharedPtr<HQEngineRTGroupWrapper> re;

	re = m_renderTargetGroups.GetItem(&params, found);

	if (!found)
	{
		//create new
		re = HQ_NEW HQEngineRTGroupWrapper();
		if (re->Init(params) != HQ_OK)
		{
			Log("Error : Could not create render target group!");
			return NULL;
		}
		m_renderTargetGroups.Add(&re->GetCreationParams(), re); 
	}

	return re;
}

HQSharedPtr<HQEngineDSBufferWrapper> HQEngineEffectManagerImpl::CreateOrGetDSBuffer(const HQEngineDSBufferWrapper::CreationParams& params)
{
	bool found;
	HQSharedPtr<HQEngineDSBufferWrapper> re;

	re = m_dsBuffers.GetItem(&params, found);

	if (!found)
	{
		//create new
		re = HQ_NEW HQEngineDSBufferWrapper();
		if (re->Init(params) != HQ_OK)
		{
			Log("Error : Could not create depth stencil buffer!");
			return NULL;
		}
		m_dsBuffers.Add(&re->GetCreationParams(), re); 
	}

	return re;
}

HQSharedPtr<HQEngineShaderProgramWrapper> HQEngineEffectManagerImpl::CreateOrGetShaderProgram(const HQEngineShaderProgramWrapper::CreationParams& params)
{
	bool found;
	HQSharedPtr<HQEngineShaderProgramWrapper> re;

	re = m_shaderPrograms.GetItem(&params, found);

	if (!found)
	{
		//create new
		re = HQ_NEW HQEngineShaderProgramWrapper();
		if (re->Init(params) != HQ_OK)
		{
			Log("Error : Could not create shader program!");
			return NULL;
		}
		m_shaderPrograms.Add(&re->GetCreationParams(), re); 
	}

	return re;
}

HQSharedPtr<HQEngineBlendStateWrapper> HQEngineEffectManagerImpl::CreateOrGetBlendState(const HQEngineBlendStateWrapper::CreationParams& params)
{
	bool found;
	HQSharedPtr<HQEngineBlendStateWrapper> re;

	re = m_blendStates.GetItem(&params, found);

	if (!found)
	{
		//create new
		re = HQ_NEW HQEngineBlendStateWrapper();
		if (re->Init(params) != HQ_OK)
		{
			Log("Error : Could not create blend state object!");
			return NULL;
		}
		m_blendStates.Add(&re->GetCreationParams(), re); 
	}

	return re;
}

HQSharedPtr<HQEngineDSStateWrapper> HQEngineEffectManagerImpl::CreateOrGetDSState(const HQEngineDSStateWrapper::CreationParams& params)
{
	bool found;
	HQSharedPtr<HQEngineDSStateWrapper> re;

	re = m_dsStates.GetItem(&params, found);

	if (!found)
	{
		//create new
		re = HQ_NEW HQEngineDSStateWrapper();
		if (re->Init(params) != HQ_OK)
		{
			Log("Error : Could not create depth stencil state object!");
			return NULL;
		}
		m_dsStates.Add(&re->GetCreationParams(), re); 
	}

	return re;
}

HQSharedPtr<HQEngineSamplerStateWrapper> HQEngineEffectManagerImpl::CreateOrGetSamplerState(const HQEngineSamplerStateWrapper::CreationParams& params)
{
	bool found;
	HQSharedPtr<HQEngineSamplerStateWrapper> re;

	re = m_samplerStates.GetItem(&params, found);

	if (!found)
	{
		//create new
		re = HQ_NEW HQEngineSamplerStateWrapper();
		if (re->Init(params) != HQ_OK)
		{
			Log("Error : Could not create sampler state object!");
			return NULL;
		}
		m_samplerStates.Add(&re->GetCreationParams(), re); 
	}

	return re;
}

HQEngineRenderEffect * HQEngineEffectManagerImpl::GetEffect(const char *name) 
{
	HQSharedPtr<HQEngineRenderEffectImpl>* ppRes = m_effects.GetItemPointer(name);
	if (ppRes == NULL)
		return NULL;
	return ppRes->GetRawPointer();
}

HQReturnVal HQEngineEffectManagerImpl::RemoveEffect(HQEngineRenderEffect *effect) 
{
	m_effects.Remove(effect->GetName());
	return HQ_OK;
}

void HQEngineEffectManagerImpl::RemoveAllEffects() 
{
	m_effects.RemoveAll();
	m_shaderPrograms.RemoveAll();
	m_renderTargetGroups.RemoveAll();
	m_dsBuffers.RemoveAll();
	m_blendStates.RemoveAll();
	m_dsStates.RemoveAll();
	m_samplerStates.RemoveAll();
}


HQReturnVal HQEngineEffectManagerImpl::SetTexture(hq_uint32 slot, HQEngineTextureResource* texture)
{
	HQEngineTextureResImpl* textureImpl = (HQEngineTextureResImpl*)texture;
	HQTexture* tid = textureImpl != NULL ? textureImpl->GetTexture() : NULL;

	return m_pRDevice->GetTextureManager()
		->SetTexture(slot, tid);
}

HQReturnVal HQEngineEffectManagerImpl::SetTextureForPixelShader(hq_uint32 slot, HQEngineTextureResource* texture)
{
	HQEngineTextureResImpl* textureImpl = (HQEngineTextureResImpl*)texture;
	HQTexture* tid = textureImpl != NULL ? textureImpl->GetTexture() : NULL;

	return m_pRDevice->GetTextureManager()
		->SetTextureForPixelShader(slot, tid);
}