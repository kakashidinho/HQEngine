/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/
#include "HQEnginePCH.h"
#include "../HQEngineApp.h"
#include "HQEngineEffectManagerImpl.h"

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
: m_programID(HQ_NULL_ID)
{
}
HQEngineShaderProgramWrapper::~HQEngineShaderProgramWrapper()
{
	if (m_programID != HQ_NULL_ID)
		m_renderDevice->GetShaderManager()->DestroyProgram(m_programID);
}

HQReturnVal HQEngineShaderProgramWrapper::Init(const CreationParams& params)
{
	hquint32 vID, gID, pID;
	vID = params.vertexShader != NULL? (params.vertexShader->GetShaderID() != HQ_NULL_ID?  params.vertexShader->GetShaderID(): HQ_NULL_VSHADER) : HQ_NULL_VSHADER;
	gID = params.geometryShader != NULL? (params.geometryShader->GetShaderID() != HQ_NULL_ID?  params.geometryShader->GetShaderID(): HQ_NULL_GSHADER) : HQ_NULL_GSHADER;
	pID = params.pixelShader != NULL? (params.pixelShader->GetShaderID() != HQ_NULL_ID?  params.pixelShader->GetShaderID(): HQ_NULL_PSHADER) : HQ_NULL_PSHADER;

	HQReturnVal re = m_renderDevice->GetShaderManager()->CreateProgram(vID, pID, gID, NULL, &m_programID);

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

/*----------controlled texture unit------------------*/
void HQEngineTextureUnit::InitD3D(HQShaderType shaderStage, hquint32 textureIdx, const HQSharedPtr<HQEngineTextureResImpl>& texture, const HQSharedPtr<HQEngineSamplerStateWrapper>& samplerState)
{
	this->unitIndex = shaderStage | textureIdx;
	this->texture = texture;
	this->samplerState = samplerState;
}

void HQEngineTextureUnit::InitGL(hquint32 textureIdx, const HQSharedPtr<HQEngineTextureResImpl>& texture, const HQSharedPtr<HQEngineSamplerStateWrapper>& samplerState)
{
	this->unitIndex =  textureIdx;
	this->texture = texture;
	this->samplerState = samplerState;
}

/*--------depth stencil buffer---------------*/
HQEngineDSBufferWrapper::HQEngineDSBufferWrapper()
: bufferID(HQ_NULL_ID)
{
}
HQEngineDSBufferWrapper::~HQEngineDSBufferWrapper()
{
	if (bufferID != HQ_NULL_ID)
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
:groupID(HQ_NULL_ID)
{
}
HQEngineRTGroupWrapper::~HQEngineRTGroupWrapper()
{
	if (groupID != HQ_NULL_ID)
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
			outputDescs[i].renderTargetID = HQ_NULL_ID;
		else
			outputDescs[i].renderTargetID = params.outputs[i].outputTexture->GetRenderTargetID();
		outputDescs[i].cubeFace = params.outputs[i].cubeFace;
	}

	hquint32 depthstencilBufID = params.dsBuffer != NULL? params.dsBuffer->bufferID : HQ_NULL_ID;

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
HQEngineRenderPassImpl::HQEngineRenderPassImpl(const char* name)
:HQNamedGraphicsRelatedObj(name)
{
}

HQEngineRenderPassImpl::~HQEngineRenderPassImpl()
{
}

void HQEngineRenderPassImpl::AddTextureUnit(const HQEngineTextureUnit& texunit)
{
	this->textureUnits.PushBack(texunit);
}

HQReturnVal HQEngineRenderPassImpl::Apply()
{
	//apply states, controlled texture units and more
	this->m_renderDevice->GetStateManager()->SetFaceCulling(this->faceCullingMode);

	this->shaderProgram->Active();
	this->renderTargetGroup->Active();
	this->dsState->Active();
	this->blendState->Active();

	this->ApplyTextureStates();

	return HQ_OK;
}

//D3D version
HQEngineRenderPassD3D::HQEngineRenderPassD3D(const char *name)
: HQEngineRenderPassImpl(name)
{
}

HQReturnVal HQEngineRenderPassD3D::ApplyTextureStates()
{
	HQLinkedList<HQEngineTextureUnit>::Iterator ite;
	this->textureUnits.GetIterator(ite);
	for (; !ite.IsAtEnd(); ++ite) //for each controlled texture unit
	{
		//set sampler state
		ite->samplerState->Apply(ite->unitIndex);
		//set texture
		m_renderDevice->GetTextureManager()->SetTexture(ite->unitIndex, ite->texture->GetTextureID());
	}
	return HQ_OK;
}

//GL version
HQEngineRenderPassGL::HQEngineRenderPassGL(const char *name)
: HQEngineRenderPassImpl(name)
{
}

HQReturnVal HQEngineRenderPassGL::ApplyTextureStates()
{
	HQLinkedList<HQEngineTextureUnit>::Iterator ite;
	this->textureUnits.GetIterator(ite);
	for (; !ite.IsAtEnd(); ++ite) //for each controlled texture unit
	{
		//set texture's sampler state
		ite->samplerState->Apply(ite->texture->GetTextureID());
		//set texture
		m_renderDevice->GetTextureManager()->SetTexture(ite->unitIndex, ite->texture->GetTextureID());
	}
	return HQ_OK;
}

/*-------------rendering effect--------------------*/
HQEngineRenderEffectImpl::HQEngineRenderEffectImpl(const char* name, 
												   HQEngineStringHashTable<HQSharedPtr<HQEngineRenderPassImpl> >& passes)
: HQNamedGraphicsRelatedObj(name), m_passes(NULL), m_numPasses(0)
{
	m_numPasses = passes.GetNumItems();
	m_passes = HQ_NEW HQSharedPtr<HQEngineRenderPassImpl> [m_numPasses];

	HQEngineStringHashTable<HQSharedPtr<HQEngineRenderPassImpl> >::Iterator ite;
	passes.GetIterator(ite);
	
	//copy passes' info
	for (int i = 0; !ite.IsAtEnd(); ++ite, ++i){
		HQSharedPtr<HQEngineRenderPassImpl> & pPass = *ite;
		m_passes[i] = pPass;
		m_passIdxMap.Add(pPass->GetName(), i);

		//set default values
		if (pPass->shaderProgram == NULL) m_passes[i]->shaderProgram = HQ_NEW HQEngineShaderProgramWrapper();
		if (pPass->renderTargetGroup == NULL) m_passes[i]->renderTargetGroup = HQ_NEW HQEngineRTGroupWrapper();
		if (pPass->blendState == NULL) m_passes[i]->blendState = HQ_NEW HQEngineBlendStateWrapper();
		if (pPass->dsState == NULL) m_passes[i]->dsState = HQ_NEW HQEngineDSStateWrapper();
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
HQEngineEffectLoadSessionImpl::HQEngineEffectLoadSessionImpl(TiXmlDocument* doc)
: m_effectXml(doc), m_type(HQ_EELT_XML)
{
	if (m_effectXml != NULL)
	{
		m_effectItem = NULL;
		m_effectGroup = m_effectXml->FirstChildElement("techniques");
		while (m_effectItem == NULL && m_effectGroup != NULL)
		{
			m_effectItem = m_effectGroup->FirstChildElement("technique");
			if (m_effectItem == NULL)//try to find in next effect group
				m_effectGroup = m_effectGroup->NextSiblingElement("techniques");
		} 
	}
	else
	{	
		m_effectGroup = m_effectItem = NULL;
	}
}

HQEngineEffectLoadSessionImpl::~HQEngineEffectLoadSessionImpl()
{
	SafeDelete(m_effectXml);
}

bool HQEngineEffectLoadSessionImpl::HasMoreEffects() const
{
	switch(m_type)
	{
	case HQ_EELT_XML:
		if (m_effectXml == NULL)
			return false;
		return (m_effectItem != NULL);

	default:
		return false;
	}
}

TiXmlElement * HQEngineEffectLoadSessionImpl::CurrentXMLEffect()
{
	return m_type == HQ_EELT_XML ? m_effectItem: NULL;
}

TiXmlElement * HQEngineEffectLoadSessionImpl::NextXMLEffect() {
	if (m_type == HQ_EELT_XML)
	{
		if (m_effectItem == NULL)
			return NULL;
		TiXmlElement * re = m_effectItem;

		m_effectItem = m_effectItem->NextSiblingElement("technique");//advance to next item

		while (m_effectItem == NULL && m_effectGroup != NULL)//try to find in next group
		{
			m_effectGroup = m_effectGroup->NextSiblingElement("techniques");
			if (m_effectGroup != NULL)
				m_effectItem = m_effectGroup->FirstChildElement("technique");
		}

		return re;
	}
	return NULL;
}

/*------------effect manager---------------*/

HQEngineEffectManagerImpl::HQEngineEffectManagerImpl(HQLogStream *stream, bool flushLog)
: HQLoggableObject(stream, "Engine's Effect Manager :", flushLog)
{
	//check if we are using OpenGL
	const char *renderer = HQEngineApp::GetInstance()->GetRenderDevice()->GetDeviceDesc();
	if (strncmp(renderer, "OpenGL", 6) == 0)
		m_isGL = true;
	else
		m_isGL = false;

	/*------------init string to value mapping table--------------*/
	//none/cw/ccw
	m_cullModeMap.Add("none", HQ_CULL_NONE);
	m_cullModeMap.Add("cw", HQ_CULL_NONE);
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
	m_blendOpMap.Add("rev add", HQ_BO_REVSUBTRACT);

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

}

HQEngineEffectManagerImpl::~HQEngineEffectManagerImpl()
{
}

HQReturnVal HQEngineEffectManagerImpl::AddEffectsFromXML(const char* fileName)
{
	HQEngineEffectLoadSession * session = this->BeginAddEffectsFromXML(fileName);
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

HQEngineEffectLoadSession* HQEngineEffectManagerImpl::BeginAddEffectsFromXML(const char* fileName)
{
	HQDataReaderStream* data_stream = HQEngineApp::GetInstance()->OpenFileForRead(fileName);
	if (data_stream == NULL)
	{
		this->Log("Error : Could not load effects from file %s! Could not open the file!", fileName);
		return NULL;
	}

	TiXmlDocument *doc = new TiXmlDocument();

	TiXmlCustomFileStream stream;
	stream.fileHandle = data_stream;
	stream.read = &HQEngineHelper::read_datastream;
	stream.seek = &HQEngineHelper::seek_datastream;
	stream.tell = &HQEngineHelper::tell_datastream;

	if (doc->LoadFile(stream) == false)
	{
		this->Log("Error : Could not load effects from file %s! %d:%d: %s", fileName, doc->ErrorRow(), doc->ErrorCol(), doc->ErrorDesc());
		delete doc;
		data_stream->Release();
		return NULL;
	}

	data_stream->Release();

	this->Log("Effects loading session from file '%s' created!", fileName);

	return HQ_NEW HQEngineEffectLoadSessionImpl(doc);
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
	case HQ_EELT_XML:
		return this->LoadEffectFromXML(resLoadSession->NextXMLEffect());
	break;
	} //switch (resLoadSession-> m_type)

	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::EndAddEffects(HQEngineEffectLoadSession* session)///for releasing loading session
{
	HQEngineEffectLoadSessionImpl * resLoadSession = static_cast <HQEngineEffectLoadSessionImpl*> (session);

	if (resLoadSession != NULL)
		delete resLoadSession;

	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::LoadEffectFromXML(TiXmlElement * effectItem)
{
	const char *effect_name = effectItem->Attribute("name");
	if (effect_name == NULL)
	{
		Log("Error : Could not load effect without any name!");
		return HQ_FAILED;
	}
	if (m_effects.GetItemPointer(effect_name) != NULL)
	{
		Log("Error : Effect '%s' already exists!", effect_name);
		return HQ_FAILED_RESOURCE_EXISTS;
	}

	HQEngineStringHashTable<HQSharedPtr<HQEngineRenderPassImpl> > passesTable;//rendering pass mapping table. TO DO: find a more efficient way to initialize the effect's passes

	TiXmlElement * pass = effectItem->FirstChildElement("pass");
	for (; pass != NULL; pass = pass->NextSiblingElement("pass"))
	{
		const char *pass_name = pass->Attribute("name");
		if (pass_name == NULL)
		{
			Log("Error : Could not load a rendering pass from '%s' effect without any name!", effect_name);
			return HQ_FAILED;
		}
		if (passesTable.GetItemPointer(pass_name) != NULL)
		{
			Log("Error : Effect '%s' already has a pass named '%s'!", effect_name, pass_name);
			return HQ_FAILED;
		}

		//now start loading the pass
		HQSharedPtr<HQEngineRenderPassImpl> newPass; 
		if (m_isGL) 
			newPass = HQ_NEW HQEngineRenderPassGL(pass_name);
		else
			newPass = HQ_NEW HQEngineRenderPassD3D(pass_name);

		if (HQFailed(this->LoadPassFromXML(pass, newPass)) )
			return HQ_FAILED;

		//insert to table
		passesTable.Add(pass_name, newPass);
	}//for (; attribute != NULL; attribute = attribute->NextSiblingElement("pass"))

	//add new effect to table
	HQSharedPtr<HQEngineRenderEffectImpl> newEffect = HQ_NEW HQEngineRenderEffectImpl(effect_name, passesTable);
	
	m_effects.Add(effect_name, newEffect);


	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::LoadPassFromXML(TiXmlElement* passItem, HQSharedPtr<HQEngineRenderPassImpl> &newPass)
{
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

	TiXmlElement * elem = passItem->FirstChildElement();
	for (; elem != NULL; elem = elem->NextSiblingElement())
	{
		const char *elemName = elem->Value();
		if (!strncmp(elemName, "texture", 7))
		{
			const char * textureResName = elem->GetText();
			hquint32 textureIdx = 0;
			if (sscanf(elemName, "texture%u",  &textureIdx) == 0)
			{
				Log("Error : invalid token %s!", elemName);
				return HQ_FAILED;
			}
			
			//get sampler state
			HQEngineSamplerStateWrapper::CreationParams samplerParams;
			if (HQFailed(this->ParseXMLSamplerState(elem, samplerParams)) )
				return HQ_FAILED;
			
			HQSharedPtr<HQEngineSamplerStateWrapper> samplerState = this->CreateOrGetSamplerState(samplerParams);
			if (samplerState == NULL)
				return HQ_FAILED;

			//get texture
			const HQSharedPtr<HQEngineTextureResImpl> texture = resManager->GetTextureResourceSharedPtr(textureResName);
			if (texture == NULL)
			{
				Log("Error: invalid texture resource named %s!", textureResName);
				return HQ_FAILED;
			}
			//controlled texture unit
			HQEngineTextureUnit texunit;

			if (m_isGL)
				texunit.InitGL(textureIdx, texture, samplerState);
			else
				texunit.InitD3D(HQ_PIXEL_SHADER, textureIdx, texture, samplerState);

			newPass->AddTextureUnit(texunit);
		}//if (!strncmp(elemName, "texture", 7))
		else if (!strcmp(elemName, "blend"))
		{
			if (HQFailed(this->ParseXMLBlendState(elem, blendStateParams)) )
				return HQ_FAILED;
			useBlendState = true;
		}
		else if (!strcmp(elemName, "stencil"))
		{
			if (HQFailed(this->ParseXMLStencilState(elem, dsStateParams)) )
				return HQ_FAILED;
		}
		else if (!strcmp(elemName, "depth"))
		{
			dsStateParams.desc.depthMode = m_depthModeMap.GetItem(elem->GetText(), mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : invalid depth=%s", elem->GetText());
				return HQ_FAILED;
			}
		}
		else if (!strcmp(elemName, "cull"))
		{
			faceCullMode = m_cullModeMap.GetItem(elem->GetText(), mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : invalid cull=%s", elem->GetText());
				return HQ_FAILED;
			}
		}
		else if (!strcmp(elemName, "custom_targets"))
		{
			if (HQFailed(this->ParseXMLRTGroup(elem, rtGroupParams)))
				return HQ_FAILED;
			useDefaultRT = false;
		}
		else if (!strcmp(elemName, "vertex_shader"))
		{
			const char *shader_res_name = elem->GetText();
			shaderProgramParams.vertexShader = resManager->GetShaderResourceSharedPtr(shader_res_name);
			if (shaderProgramParams.vertexShader == NULL)
			{
				Log("Error : invalid vertex_shader=%s!", shader_res_name);
				return HQ_FAILED;
			}
		}
		else if (!strcmp(elemName, "pixel_shader"))
		{
			const char *shader_res_name = elem->GetText();
			shaderProgramParams.pixelShader = resManager->GetShaderResourceSharedPtr(shader_res_name);
			if (shaderProgramParams.pixelShader == NULL)
			{
				Log("Error : invalid pixel_shader=%s!", shader_res_name);
				return HQ_FAILED;
			}
		}
		else if (!strcmp(elemName, "geometry_shader"))
		{
			const char *shader_res_name = elem->GetText();
			shaderProgramParams.geometryShader = resManager->GetShaderResourceSharedPtr(shader_res_name);
			if (shaderProgramParams.geometryShader == NULL)
			{
				Log("Error : invalid geometry_shader=%s!", shader_res_name);
				return HQ_FAILED;
			}
		}
	}//for (; elem != NULL; elem = elem->NextSiblingElement())

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

	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::ParseXMLStencilState(TiXmlElement *stencilElem, HQEngineDSStateWrapper::CreationParams &params)
{
	bool mappedValueFound = false;
	HQStencilOp op;
	HQStencilFunc func;
	bool has_ccw_fail_op = stencilElem->FirstChildElement("ccw_fail_op") != NULL;
	bool has_ccw_depth_fail_op = stencilElem->FirstChildElement("ccw_depth_fail_op") != NULL;
	bool has_ccw_pass_op = stencilElem->FirstChildElement("ccw_pass_op") != NULL;
	bool has_ccw_compare_func = stencilElem->FirstChildElement("ccw_compare_func") != NULL;
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

		TiXmlElement * attribute = stencilElem->FirstChildElement();
		for (; attribute != NULL; attribute = attribute->NextSiblingElement())
		{
			const char* attriName = attribute->Value();
			const char* attriValStr = attribute->GetText();

			if (!strcmp(attriName, "read_mask"))
			{
				sscanf(attriValStr, "%x", &params.desc.readMask);
			}
			else if (!strcmp(attriName, "write_mask"))
			{
				sscanf(attriValStr, "%x", &params.desc.writeMask);
			}
			else if (!strcmp(attriName, "reference_value"))
			{
				sscanf(attriValStr, "%u", &params.desc.refVal);
			}
			else if (!strcmp(attriName, "fail_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : invalid fail_op=%s!", attriValStr);
					return HQ_FAILED;
				}
				params.desc.stencilMode.failOp = op;
			}//else if (!strcmp(attriName, "fail_op"))
			else if (!strcmp(attriName, "depth_fail_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : invalid depth_fail_op=%s!", attriValStr);
					return HQ_FAILED;
				}
				params.desc.stencilMode.depthFailOp = op;
			}
			else if (!strcmp(attriName, "pass_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : invalid pass_op=%s!", attriValStr);
					return HQ_FAILED;
				}
				params.desc.stencilMode.passOp = op;
			}
			else if (!strcmp(attriName, "compare_func"))
			{
				func = m_stencilFuncMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : invalid compare_func=%s!", attriValStr);
					return HQ_FAILED;
				}
				params.desc.stencilMode.compareFunc = func;
			}
		}//for (; attribute != NULL; attribute = attribute->NextSiblingElement())
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

		TiXmlElement * attribute = stencilElem->FirstChildElement();
		for (; attribute != NULL; attribute = attribute->NextSiblingElement())
		{
			const char* attriName = attribute->Value();
			const char* attriValStr = attribute->GetText();

			if (!strcmp(attriName, "read_mask"))
			{
				sscanf(attriValStr, "%x", &params.twoSideDesc.readMask);
			}
			else if (!strcmp(attriName, "write_mask"))
			{
				sscanf(attriValStr, "%x", &params.twoSideDesc.writeMask);
			}
			else if (!strcmp(attriName, "reference_value"))
			{
				sscanf(attriValStr, "%u", &params.twoSideDesc.refVal);
			}
			else if (!strcmp(attriName, "fail_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : invalid fail_op=%s!", attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.cwFaceMode.failOp = op;
			}//else if (!strcmp(attriName, "fail_op"))
			else if (!strcmp(attriName, "depth_fail_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : invalid depth_fail_op=%s!", attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.cwFaceMode.depthFailOp = op;
			}
			else if (!strcmp(attriName, "pass_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : invalid pass_op=%s!", attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.cwFaceMode.passOp = op;
			}
			else if (!strcmp(attriName, "compare_func"))
			{
				func = m_stencilFuncMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : invalid compare_func=%s!", attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.cwFaceMode.compareFunc = func;
			}
			else if (!strcmp(attriName, "ccw_fail_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : invalid ccw_fail_op=%s!", attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.ccwFaceMode.failOp = op;
			}//else if (!strcmp(attriName, "fail_op"))
			else if (!strcmp(attriName, "ccw_depth_fail_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : invalid ccw_depth_fail_op=%s!", attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.ccwFaceMode.depthFailOp = op;
			}
			else if (!strcmp(attriName, "ccw_pass_op"))
			{
				op = m_stencilOpMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : invalid ccw_pass_op=%s!", attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.ccwFaceMode.passOp = op;
			}
			else if (!strcmp(attriName, "ccw_compare_func"))
			{
				func = m_stencilFuncMap.GetItem(attriValStr, mappedValueFound);
				if (!mappedValueFound) {
					Log("Error : invalid ccw_compare_func=%s!", attriValStr);
					return HQ_FAILED;
				}
				params.twoSideDesc.ccwFaceMode.compareFunc = func;
			}
		}//for (; attribute != NULL; attribute = attribute->NextSiblingElement())

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

HQReturnVal HQEngineEffectManagerImpl::ParseXMLBlendState(TiXmlElement* blendElem, HQEngineBlendStateWrapper::CreationParams &params)
{
	HQBlendOp op;
	HQBlendFactor factor;
	bool mappedValueFound = false;

	//initialize default values
	params.isExState = false;
	params.descEx = HQBlendStateExDesc();

	TiXmlElement * attribute = blendElem->FirstChildElement();
	for (; attribute != NULL; attribute = attribute->NextSiblingElement())
	{
		const char* attriName = attribute->Value();
		const char* attriValStr = attribute->GetText();
		if (!strcmp(attriName, "src_factor"))
		{
			factor = m_blendFactorMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : invalid src_factor=%s", attriValStr);
				return HQ_FAILED;
			}
			params.descEx.srcFactor = factor;
		}
		else if (!strcmp(attriName, "dest_factor"))
		{
			factor = m_blendFactorMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : invalid dest_factor=%s", attriValStr);
				return HQ_FAILED;
			}
			params.descEx.destFactor = factor;
		}
		else if (!strcmp(attriName, "operator"))
		{
			op = m_blendOpMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : invalid operator=%s", attriValStr);
				return HQ_FAILED;
			}
			params.isExState = true; params.descEx.blendOp = op;
		}
		else if (!strcmp(attriName, "alpha_operator"))
		{
			op = m_blendOpMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : invalid alpha_operator=%s", attriValStr);
				return HQ_FAILED;
			}
			params.isExState = true; params.descEx.alphaBlendOp = op;
		}
		else if (!strcmp(attriName, "alpha_src_factor"))
		{
			factor = m_blendFactorMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : invalid alpha_src_factor=%s", attriValStr);
				return HQ_FAILED;
			}
			params.isExState = true; params.descEx.srcAlphaFactor = factor;
		}
		else if (!strcmp(attriName, "alpha_dest_factor"))
		{
			factor = m_blendFactorMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : invalid alpha_dest_factor=%s", attriValStr);
				return HQ_FAILED;
			}
			params.isExState = true; params.descEx.destAlphaFactor = factor;
		}
	}//for (; attribute != NULL; attribute = attribute->NextSiblingElement())

	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::ParseXMLSamplerState(TiXmlElement* textureElem, HQEngineSamplerStateWrapper::CreationParams &params)
{
	HQTexAddressMode tAddrMode;
	const char defaultfilter[] = "linear";
	const char anisotropicStr[] = "anisotropic";
	const char *min_filter_str = defaultfilter;
	const char *mag_filter_str = defaultfilter;
	const char *mip_filter_str = defaultfilter;
	bool mappedValueFound = false;

	//initialize default values
	params = HQEngineSamplerStateWrapper::CreationParams();

	TiXmlAttribute * attribute = textureElem->FirstAttribute();
	for (; attribute != NULL; attribute = attribute->Next())
	{
		const char* attriName = attribute->Name();
		const char* attriValStr = attribute->Value();
		if (!strcmp(attriName, "address_u"))
		{
			tAddrMode = m_taddrModeMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : invalid address_u=%s", attriValStr);
				return HQ_FAILED;
			}
			params.addressU = tAddrMode;
		}
		else if (!strcmp(attriName, "address_v"))
		{
			tAddrMode = m_taddrModeMap.GetItem(attriValStr, mappedValueFound);
			if (!mappedValueFound)
			{
				Log("Error : invalid address_v=%s", attriValStr);
				return HQ_FAILED;
			}
			params.addressV = tAddrMode;
		}
		else if (!strcmp(attriName, "max_anisotropy"))
		{
			sscanf(attriValStr, "%u", &params.maxAnisotropy);
		}
		else if (!strcmp(attriName, "border_color")) {
			hquint32 color;
			sscanf(attriValStr, "%x", &color);
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
	}//for (; attribute != NULL; attribute = attribute->NextSiblingElement())

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
		Log("Error: unrecognized min-filer/mag-filter/mipnap-filter!");
		return HQ_FAILED;
	}
	
	
	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::ParseXMLDepthStencilBuffer(TiXmlElement* dsBufElem, HQEngineDSBufferWrapper::CreationParams &params)
{
	HQDepthStencilFormat format;
	bool mappedValueFound = false;

	format = m_dsFmtMap.GetItem(dsBufElem->GetText(), mappedValueFound);
	if (!mappedValueFound)
	{
		Log("Error : invalid depth_stencil_buffer_format=%s!", dsBufElem->GetText());
		return HQ_FAILED;
	}

	params.format = format;

	return HQ_OK;
}

HQReturnVal HQEngineEffectManagerImpl::ParseXMLRTGroup(TiXmlElement* rtGroupElem, HQEngineRTGroupWrapper::CreationParams &params)
{
	HQEngineResManagerImpl* resManager = static_cast<HQEngineResManagerImpl*> (HQEngineApp::GetInstance()->GetResourceManager());
	HQEngineDSBufferWrapper::CreationParams dsBufferParams;
	bool useDSBuffer = false;//default is not using depth stencil buffer
	hquint32 maxNumRTs = HQEngineApp::GetInstance()->GetRenderDevice()->GetMaxActiveRenderTargets();
	HQSharedArrayPtr<HQEngineRenderTargetWrapper> outputArray = HQ_NEW HQEngineRenderTargetWrapper[maxNumRTs];//auto release array
	hquint32 numOutputs = 0;
	hquint32 global_width = 0xffffffff;
	hquint32 global_height = 0xffffffff;
	hquint32 width, height;

	TiXmlElement * attribute = rtGroupElem->FirstChildElement();
	for (; attribute != NULL; attribute = attribute->NextSiblingElement())
	{
		const char* attriName = attribute->Value();
		const char* attriValStr = attribute->GetText();

		if (!strcmp(attriName, "depth_stencil_buffer_format"))
		{
			useDSBuffer = true;
			if (HQFailed(this->ParseXMLDepthStencilBuffer(attribute, dsBufferParams)))
				return HQ_FAILED;
		}
		else if (!strncmp(attriName, "target", 6))
		{
			hquint32 targetIdx = 0;
			HQEngineRenderTargetWrapper rtOutput;
			const char defaultCubeFaceStr[] = "+x";
			const char* cubeFaceStr = defaultCubeFaceStr;
			if (sscanf(attriName, "target%u", &targetIdx) != 1)
			{
				Log("Error : invald token %s!", attriName);
				return HQ_FAILED;
			}
			//get cube face
			const char *xmlCubeFaceAttrStr = attribute->Attribute("cube_face");
			if (xmlCubeFaceAttrStr != NULL) cubeFaceStr = xmlCubeFaceAttrStr;
			
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
				Log("Error : invald cube_face=%s!", cubeFaceStr);
				return HQ_FAILED;
			}

			//now get the texture resource
			rtOutput.outputTexture = resManager->GetTextureResourceSharedPtr(attriValStr);
			if (rtOutput.outputTexture == NULL || rtOutput.outputTexture->IsRenderTarget() == false)
			{
				Log("Error : invald render target resource=%s!", attriValStr);
				return HQ_FAILED;
			}
			
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
					Log("Error : invald render target resource=%s! This resource has different size from the rest of the group!", attriValStr);
					return HQ_FAILED;
				}

			}

			//everything is ok, now add to the array
			outputArray[targetIdx] = rtOutput;
			if (numOutputs <= targetIdx)
				numOutputs = targetIdx + 1;

		}//else if (!strncmp(attriName, "target", 6))

	}//for (; attribute != NULL; attribute = attribute->NextSiblingElement())

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

 HQReturnVal HQEngineEffectManagerImpl::CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDescs , 
												hq_uint32 numAttrib ,
												HQEngineShaderResource* vertexShader , 
												hq_uint32 *pInputLayoutID)
 {
	 HQEngineShaderResImpl * vshaderImpl = (HQEngineShaderResImpl*) vertexShader;
	 hquint32 vid = vshaderImpl != NULL ? vshaderImpl->GetShaderID(): HQ_NOT_USE_VSHADER;

	 return HQEngineApp::GetInstance()->GetRenderDevice()->GetVertexStreamManager()
		 ->CreateVertexInputLayout(vAttribDescs, numAttrib, vid, pInputLayoutID);
									
						
 }

HQReturnVal HQEngineEffectManagerImpl::SetTexture(hq_uint32 slot, HQEngineTextureResource* texture)
{
	HQEngineTextureResImpl* textureImpl = (HQEngineTextureResImpl*)texture;
	hquint32 tid = textureImpl != NULL ? textureImpl->GetTextureID() : HQ_NULL_ID;

	return HQEngineApp::GetInstance()->GetRenderDevice()->GetTextureManager()
		->SetTexture(slot, tid);
}

HQReturnVal HQEngineEffectManagerImpl::SetTextureForPixelShader(hq_uint32 slot, HQEngineTextureResource* texture)
{
	HQEngineTextureResImpl* textureImpl = (HQEngineTextureResImpl*)texture;
	hquint32 tid = textureImpl != NULL ? textureImpl->GetTextureID() : HQ_NULL_ID;

	return HQEngineApp::GetInstance()->GetRenderDevice()->GetTextureManager()
		->SetTextureForPixelShader(slot, tid);
}