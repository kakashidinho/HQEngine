/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/
#include "HQEnginePCH.h"
#include "../HQEngineApp.h"
#include "HQEngineEffectManagerImpl.h"

/*---------------shader program------------*/
HQEngineShaderProgramWrapper::HQEngineShaderProgramWrapper(const char* name)
: HQNamedGraphicsRelatedObj(name)
m_programID(HQ_NULL_ID)
{
}
HQEngineShaderProgramWrapper::~HQEngineShaderProgramWrapper()
{
	if (m_programID != HQ_NULL_ID)
		m_renderDevice->GetShaderManager()->DestroyProgram(m_programID);
}

HQReturnVal HQEngineShaderProgramWrapper::Init(
					const HQSharedPtr<HQEngineShaderResImpl> & vertexShader, 
					const HQSharedPtr<HQEngineShaderResImpl> & pixelShader,
					const HQSharedPtr<HQEngineShaderResImpl> & geometryShader
			)
{
	hquint32 vID, gID, pID;
	vID = vertexShader != NULL? (vertexShader->GetShaderID() != HQ_NULL_ID?  vertexShader->GetShaderID(): HQ_NULL_VSHADER) : HQ_NULL_VSHADER;
	gID = geometryShader != NULL? (geometryShader->GetShaderID() != HQ_NULL_ID?  geometryShader->GetShaderID(): HQ_NULL_GSHADER) : HQ_NULL_GSHADER;
	pID = pixelShader != NULL? (pixelShader->GetShaderID() != HQ_NULL_ID?  pixelShader->GetShaderID(): HQ_NULL_PSHADER) : HQ_NULL_PSHADER;

	HQReturnVal re = m_renderDevice->GetShaderManager()->CreateProgram(vID, pID, gID, NULL, &m_programID);

	if (!HQFailed(re))
	{
		m_vertexShader = vertexShader;
		m_pixelShader = pixelShader;
		m_geometryShader = geometryShader;
	}

	return re;
}


HQReturnVal HQEngineShaderProgramWrapper::Active(){
	return m_renderDevice->GetShaderManager()->ActiveProgram(m_programID);
}

const HQSharedPtr<HQEngineShaderResImpl>& HQEngineShaderProgramWrapper::GetShader(HQShaderType shaderStage) const
{
	switch(shaderStage)
	{
	case HQ_VERTEX_SHADER:
		return m_vertexShader;
	case HQ_GEOMETRY_SHADER:
		return m_geometryShader;
	case HQ_PIXEL_SHADER:
		return m_pixelShader;
	}

	return HQSharedPtr<HQEngineShaderResImpl>::null;
}

/*-------------blend state------------------*/
HQEngineBlendStateWrapper::HQEngineBlendStateWrapper(const char *name)
: HQNamedGraphicsRelatedObj(name),
stateID(0)
{
}
HQEngineBlendStateWrapper::~HQEngineBlendStateWrapper()
{
	if (stateID != 0)
		m_renderDevice->GetStateManager()->RemoveBlendState(stateID);
}

HQReturnVal HQEngineBlendStateWrapper::Init(const HQBlendStateDesc& _desc)
{
	HQReturnVal re = m_renderDevice->GetStateManager()->CreateBlendState(_desc, &stateID);
	if (re == HQ_OK)
	{
		isExState = false;
		desc = _desc;
	}
	return re;
}

HQReturnVal HQEngineBlendStateWrapper::Init(const HQBlendStateExDesc& _desc)
{
	HQReturnVal re = m_renderDevice->GetStateManager()->CreateBlendState(_desc, &stateID);
	if (re == HQ_OK)
	{
		isExState = true;
		descEx = _desc;
	}
	return re;
}


/*----------depth stencil state------------*/
HQEngineDSStateWrapper::HQEngineDSStateWrapper(const char* name)
:HQNamedGraphicsRelatedObj(name),
 stateID(0)
{
}
HQEngineDSStateWrapper::~HQEngineDSStateWrapper()
{
	if (stateID != 0)
		m_renderDevice->GetStateManager()->RemoveDepthStencilState(stateID);
}

HQReturnVal HQEngineDSStateWrapper::Init(const HQDepthStencilStateDesc& _desc)
{
	HQReturnVal re = m_renderDevice->GetStateManager()->CreateDepthStencilState(_desc, &stateID);
	
	if (re == HQ_OK)
	{
		isTwoSideState = false;
		this->desc = _desc;
	}
	return re;
}
HQReturnVal HQEngineDSStateWrapper::Init(const HQDepthStencilStateTwoSideDesc& _desc)
{
	HQReturnVal re = m_renderDevice->GetStateManager()->CreateDepthStencilStateTwoSide(_desc, &stateID);
	
	if (re == HQ_OK)
	{
		isTwoSideState = true;
		this->twoSideDesc = _desc;
	}
	return re;
}

/* --------------sampler state-----------------*/
HQEngineSamplerStateWrapper::HQEngineSamplerStateWrapper(const char* name)
: HQNamedGraphicsRelatedObj(name), stateID(0)
{
}
HQEngineSamplerStateWrapper::~HQEngineSamplerStateWrapper()
{
	if (stateID != 0)
		m_renderDevice->GetStateManager()->RemoveSamplerState(stateID);
}

HQReturnVal HQEngineSamplerStateWrapper::Init(const HQSamplerStateDesc& _desc)
{
	HQReturnVal re = m_renderDevice->GetStateManager()->CreateSamplerState(_desc, &stateID);
	if (re == HQ_OK)
		this->desc = _desc;

	return re;
}


/*--------depth stencil buffer---------------*/
HQEngineDSBufferWrapper::HQEngineDSBufferWrapper(const char *name)
: HQNamedGraphicsRelatedObj(name),
  bufferID(HQ_NULL_ID)
{
}
HQEngineDSBufferWrapper::~HQEngineDSBufferWrapper()
{
	if (bufferID != HQ_NULL_ID)
		m_renderDevice->GetRenderTargetManager()->RemoveDepthStencilBuffer(bufferID);
}

HQReturnVal HQEngineDSBufferWrapper::Init(HQDepthStencilFormat _format, hquint32 _width, hquint32 _height)
{
	HQReturnVal re = m_renderDevice->GetRenderTargetManager()->CreateDepthStencilBuffer(
		_width, _height,
		_format, HQ_MST_NONE,
		&bufferID);

	if (re == HQ_OK)
	{
		this->format = _format;
		this->width = _width;
		this->height = _height;
	}
	return re;
}


/*----------render targets group----------------*/
HQEngineRTGroupWrapper::HQEngineRTGroupWrapper(const char * name)
:HQNamedGraphicsRelatedObj(name),
groupID(HQ_NULL_ID), numOutputs(0), outputs(NULL)
{
}
HQEngineRTGroupWrapper::~HQEngineRTGroupWrapper()
{
	if (groupID != HQ_NULL_ID)
	{
		m_renderDevice->GetRenderTargetManager()->RemoveRenderTargetGroup(groupID);
	}

	if (outputs != NULL)
		delete[] outputs;
}

HQReturnVal HQEngineRTGroupWrapper::Init(const HQEngineRenderTargetWrapper *_outputs, hquint32 _numOutputs, const HQSharedPtr<HQEngineDSBufferWrapper>& _dsBuffer)
{
	HQRenderTargetDesc *outputDescs = HQ_NEW HQRenderTargetDesc[_numOutputs];
	//init parameters for passing to render device
	for (hquint32 i = 0; i < _numOutputs; ++i)
	{
		outputDescs[i].renderTargetID = _outputs[i].outputTexture->GetRenderTargetID();
		outputDescs[i].cubeFace = _outputs[i].cubeFace;
	}

	hquint32 depthstencilBufID = _dsBuffer != NULL? _dsBuffer->bufferID : HQ_NULL_ID;

	HQReturnVal re = m_renderDevice->GetRenderTargetManager()->CreateRenderTargetGroup(outputDescs, depthstencilBufID, _numOutputs, &groupID);

	delete[] outputDescs;

	if (re == HQ_OK)
	{
		this->numOutputs = _numOutputs;
		this->dsBuffer = _dsBuffer;

		if (this->outputs != NULL)
			delete[] this->outputs;
		this->outputs = HQ_NEW HQEngineRenderTargetWrapper[_numOutputs];
		for (hquint32 i = 0; i < _numOutputs; ++i)
			this->outputs[i] = _outputs[i];
	}

	return re;
}

HQReturnVal HQEngineRTGroupWrapper::Active()
{
	return m_renderDevice->GetRenderTargetManager()->ActiveRenderTargets(groupID);
}

/*------------rendering pass-----------------*/
HQEngineRenderPassImpl::HQEngineRenderPassImpl(const char* name)
:HQNamedGraphicsRelatedObj(name), textureUnits(NULL), numTexUnits(0)
{
}

HQEngineRenderPassImpl::~HQEngineRenderPass()
{
	if (textureUnits != NULL)
		delete[] textureUnits;
}

HQReturnVal HQEngineRenderPassImpl::Apply()
{
	//TO DO
}


/*-------------rendering effect--------------------*/
class HQEngineRenderEffectImpl: public HQNamedGraphicsRelatedObj, public HQEngineRenderEffect {
public:
	HQEngineRenderEffectImpl(const char* name);
	virtual ~HQEngineRenderEffect() ;
	virtual hquint32 GetNumPasses() const = 0;
	virtual HQEngineRenderPass* GetPassByName(const char* name);
	virtual HQEngineRenderPass* GetPass(hquint32 index);

private:
	typedef HQEngineBaseHashTable<HQEngineRenderPassImpl*> PassMapTable;
	PassMapTable m_passMap;//render pass mapping table
	HQEngineRenderPassImpl * m_passes;//render passes
};

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
}

HQEngineEffectManagerImpl::~HQEngineEffectManagerImpl()
{
}

HQReturnVal HQEngineEffectManagerImpl::AddEffectsFromXML(const char* fileName)
{
}

HQEngineEffectLoadSession* HQEngineEffectManagerImpl::BeginAddEffectsFromXML(const char* fileName)
{
}

bool HQEngineEffectManagerImpl::HasMoreEffects(HQEngineEffectLoadSession* session)
{
}

HQReturnVal HQEngineEffectManagerImpl::AddNextEffect(HQEngineEffectLoadSession* session)
{
}

HQReturnVal HQEngineEffectManagerImpl::EndAddEffects(HQEngineEffectLoadSession* session)///for releasing loading session
{
}

HQEngineRenderEffect * HQEngineEffectManagerImpl::GetEffect(const char *name) 
{
}

HQReturnVal HQEngineEffectManagerImpl::RemoveEffect(HQEngineRenderEffect *effect) 
{
}

void HQEngineEffectManagerImpl::RemoveAllEffects() 
{
}
