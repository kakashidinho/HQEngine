/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_ENGINE_EFFECT_MANAGER_IMPL_H
#define HQ_ENGINE_EFFECT_MANAGER_IMPL_H

#include "../HQEngineEffectManager.h"
#include "../HQLinkedList.h"
#include "HQEngineResManagerImpl.h"

//shader program
class HQEngineShaderProgramWrapper: public HQNamedGraphicsRelatedObj {
public:
	HQEngineShaderProgramWrapper(const char* name);
	virtual ~HQEngineShaderProgramWrapper();

	HQReturnVal Init(
					const HQSharedPtr<HQEngineShaderResImpl> & vertexShader, 
					const HQSharedPtr<HQEngineShaderResImpl> & pixelShader,
					const HQSharedPtr<HQEngineShaderResImpl> & geometryShader
			);
	virtual hquint32 GetProgramID() const {return m_programID;}
	const HQSharedPtr<HQEngineShaderResImpl>& GetShader(HQShaderType shaderStage) const;
	virtual HQReturnVal Active() ;

private:
	hquint32 m_programID;
	const HQSharedPtr<HQEngineShaderResImpl> m_vertexShader;
	const HQSharedPtr<HQEngineShaderResImpl> m_geometryShader;
	const HQSharedPtr<HQEngineShaderResImpl> m_pixelShader;
};

//blend state
class HQEngineBlendStateWrapper: public HQNamedGraphicsRelatedObj {
public:
	HQEngineBlendStateWrapper(const char *name);
	~HQEngineBlendStateWrapper();

	HQReturnVal Init(const HQBlendStateDesc& _desc);
	HQReturnVal Init(const HQBlendStateExDesc& _desc);

	bool isExState;
	union {
		HQBlendStateDesc desc;
		HQBlendStateExDesc descEx;
	};
	
	hquint32 stateID;
};

//depth stencil state
class HQEngineDSStateWrapper: public HQNamedGraphicsRelatedObj{
public:
	HQEngineDSStateWrapper(const char* name);
	~HQEngineDSStateWrapper();

	HQReturnVal Init(const HQDepthStencilStateDesc& _desc);
	HQReturnVal Init(const HQDepthStencilStateTwoSideDesc& _desc);

	bool isTwoSideState;
	union{
		HQDepthStencilStateDesc desc;
		HQDepthStencilStateTwoSideDesc twoSideDesc;
	};

	hquint32 stateID;
};

//sampler state
class HQEngineSamplerStateWrapper: public HQNamedGraphicsRelatedObj{
public:
	HQEngineSamplerStateWrapper(const char* name);
	~HQEngineSamplerStateWrapper();

	HQReturnVal Init(const HQSamplerStateDesc& _desc);

	HQSamplerStateDesc desc;

	hquint32 stateID;
};

//texture unit
struct HQEngineTextureUnit {
	hquint32 unitIndex;
	HQSharedPtr<HQEngineTextureResImpl> texture;//texture
	HQSharedPtr<HQEngineSamplerStateWrapper> samplerState;// sampling state
};

//depth stencil buffer
class HQEngineDSBufferWrapper: public HQNamedGraphicsRelatedObj {
public:
	HQEngineDSBufferWrapper(const char *name);
	~HQEngineDSBufferWrapper();

	HQReturnVal Init(HQDepthStencilFormat _format, hquint32 _width, hquint32 _height);


	HQDepthStencilFormat format;
	hquint32 width, height;
	hquint32 bufferID;
};

//render target
struct HQEngineRenderTargetWrapper {
	SharedPtr<HQEngineTextureResImpl> outputTexture;
	HQCubeTextureFace cubeFace;
};

//render targets group
class HQEngineRTGroupWrapper: public HQNamedGraphicsRelatedObj {
public:
	HQEngineRTGroupWrapper(const char * name);
	~HQEngineRTGroupWrapper();

	HQReturnVal Init(const HQEngineRenderTargetWrapper *_outputs, hquint32 _numOutputs, const HQSharedPtr<HQEngineDSBufferWrapper>& _dsBuffer);

	HQReturnVal Active();

	hquint32 groupID;
	HQEngineRenderTargetWrapper * outputs;
	hquint32 numOutputs;
	HQSharedPtr<HQEngineDSBufferWrapper> dsBuffer;//depth stencil buffer
};

#ifdef WIN32
#	pragma warning( push )
#	pragma warning( disable : 4250 )//dominance inheritance of GetName()
#endif

//rendering pass
struct HQEngineRenderPassImpl : public HQNamedGraphicsRelatedObj, public HQEngineRenderPass{
	HQEngineRenderPassImpl(const char* name);
	virtual ~HQEngineRenderPass();
	virtual HQReturnVal Apply();

	HQSharedPtr<HQEngineShaderProgramWrapper> shaderProgram;//shader program
	HQSharedPtr<HQEngineRTGroupWrapper> renderTargetGroup;//render targets. 
	HQSharedPtr<HQEngineBlendStateWrapper> blendState;//blend state
	HQSharedPtr<HQEngineDSStateWrapper> dsState;//depth stencil state
	
	HQLinkedList<HQEngineTextureUnit> textureUnits; //controlled texture units in this pass
	HQCullMode faceCullingMode;

};

//rendering effect
class HQEngineRenderEffectImpl: public HQNamedGraphicsRelatedObj, public HQEngineRenderEffect {
public:
	HQEngineRenderEffectImpl(const char* name);
	virtual ~HQEngineRenderEffectImpl() ;
	virtual hquint32 GetNumPasses() const = 0;
	virtual HQEngineRenderPass* GetPassByName(const char* name);
	virtual hquint32 GetPassIndexByName(const char* name);
	virtual HQEngineRenderPass* GetPass(hquint32 index);
	
	virtual HQReturnVal AddPass(const HQSharedPtr<HQEngineRenderPassImpl>& pass);
	virtual void Finalize();//complete the effect's initialization. AddPass() must not be called again after this method finishes
private:
	typedef HQEngineBaseHashTable<HQEngineRenderPassImpl*> PassMapTable;
	PassMapTable m_passMap;//render pass mapping table
	HQSharedPtr<HQEngineRenderPassImpl> * m_passes;//render passes
	hquint32 m_numPasses;

	typedef HQLinkedList<HQSharedPtr<HQEngineRenderPassImpl> > PassList;
	PassList m_tempList;//temp passes list during initialization
};

#ifdef WIN32
#	pragma warning( pop )
#endif

//effect loading session
enum HQEngineEffectLoadType {
	HQ_EELT_XML
};

class HQEngineEffectLoadSessionImpl: public HQEngineEffectLoadSession{
public:
	HQEngineEffectLoadSessionImpl(TiXmlDocument* doc);
	~HQEngineEffectLoadSessionImpl();
	bool HasMoreEffects() const;
	TiXmlElement * CurrentXMLEffect();
	TiXmlElement * NextXMLEffect();//advances to next effect item and returns current effect item

	const HQEngineEffectLoadType m_type;
	union {
		struct{
			TiXmlDocument * m_effectXml;
			TiXmlElement* m_effectGroup;
			TiXmlElement* m_effectItem;
		};
	};
};

//effect manager
class HQEngineEffectManagerImpl: public  HQEngineEffectManager, public HQLoggableObject
{
public:
	HQEngineEffectManagerImpl(HQLogStream* stream, bool flushLog);
	virtual ~HQEngineEffectManagerImpl() ;

	virtual HQReturnVal AddEffectsFromXML(const char* fileName);
	virtual HQEngineEffectLoadSession* BeginAddEffectsFromXML(const char* fileName);
	virtual bool HasMoreEffects(HQEngineEffectLoadSession* session);
	virtual HQReturnVal AddNextEffect(HQEngineEffectLoadSession* session);
	virtual HQReturnVal EndAddEffects(HQEngineEffectLoadSession* session);///for releasing loading session

	virtual HQEngineRenderEffect * GetEffect(const char *name) ;
	virtual HQReturnVal RemoveEffect(HQEngineRenderEffect *effect) ;
	virtual void RemoveAllEffects() ;

private:
	typedef HQEngineBaseHashTable<HQSharedPtr<HQEngineShaderProgramWrapper> > ProgramTable;
	typedef HQEngineBaseHashTable<HQSharedPtr<HQEngineRTGroupWrapper> > RTGroupTable;
	typedef HQEngineBaseHashTable<HQSharedPtr<HQEngineBlendStateWrapper> > BlendStateTable;
	typedef HQEngineBaseHashTable<HQSharedPtr<HQEngineDSStateWrapper> > DSStateTable;
	typedef HQEngineBaseHashTable<HQSharedPtr<HQEngineSamplerStateWrapper> > SamplerStateTable;
	typedef HQEngineBaseHashTable<HQSharedPtr<HQEngineRenderEffectImpl> > EffectTable;

	ProgramTable m_shaderPrograms;//shader programs
	RTGroupTable m_renderTargetGroups;//render targets. 
	BlendStateTable m_blendStates;//blend states
	DSStateTable m_dsStates;//depth stencil states
	SamplerStateTable m_samplerStates;//sampler states

	EffectTable m_effects;//rendering effects

};

#endif