/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_ENGINE_RES_MANAGER_IMPL_H
#define HQ_ENGINE_RES_MANAGER_IMPL_H

#include "../HQEngineResManager.h"
#include "../HQLoggableObject.h"
#include "../../../tinyxml/tinyxml.h"
#include "HQEngineCommonInternal.h"

#ifdef WIN32
#	pragma warning( push )
#	pragma warning( disable : 4250 )//dominance inheritance of GetName()
#endif

//texture resource
class HQEngineTextureResImpl: public HQNamedGraphicsRelatedObj, public HQEngineTextureResource{
public:
	HQEngineTextureResImpl();
	virtual ~HQEngineTextureResImpl();

	void SetTextureID(hquint32 textureID) {m_textureID = textureID;}
	hquint32 GetTextureID() const {return m_textureID;}
	hquint32 GetRenderTargetID() const {return m_renderTargetID;}
	virtual hquint32 GetTexture2DSize(hquint32 &width, hquint32 &height) const;
	virtual bool IsRenderTarget() const {return m_renderTargetID != HQ_NULL_ID;}

private:
	hquint32 m_textureID;
	hquint32 m_renderTargetID;
};

//shader resource
class HQEngineShaderResImpl: public HQNamedGraphicsRelatedObj, public HQEngineShaderResource{
public:
	HQEngineShaderResImpl();
	virtual ~HQEngineShaderResImpl();

	void Init(HQShaderType type, hquint32 shaderID) {m_shaderID = shaderID; m_shaderType = type;}
	virtual hquint32 GetShaderID() const {return m_shaderID;}
	virtual HQShaderType GetShaderType() const {return m_shaderType;}

private:
	hquint32 m_shaderID;
	HQShaderType m_shaderType;
};

//shader program
class HQEngineShaderProgramResImpl: public HQNamedGraphicsRelatedObj, public HQEngineShaderProgramResource {
public:
	HQEngineShaderProgramResImpl();
	virtual ~HQEngineShaderProgramResImpl();

	void SetProgramID(hquint32 programID) {m_programID = programID;}
	virtual hquint32 GetProgramID() const {return m_programID;}
	virtual HQReturnVal Active() ;

private:
	hquint32 m_programID;
};

#ifdef WIN32
#	pragma warning( pop )
#endif

enum HQEngineResLoadType {
	HQ_ERLT_XML
};

//resource loading session
class HQEngineResLoadSessionImpl: public HQEngineResLoadSession{
public:
	HQEngineResLoadSessionImpl(TiXmlDocument* doc);
	~HQEngineResLoadSessionImpl();
	bool HasMoreResources() const;
	TiXmlElement * CurrentXMResource();
	TiXmlElement * NextXMLResource();//advances to next resource item and returns current resource item

	const HQEngineResLoadType m_type;
	union {
		struct{
			TiXmlDocument * m_resXml;
			TiXmlElement* m_resGroup;
			TiXmlElement* m_resourceIem;
		};
	};
	const char* GetResScriptDir() const {return m_resScriptDir;}
private:
	char * m_resScriptDir;//directory contains the resource script
};

//resource manager
class HQEngineResManagerImpl: public HQEngineResManager, public HQLoggableObject {
public:
	HQEngineResManagerImpl(HQLogStream* stream, bool flushLog);
	~HQEngineResManagerImpl();

	virtual HQReturnVal AddResourcesFromXML(HQDataReaderStream * data_stream);
	virtual HQEngineResLoadSession* BeginAddResourcesFromXML(HQDataReaderStream * data_stream);
	virtual bool HasMoreResources(HQEngineResLoadSession* session);
	virtual HQReturnVal AddNextResource(HQEngineResLoadSession* session);
	virtual HQReturnVal EndAddResources(HQEngineResLoadSession* session);
	virtual HQEngineTextureResource * GetTextureResource(const char* name);
	virtual HQEngineShaderResource * GetShaderResource(const char* name);
	virtual HQEngineShaderProgramResource * GetShaderProgramResource(const char* name);

	virtual HQReturnVal RemoveTextureResource(HQEngineTextureResource* res);
	virtual HQReturnVal RemoveShaderResource(HQEngineShaderResource* res);
	virtual HQReturnVal RemoveShaderProgramResource(HQEngineShaderProgramResource* res);
	virtual void RemoveAllResources();
private:
	HQReturnVal LoadResourceFromXML(TiXmlElement* resource);
	HQReturnVal LoadTextureFromXML(TiXmlElement* textureItem);
	HQReturnVal LoadShaderFromXML(TiXmlElement* textureItem);
	HQReturnVal LoadShaderProgramFromXML(TiXmlElement* textureItem);


	HQEngineBaseHashTable<HQEngineTextureResImpl> m_textures;
	HQEngineBaseHashTable<HQEngineShaderResImpl> m_shaders;
	HQEngineBaseHashTable<HQEngineShaderProgramResImpl> m_shaderPrograms;
};



#endif