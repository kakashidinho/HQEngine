/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQEnginePCH.h"
#include "../HQEngineApp.h"
#include "HQEngineResManagerImpl.h"

/*------------texture resource----------*/
HQEngineTextureResImpl::HQEngineTextureResImpl()
:
m_textureID(HQ_NULL_ID)
{
}

HQEngineTextureResImpl::~HQEngineTextureResImpl()
{
	if (m_textureID != HQ_NULL_ID)
		m_renderDevice->GetTextureManager()->RemoveTexture(m_textureID);
}

hquint32 HQEngineTextureResImpl::GetTexture2DSize(hquint32 &width, hquint32 &height) const
{
	return m_renderDevice->GetTextureManager()->GetTexture2DSize(m_textureID, width, height);
}


/*-----------shader resource------------------------------*/
HQEngineShaderResImpl::HQEngineShaderResImpl()
:m_shaderID(HQ_NULL_ID)
{
}
HQEngineShaderResImpl::~HQEngineShaderResImpl()
{
	if (m_shaderID != HQ_NULL_ID)
		m_renderDevice->GetShaderManager()->DestroyShader(m_shaderID);
}

/*---------------shader program resource------------*/
HQEngineShaderProgramResImpl::HQEngineShaderProgramResImpl()
:m_programID(HQ_NULL_ID)
{
}
HQEngineShaderProgramResImpl::~HQEngineShaderProgramResImpl()
{
	if (m_programID != HQ_NULL_ID)
		m_renderDevice->GetShaderManager()->DestroyProgram(m_programID);
}

HQReturnVal HQEngineShaderProgramResImpl::Active(){
	return m_renderDevice->GetShaderManager()->ActiveProgram(m_programID);
}

/*----------------resource loading session----------------*/
HQEngineResLoadSessionImpl::HQEngineResLoadSessionImpl(TiXmlDocument* doc)
: m_resXml(doc), m_type(HQ_ERLT_XML)
{
	if (m_resXml != NULL)
	{
		m_resourceIem = NULL;
		m_resGroup = m_resXml->FirstChildElement("resources");
		while (m_resourceIem == NULL && m_resGroup != NULL)
		{
			m_resourceIem = m_resGroup->FirstChildElement();
			if (m_resourceIem == NULL)//try to find in next resource group
				m_resGroup = m_resGroup->NextSiblingElement("resources");
		} 
	}
	else
	{	
		m_resGroup = m_resourceIem = NULL;
	}
}

HQEngineResLoadSessionImpl::~HQEngineResLoadSessionImpl()
{
	SafeDelete(m_resXml);
}

bool HQEngineResLoadSessionImpl::HasMoreResources() const
{
	switch(m_type)
	{
	case HQ_ERLT_XML:
		if (m_resXml == NULL)
			return false;
		return (m_resourceIem != NULL);

	default:
		return false;
	}
}

TiXmlElement * HQEngineResLoadSessionImpl::CurrentXMResource()
{
	return m_type == HQ_ERLT_XML ? m_resourceItem: NULL;
}

TiXmlElement * HQEngineResLoadSessionImpl::NextXMLResource() {
	if (m_type == HQ_ERLT_XML)
	{
		if (m_resourceIem == NULL)
			return NULL;
		TiXmlElement * re = m_resourceIem;

		m_resourceIem = m_resourceIem->NextSiblingElement();//advance to next item

		while (m_resourceIem == NULL && m_resGroup != NULL)//try to find in next group
		{
			m_resGroup = m_resGroup->NextSiblingElement("resources");
			if (m_resGroup != NULL)
				m_resourceIem = m_resGroup->FirstChildElement();
		}

		return re;
	}
	return NULL;
}

/*-----------------resource manager-----------------------*/
HQEngineResManagerImpl::HQEngineResManagerImpl(HQLogStream* logStream, bool flushLog)
: HQLoggableObject(logStream, "Engine's Resource Manager :", flushLog)
{

	this->Log("Init done!");

}

HQEngineResManagerImpl::~HQEngineResManagerImpl()
{
	this->Log("Released!");
}

HQReturnVal HQEngineResManagerImpl::AddResourcesFromXML(HQDataReaderStream * data_stream)
{
	
	HQEngineResLoadSession * session = this->BeginAddResourcesFromXML(data_stream);
	if (stream == NULL)

	

	return re;
}

HQEngineResLoadSession* HQEngineResManagerImpl::BeginAddResourcesFromXML(HQDataReaderStream * data_stream)
{
	TiXmlDocument *doc = new TiXmlDocument();

	TiXmlCustomFileStream stream;
	stream.fileHandle = data_stream;
	stream.read = &HQEngineHelper::read_datastream;
	stream.seek = &HQEngineHelper::seek_datastream;
	stream.tell = &HQEngineHelper::tell_datastream;

	if (doc->LoadFile(stream) == false)
	{
		if (data_stream->GetName() != NULL)
			this->Log("Error: Cannot load resources from stream: \"%s\"!", data_stream->GetName());
		else
			this->Log("Error: Cannot load resources!");
		delete doc;
		return NULL;
	}

	return HQ_NEW HQEngineResLoadSessionImpl(doc);
}

bool HQEngineResManagerImpl::HasMoreResources(HQEngineResLoadSession* session)
{
	HQEngineResLoadSessionImpl * resLoadSession = static_cast <HQEngineResLoadSessionImpl*> (session);
	if (resLoadSession == NULL)
		return false;
	return resLoadSession->HasMoreResources();
}

HQReturnVal HQEngineResManagerImpl::AddNextResource(HQEngineResLoadSession* session)
{
	HQEngineResLoadSessionImpl * resLoadSession = static_cast <HQEngineResLoadSessionImpl*> (session);

	if (resLoadSession->HasMoreResources() == false)
		return HQ_FAILED_NO_MORE_RESOURCE;
	switch (resLoadSession-> m_type)
	{
	case HQ_ERLT_XML:
		return this->LoadResourceFromXML(resLoadSession->NextXMLResource());
	break;
	} //switch (resLoadSession-> m_type)

	return HQ_OK;
}

HQReturnVal HQEngineResManagerImpl::EndAddResources(HQEngineResLoadSession* session)
{
	HQEngineResLoadSessionImpl * resLoadSession = static_cast <HQEngineResLoadSessionImpl*> (session);

	if (resLoadSession != NULL)
		delete resLoadSession;

	return HQ_OK;
}

HQReturnVal HQEngineResManagerImpl::LoadResourceFromXML(TiXmlElement* item)
{
	HQReturnVal re = HQ_OK;
	if (strcmp(item->Value(), "texture") == 0)
	{
		//texture resource
		if (LoadTextureFromXML(item) != HQ_OK)
			re = HQ_FAILED;
	}
	else if (strcmp(item->Value(), "shader") == 0)
	{
		//shader resource
		if (LoadShaderFromXML(item) != HQ_OK)
			re = HQ_FAILED;
	}
	else if (strcmp(item->Value(), "shader_program") == 0)
	{
		//shader program resource
		if (LoadShaderProgramFromXML(item) != HQ_OK)
			re = HQ_FAILED;
	}
		
	return re;
}

HQReturnVal HQEngineResManagerImpl::LoadTextureFromXML(TiXmlElement* textureItem)
{
	return HQ_OK;
}

HQReturnVal HQEngineResManagerImpl::LoadShaderFromXML(TiXmlElement* textureItem)
{
	return HQ_OK;
}

HQReturnVal HQEngineResManagerImpl::LoadShaderProgramFromXML(TiXmlElement* textureItem)
{
	return HQ_OK;
}

HQEngineTextureResource * HQEngineResManagerImpl::GetTextureResource(const char* name)
{
	return m_textures.GetItemPointer(name);
}

HQEngineShaderResource * HQEngineResManagerImpl::GetShaderResource(const char* name)
{
	return m_shaders.GetItemPointer(name);
}

HQEngineShaderProgramResource * HQEngineResManagerImpl::GetShaderProgramResource(const char* name)
{
	return m_shaderPrograms.GetItemPointer(name);
}


HQReturnVal HQEngineResManagerImpl::RemoveTextureResource(HQEngineTextureResource* res)
{
	m_textures.Remove(res->GetName());
	return HQ_OK;//TO DO: error report
}

HQReturnVal HQEngineResManagerImpl::RemoveShaderResource(HQEngineShaderResource* res)
{
	m_shaders.Remove(res->GetName());
	return HQ_OK;//TO DO: error report
}

HQReturnVal HQEngineResManagerImpl::RemoveShaderProgramResource(HQEngineShaderProgramResource* res)
{
	m_shaderPrograms.Remove(res->GetName());
	return HQ_OK;//TO DO: error report
}

void HQEngineResManagerImpl::RemoveAllResources()
{
	m_textures.RemoveAll();
	m_shaders.RemoveAll();
	m_shaderPrograms.RemoveAll();
}
