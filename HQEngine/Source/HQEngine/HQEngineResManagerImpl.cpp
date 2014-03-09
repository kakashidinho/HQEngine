/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQEnginePCH.h"
#include "../HQEngineApp.h"
#include "../HQGrowableArray.h"
#include "HQEngineResManagerImpl.h"

//TO DO: reload resources when device lost

/*------------texture resource----------*/
HQEngineTextureResImpl::HQEngineTextureResImpl(const char* name)
: HQNamedGraphicsRelatedObj(name),
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
HQEngineShaderResImpl::HQEngineShaderResImpl(const char* name)
: HQNamedGraphicsRelatedObj(name),
m_shaderID(HQ_NULL_ID)
{
}
HQEngineShaderResImpl::~HQEngineShaderResImpl()
{
	if (m_shaderID != HQ_NULL_ID)
		m_renderDevice->GetShaderManager()->DestroyShader(m_shaderID);
}

/*----------------resource loading session----------------*/
HQEngineResLoadSessionImpl::HQEngineResLoadSessionImpl(TiXmlDocument* doc)
: m_resXml(doc), m_type(HQ_ERLT_XML)
{
	if (m_resXml != NULL)
	{
		m_resourceItem = NULL;
		m_resGroup = m_resXml->FirstChildElement("resources");
		while (m_resourceItem == NULL && m_resGroup != NULL)
		{
			m_resourceItem = m_resGroup->FirstChildElement();
			if (m_resourceItem == NULL)//try to find in next resource group
				m_resGroup = m_resGroup->NextSiblingElement("resources");
		} 
	}
	else
	{	
		m_resGroup = m_resourceItem = NULL;
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
		return (m_resourceItem != NULL);

	default:
		return false;
	}
}

TiXmlElement * HQEngineResLoadSessionImpl::CurrentXMLResource()
{
	return m_type == HQ_ERLT_XML ? m_resourceItem: NULL;
}

TiXmlElement * HQEngineResLoadSessionImpl::NextXMLResource() {
	if (m_type == HQ_ERLT_XML)
	{
		if (m_resourceItem == NULL)
			return NULL;
		TiXmlElement * re = m_resourceItem;

		m_resourceItem = m_resourceItem->NextSiblingElement();//advance to next item

		while (m_resourceItem == NULL && m_resGroup != NULL)//try to find in next group
		{
			m_resGroup = m_resGroup->NextSiblingElement("resources");
			if (m_resGroup != NULL)
				m_resourceItem = m_resGroup->FirstChildElement();
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

HQReturnVal HQEngineResManagerImpl::AddResourcesFromXML(const char* fileName)
{
	HQEngineResLoadSession * session = this->BeginAddResourcesFromXML(fileName);
	if (session == NULL)
		return HQ_FAILED;
	
	HQReturnVal re = HQ_OK;

	while (this->HasMoreResources(session)){
		if (this->AddNextResource(session) != HQ_OK)
			re = HQ_FAILED;
	}

	this->EndAddResources(session);
	

	return re;
}

HQEngineResLoadSession* HQEngineResManagerImpl::BeginAddResourcesFromXML(const char* fileName)
{
	HQDataReaderStream* data_stream = HQEngineApp::GetInstance()->OpenFileForRead(fileName);
	if (data_stream == NULL)
	{
		this->Log("Error : Could not load resources from file %s!", fileName);
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
		this->Log("Error : Could not load resources from file %s!", fileName);
		delete doc;
		data_stream->Release();
		return NULL;
	}

	data_stream->Release();

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
		
	return re;
}

HQReturnVal HQEngineResManagerImpl::LoadTextureFromXML(TiXmlElement* textureItem)
{
	/*
	Texture resource format:
	<texture name="texture1">
		<type>2d/cube</type> <!-- default to 2D if not specified -->
		<render_target> <!-- info for render target -->
			<size> 
				<width>100</width>
				<height>100</height>
			</size>
			<format>
				r32f/r16f/rgba32f/rgba16f/rg32f/rg16f/rgba8ub/r8ui/a8ui
			</format>
		</render_target>
		<src generate_mipmap="true/false">image.bmp</src> <!-- image file, "src", "cube_src" and "renderTarget" elements are mutually exclusive -->
		<cube_src generate_mipmap="true/false">
			<positive_x>img1.bmp</positive_x>
			<negative_x>img2.bmp</negative_x>
			<positive_y>img3.bmp</positive_y>
			<negative_y>img4.bmp</negative_y>
			<positive_z>img5.bmp</positive_z>
			<negative_z>img6.bmp</negative_z>
		</cube_src>
	</texture>
	*/
	HQTextureType textureType = HQ_TEXTURE_2D;

	//get type info
	TiXmlElement * typeInfo = textureItem->FirstChildElement("type");
	if (typeInfo != NULL)
	{
		if (strcmp(typeInfo->GetText(), "cube") == 0 )
		{
			textureType = HQ_TEXTURE_CUBE;
		}
	}
	//
	const char * res_Name = NULL;
	bool genMipmap = false;
	TiXmlElement * texture_info = NULL;
	TiXmlElement * texture_info_elem = NULL;

	res_Name = textureItem->Attribute("name");
	//if texture is render target
	texture_info = textureItem->FirstChildElement("render_target");
	if (texture_info != NULL)
	{
		if (res_Name == NULL)
		{
			Log("Error : Cannot create render target texture resource without name!");
			return HQ_FAILED;
		}
		hquint32 width, height;
		HQRenderTargetFormat format;
		bool hasMipmap;
		const char* hasMipmapStr = texture_info->Attribute("has_mipmap");
		hasMipmap = hasMipmapStr != NULL? strcmp(hasMipmapStr, "true") == 0: false;

		//size
		texture_info_elem = texture_info->FirstChildElement("size");
		if (texture_info_elem == NULL)
		{
			Log("Error : Render target texture missing size info!");
			return HQ_FAILED;
		}
		TiXmlElement* widthInfo = texture_info_elem->FirstChildElement("width");
		TiXmlElement* heightInfo = texture_info_elem->FirstChildElement("height");
		if (widthInfo == NULL || heightInfo == NULL)
		{
			Log("Error : Render target texture missing size info!");
			return HQ_FAILED;
		}
		sscanf(widthInfo->GetText(), "%u", &width);
		sscanf(heightInfo->GetText(), "%u", &height);

		//format
		TiXmlElement* formatInfo = texture_info->FirstChildElement("format");
		if (formatInfo == NULL)
		{
			Log("Error : Render target texture missing format info!");
			return HQ_FAILED;
		}

		const char *formatStr = formatInfo->GetText();
		if (!strcmp(formatStr, "r32f"))
			format = HQ_RTFMT_R_FLOAT32;
		else if (!strcmp(formatStr, "r16f"))
			format = HQ_RTFMT_R_FLOAT16;
		else if (!strcmp(formatStr, "rg32f"))
			format = HQ_RTFMT_RG_FLOAT64;
		else if (!strcmp(formatStr, "rg16f"))
			format = HQ_RTFMT_RG_FLOAT32;
		else if (!strcmp(formatStr, "rgba32f"))
			format = HQ_RTFMT_RGBA_FLOAT128;
		else if (!strcmp(formatStr, "rgba16f"))
			format = HQ_RTFMT_RGBA_FLOAT64;
		else if (!strcmp(formatStr, "rgba8ub"))
			format = HQ_RTFMT_RGBA_32;
		else if (!strcmp(formatStr, "r8ui"))
			format = HQ_RTFMT_R_UINT8;
		else if (!strcmp(formatStr, "a8ui"))
			format = HQ_RTFMT_A_UINT8;
		else
		{
			Log("Error : Render target texture has invalid format info!");
			return HQ_FAILED;
		}

		return this->AddRenderTargetTextureResource(res_Name,
														width, height,
														hasMipmap,
														format,
														HQ_MST_NONE,
														textureType);
	}//if (texture_info != NULL)
	else
	{
		texture_info = textureItem->FirstChildElement("src");
		if (texture_info != NULL)
		{
			const char * src_file_name = NULL;
			//if texture source is one image only
			src_file_name = texture_info->GetText();

			//generate mipmap for image?
			const char* genMipStr = texture_info->Attribute("generate_mipmap");
			genMipmap = genMipStr != NULL? strcmp(genMipStr, "true") == 0 : false;

			if (res_Name == NULL) res_Name = src_file_name;

			return this->AddTextureResource(res_Name,
											src_file_name,
											genMipmap,
											textureType);
		}
		else if (textureType == HQ_TEXTURE_CUBE) {
			texture_info = textureItem->FirstChildElement("cube_src");
			if (texture_info != NULL)
			{
				//if texture source are 6 images
				const char * src_file_names[6] = {NULL};
				if (res_Name == NULL)
				{
					Log("Error : Cannot create render target texture resource without name!");
					return HQ_FAILED;
				}

				const char* genMipStr = texture_info->Attribute("generate_mipmap");
				genMipmap = genMipStr != NULL? strcmp(genMipStr, "true") == 0 : false;

				texture_info_elem = texture_info->FirstChildElement("positive_x");
				if (texture_info_elem != NULL)
					src_file_names[0] = texture_info_elem->GetText();
				else
				{
					Log("Error : Cube Texture Reource missing positive_x image!");
					return HQ_FAILED;
				}

				texture_info_elem = texture_info->FirstChildElement("negative_x");
				if (texture_info_elem != NULL)
					src_file_names[1] = texture_info_elem->GetText();
				else
				{
					Log("Error : Cube Texture Reource missing negative_x image!");
					return HQ_FAILED;
				}

				texture_info_elem = texture_info->FirstChildElement("positive_y");
				if (texture_info_elem != NULL)
					src_file_names[2] = texture_info_elem->GetText();
				else
				{
					Log("Error : Cube Texture Reource missing positive_y image!");
					return HQ_FAILED;
				}

				texture_info_elem = texture_info->FirstChildElement("negative_y");
				if (texture_info_elem != NULL)
					src_file_names[3] = texture_info_elem->GetText();
				else
				{
					Log("Error : Cube Texture Reource missing negative_y image!");
					return HQ_FAILED;
				}

				texture_info_elem = texture_info->FirstChildElement("positive_z");
				if (texture_info_elem != NULL)
					src_file_names[4] = texture_info_elem->GetText();
				else
				{
					Log("Error : Cube Texture Reource missing positive_z image!");
					return HQ_FAILED;
				}

				texture_info_elem = texture_info->FirstChildElement("negative_z");
				if (texture_info_elem != NULL)
					src_file_names[5] = texture_info_elem->GetText();
				else
				{
					Log("Error : Cube Texture Reource missing negative_z image!");
					return HQ_FAILED;
				}

				return this->AddCubeTextureResource(res_Name,
					src_file_names,
					genMipmap);
			}//"cube_src" element
		}//"src" element
	}//"render_target" element

	this->Log("Error : Resource lacks informations.");

	return HQ_FAILED;
}

HQReturnVal HQEngineResManagerImpl::LoadShaderFromXML(TiXmlElement* shaderItem)
{
	/*
	Shader resource format:
	<shader name="shader1"> <!-- name will be default to source file name -->
		<type>vertex shader</type> <!-- vertex/pixel/geometry shader. default to vertex -->
		<src_type>hlsl/glsl/cg/bytecode</src_type> <!-- default is cg -->
		<entry>main</entry> <!-- entry function. default is 'main'. ignored in bytecode/glsl src_type -->
		<src>file.txt</src> <!-- source file -->
		<definition name="name1">value</definition> <!-- macro definition for compiling the source code. Can have more than one or none. -->
	</shader>
	*/
	const char* res_name = shaderItem->Attribute("name");
	const char* src_file = NULL;
	const char defaultEntry[] = "main";
	const char *entry = defaultEntry;
	HQShaderType shaderType = HQ_VERTEX_SHADER;
	HQShaderCompileMode compileMode = HQ_SCM_CG;
	bool byteCode = false;
	HQGrowableArray<HQShaderMacro> macros;

	TiXmlElement * item_elem = shaderItem->FirstChildElement();
	while(item_elem != NULL)
	{
		const char *elemName = item_elem->Value();
		const char *elemStr = item_elem->GetText();
		//source file
		if (!strcmp(elemName, "src"))
		{
			src_file = elemStr;
		}
		else if (!strcmp(elemName, "type"))
		{
			if (!strcmp(elemStr, "vertex shader"))
				shaderType = HQ_VERTEX_SHADER;
			else if (!strcmp(elemStr, "pixel shader"))
				shaderType = HQ_PIXEL_SHADER;
			else if (!strcmp(elemStr, "geometry shader"))
				shaderType = HQ_GEOMETRY_SHADER;
			else
			{
				Log("Error : unknown shader type %s!", elemStr);
				return HQ_FAILED;
			}
		}//else if (!strcmp(elemName, "type"))
		else if (!strcmp(elemName, "src_type"))
		{
			byteCode = false;
			if (!strcmp(elemStr, "cg"))
			{
				compileMode = HQ_SCM_CG;
			}
			else if (!strcmp(elemStr, "hlsl"))
			{
				compileMode = HQ_SCM_HLSL_10;
			}
			else if (!strcmp(elemStr, "glsl"))
			{
				compileMode = HQ_SCM_GLSL;
			}
			else if (!strcmp(elemStr, "bytecode"))
			{
				byteCode = true;
			}
			else
			{
				Log("Error : unknown shader source type %s!", elemStr);
				return HQ_FAILED;
			}
		}//else if (!strcmp(elemName, "src_type"))
		else if (!strcmp(elemName, "entry"))
		{
			entry = elemStr;
		}//else if (!strcmp(elemName, "entry"))
		else if (!strcmp(elemName, "definition"))
		{
			const char emptyDef[] = "";
			HQShaderMacro newMacro;
			newMacro.name = item_elem->Attribute("name");
			newMacro.definition = elemStr != NULL? elemStr : emptyDef;
			if (newMacro.name == NULL)
			{
				Log("Warning : Shader resource loading ignored no named definition!");
			}
			else{
				macros.Add(newMacro);
			}
		}//else if (!strcmp(elemName, "definition"))

		item_elem = item_elem->NextSiblingElement();
	}//while(item_elem != NULL)
	
	if (src_file == NULL)
	{
		Log("Error : shader resource missing source file!");
		return HQ_FAILED;
	}
	if (res_name == NULL) res_name = src_file;//if there is no name. default name is source file name
	
	if (byteCode)
		return this->AddShaderResourceFromByteCode(res_name, shaderType, src_file);
	else
	{
		HQShaderMacro end = {NULL, NULL};
		macros.Add(end);
		return this->AddShaderResource(
							res_name,
							shaderType,
							compileMode,
							src_file,
							macros,
							entry);
	}

	return HQ_OK;
}

HQReturnVal HQEngineResManagerImpl::AddTextureResource(const char *name,
											const char * image_file,
											bool generateMipmap,
											HQTextureType type
											)
{
	if (m_textures.GetItemPointer(name) != NULL)
	{
		this->Log("Error : could not create already existing texture resource named %s", name);
		return HQ_FAILED_RESOURCE_EXISTS;
	}
	//open data stream
	HQDataReaderStream * stream = HQEngineApp::GetInstance()->OpenFileForRead(image_file);
	if (stream == NULL)
	{
		Log("Error : could not load texture resource from %s!", image_file);
		return HQ_FAILED;
	}

	//create texture
	hquint32 textureID;
	HQReturnVal re = HQEngineApp::GetInstance()->GetRenderDevice()->GetTextureManager()
		->AddTexture(stream, 1.0f, NULL, 0, generateMipmap, type, &textureID);
	stream->Release();

	if (HQFailed(re))
		return re;

	//succeeded, now create resource
	m_textures.Add(name, HQ_NEW HQEngineTextureResImpl(name));
	HQEngineTextureResImpl *newRes = (HQEngineTextureResImpl*) this->GetTextureResource(name);

	newRes->Init(textureID);

	return HQ_OK;
}

HQReturnVal HQEngineResManagerImpl::AddCubeTextureResource(const char *name,
											const char * image_files[6],
											bool generateMipmap
											)
{
	if (m_textures.GetItemPointer(name) != NULL)
	{
		this->Log("Error : could not create already existing texture resource named %s", name);
		return HQ_FAILED_RESOURCE_EXISTS;
	}
	//open data streams
	HQDataReaderStream * streams[6] = {NULL};
	
	for (int i = 0; i < 6; ++i)
	{
		streams[i] = HQEngineApp::GetInstance()->OpenFileForRead(image_files[i]);
		if (streams[i] == NULL)
		{
			for (int j = 0; j < i; ++j)
				streams[j]->Release();//close previous streams
			Log("Error : could not load texture resource from %s!", image_files[i]);
			return HQ_FAILED;
		}
	}

	//create texture
	hquint32 textureID;
	HQReturnVal re = HQEngineApp::GetInstance()->GetRenderDevice()->GetTextureManager()
		->AddCubeTexture(streams, 1.f, NULL, 0, generateMipmap, &textureID);
	for (int j = 0; j < 6; ++j)
		streams[j]->Release();//close streams

	if (HQFailed(re))
		return re;

	//succeeded, now create resource
	m_textures.Add(name, HQ_NEW HQEngineTextureResImpl(name));
	HQEngineTextureResImpl *newRes = (HQEngineTextureResImpl*) this->GetTextureResource(name);

	newRes->Init(textureID);

	return HQ_OK;
}

HQReturnVal HQEngineResManagerImpl::AddRenderTargetTextureResource(
								  const char *name,
								  hq_uint32 width , hq_uint32 height,
								  bool hasMipmaps,
								  HQRenderTargetFormat format , 
								  HQMultiSampleType multisampleType,
								  HQTextureType textureType)
{
	
	if (m_textures.GetItemPointer(name) != NULL)
	{
		this->Log("Error : could not create already existing texture resource named %s", name);
		return HQ_FAILED_RESOURCE_EXISTS;
	}

	//create render target
	hquint32 textureID, renderTargetID;
	HQReturnVal re = HQEngineApp::GetInstance()->GetRenderDevice()->GetRenderTargetManager()
		->CreateRenderTargetTexture(width, height, hasMipmaps, format, multisampleType, textureType, 
									&renderTargetID, &textureID);

	if (HQFailed(re))
		return re;

	//succeeded, now create resource
	m_textures.Add(name, HQ_NEW HQEngineTextureResImpl(name));
	HQEngineTextureResImpl *newRes = (HQEngineTextureResImpl*) this->GetTextureResource(name);

	newRes->Init(textureID, renderTargetID);

	return HQ_OK;
}

HQReturnVal HQEngineResManagerImpl::AddShaderResource(
									 const char * name,
									 HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char * source_file,
									 const HQShaderMacro * pDefines,
									 const char* entryFunctionName)
{
	if (m_shaders.GetItemPointer(name) != NULL)
	{
		this->Log("Error : could not create already existing shader resource named %s", name);
		return HQ_FAILED_RESOURCE_EXISTS;
	}
	//open data stream
	HQDataReaderStream * stream = HQEngineApp::GetInstance()->OpenFileForRead(source_file);
	if (stream == NULL)
	{
		Log("Error : could not load shader resource from %s!", source_file);
		return HQ_FAILED;
	}

	//create shader
	hquint32 shaderID;
	HQReturnVal re = HQEngineApp::GetInstance()->GetRenderDevice()->GetShaderManager()
		->CreateShaderFromStream(type, compileMode, stream, pDefines, entryFunctionName, &shaderID);
	stream->Release();

	if (HQFailed(re))
		return re;

	//succeeded, now create resource
	m_shaders.Add(name, HQ_NEW HQEngineShaderResImpl(name));
	HQEngineShaderResImpl *newRes = (HQEngineShaderResImpl*)this->GetShaderResource(name);

	newRes->Init(type, shaderID);

	return HQ_OK;
}

HQReturnVal HQEngineResManagerImpl::AddShaderResourceFromByteCode(
								 const char *name,
								 HQShaderType type,
								 const char * source_file)
{
	if (m_shaders.GetItemPointer(name) != NULL)
	{
		this->Log("Error : could not create already existing shader resource named %s", name);
		return HQ_FAILED_RESOURCE_EXISTS;
	}
	//open data stream
	HQDataReaderStream * stream = HQEngineApp::GetInstance()->OpenFileForRead(source_file);
	if (stream == NULL)
	{
		Log("Error : could not load shader resource from %s!", source_file);
		return HQ_FAILED;
	}

	//create shader
	hquint32 shaderID;
	HQReturnVal re = HQEngineApp::GetInstance()->GetRenderDevice()->GetShaderManager()
		->CreateShaderFromByteCodeStream(type, stream, &shaderID);
	stream->Release();

	if (HQFailed(re))
		return re;

	//succeeded, now create resource
	m_shaders.Add(name, HQ_NEW HQEngineShaderResImpl(name));
	HQEngineShaderResImpl *newRes = (HQEngineShaderResImpl*)this->GetShaderResource(name);

	newRes->Init(type, shaderID);

	return HQ_OK;
}

HQEngineTextureResource * HQEngineResManagerImpl::GetTextureResource(const char* name)
{
	HQSharedPtr<HQEngineTextureResImpl>* ppRes = m_textures.GetItemPointer(name);
	if (ppRes == NULL)
		return NULL;
	return ppRes->GetRawPointer();
}

const HQSharedPtr<HQEngineTextureResImpl>& HQEngineResManagerImpl::GetTextureResourceSharedPtr(const char* name)
{
	bool found = false;
	return m_textures.GetItem(name, found);
}

const HQSharedPtr<HQEngineShaderResImpl>& HQEngineResManagerImpl::GetShaderResourceSharedPtr(const char* name)
{
	bool found = false;
	return m_shaders.GetItem(name, found);
}

HQEngineShaderResource * HQEngineResManagerImpl::GetShaderResource(const char* name)
{
	HQSharedPtr<HQEngineShaderResImpl>* ppRes = m_shaders.GetItemPointer(name);
	if (ppRes == NULL)
		return NULL;
	return ppRes->GetRawPointer();
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


void HQEngineResManagerImpl::RemoveAllResources()
{
	m_textures.RemoveAll();
	m_shaders.RemoveAll();
}
