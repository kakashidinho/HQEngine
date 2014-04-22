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

#include <sstream>

//TO DO: reload resources when device lost

//parser's functions and variables
extern HQDataReaderStream * hqengine_res_parser_input_stream ;
extern std::stringstream	* hqengine_res_parser_log_stream ;
extern HQEngineResParserNode * hqengine_res_parser_root_result;

extern int hqengine_res_parser_scan();
extern void hqengine_res_parser_recover_from_error();
extern void hqengine_res_parser_clean_up();

/*------------texture resource----------*/
HQEngineTextureResImpl::HQEngineTextureResImpl(const char* name)
: HQNamedGraphicsRelatedObj(name),
m_texture(NULL), m_renderTarget(NULL)
{
}

HQEngineTextureResImpl::~HQEngineTextureResImpl()
{
	if (m_texture != NULL)
		m_renderDevice->GetTextureManager()->RemoveTexture(m_texture);
	if (m_renderTarget != NULL)
		m_renderDevice->GetRenderTargetManager()->RemoveRenderTarget(m_renderTarget);
}

void HQEngineTextureResImpl::GetTexture2DSize(hquint32 &width, hquint32 &height) const
{
	width = m_texture->GetWidth();
	height = m_texture->GetHeight();
}


/*-----------shader resource------------------------------*/
HQEngineShaderResImpl::HQEngineShaderResImpl(const char* name)
: HQNamedGraphicsRelatedObj(name),
m_shader(NULL)
{
}
HQEngineShaderResImpl::~HQEngineShaderResImpl()
{
	if (m_shader != NULL)
		m_renderDevice->GetShaderManager()->RemoveShader(m_shader);
}

/*--------------buffer resource----------------*/
HQEngineShaderBufferResImpl::HQEngineShaderBufferResImpl(const char *name)
:HQNamedGraphicsRelatedObj(name),
m_buffer(NULL)
{

}
HQEngineShaderBufferResImpl::~HQEngineShaderBufferResImpl()
{
	if (m_buffer != NULL)
	{
		switch (m_type)
		{
		case HQ_ESBT_VERTEX:
			HQEngineApp::GetInstance()->GetRenderDevice()->GetVertexStreamManager()
				->RemoveVertexBuffer(m_buffer);
			break;
		case HQ_ESBT_INDEX:
			HQEngineApp::GetInstance()->GetRenderDevice()->GetVertexStreamManager()
				->RemoveIndexBuffer(m_buffer);
			break;//case HQ_ESBT_INDEX:
		case HQ_ESBT_COMPUTE_INDIRECT:
		case HQ_ESBT_DRAW_INDIRECT:
		case HQ_ESBT_DRAW_INDEXED_INDIRECT:
		case HQ_ESBT_SHADER_USE_ONLY:
			HQEngineApp::GetInstance()->GetRenderDevice()->GetShaderManager()
				->RemoveBufferUAV(m_buffer);
			break;
		}
	}
}

void HQEngineShaderBufferResImpl::Init(HQEngineShaderBufferType type, HQBufferUAV* buffer, hquint32 numElements, hquint32 elementSize)
{
	m_type = type;
	m_buffer = buffer;
	m_elementSize = elementSize;
	m_numElements = numElements;
}

/*----------------resource loading session----------------*/
HQEngineResLoadSessionImpl::HQEngineResLoadSessionImpl(HQEngineResParserNode* root)
: m_root(root), m_type(HQ_ERLT_STANDARD)
{
	if (m_root != NULL)
	{
		m_resourceItem = NULL;
		m_resGroup = m_root->GetFirstChild("resources");
		while (m_resourceItem == NULL && m_resGroup != NULL)
		{
			m_resourceItem = m_resGroup->GetFirstChild();
			if (m_resourceItem == NULL)//try to find in next resource group
				m_resGroup = m_resGroup->GetNextSibling("resources");
		} 
	}
	else
	{	
		m_resGroup = m_resourceItem = NULL;
	}
}

HQEngineResLoadSessionImpl::~HQEngineResLoadSessionImpl()
{
	HQEngineResParserNode::DeleteTree(m_root);
}

bool HQEngineResLoadSessionImpl::HasMoreResources() const
{
	switch(m_type)
	{
	case HQ_ERLT_STANDARD:
		if (m_root == NULL)
			return false;
		return (m_resourceItem != NULL);

	default:
		return false;
	}
}

const HQEngineResParserNode * HQEngineResLoadSessionImpl::CurrentResource()
{
	return m_type == HQ_ERLT_STANDARD ? m_resourceItem: NULL;
}

const HQEngineResParserNode * HQEngineResLoadSessionImpl::NextResource() {
	if (m_type == HQ_ERLT_STANDARD)
	{
		if (m_resourceItem == NULL)
			return NULL;
		const HQEngineResParserNode * re = m_resourceItem;

		m_resourceItem = m_resourceItem->GetNextSibling();//advance to next item

		while (m_resourceItem == NULL && m_resGroup != NULL)//try to find in next group
		{
			m_resGroup = m_resGroup->GetNextSibling("resources");
			if (m_resGroup != NULL)
				m_resourceItem = m_resGroup->GetFirstChild();
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

void HQEngineResManagerImpl::SetSuffix(const char* suffix)
{
	if (suffix)
		m_suffix = suffix;
}

HQReturnVal HQEngineResManagerImpl::AddResourcesFromFile(const char* fileName)
{
	HQEngineResLoadSession * session = this->BeginAddResourcesFromFile(fileName);
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

HQEngineResLoadSession* HQEngineResManagerImpl::BeginAddResourcesFromFile(const char* raw_fileName)
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
			this->Log("Error : Could not load resources from file %s! Could not open the file!", raw_fileName);
			return NULL;
		}
	}

	//prepare the parser
	std::stringstream log_stream;
	hqengine_res_parser_input_stream = data_stream;
	hqengine_res_parser_log_stream = &log_stream;

	//now parse the script
	if (hqengine_res_parser_scan())
	{
		this->Log("Error : Could not load resources from file %s! %s", raw_fileName, hqengine_res_parser_log_stream->str().c_str());
		HQEngineHelper::GlobalPoolReleaseAll();
		data_stream->Release();
		return NULL;
	}
	data_stream->Release();

	HQEngineResParserNode *result = hqengine_res_parser_root_result;
	hqengine_res_parser_root_result = NULL;

	this->Log("Resource loading session from file '%s' started!", raw_fileName);

	return HQ_NEW HQEngineResLoadSessionImpl(result);
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
	case HQ_ERLT_STANDARD:
		return this->LoadResource(resLoadSession->NextResource());
	break;
	} //switch (resLoadSession-> m_type)

	return HQ_OK;
}

HQReturnVal HQEngineResManagerImpl::EndAddResources(HQEngineResLoadSession* session)
{
	HQEngineResLoadSessionImpl * resLoadSession = static_cast <HQEngineResLoadSessionImpl*> (session);

	if (resLoadSession != NULL)
		delete resLoadSession;

	//release all memory blocks allocated for parser
	HQEngineHelper::GlobalPoolReleaseAll();

	return HQ_OK;
}

HQReturnVal HQEngineResManagerImpl::LoadResource(const HQEngineResParserNode* item)
{
	HQReturnVal re = HQ_OK;
	if (strcmp(item->GetType(), "texture") == 0)
	{
		//texture resource
		if (LoadTexture(item) != HQ_OK)
			re = HQ_FAILED;
	}
	else if (strcmp(item->GetType(), "texture_uav") == 0)
	{
		//texture resource
		if (LoadTextureUAV(item) != HQ_OK)
			re = HQ_FAILED;
	}
	else if (strcmp(item->GetType(), "render_target") == 0)
	{
		//texture resource
		if (LoadTexture(item, true) != HQ_OK)
			re = HQ_FAILED;
	}
	else if (strcmp(item->GetType(), "shader") == 0)
	{
		//shader resource
		if (LoadShader(item) != HQ_OK)
			re = HQ_FAILED;
	}

	else if (strcmp(item->GetType(), "buffer_uav") == 0)
	{
		//buffer resource
		if (LoadBuffer(item) != HQ_OK)
			re = HQ_FAILED;
	}
		
	return re;
}

HQReturnVal HQEngineResManagerImpl::LoadTexture(const HQEngineResParserNode* texture_info, bool renderTarget)
{
	HQTextureType textureType = HQ_TEXTURE_2D;
	typedef HQEngineResParserNode::ValueType NodeAttrType;

	//get type info
	const HQEngineResParserNode * typeInfo = texture_info->GetFirstChild("type");
	if (typeInfo != NULL)
	{
		const NodeAttrType &value = typeInfo->GetAttribute("value");
		const char *valueStr = value.GetAsString();
		if (valueStr != NULL)
		{
			if (strcmp(valueStr, "cube") == 0 )
				textureType = HQ_TEXTURE_CUBE;
		}
	}
	//
	const char * res_Name = NULL;
	bool genMipmap = false;
	const HQEngineResParserNode * texture_info_elem = NULL;

	res_Name = texture_info->GetStrAttribute("name");
	//if texture is render target
	if (renderTarget)
	{
		if (res_Name == NULL)
		{
			Log("Error : %d : Cannot create render target texture resource without name!", texture_info->GetSourceLine());
			return HQ_FAILED;
		}
		hquint32 width, height;
		HQRenderTargetFormat format;
		bool hasMipmap;
		const char* hasMipmapStr = texture_info->GetStrAttribute("has_mipmap");
		hasMipmap = hasMipmapStr != NULL? strcmp(hasMipmapStr, "true") == 0: false;

		//size
		const HQEngineResParserNode* widthInfo = texture_info->GetFirstChild("width");
		const HQEngineResParserNode* heightInfo = texture_info->GetFirstChild("height");
		if (widthInfo == NULL || heightInfo == NULL)
		{
			Log("Error : %d : Render target texture missing size info!", texture_info->GetSourceLine());
			return HQ_FAILED;
		}
		const hqint32 * wPtr = widthInfo->GetIntAttributePtr("value");
		const hqint32 * hPtr = heightInfo->GetIntAttributePtr("value");

		if (wPtr == NULL || hPtr == NULL)
		{
			Log("Error : %d : Render target texture has invalid size info!", texture_info->GetSourceLine());
			return HQ_FAILED;
		}

		width = (hquint32)*wPtr;
		height = (hquint32)*hPtr;

		//format
		const HQEngineResParserNode* formatInfo = texture_info->GetFirstChild("format");
		if (formatInfo == NULL)
		{
			Log("Error : %d : Render target texture missing format info!", texture_info->GetSourceLine());
			return HQ_FAILED;
		}

		const char *formatStr = formatInfo->GetStrAttribute("value");
		if (formatStr == NULL)
		{
			Log("Error : %d : Render target texture has invalid format info!", texture_info->GetSourceLine());
			return HQ_FAILED;
		}

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
			Log("Error : %d : Render target texture has invalid format info!", texture_info->GetSourceLine());
			return HQ_FAILED;
		}

		return this->AddRenderTargetTextureResource(res_Name,
														width, height,
														hasMipmap,
														format,
														HQ_MST_NONE,
														textureType);
	}//if (renderTarget)
	else
	{
		const HQEngineResParserNode * texture_image_info = texture_info->GetFirstChild("image");
		if (texture_image_info != NULL)
		{
			const char * src_file_name = NULL;
			//if texture source is one image only
			src_file_name = texture_image_info->GetStrAttribute("value");
			if (src_file_name == NULL)
			{
				Log("Error : %d : texture has invalid source image!", texture_image_info->GetSourceLine());
				return HQ_FAILED;
			}


			//generate mipmap for image?
			const char* genMipStr = texture_info->GetStrAttribute("generate_mipmap");
			genMipmap = genMipStr != NULL? strcmp(genMipStr, "true") == 0 : false;

			if (res_Name == NULL) res_Name = src_file_name;

			return this->AddTextureResource(res_Name,
											src_file_name,
											genMipmap,
											textureType);
		}//if (texture_image_info != NULL)
		else {
			//if texture source are 6 images
			const char * src_file_names[6] = {NULL};
			if (res_Name == NULL)
			{
				Log("Error : %d : Cannot create render target texture resource without name!", texture_info->GetSourceLine());
				return HQ_FAILED;
			}

			const char* genMipStr = texture_info->GetStrAttribute("generate_mipmap");
			genMipmap = genMipStr != NULL? strcmp(genMipStr, "true") == 0 : false;

			texture_info_elem = texture_info->GetFirstChild("positive_x");
			if (texture_info_elem != NULL)
				src_file_names[0] = texture_info_elem->GetStrAttribute("value");
			else
			{
				Log("Error : %d : Cube Texture Reource missing positive_x image!", texture_info->GetSourceLine());
				return HQ_FAILED;
			}

			texture_info_elem = texture_info->GetFirstChild("negative_x");
			if (texture_info_elem != NULL)
				src_file_names[1] = texture_info_elem->GetStrAttribute("value");
			else
			{
				Log("Error : %d : Cube Texture Reource missing negative_x image!", texture_info->GetSourceLine());
				return HQ_FAILED;
			}

			texture_info_elem = texture_info->GetFirstChild("positive_y");
			if (texture_info_elem != NULL)
				src_file_names[2] = texture_info_elem->GetStrAttribute("value");
			else
			{
				Log("Error : %d : Cube Texture Reource missing positive_y image!", texture_info->GetSourceLine());
				return HQ_FAILED;
			}

			texture_info_elem = texture_info->GetFirstChild("negative_y");
			if (texture_info_elem != NULL)
				src_file_names[3] = texture_info_elem->GetStrAttribute("value");
			else
			{
				Log("Error : %d : Cube Texture Reource missing negative_y image!", texture_info->GetSourceLine());
				return HQ_FAILED;
			}

			texture_info_elem = texture_info->GetFirstChild("positive_z");
			if (texture_info_elem != NULL)
				src_file_names[4] = texture_info_elem->GetStrAttribute("value");
			else
			{
				Log("Error : %d : Cube Texture Reource missing positive_z image!", texture_info->GetSourceLine());
				return HQ_FAILED;
			}

			texture_info_elem = texture_info->GetFirstChild("negative_z");
			if (texture_info_elem != NULL)
				src_file_names[5] = texture_info_elem->GetStrAttribute("value");
			else
			{
				Log("Error : %d : Cube Texture Reource missing negative_z image!", texture_info->GetSourceLine());
				return HQ_FAILED;
			}

			return this->AddCubeTextureResource(res_Name,
				src_file_names,
				genMipmap);
		}//cube 6 images case
	}//else of if (renderTarget)

	this->Log("Error : %d : Resource lacks informations.", texture_info->GetSourceLine());

	return HQ_FAILED;
}

HQReturnVal HQEngineResManagerImpl::LoadTextureUAV(const HQEngineResParserNode* texture_info)
{
	typedef HQEngineResParserNode::ValueType NodeAttrType;

	const char * res_Name = NULL;

	res_Name = texture_info->GetStrAttribute("name");
	if (res_Name == NULL)
	{
		Log("Error : %d : Cannot create UAV texture resource without name!", texture_info->GetSourceLine());
		return HQ_FAILED;
	}

	hquint32 width, height;
	HQTextureUAVFormat format;
	bool hasMipmap;
	const char* hasMipmapStr = texture_info->GetStrAttribute("has_mipmap");
	hasMipmap = hasMipmapStr != NULL ? strcmp(hasMipmapStr, "true") == 0 : false;

	//size
	const HQEngineResParserNode* widthInfo = texture_info->GetFirstChild("width");
	const HQEngineResParserNode* heightInfo = texture_info->GetFirstChild("height");
	if (widthInfo == NULL || heightInfo == NULL)
	{
		Log("Error : %d : UAV texture missing size info!", texture_info->GetSourceLine());
		return HQ_FAILED;
	}
	const hqint32 * wPtr = widthInfo->GetIntAttributePtr("value");
	const hqint32 * hPtr = heightInfo->GetIntAttributePtr("value");

	if (wPtr == NULL || hPtr == NULL)
	{
		Log("Error : %d : UAV texture has invalid size info!", texture_info->GetSourceLine());
		return HQ_FAILED;
	}

	width = (hquint32)*wPtr;
	height = (hquint32)*hPtr;

	//format
	const HQEngineResParserNode* formatInfo = texture_info->GetFirstChild("format");
	if (formatInfo == NULL)
	{
		Log("Error : %d : UAV texture missing format info!", texture_info->GetSourceLine());
		return HQ_FAILED;
	}

	const char *formatStr = formatInfo->GetStrAttribute("value");
	if (formatStr == NULL)
	{
		Log("Error : %d : UAV texture has invalid format info!", texture_info->GetSourceLine());
		return HQ_FAILED;
	}

	if (!strcmp(formatStr, "r32f"))
		format = HQ_UAVTFMT_R32_FLOAT;
	else if (!strcmp(formatStr, "r32i"))
		format = HQ_UAVTFMT_R32_INT;
	else if (!strcmp(formatStr, "r32ui"))
		format = HQ_UAVTFMT_R32_UINT;
	else if (!strcmp(formatStr, "r16f"))
		format = HQ_UAVTFMT_R16_FLOAT;
	else if (!strcmp(formatStr, "rg32f"))
		format = HQ_UAVTFMT_R32G32_FLOAT;
	else if (!strcmp(formatStr, "rg32i"))
		format = HQ_UAVTFMT_R32G32_INT;
	else if (!strcmp(formatStr, "rg32ui"))
		format = HQ_UAVTFMT_R32G32_UINT;
	else if (!strcmp(formatStr, "rg16f"))
		format = HQ_UAVTFMT_R16G16_FLOAT;
	else if (!strcmp(formatStr, "rgba32f"))
		format = HQ_UAVTFMT_R32G32B32A32_FLOAT;
	else if (!strcmp(formatStr, "rgba32i"))
		format = HQ_UAVTFMT_R32G32B32A32_INT;
	else if (!strcmp(formatStr, "rgba32ui"))
		format = HQ_UAVTFMT_R32G32B32A32_UINT;
	else if (!strcmp(formatStr, "rgba16f"))
		format = HQ_UAVTFMT_R16G16B16A16_FLOAT;
	else if (!strcmp(formatStr, "rgba8ub"))
		format = HQ_UAVTFMT_R8G8B8A8_UNORM;
	else
	{
		Log("Error : %d : UAV texture has invalid format info!", texture_info->GetSourceLine());
		return HQ_FAILED;
	}

	return this->AddTextureUAVResource(res_Name,
		format,
		width, height,
		hasMipmap);
}

HQReturnVal HQEngineResManagerImpl::LoadShader(const HQEngineResParserNode* shaderItem)
{
	const char* res_name = shaderItem->GetStrAttribute("name");
	const char* src_file = NULL;
	const char defaultEntry[] = "main";
	const char *entry = defaultEntry;
	HQShaderType shaderType = HQ_VERTEX_SHADER;
	HQShaderCompileMode compileMode = HQ_SCM_CG;
	bool byteCode = false;
	bool debug = false;
	HQGrowableArray<HQShaderMacro> macros;

	const HQEngineResParserNode * item_elem = shaderItem->GetFirstChild();
	while(item_elem != NULL)
	{
		int elemLine = item_elem->GetSourceLine();
		const char *elemName = item_elem->GetType();
		const char *elemStr = item_elem->GetStrAttribute("value");
		//source file
		if (!strcmp(elemName, "src"))
		{
			src_file = elemStr;
		}
		else if (!strcmp(elemName, "type"))
		{
			if (!strcmp(elemStr, "vertex"))
				shaderType = HQ_VERTEX_SHADER;
			else if (!strcmp(elemStr, "pixel"))
				shaderType = HQ_PIXEL_SHADER;
			else if (!strcmp(elemStr, "geometry"))
				shaderType = HQ_GEOMETRY_SHADER;
			else if (!strcmp(elemStr, "compute"))
				shaderType = HQ_COMPUTE_SHADER;
			else
			{
				Log("Error : %d : unknown shader type %s!", elemLine, elemStr);
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
				Log("Error : %d : unknown shader source type %s!", elemLine, elemStr);
				return HQ_FAILED;
			}
		}//else if (!strcmp(elemName, "src_type"))
		else if (!strcmp(elemName, "entry"))
		{
			entry = elemStr;
		}//else if (!strcmp(elemName, "entry"))
		else if (!strcmp(elemName, "debug"))
		{
			debug = strcmp(elemStr, "true") == 0;
		}
		else if (!strcmp(elemName, "definition"))
		{
			const char emptyDef[] = "";
			HQShaderMacro newMacro;
			newMacro.name = item_elem->GetStrAttribute("name");
			newMacro.definition = elemStr != NULL? elemStr : emptyDef;
			if (newMacro.name == NULL)
			{
				Log("Warning : %d : Shader resource loading ignored no named definition!", elemLine);
			}
			else{
				macros.Add(newMacro);
			}
		}//else if (!strcmp(elemName, "definition"))

		item_elem = item_elem->GetNextSibling();
	}//while(item_elem != NULL)
	
#if defined DEBUG || defined _DEBUG
	if (debug)
	{
		switch (compileMode)
		{
		case HQ_SCM_CG:
			compileMode = HQ_SCM_CG_DEBUG;
			break;
		case HQ_SCM_GLSL:
			compileMode = HQ_SCM_GLSL_DEBUG;
			break;
		case HQ_SCM_HLSL_10:
			compileMode = HQ_SCM_HLSL_10_DEBUG;
			break;
		}
	}
#endif

	if (src_file == NULL)
	{
		Log("Error : %d : shader resource missing source file!", shaderItem->GetSourceLine());
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

HQReturnVal HQEngineResManagerImpl::LoadBuffer(const HQEngineResParserNode* bufferItem)
{
	const char* res_name = bufferItem->GetStrAttribute("name");
	HQEngineShaderBufferType type = HQ_ESBT_SHADER_USE_ONLY;
	hquint32 numElements = 1;
	hquint32 elementSize = 4;

	if (res_name == NULL)
	{
		Log("Error : %d : Cannot create buffer resource without name!", bufferItem->GetSourceLine());
		return HQ_FAILED;
	}

	const HQEngineResParserNode * item_elem = bufferItem->GetFirstChild();
	while (item_elem != NULL)
	{
		int elemLine = item_elem->GetSourceLine();
		const char *elemName = item_elem->GetType();
		const char *elemStr = item_elem->GetStrAttribute("value");
		const hqint32 * valuePtr = item_elem->GetIntAttributePtr("value");
		if (!strcmp(elemName, "type"))
		{
			//vertex/index/draw_indirect/draw_indexed_indirect/compute_indirect
			if (!strcmp(elemStr, "vertex"))
				type = HQ_ESBT_VERTEX;
			else if (!strcmp(elemStr, "index"))
				type = HQ_ESBT_INDEX;
			else if (!strcmp(elemStr, "draw_indirect"))
				type = HQ_ESBT_DRAW_INDIRECT;
			else if (!strcmp(elemStr, "draw_indexed_indirect"))
				type = HQ_ESBT_DRAW_INDEXED_INDIRECT;
			else if (!strcmp(elemStr, "compute_indirect"))
				type = HQ_ESBT_COMPUTE_INDIRECT;
			else
			{
				Log("Error : %d : unknown buffer type '%s'!", elemLine, elemStr);
				return HQ_FAILED;
			}
		}//if (!strcmp(elemName, "type"))
		else if (!strcmp(elemName, "num_elements"))
		{
			if (valuePtr != NULL)
				numElements = *valuePtr;
			else
			{
				Log("Error : %d : invalid num_elements='%s'!", elemLine, elemStr);
				return HQ_FAILED;
			}
		}//else if (!strcmp(elemName, "num_elements"))
		else if (!strcmp(elemName, "element_size"))
		{
			if (valuePtr != NULL)
				elementSize = *valuePtr;
			else
			{
				Log("Error : %d : invalid element_size='%s'!", elemLine, elemStr);
				return HQ_FAILED;
			}
		}//else if (!strcmp(elemName, "element_size"))

		item_elem = item_elem->GetNextSibling();//next element
	}//while(item_elem != NULL)

	return this->AddShaderBufferResource(res_name, type, numElements, elementSize, NULL);
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
	HQTexture* textureID;
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
	HQTexture* textureID;
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

HQReturnVal HQEngineResManagerImpl::AddTextureUAVResource(const char *name,
	HQTextureUAVFormat format,
	hquint32 width, hquint32 height,
	bool hasMipmap
	)
{
	if (m_textures.GetItemPointer(name) != NULL)
	{
		this->Log("Error : could not create already existing texture resource named %s", name);
		return HQ_FAILED_RESOURCE_EXISTS;
	}
	//create new one
	HQTexture * texture;
	HQReturnVal re = HQEngineApp::GetInstance()->GetRenderDevice()->GetTextureManager()
		->AddTextureUAV(format, width, height, hasMipmap, &texture);

	if (HQFailed(re))
		return re;

	//succeeded, now create resource
	m_textures.Add(name, HQ_NEW HQEngineTextureResImpl(name));
	HQEngineTextureResImpl *newRes = (HQEngineTextureResImpl*) this->GetTextureResource(name);

	newRes->Init(texture, NULL);

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
	HQTexture* textureID;
	HQRenderTargetView* renderTargetID;
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
	HQShaderObject* shaderID;
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
	HQShaderObject* shaderID;
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


HQReturnVal HQEngineResManagerImpl::AddShaderBufferResource(
	const char *name,
	HQEngineShaderBufferType type,
	hquint32 numElements,
	hquint32 elementSize,
	void * initData
	)
{
	if (m_buffers.GetItemPointer(name) != NULL)
	{
		this->Log("Error : could not create already existing shader buffer resource named %s", name);
		return HQ_FAILED_RESOURCE_EXISTS;
	}

	//create shader buffer
	HQBufferUAV * buffer;
	HQReturnVal re = HQ_FAILED_INVALID_PARAMETER;
	switch (type)
	{
	case HQ_ESBT_VERTEX:
		re = HQEngineApp::GetInstance()->GetRenderDevice()->GetVertexStreamManager()
			->CreateVertexBufferUAV(initData, elementSize, numElements, &buffer);
		break;
	case HQ_ESBT_INDEX:
	{
			HQIndexDataType dataType;
			switch (elementSize){
			case 2:
				dataType = HQ_IDT_USHORT;
				break;
			case 4:
				dataType = HQ_IDT_UINT;
				break;
			default:
				Log("Error : could not create index buffer resource named %s with element size=%u!", name, elementSize);
				return HQ_FAILED_INVALID_PARAMETER;
			}
			re = HQEngineApp::GetInstance()->GetRenderDevice()->GetVertexStreamManager()
							  ->CreateIndexBufferUAV(initData, numElements, dataType, &buffer);
	}
		break;//case HQ_ESBT_INDEX:
	case HQ_ESBT_COMPUTE_INDIRECT:
		elementSize = 0;//ignore
		re = HQEngineApp::GetInstance()->GetRenderDevice()->GetShaderManager()
			->CreateComputeIndirectArgs(numElements, initData, &buffer);
		break;
	case HQ_ESBT_DRAW_INDIRECT:
		elementSize = 0;//ignore
		re = HQEngineApp::GetInstance()->GetRenderDevice()->GetShaderManager()
			->CreateDrawIndirectArgs(numElements, initData, &buffer);
		break;
	case HQ_ESBT_DRAW_INDEXED_INDIRECT:
		elementSize = 0;//ignore
		re = HQEngineApp::GetInstance()->GetRenderDevice()->GetShaderManager()
			->CreateDrawIndexedIndirectArgs(numElements, initData, &buffer);
		break;
	case HQ_ESBT_SHADER_USE_ONLY:
		re = HQEngineApp::GetInstance()->GetRenderDevice()->GetShaderManager()
			->CreateBufferUAV(numElements, elementSize, initData, &buffer);
		break;
	}

	if (HQFailed(re))
		return re;

	//succeeded, now create resource
	m_buffers.Add(name, HQ_NEW HQEngineShaderBufferResImpl(name));
	HQEngineShaderBufferResImpl *newRes = static_cast<HQEngineShaderBufferResImpl*>
		(this->GetShaderBufferResource(name));

	newRes->Init(type, buffer, numElements, elementSize);

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

const HQSharedPtr<HQEngineShaderBufferResImpl>& HQEngineResManagerImpl::GetShaderBufferResourceSharedPtr(const char* name)
{
	bool found = false;
	return m_buffers.GetItem(name, found);
}

HQEngineShaderResource * HQEngineResManagerImpl::GetShaderResource(const char* name)
{
	HQSharedPtr<HQEngineShaderResImpl>* ppRes = m_shaders.GetItemPointer(name);
	if (ppRes == NULL)
		return NULL;
	return ppRes->GetRawPointer();
}

HQEngineShaderBufferResource * HQEngineResManagerImpl::GetShaderBufferResource(const char *name)
{
	HQSharedPtr<HQEngineShaderBufferResImpl>* ppRes = m_buffers.GetItemPointer(name);
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

HQReturnVal  HQEngineResManagerImpl::RemoveShaderBufferResource(HQEngineShaderBufferResource* res)
{
	m_buffers.Remove(res->GetName());
	return HQ_OK;//TO DO: error report
}


void HQEngineResManagerImpl::RemoveAllResources()
{
	m_textures.RemoveAll();
	m_shaders.RemoveAll();
	m_buffers.RemoveAll();
}


HQReturnVal HQEngineResManagerImpl::CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDescs,
	hq_uint32 numAttrib,
	const char* vertexShaderName,
	HQVertexLayout **pInputLayoutID)
{
	HQEngineShaderResource* vertexShader = this->GetShaderResource(vertexShaderName);
	HQEngineShaderResImpl * vshaderImpl = (HQEngineShaderResImpl*)vertexShader;
	HQShaderObject* vid = vshaderImpl != NULL ? vshaderImpl->GetShader() : NULL;

	return HQEngineApp::GetInstance()->GetRenderDevice()->GetVertexStreamManager()
		->CreateVertexInputLayout(vAttribDescs, numAttrib, vid, pInputLayoutID);


}