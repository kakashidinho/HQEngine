/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_ENGINE_RES_MANAGER_H
#define HQ_ENGINE_RES_MANAGER_H

#include "HQRendererCoreType.h"
#include "HQReturnVal.h"
#include "HQEngineCommon.h"
#include "HQDataStream.h"

#include <string.h>
#include <stdio.h>


//texture
class HQEngineTextureResource: public virtual HQEngineNamedObj {
public:
	virtual void GetTexture2DSize(hquint32 &width, hquint32 &height) const = 0;
	virtual bool IsRenderTarget() const = 0;

	virtual HQTexture* GetTexture() const = 0;///caution: don't release this returned pointer by HQTextureManager interface
protected:
	virtual ~HQEngineTextureResource() {}
};

//shader 
class HQEngineShaderResource: public virtual HQEngineNamedObj {
public:
	virtual HQShaderType GetShaderType() const = 0;
	virtual HQShaderObject* GetShader() const = 0;///caution: don't release this returned pointer by HQShaderManager interface
protected:
	virtual ~HQEngineShaderResource() {}
};

//buffer
enum HQEngineShaderBufferType{
	HQ_ESBT_VERTEX,
	HQ_ESBT_INDEX,
	HQ_ESBT_DRAW_INDIRECT,
	HQ_ESBT_DRAW_INDEXED_INDIRECT,
	HQ_ESBT_COMPUTE_INDIRECT,
	HQ_ESBT_SHADER_USE_ONLY,
};

class HQEngineShaderBufferResource : public virtual HQEngineNamedObj{
public:
	virtual HQEngineShaderBufferType GetType() const = 0;
	virtual HQBufferUAV* GetBuffer() const = 0;
	virtual hquint32 GetNumElements() const = 0;
	virtual hquint32 GetElementSize() const = 0;
protected:
	virtual ~HQEngineShaderBufferResource() {}
};


class HQEngineResLoadSession {
protected:
	virtual ~HQEngineResLoadSession() {}
};
//resource manager
class HQEngineResManager {
public:
	///
	///Set file's suffix name. When loading a resource file, if it cannot be not found, 
	///resource manager will try to load another file with specified suffix name. 
	///For example, if suffix set to "GL", if a file "resource.script" can't be found, 
	///"resourceGL.script" will be searched. By default, suffix is renderer type passing to HQEngineApp's Window creation functions
	///
	virtual void SetSuffix(const char* suffix) = 0;
	
	virtual HQReturnVal AddResourcesFromFile(const char* fileName) = 0;
	virtual HQEngineResLoadSession* BeginAddResourcesFromFile(const char* fileName) = 0;
	virtual bool HasMoreResources(HQEngineResLoadSession* session) = 0;
	virtual HQReturnVal AddNextResource(HQEngineResLoadSession* session) = 0;
	virtual HQReturnVal EndAddResources(HQEngineResLoadSession* session) = 0;///for releasing loading session
	
	///
	///return HQ_FAILED_RESOURCE_EXISTS if there is an existing resource with the same name
	///
	virtual HQReturnVal AddTextureResource(const char *name,
											const char * image_file,
											bool generateMipmap,
											HQTextureType type
											) = 0;

	///
	///return HQ_FAILED_RESOURCE_EXISTS if there is an existing resource with the same name
	///
	virtual HQReturnVal AddTextureResource(const char *name, HQTexture* pTexture) = 0;
	///
	///return HQ_FAILED_RESOURCE_EXISTS if there is an existing resource with the same name
	///
	virtual HQReturnVal AddCubeTextureResource(const char *name,
											const char * image_files[6],
											bool generateMipmap
											) = 0;

	///
	///Create unordered access 2d texture
	///return HQ_FAILED_RESOURCE_EXISTS if there is an existing resource with the same name
	///
	virtual HQReturnVal AddTextureUAVResource(const char *name,
		HQTextureUAVFormat format,
		hquint32 width, hquint32 height,
		bool hasMipmap
		) = 0;

	///
	///create render target texture resource. 
	///{pRenderTargetID_Out} - will store ID of newly created render target. 
	///{pTextureID_Out} - will store ID of texture in material manager. 
	///{hasMipmaps} - this texture has full range mipmap or not. 
	///Note : 
	///-if {textureType} = HQ_TEXTURE_CUBE , new texture will be created with size {width} x {width}. 
	///-openGL ES 2.0 device always create texture with full range mipmap levels. 
	///-return HQ_FAILED_RESOURCE_EXISTS if there is an existing resource with the same name. 
	///-return HQ_FAILED_FORMAT_NOT_SUPPORT if {format} is not supported. 
	///-return HQ_FAILED_MULTISAMPLE_TYPE_NOT_SUPPORT if {multisampleType} is not supported. 
	///
	virtual HQReturnVal AddRenderTargetTextureResource(
								  const char *name,
								  hq_uint32 width , hq_uint32 height,
								  bool hasMipmaps,
								  HQRenderTargetFormat format , 
								  HQMultiSampleType multisampleType,
								  HQTextureType textureType) = 0;

	///
	///{pDefines} - pointer đến dãy các shader macro, phần tử cuối phải có cả 2 thành phần {name} và {definition} là NULL để chỉ kết thúc dãy
	///
	virtual HQReturnVal AddShaderResource(
									 const char * name,
									 HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char * source_file,
									 const HQShaderMacro * pDefines,
									 const char* entryFunctionName) = 0;

	virtual HQReturnVal AddShaderResourceFromByteCode(
									 const char *name,
									 HQShaderType type,
									 const char * source_file) = 0;


	///
	///create unordered access supported buffer to be able to be read and written in shader. 
	///Note that {elementSize} is only relevant in HQ_ESBT_SHADER_USE_ONLY/HQ_ESBT_VERTEX/HQ_ESBT_INDEX 
	///type
	///
	virtual HQReturnVal AddShaderBufferResource(
			const char *name,
			HQEngineShaderBufferType type,
			hquint32 numElements,
			hquint32 elementSize,
			void * initData = NULL
		) = 0;

	///
	///{vertexShaderResourceName} is ignored in D3D9 device. if {vertexShaderResourceName} = NULL, this method will create 
	///input layout for fixed function shader. D3D11 & GL only accepts the following layout: 
	///position (x,y,z); color (r,g,b,a); normal (x,y,z); texcoords (u,v)
	///
	virtual HQReturnVal CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDescs,
		hq_uint32 numAttrib,
		const char* vertexShaderResourceName,
		HQVertexLayout **pInputLayoutID) = 0;

	virtual HQEngineTextureResource * GetTextureResource(const char* name) = 0;
	virtual HQEngineShaderResource * GetShaderResource(const char* name) = 0;
	virtual HQEngineShaderBufferResource * GetShaderBufferResource(const char *name) = 0;

	virtual HQReturnVal RemoveTextureResource(HQEngineTextureResource* res) = 0;
	virtual HQReturnVal RemoveShaderResource(HQEngineShaderResource* res) = 0;
	virtual HQReturnVal RemoveShaderBufferResource(HQEngineShaderBufferResource* res) = 0;
	virtual void RemoveAllResources() = 0;
protected:
	virtual ~HQEngineResManager() {}
};


#endif