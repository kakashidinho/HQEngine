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


class HQEngineResLoadSession {
protected:
	virtual ~HQEngineResLoadSession() {}
};
//resource manager
class HQEngineResManager {
public:
	

	/*XML format
	<resource>
		<texture>
		</texture>
		<shader>
		</shader>
	</resources>
	*/
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
	virtual HQReturnVal AddCubeTextureResource(const char *name,
											const char * image_files[6],
											bool generateMipmap
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

	virtual HQEngineTextureResource * GetTextureResource(const char* name) = 0;
	virtual HQEngineShaderResource * GetShaderResource(const char* name) = 0;

	virtual HQReturnVal RemoveTextureResource(HQEngineTextureResource* res) = 0;
	virtual HQReturnVal RemoveShaderResource(HQEngineShaderResource* res) = 0;
	virtual void RemoveAllResources() = 0;
protected:
	virtual ~HQEngineResManager() {}
};


#endif