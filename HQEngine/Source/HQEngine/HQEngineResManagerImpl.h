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
#include "../HQSharedPointer.h"
#include "HQEngineCommonInternal.h"
#include "HQEngineResParserCommon.h"

#ifdef WIN32
#	pragma warning( push )
#	pragma warning( disable : 4250 )//dominance inheritance of GetName()
#endif

//texture resource
class HQEngineTextureResImpl: public HQNamedGraphicsRelatedObj, public HQEngineTextureResource{
public:
	HQEngineTextureResImpl(const char* name);
	virtual ~HQEngineTextureResImpl();

	void Init(HQTexture* texture, HQRenderTargetView* renderTarget = NULL) {m_texture = texture; m_renderTarget = renderTarget;}
	virtual HQTexture* GetTexture() const  { return m_texture; }
	virtual HQRenderTargetView* GetRenderTargetView() const { return m_renderTarget; }
	virtual void GetTexture2DSize(hquint32 &width, hquint32 &height) const;
	virtual bool IsRenderTarget() const {return m_renderTarget != NULL;}

	virtual void* GetRawHandle() { return m_texture->GetRawHandle(); }

private:
	HQTexture* m_texture;
	HQRenderTargetView* m_renderTarget;
};

//shader resource
class HQEngineShaderResImpl: public HQNamedGraphicsRelatedObj, public HQEngineShaderResource{
public:
	HQEngineShaderResImpl(const char* name);
	virtual ~HQEngineShaderResImpl();

	void Init(HQShaderType type, HQShaderObject* shaderID) { m_shader = shaderID; }
	virtual HQShaderObject* GetShader() const { return m_shader; }
	virtual HQShaderType GetShaderType() const { return m_shader->GetType(); }

private:
	HQShaderObject* m_shader;
};

//buffer resource
class HQEngineShaderBufferResImpl : public virtual HQNamedGraphicsRelatedObj, public HQEngineShaderBufferResource{
public:
	HQEngineShaderBufferResImpl(const char *name);
	virtual ~HQEngineShaderBufferResImpl();

	void Init(HQEngineShaderBufferType type, HQBufferUAV* buffer, hquint32 numElements, hquint32 elementSize);

	virtual HQEngineShaderBufferType GetType() const { return m_type; }
	virtual HQBufferUAV* GetBuffer() const { return m_buffer; }
	virtual hquint32 GetNumElements() const { return m_numElements; }
	virtual hquint32 GetElementSize() const { return m_elementSize; }
protected:
	HQBufferUAV * m_buffer;

	HQEngineShaderBufferType m_type;
	hquint32 m_numElements;
	hquint32 m_elementSize;
};

#ifdef WIN32
#	pragma warning( pop )
#endif

enum HQEngineResLoadType {
	HQ_ERLT_STANDARD
};

//resource loading session
class HQEngineResLoadSessionImpl: public HQEngineResLoadSession{
public:
	HQEngineResLoadSessionImpl(HQEngineResParserNode* root);
	~HQEngineResLoadSessionImpl();
	bool HasMoreResources() const;
	const HQEngineResParserNode * CurrentResource();
	const HQEngineResParserNode * NextResource();//advances to next resource item and returns current resource item

	HQEngineResLoadType m_type;
	union {
		struct{
			HQEngineResParserNode * m_root;
			const HQEngineResParserNode* m_resGroup;
			const HQEngineResParserNode* m_resourceItem;
		};
	};
};

//resource manager
class HQEngineResManagerImpl: public HQEngineResManager, public HQLoggableObject {
public:
	HQEngineResManagerImpl(HQLogStream* stream, bool flushLog);
	~HQEngineResManagerImpl();

	virtual void SetSuffix(const char* suffix);

	virtual HQReturnVal AddResourcesFromFile(const char* fileName);
	virtual HQEngineResLoadSession* BeginAddResourcesFromFile(const char* fileName);
	virtual bool HasMoreResources(HQEngineResLoadSession* session);
	virtual HQReturnVal AddNextResource(HQEngineResLoadSession* session);
	virtual HQReturnVal EndAddResources(HQEngineResLoadSession* session);

	virtual HQReturnVal AddTextureResource(const char *name,
											const char * image_file,
											bool generateMipmap,
											HQTextureType type
											) ;

	virtual HQReturnVal AddTextureResource(const char *name, HQTexture* pTexture);

	virtual HQReturnVal AddCubeTextureResource(const char *name,
											const char * image_files[6],
											bool generateMipmap
											);

	///
	///add unordered access texture. 
	///{depth} is array size if texture is array texture
	///
	virtual HQReturnVal AddTextureUAVResource(const char *name,
		HQTextureUAVFormat format,
		hquint32 width, hquint32 height,
		hquint32 depth,
		bool hasMipmap,
		HQTextureType textureType
		);

	virtual HQReturnVal AddRenderTargetTextureResource(
								  const char *name,
								  hq_uint32 width, hq_uint32 height, hq_uint32 arraySize,
								  bool hasMipmaps,
								  HQRenderTargetFormat format , 
								  HQMultiSampleType multisampleType,
								  HQTextureType textureType);

	virtual HQReturnVal AddShaderResource(
									 const char * name,
									 HQShaderType type,
									 HQShaderCompileMode compileMode,
									 const char * source_file,
									 const HQShaderMacro * pDefines,
									 const char* entryFunctionName);

	virtual HQReturnVal AddShaderResourceFromByteCode(
									 const char *name,
									 HQShaderType type,
									 const char * source_file);

	virtual HQReturnVal AddShaderBufferResource(
		const char *name,
		HQEngineShaderBufferType type,
		hquint32 numElements,
		hquint32 elementSize,
		void * initData
		);

	virtual HQEngineTextureResource * GetTextureResource(const char* name);
	virtual HQEngineShaderResource * GetShaderResource(const char* name);
	virtual HQEngineShaderBufferResource * GetShaderBufferResource(const char *name);

	virtual HQReturnVal CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDescs,
		hq_uint32 numAttrib,
		const char* vertexShaderResourceName,
		HQVertexLayout **pInputLayoutID);

	const HQSharedPtr<HQEngineTextureResImpl>& GetTextureResourceSharedPtr(const char* name);
	const HQSharedPtr<HQEngineShaderResImpl>& GetShaderResourceSharedPtr(const char* name);
	const HQSharedPtr<HQEngineShaderBufferResImpl>& GetShaderBufferResourceSharedPtr(const char* name);

	virtual HQReturnVal RemoveTextureResource(HQEngineTextureResource* res);
	virtual HQReturnVal RemoveShaderResource(HQEngineShaderResource* res);
	virtual HQReturnVal RemoveShaderBufferResource(HQEngineShaderBufferResource* res);
	virtual void RemoveAllResources();
private:
	HQReturnVal LoadResource(const HQEngineResParserNode* resource);
	HQReturnVal LoadTextureUAV(const HQEngineResParserNode* textureItem);
	HQReturnVal LoadTexture(const HQEngineResParserNode* textureItem, bool renderTarget = false);
	HQReturnVal LoadShader(const HQEngineResParserNode* shaderItem);
	HQReturnVal LoadBuffer(const HQEngineResParserNode* bufferItem);


	HQClosedStringPrimeHashTable<HQSharedPtr<HQEngineTextureResImpl> > m_textures;
	HQClosedStringPrimeHashTable<HQSharedPtr<HQEngineShaderResImpl> > m_shaders;
	HQClosedStringPrimeHashTable<HQSharedPtr<HQEngineShaderBufferResImpl> > m_buffers;

	std::string m_suffix;
};



#endif