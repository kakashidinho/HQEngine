/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "HQDeviceGL.h"

#include <math.h>
#ifndef max
#define max(a,b) ((a>b)?a:b)
#endif

#ifdef glTexBuffer
#undef glTexBuffer
#endif

//size of 2d/cube texture
struct HQTexture2DDesc
{
	hquint32 width , height;
};

#define PVRTC_DECOMPRESSION_SUPPORT 0

#if PVRTC_DECOMPRESSION_SUPPORT
#	error need implement
#endif

/*---------texture buffer------------------*/
typedef void (GLAPIENTRY * PFNGLTEXBUFFERPROC) (GLenum target, GLenum internalformat, GLuint buffer);
void GLAPIENTRY glTexBufferDummy(GLenum target, GLenum internalformat, GLuint buffer){}
PFNGLTEXBUFFERPROC glTexBuffer ;

HQTextureGL::HQTextureGL(HQTextureType type ):HQTexture()
{
	this->type = type;
	
	switch (type)
	{
	case HQ_TEXTURE_2D:
		this->textureTarget = GL_TEXTURE_2D;
		break;
	case HQ_TEXTURE_CUBE:
		this->textureTarget = GL_TEXTURE_CUBE_MAP;
		break;
	case HQ_TEXTURE_BUFFER:
		this->textureTarget = GL_TEXTURE_BUFFER;
		break;
	}

	this->pData = new GLuint();
	if (this->pData)
		glGenTextures(1,(GLuint*)pData);

	this->textureDesc = NULL;
}
HQTextureGL::~HQTextureGL()
{
	if(pData != NULL)
	{
		GLuint *p = (GLuint*)pData;
#if defined DEVICE_LOST_POSSIBLE
		if (!g_pOGLDev->IsDeviceLost())
#endif
			glDeleteTextures(1 , p);
		delete p;
	}
	if (textureDesc != NULL)
		free (textureDesc);
}

#ifndef GLES
struct HQTextureBufferGL : public HQTextureGL
{
	HQTextureBufferGL(HQTextureManagerGL * manager) : HQTextureGL(HQ_TEXTURE_BUFFER)
	{
		this->manager = manager;
		glGenBuffers(1 , &buffer);
	}
	~HQTextureBufferGL()
	{
		if (manager->GetCurrentBoundTBuffer() == buffer)
			manager->InvalidateCurrentBoundTBuffer();
		if (buffer != 0)
		{
#if defined DEVICE_LOST_POSSIBLE
			if (!g_pOGLDev->IsDeviceLost())
#endif
				glDeleteBuffers(1 , &buffer);
		}
	}

	GLuint buffer;
	HQTextureManagerGL * manager;
	hq_uint32 size;
	GLenum usage;
};
#endif
/*---------helper functions--------*/

namespace helper
{

//************************************************************
//định dạng của texture tương ứng với định dạng của pixel ảnh
//************************************************************
	static GLint GetTextureFmt(const SurfaceFormat imgFmt)
	{
		switch (imgFmt)
		{
		case FMT_R8G8B8: case FMT_B8G8R8:
			return GL_RGB8;
		case FMT_A8R8G8B8:
			return GL_RGBA8;
		case FMT_X8R8G8B8:
			return GL_RGB8;
		case FMT_R5G6B5: case FMT_B5G6R5:
			return GL_RGB8;
		case FMT_L8:
			return GL_LUMINANCE8;
		case FMT_A8L8:
			return GL_LUMINANCE8_ALPHA8;
		case FMT_A8:
			return GL_ALPHA8;
		case FMT_S3TC_DXT1:
			return GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
		case FMT_S3TC_DXT3:
			return GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
		case FMT_S3TC_DXT5:
			return GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
		case FMT_A8B8G8R8:
			return GL_RGBA8;
		case FMT_X8B8G8R8:
			return GL_RGB8;
#ifdef GLES
		case FMT_ETC1:
			return GL_ETC1_RGB8_OES;
		case FMT_PVRTC_RGB_2BPP:
			return GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG;
		case FMT_PVRTC_RGB_4BPP:
			return GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG;
		case FMT_PVRTC_RGBA_2BPP:
			return GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;
		case FMT_PVRTC_RGBA_4BPP:
			return GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;
#endif
		default:
			return -1;
		}
	}

	void GetImageDataInfo(const SurfaceFormat imgFmt,GLenum &format,GLenum &dataType)
	{
		switch (imgFmt)
		{
		case FMT_R8G8B8: case FMT_B8G8R8:
			format=GL_RGB;
			dataType = GL_UNSIGNED_BYTE;
			break;
		case FMT_A8R8G8B8:
			format=GL_RGBA;
			dataType = GL_UNSIGNED_BYTE;
			break;
		case FMT_X8R8G8B8:
			format=GL_RGBA;
			dataType = GL_UNSIGNED_BYTE;
			break;
		case FMT_R5G6B5: case FMT_B5G6R5:
			format=GL_RGB;
			dataType = GL_UNSIGNED_SHORT_5_6_5;
			break;
		case FMT_L8:
			format=GL_LUMINANCE;
			dataType = GL_UNSIGNED_BYTE;
			break;
		case FMT_A8L8:
			format=GL_LUMINANCE_ALPHA;
			dataType = GL_UNSIGNED_BYTE;
			break;
		case FMT_A8:
			format=GL_ALPHA;
			dataType = GL_UNSIGNED_BYTE;
			break;
		case FMT_S3TC_DXT1:
			format=GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
			dataType = GL_UNSIGNED_BYTE;
			break;
		case FMT_S3TC_DXT3:
			format=GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
			dataType = GL_UNSIGNED_BYTE;
			break;
		case FMT_S3TC_DXT5:
			format=GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
			dataType = GL_UNSIGNED_BYTE;
			break;
		case FMT_A8B8G8R8:
			format=GL_RGBA;
			dataType = GL_UNSIGNED_BYTE;
			break;
		case FMT_X8B8G8R8:
			format=GL_RGBA;
			dataType = GL_UNSIGNED_BYTE;
			break;
#ifdef GLES
		case FMT_ETC1:
			format = GL_ETC1_RGB8_OES;
			dataType = GL_UNSIGNED_BYTE;
			break;
		case FMT_PVRTC_RGB_2BPP:
			format = GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG;
			dataType = GL_UNSIGNED_BYTE;
			break;
		case FMT_PVRTC_RGB_4BPP:
			format = GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG;
			dataType = GL_UNSIGNED_BYTE;
			break;
		case FMT_PVRTC_RGBA_2BPP:
			format = GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;
			dataType = GL_UNSIGNED_BYTE;
			break;
		case FMT_PVRTC_RGBA_4BPP:
			format = GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;
			dataType = GL_UNSIGNED_BYTE;
			break;
#endif
		default:
			return;
		}
	}

};


GLenum GetTextureBufferFormat(HQTextureBufferFormat format )
{
#ifdef GLES
	return 0;
#else
	switch (format)
	{
	case HQ_TBFMT_R16_FLOAT :
		//texelSize = 2 ;
		if (GLEW_VERSION_3_0)
			return GL_R16F;
		if (GLEW_ARB_texture_float)
		{
			if(GLEW_ARB_texture_rg)
				return GL_R16F;
			return GL_LUMINANCE16F_ARB;
		}
	case HQ_TBFMT_R16G16B16A16_FLOAT  :
		//texelSize = 8;
		if (GLEW_VERSION_3_0)
			return GL_RGBA16F;
		if (GLEW_ARB_texture_float)
			return GL_RGBA16F_ARB;
	case HQ_TBFMT_R32_FLOAT  :
		//texelSize = 4;
		if (GLEW_VERSION_3_0)
			return GL_R32F;
		if (GLEW_ARB_texture_float)
		{
			if(GLEW_ARB_texture_rg)
				return GL_R32F;
			return GL_LUMINANCE32F_ARB;
		}
	case HQ_TBFMT_R32G32B32_FLOAT:
		//texelSize = 12;
		if (GLEW_VERSION_3_0 || GLEW_ARB_texture_float)
			return GL_RGB32F_ARB;
	case HQ_TBFMT_R32G32B32A32_FLOAT  :
		//texelSize = 16;
		if (GLEW_VERSION_3_0 || GLEW_ARB_texture_float)
			return GL_RGBA32F_ARB;
	case HQ_TBFMT_R8_INT :
		//texelSize = 1;
		if (GLEW_VERSION_3_0)
			return GL_R8I;
		if (GLEW_EXT_texture_integer)
		{
			if(GLEW_ARB_texture_rg)
				return GL_R8I;
			return GL_LUMINANCE8I_EXT;
		}
	case HQ_TBFMT_R8G8B8A8_INT :
		//texelSize = 4;
		if (GLEW_EXT_texture_integer)
			return GL_RGBA8I_EXT;
	case HQ_TBFMT_R8_UINT  :
		//texelSize = 1;
		if (GLEW_VERSION_3_0)
			return GL_R8UI;
		if (GLEW_EXT_texture_integer)
		{
			if(GLEW_ARB_texture_rg)
				return GL_R8UI;
			return GL_LUMINANCE8UI_EXT;
		}
	case HQ_TBFMT_R8G8B8A8_UINT  :
		//texelSize = 4;
		if (GLEW_EXT_texture_integer)
			return GL_RGBA8UI_EXT;
	case HQ_TBFMT_R16_INT :
		//texelSize = 2;
		if (GLEW_VERSION_3_0)
			return GL_R16I;
		if (GLEW_EXT_texture_integer)
		{
			if(GLEW_ARB_texture_rg)
				return GL_R16I;
			return GL_LUMINANCE16I_EXT;
		}
	case HQ_TBFMT_R16G16B16A16_INT :
		//texelSize = 8;
		if (GLEW_EXT_texture_integer)
			return GL_RGBA16UI_EXT;
	case HQ_TBFMT_R16_UINT :
		//texelSize = 2;
		if (GLEW_VERSION_3_0)
			return GL_R16UI;
		if (GLEW_EXT_texture_integer)
		{
			if(GLEW_ARB_texture_rg)
				return GL_R16UI;
			return GL_LUMINANCE16UI_EXT;
		}
	case HQ_TBFMT_R16G16B16A16_UINT :
		//texelSize = 8;
		if (GLEW_EXT_texture_integer)
			return GL_RGBA16UI_EXT;
	case HQ_TBFMT_R32_INT :
		//texelSize = 4;
		if (GLEW_VERSION_3_0)
			return GL_R32I;
		if (GLEW_EXT_texture_integer)
		{
			if(GLEW_ARB_texture_rg)
				return GL_R32I;
			return GL_LUMINANCE32I_EXT;
		}
	case HQ_TBFMT_R32G32B32A32_INT :
		//texelSize = 16;
		if (GLEW_EXT_texture_integer)
			return GL_RGBA32I_EXT;
	case HQ_TBFMT_R32_UINT :
		//texelSize = 4;
		if (GLEW_VERSION_3_0)
			return GL_R32UI;
		if (GLEW_EXT_texture_integer)
		{
			if(GLEW_ARB_texture_rg)
				return GL_R32UI;
			return GL_LUMINANCE32UI_EXT;
		}
	case HQ_TBFMT_R32G32B32A32_UINT :
		//texelSize = 16;
		if (GLEW_EXT_texture_integer)
			return GL_RGBA32UI_EXT;
	case HQ_TBFMT_R8_UNORM  :
		//texelSize = 1;
		if (GLEW_VERSION_3_0 || GLEW_ARB_texture_rg)
			return GL_R8;
		return GL_LUMINANCE8;
	case HQ_TBFMT_R8G8B8A8_UNORM  :
		//texelSize = 4;
		return GL_RGBA8;
	case HQ_TBFMT_R16_UNORM :
		//texelSize = 2;
		if (GLEW_VERSION_3_0 || GLEW_ARB_texture_rg)
			return GL_R16;
		return GL_LUMINANCE16;
	case HQ_TBFMT_R16G16B16A16_UNORM :
		//texelSize = 8;
		return GL_RGBA16;
	default:
		return 0;
	}
#endif
}

/*
//HQTextureManagerGL class
*/
/*
//constructor
*/
HQTextureManagerGL::HQTextureManagerGL(
					hq_uint32 maxTextureUnits,
					HQLogStream* logFileStream ,
					bool flushLog)
:HQBaseTextureManager(logFileStream  , "GL Texture Manager:" ,flushLog)
{
	this->bitmap.SetLoadedOutputRGBLayout(LAYOUT_BGR);
	this->bitmap.SetLoadedOutputRGB16Layout(LAYOUT_RGB);

	this->maxTextureUnits = maxTextureUnits;
	this->texUnits = new HQTextureUnitInfoGL[maxTextureUnits];
	this->activeTexture = 0;

#ifndef GLES
	this->currentBoundTBuffer = 0;

	/*------------------*/
	glTexBuffer = (PFNGLTEXBUFFERPROC )gl_GetProcAddress("glTexBuffer");
	if (glTexBuffer == NULL)
		glTexBuffer = (PFNGLTEXBUFFERPROC )gl_GetProcAddress("glTexBufferARB");
	if (glTexBuffer == NULL)
		glTexBuffer = (PFNGLTEXBUFFERPROC )gl_GetProcAddress("glTexBufferEXT");
	if (glTexBuffer == NULL)
		glTexBuffer = &glTexBufferDummy;
#endif
	Log("Init done!");
	LogTextureCompressionSupportInfo();
}
/*
Destructor
*/
HQTextureManagerGL::~HQTextureManagerGL()
{
	SafeDeleteArray(this->texUnits);
}

HQTexture * HQTextureManagerGL::CreateNewTextureObject(HQTextureType type)
{
#ifdef GLES
	if(type == HQ_TEXTURE_CUBE && !GLEW_VERSION_2_0)//cube texture not supported
	{
		Log("Cube texture is not supported!");
		return NULL;
	}
#endif
	
    GLuint currentBoundTex;

	switch (type)
	{
	case HQ_TEXTURE_2D:
		currentBoundTex = this->texUnits[this->activeTexture].GetTexture2DGL();
		break;
	case HQ_TEXTURE_CUBE:
		currentBoundTex = this->texUnits[this->activeTexture].GetTextureCubeGL();
		break;
	case HQ_TEXTURE_BUFFER:
		currentBoundTex = this->texUnits[this->activeTexture].GetTextureBufferGL();
		break;
    default:
		return NULL;
	}

	HQTextureGL *newTex;

#ifndef GLES
	if (type == HQ_TEXTURE_BUFFER)
		newTex = new HQTextureBufferGL(this);
	else
#endif
		newTex = new HQTextureGL(type);


	glBindTexture(newTex->textureTarget ,*(GLuint*)newTex->pData );

	if (type != HQ_TEXTURE_BUFFER)
	{
		//set default sampler state
		HQStateManagerGL* pStateManager = (HQStateManagerGL*)g_pOGLDev->GetStateManager();
		newTex->pSamplerState = pStateManager->GetSamplerState(0);

		glTexParameteri(newTex->textureTarget,GL_TEXTURE_WRAP_S , GL_REPEAT);
		glTexParameteri(newTex->textureTarget,GL_TEXTURE_WRAP_T , GL_REPEAT);
		glTexParameteri(newTex->textureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(newTex->textureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	}
	//re-bind old texture
	glBindTexture(newTex->textureTarget , currentBoundTex);

	return newTex;
}


HQReturnVal HQTextureManagerGL::SetTexture(hq_uint32 slot , hq_uint32 textureID)
{
#if defined _DEBUG || defined DEBUG
	if (slot >= this->maxTextureUnits)
	{
		hquint32 textureSlot = slot & 0xfffffff;
		switch (slot & 0xf0000000){
			case HQ_VERTEX_SHADER:
				Log("SetTexture() Error : Did you mistakenly bitwise OR %u with HQ_VERTEX_SHADER!", textureSlot);
				break;
			case HQ_PIXEL_SHADER:
				Log("SetTexture() Error : Did you mistakenly bitwise OR %u with HQ_PIXEL_SHADER!", textureSlot);
				break;
			case HQ_GEOMETRY_SHADER:
				Log("SetTexture() Error : Did you mistakenly bitwise OR %u with HQ_GEOMETRY_SHADER!", textureSlot);
				break;
			default:
				Log("SetTexture() Error : {slot=%u} out of range!", slot);
		}
		return HQ_FAILED;
	}
#endif

	HQSharedPtr<HQTexture> pTexture = this->textures.GetItemPointer(textureID);
	HQTextureGL* pTextureRawPtr = (HQTextureGL*)pTexture.GetRawPointer();
	if (pTextureRawPtr == NULL)
		return HQ_OK;

	HQTextureUnitInfoGL &texUnitInfo =  this->texUnits[slot];

#if 1
	HQSharedPtr<HQTexture> &currentTexture = texUnitInfo.texture[ pTextureRawPtr->type ];
	if (pTexture != currentTexture)
	{
		GLuint &textureName = *((GLuint*)pTextureRawPtr->pData);
		this->BindTexture(slot , pTextureRawPtr->textureTarget , textureName);
		
		currentTexture = pTexture;
	}

#else //switch version , last update : 23/6/11
	switch (pTexture->type)
	{
	case GL_TEXTURE_2D:
		if (pTexture != texUnitInfo.texture[HQ_TEXTURE_2D])
		{
			GLuint textureName = *((GLuint*)pTexture->pData);
			this->BindTexture(slot , GL_TEXTURE_2D , textureName);
			texUnitInfo.texture[HQ_TEXTURE_2D] = pTexture;
		}
		break;
	case GL_TEXTURE_CUBE:
		if (pTexture != texUnitInfo.texture[HQ_TEXTURE_CUBE])
		{
			GLuint textureName = *((GLuint*)pTexture->pData);
			this->BindTexture(slot , GL_TEXTURE_CUBE_MAP , textureName);
			texUnitInfo.texture[HQ_TEXTURE_CUBE] = pTexture;
		}
		break;
	case GL_TEXTURE_BUFFER:
		if (pTexture != texUnitInfo.texture[HQ_TEXTURE_BUFFER])
		{
			GLuint textureName = *((GLuint*)pTexture->pData);
			this->BindTexture(slot , GL_TEXTURE_BUFFER , textureName);
			texUnitInfo.texture[HQ_TEXTURE_BUFFER] = pTexture;
		}
		break;
	}
#endif
	return HQ_OK;
}


HQReturnVal HQTextureManagerGL::SetTextureForPixelShader(hq_uint32 slot , hq_uint32 textureID)
{
#if defined _DEBUG || defined DEBUG
	if (slot >= this->maxTextureUnits)
	{
		hquint32 textureSlot = slot & 0xfffffff;
		switch (slot & 0xf0000000){
			case HQ_VERTEX_SHADER:
				Log("SetTextureForPixelShader() Error : Did you mistakenly bitwise OR %u with HQ_VERTEX_SHADER!", textureSlot);
				break;
			case HQ_PIXEL_SHADER:
				Log("SetTextureForPixelShader() Error : Did you mistakenly bitwise OR %u with HQ_PIXEL_SHADER!", textureSlot);
				break;
			case HQ_GEOMETRY_SHADER:
				Log("SetTextureForPixelShader() Error : Did you mistakenly bitwise OR %u with HQ_GEOMETRY_SHADER!", textureSlot);
				break;
			default:
				Log("SetTextureForPixelShader() Error : {slot=%u} out of range!", slot);
		}
		return HQ_FAILED;
	}
#endif

	HQSharedPtr<HQTexture> pTexture = this->textures.GetItemPointer(textureID);
	HQTextureGL* pTextureRawPtr = (HQTextureGL*)pTexture.GetRawPointer();
	if (pTextureRawPtr == NULL)
		return HQ_OK;

	HQTextureUnitInfoGL &texUnitInfo =  this->texUnits[slot];

#if 1
	HQSharedPtr<HQTexture> &currentTexture = texUnitInfo.texture[ pTextureRawPtr->type ];
	if (pTexture != currentTexture)
	{
		GLuint &textureName = *((GLuint*)pTextureRawPtr->pData);
		this->BindTexture(slot , pTextureRawPtr->textureTarget , textureName);
		
		currentTexture = pTexture;
	}

#else //switch version , last update : 23/6/11
	switch (pTexture->type)
	{
	case GL_TEXTURE_2D:
		if (pTexture != texUnitInfo.texture[HQ_TEXTURE_2D])
		{
			GLuint textureName = *((GLuint*)pTexture->pData);
			this->BindTexture(slot , GL_TEXTURE_2D , textureName);
			texUnitInfo.texture[HQ_TEXTURE_2D] = pTexture;
		}
		break;
	case GL_TEXTURE_CUBE:
		if (pTexture != texUnitInfo.texture[HQ_TEXTURE_CUBE])
		{
			GLuint textureName = *((GLuint*)pTexture->pData);
			this->BindTexture(slot , GL_TEXTURE_CUBE_MAP , textureName);
			texUnitInfo.texture[HQ_TEXTURE_CUBE] = pTexture;
		}
		break;
	case GL_TEXTURE_BUFFER:
		if (pTexture != texUnitInfo.texture[HQ_TEXTURE_BUFFER])
		{
			GLuint textureName = *((GLuint*)pTexture->pData);
			this->BindTexture(slot , GL_TEXTURE_BUFFER , textureName);
			texUnitInfo.texture[HQ_TEXTURE_BUFFER] = pTexture;
		}
		break;
	}
#endif
	return HQ_OK;
}

/*
Load texture from file
*/
HQReturnVal HQTextureManagerGL::LoadTextureFromFile(HQTexture * pTex)
{
	//các thông tin cơ sở
	hq_uint32 w,h;//width,height
	short bpp;//bits per pixel
	SurfaceFormat format;//định dạng
	SurfaceComplexity complex;//độ phức tạp
	ImgOrigin origin;//vị trí pixel đầu

	//load bitmap file
	int result=bitmap.Load(pTex->fileName);

	char errBuffer[128];

	if(result!=IMG_OK)
	{
		if (result == IMG_FAIL_NOT_ENOUGH_CUBE_FACES)
		{
			Log("Load cube texture from file %s error : File doesn't have enough 6 cube faces!",pTex->fileName);
			return HQ_FAILED_NOT_ENOUGH_CUBE_FACES;
		}
		bitmap.GetErrorDesc(result,errBuffer);
		Log("Load texture from file %s error : %s",pTex->fileName,errBuffer);
		return HQ_FAILED;
	}
	w=bitmap.GetWidth();
	h=bitmap.GetHeight();
	bpp=bitmap.GetBits();
	format=bitmap.GetSurfaceFormat();
	origin=bitmap.GetPixelOrigin();
	complex=bitmap.GetSurfaceComplex();

	if(complex.dwComplexFlags & SURFACE_COMPLEX_VOLUME)//it's volume texture
	{
		Log("Load texture from file error :can't load volume texture",errBuffer);
		return HQ_FAILED;
	}

#ifdef GLES
	if(!GLEW_OES_texture_non_power_of_two)//kích thước texture phải là lũy thừa của 2
	{
		hq_uint32 exp;
		bool needResize=false;
		if(!IsPowerOfTwo(w,&exp))//chiều rộng không là lũy thừa của 2
		{
			needResize=true;
			w=0x1 << exp;//2^exp
		}
		if(!IsPowerOfTwo(h,&exp))//chiều cao không là lũy thừa của 2
		{
			needResize=true;
			h=0x1 << exp;//2^exp
		}
		if(needResize)
		{
			if (pTex->type == HQ_TEXTURE_CUBE)
			{
				Log("Load cube texture from files %s error : Dimension need to be power of 2");
				return HQ_FAILED;
			}
			Log("Now trying to resize image dimesions to power of two dimensions");
			unsigned long size=bitmap.GetFirstLevelSize();

			hq_ubyte8 *pData=new hq_ubyte8[size];
			if(!pData)
			{
				Log("Memory allocation failed");
				return HQ_FAILED;
			}
			//copy first level pixel data
			memcpy(pData,bitmap.GetPixelData(),size);
			//loại bỏ các thuộc tính phức tạp
			memset(&complex,0,sizeof(SurfaceComplexity));

			bitmap.Set(pData,bitmap.GetWidth(),bitmap.GetHeight(),bpp,size, format,origin,complex);
			//giải nén nếu hình ảnh ở dạng nén ,làm thế mới resize hình ảnh dc
			
			if(bitmap.IsCompressed() &&  (result = bitmap.DeCompress())!=IMG_OK)
			{
				if (result == IMG_FAIL_MEM_ALLOC)
					Log("Memory allocation failed when attempt to decompressing compressed data!");
				else
					Log("Couldn't decompress compressed data!");
				return HQ_FAILED;
			}
			//phóng to hình ảnh lên thành kích thước lũy thừa của 2
			bitmap.Scalei(w,h);
			
			//chỉnh lại thông tin cơ sở
			format=bitmap.GetSurfaceFormat();
			bpp=bitmap.GetBits();
		}//if (need resize)
	}//if (must power of two)
#endif

	if((!GLEW_EXT_texture_compression_s3tc) &&
		(format ==FMT_S3TC_DXT1 || format ==FMT_S3TC_DXT3 || format ==FMT_S3TC_DXT5))//không hỗ trợ dạng nén
	{
		Log("DXT compressed texture is not supported!Now trying to decompress it!");
		
		//giải nén
		
		if(bitmap.DeCompressDXT(true)==IMG_FAIL_MEM_ALLOC)
		{
			Log("Memory allocation failed when attempt to decompressing DXTn compressed data!");
			return HQ_FAILED;
		}
		//chỉnh lại thông tin cơ sở
		format=bitmap.GetSurfaceFormat();
		bpp=bitmap.GetBits();
	}//if (not support DXT)

	if (bitmap.IsPVRTC()//not support PVRTC
#ifdef GLES
		&& !GLEW_IMG_texture_compression_pvrtc
#endif
		)
#if PVRTC_DECOMPRESSION_SUPPORT
#	error need implement
#else
	{
		Log("Error : PVRTC texture is not supported!");
		return HQ_FAILED_FORMAT_NOT_SUPPORT;
	}//not support PVRTC
#endif

	if (format == FMT_ETC1
#ifdef GLES
	 && !GLEW_OES_compressed_ETC1_RGB8_texture
#endif
	)
	{
		Log("ETC compressed texture is not supported!Now trying to decompress it!");
		
		//giải nén
		
		if(bitmap.DeCompressETC(true)==IMG_FAIL_MEM_ALLOC)
		{
			Log("Memory allocation failed when attempt to decompressing ETC compressed data!");
			return HQ_FAILED;
		}
		//chỉnh lại thông tin cơ sở
		format=bitmap.GetSurfaceFormat();
		bpp=bitmap.GetBits();
	}//if (not support ETC)

	hq_uint32 nMips = 1;//số lượng mipmap level
	if(generateMipmap)
	{
		//full range mipmap
		if(IsPowerOfTwo(max(w,h),&nMips))//nMips=1+floor(log2(max(w,h)))
			nMips++;

		else if ((complex.dwComplexFlags & SURFACE_COMPLEX_MIPMAP)!=0 && complex.nMipMap>1)//nếu trong file ảnh có chứa nhiều hơn 1 mipmap level
		{
			nMips=complex.nMipMap;
		}

	}
	if(format==FMT_R8G8B8A8 || format==FMT_B8G8R8A8)
		bitmap.FlipRGBA();//đưa alpha byte lên thành byte có trọng số lớn nhất
	//đưa pixel đầu tiên lên vị trí góc trên bên trái
	result = bitmap.SetPixelOrigin(ORIGIN_TOP_LEFT);
	
	if(result!=IMG_OK)
	{
		if (bitmap.IsCompressed())
		{
			Log ("Warning : Couldn't flip texture's image origin to top left corner because it's compressed!");
		}
		else
		{
			bitmap.GetErrorDesc(result,errBuffer);
			Log("Error when trying to flip texture : %s",errBuffer);
		}
	}

	if(CreateTexture((pTex->alpha<1.0f || (pTex->colorKey!=NULL && pTex->nColorKey>0)),
					 nMips,pTex)!=HQ_OK)
	{
		return HQ_FAILED;
	}

	return HQ_OK;
}

/*
Load cube texture from 6 files
*/
HQReturnVal HQTextureManagerGL::LoadCubeTextureFromFiles(const char *fileNames[6] , HQTexture * pTex)
{
	//các thông tin cơ sở
	hq_uint32 w,h;//width,height
	short bpp;//bits per pixel
	SurfaceFormat format;//định dạng
	SurfaceComplexity complex;//độ phức tạp
	ImgOrigin origin;//vị trí pixel đầu

	//load bitmap file
	int result=bitmap.LoadCubeFaces(fileNames , ORIGIN_TOP_LEFT , this->generateMipmap);

	char errBuffer[128];

	if(result == IMG_FAIL_CANT_GENERATE_MIPMAPS)
	{
		Log("Load cube texture from files %s warning : can't generate mipmaps",pTex->fileName);
		this->generateMipmap = false;
	}
	else if(result!=IMG_OK)
	{
		bitmap.GetErrorDesc(result,errBuffer);
		Log("Load cube texture from files %s error : %s",pTex->fileName,errBuffer);
		return HQ_FAILED;
	}
	w=bitmap.GetWidth();
	h=bitmap.GetHeight();
	bpp=bitmap.GetBits();
	format=bitmap.GetSurfaceFormat();
	origin=bitmap.GetPixelOrigin();
	complex=bitmap.GetSurfaceComplex();



	if((!GLEW_EXT_texture_compression_s3tc) &&
		(format ==FMT_S3TC_DXT1 || format ==FMT_S3TC_DXT3 || format ==FMT_S3TC_DXT5))//không hỗ trợ dạng nén
	{
		Log("DXT compressed texture is not supported!Now trying to decompress it!");
		
		//giải nén
		if(bitmap.DeCompressDXT(true)==IMG_FAIL_MEM_ALLOC)
		{
			Log("Memory allocation failed when attempt to decompressing DXTn compressed data!");
			return HQ_FAILED;
		}
		//chỉnh lại thông tin cơ sở
		format=bitmap.GetSurfaceFormat();
		bpp=bitmap.GetBits();

	}//if (not support DXT)
	
	if (bitmap.IsPVRTC()//not support PVRTC
#ifdef GLES
		&& !GLEW_IMG_texture_compression_pvrtc
#endif
		)
#if PVRTC_DECOMPRESSION_SUPPORT
#	error need implement
#else
	{
		Log("Error : PVRTC texture is not supported!");
		return HQ_FAILED_FORMAT_NOT_SUPPORT;
	}//not support PVRTC
#endif
	

	if (format == FMT_ETC1
#ifdef GLES
	 && !GLEW_OES_compressed_ETC1_RGB8_texture
#endif
	)
	{
		Log("ETC compressed texture is not supported!Now trying to decompress it!");
		
		//giải nén
		
		if(bitmap.DeCompressETC(true)==IMG_FAIL_MEM_ALLOC)
		{
			Log("Memory allocation failed when attempt to decompressing ETC compressed data!");
			return HQ_FAILED;
		}
		//chỉnh lại thông tin cơ sở
		format=bitmap.GetSurfaceFormat();
		bpp=bitmap.GetBits();
	}//if (not support ETC)


#ifdef GLES
	if(!GLEW_OES_texture_non_power_of_two)//kích thước texture phải là lũy thừa của 2
	{
		hq_uint32 exp;
		bool needResize=false;
		if(!IsPowerOfTwo(w,&exp))//chiều rộng không là lũy thừa của 2
		{
			needResize=true;
			w=0x1 << exp;//2^exp
		}
		if(!IsPowerOfTwo(h,&exp))//chiều cao không là lũy thừa của 2
		{
			needResize=true;
			h=0x1 << exp;//2^exp
		}
		if(needResize)
		{
			Log("Load cube texture from files %s error : Dimension need to be power of 2");
			return HQ_FAILED;
		}//if (need resize)
	}//if (must power of two)
#endif

	hq_uint32 nMips = 1;//số lượng mipmap level
	if(generateMipmap)
	{
		if ((complex.dwComplexFlags & SURFACE_COMPLEX_MIPMAP)!=0 && complex.nMipMap>1)//nếu trong file ảnh có chứa nhiều hơn 1 mipmap level
		{
			nMips=complex.nMipMap;
		}

	}
	if(format==FMT_R8G8B8A8 || format==FMT_B8G8R8A8)
		bitmap.FlipRGBA();//đưa alpha byte lên thành byte có trọng số lớn nhất


	if(CreateTexture((pTex->alpha<1.0f || (pTex->colorKey!=NULL && pTex->nColorKey>0)),
					 nMips,pTex)!=HQ_OK)
	{
		return HQ_FAILED;
	}

	return HQ_OK;
}


HQReturnVal HQTextureManagerGL::CreateSingleColorTexture(HQTexture *pTex,HQColorui color)
{
	GLuint *pTextureName = (GLuint*)pTex->pData;
	glBindTexture(GL_TEXTURE_2D , *pTextureName);

	glPixelStorei( GL_UNPACK_ALIGNMENT, 1);
#ifndef GLES
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1 );
#endif

	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,1,1,0,GL_RGBA,GL_UNSIGNED_BYTE,&color);

	glBindTexture(GL_TEXTURE_2D, this->texUnits[this->activeTexture].GetTexture2DGL());//bind old texture

	return HQ_OK;
}
/*
create texture object from pixel data
*/
HQReturnVal HQTextureManagerGL::CreateTexture(bool changeAlpha,hq_uint32 numMipmaps,HQTexture * pTex)
{
	SurfaceComplexity complex=bitmap.GetSurfaceComplex();
	SurfaceFormat format=bitmap.GetSurfaceFormat();

	if(bitmap.IsCompressed() && complex.nMipMap < 2 && numMipmaps > 1 )//dữ liệu ảnh dạng nén và chỉ có 1 mipmap level sẵn có,trong khi ta cần nhiều hơn 1 mipmap level
	{
		if (bitmap.DeCompress(true) == IMG_FAIL_NOT_SUPPORTED)//nếu giải nén vẫn không được
			numMipmaps=1;//cho số mipmap level bằng 1
	}

	if(changeAlpha)//có thay đổi alpha
	{
		int result=IMG_OK;
		if (bitmap.IsCompressed())
			result = bitmap.DeCompress(true);
		if(result==IMG_OK)
		{
			switch(bitmap.GetSurfaceFormat())
			{
			case FMT_R5G6B5:
				result=bitmap.RGB16ToRGBA(true);//R5G6B5 => A8B8G8R8
				break;
			case FMT_B5G6R5:
				result=bitmap.RGB16ToRGBA();//B5G6R5 => A8B8G8R8
				break;
			case FMT_R8G8B8:
				result=bitmap.RGB24ToRGBA(true);
				break;
			case FMT_B8G8R8:
				result=bitmap.RGB24ToRGBA();
				break;
			case FMT_L8:
				result=bitmap.L8ToAL16();//8 bit greyscale => 8 bit greyscale,8 bit alpha
				break;
			}
			if(result==IMG_OK)
				this->ChangePixelData(pTex);
		}
	}

	format=bitmap.GetSurfaceFormat();

	if(format==FMT_A8R8G8B8 || format==FMT_X8R8G8B8 || format==FMT_R8G8B8 || format==FMT_B5G6R5)
		bitmap.FlipRGB();//BGR->RGB

	if (pTex->type == HQ_TEXTURE_2D || pTex->type == HQ_TEXTURE_CUBE)
	{
		
		HQTexture2DDesc * l_textureDesc = (HQTexture2DDesc *)malloc (sizeof(HQTexture2DDesc));
		l_textureDesc->width = bitmap.GetWidth();
		l_textureDesc->height = bitmap.GetHeight();
		((HQTextureGL*)pTex)->textureDesc = l_textureDesc;
	}

	switch(pTex->type)
	{
	case HQ_TEXTURE_2D:
		return this->Create2DTexture(numMipmaps , pTex);
	case HQ_TEXTURE_CUBE:
		return this->CreateCubeTexture(numMipmaps , pTex);
	}
	return HQ_FAILED;
}
HQReturnVal HQTextureManagerGL::Create2DTexture(hq_uint32 numMipmaps,HQTexture * pTex)
{
	bool onlyFirstLevel = false;

	SurfaceComplexity complex=bitmap.GetSurfaceComplex();

	GLuint *pTextureName = (GLuint*)pTex->pData;
	glBindTexture(GL_TEXTURE_2D , *pTextureName);

	glPixelStorei( GL_UNPACK_ALIGNMENT, 1);
#ifndef GLES
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, (GLint)(numMipmaps-1));
#endif
	hq_uint32 w=bitmap.GetWidth();
	hq_uint32 h=bitmap.GetHeight();

	hquint32 exp = 0;

	//non power of 2 texture
	if(!IsPowerOfTwo(w,&exp) || !IsPowerOfTwo(h,&exp))
	{
		if (!g_pOGLDev->IsNpotTextureFullySupported(HQ_TEXTURE_2D))//only 1 mipmap level is allowed
		{
			onlyFirstLevel = true;
			if (pTex->fileName != NULL)
				Log("warning : only 1 mipmap level is allowed to use in a non power of 2 texture from file %s",pTex->fileName);
			else
				Log("warning : only 1 mipmap level is allowed to use in a non power of 2 texture ");
		}
	}

	hq_uint32 lvlSize;//độ lớn của 1 mipmap level trong dữ liệu ảnh

	SurfaceFormat format=bitmap.GetSurfaceFormat();
	//định dạng của texture lấy tương ứng theo định dạng của pixel data
	GLint textureFmt=helper::GetTextureFmt(format);

	GLenum dataFmt,dataType;
	helper::GetImageDataInfo(format,dataFmt,dataType);

	bool compressed= bitmap.IsCompressed();

	//tạo bộ mipmap
	hq_ubyte8 *pCur =bitmap.GetPixelData();
	bool fullMipmaps = false;

	for(hq_uint32 level=0;level<numMipmaps;++level)
	{
		if (w == 1 && h == 1)
			fullMipmaps = true;

		lvlSize=bitmap.CalculateSize(w,h);
		
		if (level == 0 || !onlyFirstLevel)
		{
			if(!compressed)
				glTexImage2D(GL_TEXTURE_2D,level,textureFmt,w,h,0,dataFmt,dataType,pCur);
			else
				glCompressedTexImage2D(GL_TEXTURE_2D,level,textureFmt,w,h,0,lvlSize,pCur);
		}

		if(w > 1) w >>=1; //w/=2
		if(h > 1) h >>=1; //h/=2


		if(complex.nMipMap < 2) //nếu trong dữ liệu ảnh chỉ có sẵn 1 mipmap level,tự tạo dữ liệu ảnh ở level thấp hơn bằng cách phóng nhỏ hình ảnh
		{
			bitmap.Scalei(w,h);
			pCur=bitmap.GetPixelData();
		}
		else
			pCur +=lvlSize;
	}//for (level)

#ifdef GLES
	if (!onlyFirstLevel && (this->generateMipmap || (numMipmaps > 1 && !fullMipmaps) ))
		glGenerateMipmap(GL_TEXTURE_2D);
#endif

	glBindTexture(GL_TEXTURE_2D,this->texUnits[this->activeTexture].GetTexture2DGL());//bind old texture

	return HQ_OK;
}
HQReturnVal HQTextureManagerGL::CreateCubeTexture(hq_uint32 numMipmaps,HQTexture * pTex)
{
	bool onlyFirstLevel = false;

	SurfaceComplexity complex=bitmap.GetSurfaceComplex();
	if((complex.dwComplexFlags & SURFACE_COMPLEX_CUBE) != 0 &&
		complex.nMipMap < 2 && numMipmaps > 1 )//dữ liệu ảnh dạng cube map và chỉ có 1 mipmap level sẵn có,trong khi ta cần nhiều hơn 1 mipmap level
		numMipmaps = 1;

	GLuint *pTextureName = (GLuint*)pTex->pData;
	glBindTexture(GL_TEXTURE_CUBE_MAP , *pTextureName);

	glPixelStorei( GL_UNPACK_ALIGNMENT, 1);
#ifndef GLES
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, (GLint)(numMipmaps-1));
#endif

	hq_uint32 Width=bitmap.GetWidth();
	hquint32 Exp;

	//non power of 2 texture
	if(!IsPowerOfTwo(Width,&Exp))
	{
		if (!g_pOGLDev->IsNpotTextureFullySupported(HQ_TEXTURE_CUBE))//only 1 mipmap level is allowed
		{
			onlyFirstLevel = true;
			if (pTex->fileName != NULL)
				Log("warning : only 1 mipmap level is allowed to use in a non power of 2 texture from file %s",pTex->fileName);
			else
				Log("warning : only 1 mipmap level is allowed to use in a non power of 2 texture ");
		}
	}

	hq_uint32 lvlSize;//độ lớn của 1 mipmap level trong dữ liệu ảnh

	SurfaceFormat format=bitmap.GetSurfaceFormat();
	//định dạng của texture lấy tương ứng theo định dạng của pixel data
	GLint textureFmt=helper::GetTextureFmt(format);

	GLenum dataFmt,dataType;
	helper::GetImageDataInfo(format,dataFmt,dataType);

	bool compressed= bitmap.IsCompressed();

	//tạo bộ mipmap
	hq_ubyte8 *pCur =bitmap.GetPixelData();
	bool fullMipmaps = false;

	for (int face = 0 ; face < 6 ; ++face)
	{
		hq_uint32 w = Width;
		for(hq_uint32 level=0;level<numMipmaps;++level)
		{
			if (w == 1)
				fullMipmaps = true;

			lvlSize=bitmap.CalculateSize(w,w);

			if (level == 0 || !onlyFirstLevel)
			{
				if(!compressed)
					glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,level,textureFmt,w,w,0,dataFmt,dataType,pCur);
				else
					glCompressedTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,level,textureFmt,w,w,0,lvlSize,pCur);
			}
			if(w > 1) w >>=1; //w/=2

			pCur +=lvlSize;
		}//for (level)
	}//for (face)

#ifdef GLES
	if (!onlyFirstLevel && (this->generateMipmap || 
		(numMipmaps > 1 && !fullMipmaps)))
		glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
#endif
	glBindTexture(GL_TEXTURE_CUBE_MAP,this->texUnits[this->activeTexture].GetTextureCubeGL());//bind old texture

	return HQ_OK;
}



/*
set alphakey
*/
HQReturnVal HQTextureManagerGL::SetAlphaValue(hq_ubyte8 R,hq_ubyte8 G,hq_ubyte8 B,hq_ubyte8 A)
{
	hq_ubyte8* pData=bitmap.GetPixelData();
	SurfaceFormat format=bitmap.GetSurfaceFormat();
	DWORD color;

	if(pData==NULL)
		return HQ_FAILED;

	switch(format)
	{
	case FMT_A8R8G8B8:
		color=COLOR_ARGB(A,R,G,B);
		break;
	case FMT_A8B8G8R8:
		color=COLOR_ARGB(A,B,G,R);
		break;
	case FMT_A8L8:
		color=COLOR_ARGB(A,R,0,0);
		break;
	default:
		return HQ_FAILED;
	}

	hq_ubyte8 *pCur=pData;
	hq_ubyte8 *pEnd=pData + bitmap.GetImgSize();

	hq_ushort16 pixelSize=bitmap.GetBits()/8;

	for(;pCur < pEnd; pCur+=pixelSize)//với từng pixel
	{
		if(format==FMT_A8L8)
		{
			if(*pCur==R)
				*((hq_ushort16*)pCur)=(hq_ushort16)(color >> 16);//(A,R) => (A,L)
		}
		else{
			DWORD * pPixel=(DWORD*)pCur;
			if(((*pPixel) & 0xffffff )== (color & 0xffffff))//cùng giá trị RGB
				*pPixel=color;
		}
	}

	return HQ_OK;
}

/*
set transparency
*/
HQReturnVal HQTextureManagerGL::SetTransparency(hq_float32 alpha)
{
	hq_ubyte8* pData=bitmap.GetPixelData();
	SurfaceFormat format=bitmap.GetSurfaceFormat();

	if(pData==NULL)
		return HQ_FAILED;

	hq_ubyte8 *pCur=pData;
	hq_ubyte8 *pEnd=pData + bitmap.GetImgSize();

	hq_ushort16 pixelSize=bitmap.GetBits()/8;
	hq_ubyte8 A=(hq_ubyte8)(alpha*255);//alpha range 0->255
	for(;pCur < pEnd; pCur+=pixelSize)//với từng pixel
	{
		switch(format)
		{
		case FMT_A8R8G8B8:case FMT_A8B8G8R8:
			if(pCur[3] > A)
				pCur[3]=A;
			break;
		case FMT_A8L8:
			if(pCur[1] > A )
				pCur[1]=A;
			break;
		default:
			return HQ_FAILED;
		}
	}
	return HQ_OK;
}
#ifndef GLES

HQReturnVal HQTextureManagerGL::CreateTextureBuffer(HQTexture *pTex ,HQTextureBufferFormat format , hq_uint32 size , void *initData, bool isDynamic)
{
	if (!g_pOGLDev->IsTextureBufferFormatSupported(format))
		return HQ_FAILED_FORMAT_NOT_SUPPORT;
	GLenum glFormat = GetTextureBufferFormat(format);

	HQTextureBufferGL * tbuffer = (HQTextureBufferGL *)pTex;
	tbuffer->usage = _GL_DRAW_BUFFER_USAGE( isDynamic);
	tbuffer->size = size;

	GLuint texture = *(GLuint*)tbuffer->pData;

	this->BindTextureBuffer(tbuffer->buffer);

	glBufferData(GL_TEXTURE_BUFFER , size , initData , tbuffer->usage);

	GLenum error = glGetError();
	if (GL_OUT_OF_MEMORY == error)
		return HQ_FAILED_MEM_ALLOC;

	else if (error != GL_NO_ERROR)
		return HQ_FAILED;

	this->BindTexture(this->activeTexture , GL_TEXTURE_BUFFER , texture);

	glTexBuffer(GL_TEXTURE_BUFFER , glFormat , tbuffer->buffer);

	glBindTexture(GL_TEXTURE_BUFFER , this->texUnits[activeTexture].GetTextureBufferGL()); //re-bind old texture

	return HQ_OK;
}
HQReturnVal HQTextureManagerGL::MapTextureBuffer(hq_uint32 textureID , void **ppData )
{
	HQTexture* pTexture = this->textures.GetItemRawPointer(textureID);
#if defined _DEBUG || defined DEBUG
	if (pTexture == NULL || pTexture->type != HQ_TEXTURE_BUFFER)
		return HQ_FAILED_INVALID_ID;
	if (ppData == NULL)
		return HQ_FAILED;
#endif

	HQTextureBufferGL * tbuffer = (HQTextureBufferGL *)pTexture;

	this->BindTextureBuffer(tbuffer->buffer);

	glBufferData(GL_TEXTURE_BUFFER , tbuffer->size , NULL , tbuffer->usage);//discard old data

	*ppData = glMapBuffer( GL_TEXTURE_BUFFER , GL_WRITE_ONLY);
	
	return HQ_OK;
}
HQReturnVal HQTextureManagerGL::UnmapTextureBuffer(hq_uint32 textureID)
{
	HQTexture* pTexture = this->textures.GetItemRawPointer(textureID);
#if defined _DEBUG || defined DEBUG	
	if (pTexture == NULL || pTexture->type != HQ_TEXTURE_BUFFER)
		return HQ_FAILED_INVALID_ID;
#endif

	HQTextureBufferGL * tbuffer = (HQTextureBufferGL *)pTexture;

	this->BindTextureBuffer(tbuffer->buffer);
	glUnmapBuffer(GL_TEXTURE_BUFFER);
	return HQ_OK;
}

#endif//#ifndef GLES



HQReturnVal HQTextureManagerGL::GetTexture2DSize(hq_uint32 textureID, hquint32 &width, hquint32& height)
{
	HQSharedPtr<HQTexture> pTexture = this->textures.GetItemPointer(textureID);
	if (pTexture == NULL)
		return HQ_FAILED_INVALID_ID;

	switch (pTexture->type)
	{
	case HQ_TEXTURE_2D:case HQ_TEXTURE_CUBE:
		{
			HQTexture2DDesc * l_textureDesc = (HQTexture2DDesc *)((HQTextureGL*)pTexture.GetRawPointer())->textureDesc;
			width = l_textureDesc->width;
			height = l_textureDesc->height;
		}
		break;
	default:
		return HQ_FAILED;
	}
	return HQ_OK;
}

HQTextureCompressionSupport HQTextureManagerGL::IsCompressionSupported(HQTextureType textureType,HQTextureCompressionFormat type)
{
	switch (type)
	{
	case HQ_TC_S3TC_DTX1:
	case HQ_TC_S3TC_DXT3:
	case HQ_TC_S3TC_DXT5:
		if(GLEW_EXT_texture_compression_s3tc)
			return HQ_TCS_ALL;
		else
			return HQ_TCS_SW;
	case HQ_TC_ETC1: 
#ifdef GLES
		if (GLEW_OES_compressed_ETC1_RGB8_texture && (textureType == HQ_TEXTURE_2D || textureType == HQ_TEXTURE_CUBE))
			return HQ_TCS_ALL;
#endif
		return HQ_TCS_SW;
	case HQ_TC_PVRTC_RGB_2BPP :
	case HQ_TC_PVRTC_RGB_4BPP: 
	case HQ_TC_PVRTC_RGBA_2BPP:
	case HQ_TC_PVRTC_RGBA_4BPP:
#ifdef GLES
		if (GLEW_IMG_texture_compression_pvrtc)
#	if PVRTC_DECOMPRESSION_SUPPORT
			return HQ_TCS_ALL;
#	else
			return HQ_TCS_HW;
#	endif
#endif
#if PVRTC_DECOMPRESSION_SUPPORT
		return HQ_TCS_SW;
#else
		return HQ_TCS_NONE;
#endif
	default:
		return HQ_TCS_NONE;
	}
}

HQReturnVal HQTextureManagerGL::RemoveTexture(hq_uint32 ID)
{
	const HQSharedPtr<HQTexture> pTex = this->GetTexture(ID);
	if (pTex == NULL)
		return HQ_FAILED_INVALID_ID;

	HQTextureGL* pTextureRawPtr = (HQTextureGL*)pTex.GetRawPointer();
	//check if this texture is bound to any texture unit
	for (hquint32 i = 0; i < this->maxTextureUnits ; ++i)
	{
		if (this->texUnits[i].texture[pTextureRawPtr->type] == pTex)
			this->texUnits[i].texture[pTextureRawPtr->type] = HQSharedPtr<HQTexture>::null;
	}

	return HQBaseTextureManager::RemoveTexture(ID);
}
void HQTextureManagerGL::RemoveAllTexture()
{
	for (hquint32 i = 0; i < this->maxTextureUnits ; ++i)
	{
		this->texUnits[i].texture[HQ_TEXTURE_2D] = HQSharedPtr<HQTexture>::null;
		this->texUnits[i].texture[HQ_TEXTURE_CUBE] = HQSharedPtr<HQTexture>::null;
		this->texUnits[i].texture[HQ_TEXTURE_BUFFER] = HQSharedPtr<HQTexture>::null;
	}
	HQBaseTextureManager::RemoveAllTexture();
}

#if defined DEVICE_LOST_POSSIBLE
void HQTextureManagerGL::OnLost()
{
	this->RemoveAllTexture();
	this->activeTexture = 0;
}
#endif


HQBaseRawPixelBuffer* HQTextureManagerGL::CreatePixelBufferImpl(HQRawPixelFormat intendedFormat, hquint32 width, hquint32 height)
{
	switch (intendedFormat)
	{
	case HQ_RPFMT_R8G8B8A8:
	case HQ_RPFMT_B8G8R8A8:
		return HQ_NEW HQBaseRawPixelBuffer(HQ_RPFMT_R8G8B8A8, width, height);
		break;
	default:
		return HQ_NEW HQBaseRawPixelBuffer(intendedFormat, width, height);
	}
}

HQReturnVal HQTextureManagerGL::CreateTexture(HQTexture *pTex, const HQBaseRawPixelBuffer* color)
{

	color->MakeWrapperBitmap(bitmap);


	hquint32 w=bitmap.GetWidth();
	hquint32 h=bitmap.GetHeight();
	hquint32 bpp=bitmap.GetBits();
	SurfaceFormat format=bitmap.GetSurfaceFormat();
	ImgOrigin origin=bitmap.GetPixelOrigin();
	SurfaceComplexity complex=bitmap.GetSurfaceComplex();


	
#ifdef GLES
	if(!GLEW_OES_texture_non_power_of_two)//kích thước texture phải là lũy thừa của 2
	{
		int result;
		hq_uint32 exp;
		bool needResize=false;
		if(!IsPowerOfTwo(w,&exp))//chiều rộng không là lũy thừa của 2
		{
			needResize=true;
			w=0x1 << exp;//2^exp
		}
		if(!IsPowerOfTwo(h,&exp))//chiều cao không là lũy thừa của 2
		{
			needResize=true;
			h=0x1 << exp;//2^exp
		}
		if(needResize)
		{
			if (pTex->type == HQ_TEXTURE_CUBE)
			{
				Log("Load cube texture from files %s error : Dimension need to be power of 2");
				return HQ_FAILED;
			}
			Log("Now trying to resize image dimesions to power of two dimensions");
			unsigned long size=bitmap.GetFirstLevelSize();

			hq_ubyte8 *pData=new hq_ubyte8[size];
			if(!pData)
			{
				Log("Memory allocation failed");
				return HQ_FAILED;
			}
			//copy first level pixel data
			memcpy(pData,bitmap.GetPixelData(),size);
			//loại bỏ các thuộc tính phức tạp
			memset(&complex,0,sizeof(SurfaceComplexity));

			bitmap.Set(pData,bitmap.GetWidth(),bitmap.GetHeight(),bpp,size, format,origin,complex);
			//giải nén nếu hình ảnh ở dạng nén ,làm thế mới resize hình ảnh dc
			
			if(bitmap.IsCompressed() &&  (result = bitmap.DeCompress())!=IMG_OK)
			{
				if (result == IMG_FAIL_MEM_ALLOC)
					Log("Memory allocation failed when attempt to decompressing compressed data!");
				else
					Log("Couldn't decompress compressed data!");
				return HQ_FAILED;
			}
			//phóng to hình ảnh lên thành kích thước lũy thừa của 2
			bitmap.Scalei(w,h);
			
			//chỉnh lại thông tin cơ sở
			format=bitmap.GetSurfaceFormat();
			bpp=bitmap.GetBits();
		}//if (need resize)
	}//if (must power of two)
#endif

	hq_uint32 nMips = 1;//số lượng mipmap level
	if(generateMipmap)
	{
		//full range mipmap
		if(IsPowerOfTwo(max(w,h),&nMips))//nMips=1+floor(log2(max(w,h)))
			nMips++;

		else if ((complex.dwComplexFlags & SURFACE_COMPLEX_MIPMAP)!=0 && complex.nMipMap>1)//nếu trong file ảnh có chứa nhiều hơn 1 mipmap level
		{
			nMips=complex.nMipMap;
		}

	}

	
	if(CreateTexture((pTex->alpha<1.0f || (pTex->colorKey!=NULL && pTex->nColorKey>0)),nMips,pTex)!=HQ_OK)
	{
		return HQ_FAILED;
	}

	return HQ_OK;
}
