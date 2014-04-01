/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQPlatformDef.h"
#include "../HQEngineCustomHeap.h"
#include "HQTextureManagerBaseImpl.h"

static void GetTextureCompressionSupportString(char *str, HQTextureCompressionSupport support)
{
	switch (support)
	{
	case HQ_TCS_ALL: strcpy(str, "all"); break;
	case HQ_TCS_HW: strcpy(str, "hardware"); break;
	case HQ_TCS_SW: strcpy(str, "software"); break;
	case HQ_TCS_NONE: strcpy(str, "none"); break;
	}
}

static void GetTextureCompressionFormatString(char *str, HQTextureCompressionFormat format)
{
	switch (format)
	{
	case HQ_TC_S3TC_DTX1: strcpy(str, "S3TC DXT1"); break;
	case HQ_TC_S3TC_DXT3: strcpy(str, "S3TC DXT3"); break;
	case HQ_TC_S3TC_DXT5: strcpy(str, "S3TC DXT5"); break;
	case HQ_TC_ETC1:  strcpy(str, "ETC1"); break;
	case HQ_TC_PVRTC_RGB_2BPP : strcpy(str, "PVRTC RGB 2BPP"); break;
	case HQ_TC_PVRTC_RGB_4BPP: strcpy(str, "PVRTC RGB 4BPP"); break;
	case HQ_TC_PVRTC_RGBA_2BPP: strcpy(str, "PVRTC RGBA 2BPP"); break;
	case HQ_TC_PVRTC_RGBA_4BPP: strcpy(str, "PVRTC RGBA 4BPP"); break;
	}
}


/*--------------------------------*/

HQBaseRawPixelBuffer::HQBaseRawPixelBuffer(HQRawPixelFormat format, hquint32 width, hquint32 height)
:m_format(format), m_width(width), m_height(height), m_pixelSize(0)
{
	switch (format)
	{
	case HQ_RPFMT_R8G8B8A8:
	case HQ_RPFMT_B8G8R8A8:
		m_pixelSize = 4;
		break;
	case HQ_RPFMT_R5G6B5:
	case HQ_RPFMT_L8A8:
		m_pixelSize = 2;
		break;
	case HQ_RPFMT_A8:
		m_pixelSize = 1;
		break;
	}

	m_pData = new hqubyte8[m_pixelSize * m_width * m_height];
	m_rowSize = m_pixelSize * m_width;
}


HQBaseRawPixelBuffer::~HQBaseRawPixelBuffer()
{
	delete[] m_pData;
}

void HQBaseRawPixelBuffer::SetPixelf(hquint32 x, hquint32 y, float r, float g, float b, float a)
{
	hqubyte8 *pPixel = GetPixel(x, y);

	switch (m_format)
	{
	case HQ_RPFMT_R8G8B8A8:
		pPixel[0] = (hqubyte8)(r * 255.0f);
		pPixel[1] = (hqubyte8)(g * 255.0f);
		pPixel[2] = (hqubyte8)(b * 255.0f);
		pPixel[3] = (hqubyte8)(a * 255.0f);
		break;
	case HQ_RPFMT_B8G8R8A8:
		pPixel[0] = (hqubyte8)(b * 255.0f);
		pPixel[1] = (hqubyte8)(g * 255.0f);
		pPixel[2] = (hqubyte8)(r * 255.0f);
		pPixel[3] = (hqubyte8)(a * 255.0f);
		break;
	case HQ_RPFMT_R5G6B5:
		{
			hqushort16 *pPixel16bit = (hqushort16*)pPixel;
			*pPixel16bit = RGB16((hquint32)(r * 31.f), (hquint32)(g * 63.f), (hquint32)(b * 31.f));
		}
		break;
	case HQ_RPFMT_L8A8:
		pPixel[0] = (hqubyte8)(r * 255.0f);
		pPixel[1] = (hqubyte8)(a * 255.0f);
		break;
	case HQ_RPFMT_A8:
		pPixel[0] = (hqubyte8)(a * 255.0f);
		break;
	}
}

void HQBaseRawPixelBuffer::SetPixel(hquint32 x, hquint32 y, hqubyte8 r, hqubyte8 g, hqubyte8 b, hqubyte8 a)
{
	hqubyte8 *pPixel = GetPixel(x, y);

	switch (m_format)
	{
	case HQ_RPFMT_R8G8B8A8:
		pPixel[0] = r ;
		pPixel[1] = g ;
		pPixel[2] = b ;
		pPixel[3] = a ;
		break;
	case HQ_RPFMT_B8G8R8A8:
		pPixel[0] = b ;
		pPixel[1] = g ;
		pPixel[2] = r ;
		pPixel[3] = a ;
		break;
	case HQ_RPFMT_R5G6B5:
		{
			hqushort16 *pPixel16bit = (hqushort16*)pPixel;
			hquint32 Rval = (hquint32)(r * 31.f / 255.f);
			hquint32 Gval = (hquint32)(g * 63.f / 255.f);
			hquint32 Bval = (hquint32)(b * 31.f / 255.f);
			*pPixel16bit = RGB16(Rval, Gval, Bval);
		}
		break;
	case HQ_RPFMT_L8A8:
		pPixel[0] = r;
		pPixel[1] = a;
		break;
	case HQ_RPFMT_A8:
		pPixel[0] = a;
		break;
	}
}

HQColor HQBaseRawPixelBuffer::GetPixelData(int x, int y) const
{
	HQColor color;
	const hqubyte8 *pPixel = GetPixel(x, y);

	switch (m_format)
	{
	case HQ_RPFMT_R8G8B8A8:
		color.r =  pPixel[0]/ 255.f ;
		color.g =  pPixel[1]/ 255.f ;
		color.b =  pPixel[2]/ 255.f ;
		color.a =  pPixel[3]/ 255.f ;
		break;
	case HQ_RPFMT_B8G8R8A8:
		color.r =  pPixel[2]/ 255.f ;
		color.g =  pPixel[1]/ 255.f ;
		color.b =  pPixel[1]/ 255.f ;
		color.a =  pPixel[3]/ 255.f ;
		break;
	case HQ_RPFMT_R5G6B5:
		{
			const hqushort16 &pixel16bit = *(hqushort16*)pPixel;

			color.r =  GetRfromRGB16(pixel16bit)/ 31.f ;
			color.g =  GetGfromRGB16(pixel16bit)/ 63.f ;
			color.b =  GetBfromRGB16(pixel16bit)/ 31.f ;
			color.a =  1.0f ;
		}
		break;
	case HQ_RPFMT_L8A8:
		color.r =  pPixel[0]/ 255.f ;
		color.g =  pPixel[0]/ 255.f ;
		color.b =  pPixel[0]/ 255.f ;
		color.a =  pPixel[1]/ 255.f ;
		break;
	case HQ_RPFMT_A8:
		color.r =  0 ;
		color.g =  0 ;
		color.b =  0 ;
		color.a =  pPixel[0]/ 255.f ;
		break;
	}


	return color;
}

hqubyte8* HQBaseRawPixelBuffer::GetPixel(hquint32 x, hquint32 y)
{
	return m_pData + m_rowSize * y + m_pixelSize * x;
}

const hqubyte8* HQBaseRawPixelBuffer::GetPixel(hquint32 x, hquint32 y) const
{
	return m_pData + m_rowSize * y + m_pixelSize * x;
}

void HQBaseRawPixelBuffer::MakeWrapperBitmap( Bitmap &bitmap) const
{
	SurfaceFormat bitmapFmt;
	switch (m_format)
	{
	case HQ_RPFMT_R8G8B8A8:
		bitmapFmt = FMT_A8B8G8R8;
		break;
	case HQ_RPFMT_B8G8R8A8:
		bitmapFmt = FMT_A8R8G8B8;
		break;
	case HQ_RPFMT_R5G6B5:
		bitmapFmt = FMT_R5G6B5;
		break;
	case HQ_RPFMT_L8A8:
		bitmapFmt = FMT_A8L8;
		break;
	case HQ_RPFMT_A8:
		bitmapFmt = FMT_A8;
		break;
	default:
		bitmapFmt = FMT_UNKNOWN;
	}

	SurfaceComplexity complex;
	complex.nMipMap = 1;

	bitmap.Wrap((hqubyte8*)GetPixelData(), 
		GetWidth(), 
		GetHeight(), 
		GetBitsPerPixel(), 
		GetBufferSize(), 
		bitmapFmt, 
		GetFirstPixelPosition(),
		complex);
}

ImgOrigin HQBaseRawPixelBuffer::GetFirstPixelPosition() const
{
	return ORIGIN_TOP_LEFT;
}

/*--------------------------------*/
/*
Destructor
*/
HQBaseTextureManager::~HQBaseTextureManager()
{
	RemoveAllTexture();
	Log("Released!");
}
/*
release
*/
HQReturnVal HQBaseTextureManager::RemoveTexture(HQTexture* ID)
{
	return (HQReturnVal)this->textures.Remove(ID);
}

void HQBaseTextureManager::RemoveAllTexture()
{
	this->textures.RemoveAll();
}

/*
get texture
*/
const HQSharedPtr<HQBaseTexture> HQBaseTextureManager::GetTextureSharedPtr(HQTexture* ID)
{
	return this->textures.GetItemPointer(ID);
}

const HQSharedPtr<HQBaseTexture> HQBaseTextureManager::GetTextureSharedPtrAt(hquint32 resourceIndex)
{
	return this->textures.ParentType::GetItemPointer(resourceIndex);
}

HQReturnVal HQBaseTextureManager::AddSingleColorTexture(HQColorui color, HQTexture** pTextureID)
{
	HQTexture* texID;
	HQItemManager<HQBaseTexture>::Iterator ite;
	this->textures.GetIterator(ite);
	HQBaseTexture *pNewTex = this->CreateNewTextureObject(HQ_TEXTURE_2D);

	pNewTex->nColorKey=0;
	pNewTex->colorKey=NULL;
	pNewTex->alpha=1.0f;


	HQReturnVal result=this->CreateSingleColorTexture(pNewTex,color);
	if(result!=HQ_OK)
	{
		//giải phóng bộ nhớ
		delete pNewTex;
		return result;
	}
	if(this->textures.AddItem(pNewTex,&texID)==false)
	{
		delete pNewTex;
		return HQ_FAILED_MEM_ALLOC;
	}
	if(pTextureID)
		*pTextureID = texID;

	return HQ_OK;
}

#ifndef HQ_OPENGLES

HQReturnVal HQBaseTextureManager::AddTextureBuffer(HQTextureBufferFormat format, hq_uint32 size, void *initData, bool isDynamic, HQTextureBuffer** pTextureID)
{
	HQBaseTexture *pNewTex = this->CreateNewTextureObject(HQ_TEXTURE_BUFFER);

	HQReturnVal result=this->CreateTextureBuffer(pNewTex , format , size , initData, isDynamic);
	if(result!=HQ_OK)
	{
		//giải phóng bộ nhớ
		delete pNewTex;
		return result;
	}
	if(this->textures.AddItem(pNewTex,pTextureID)==false)
	{
		delete pNewTex;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}
#endif


HQRawPixelBuffer* HQBaseTextureManager::CreatePixelBuffer(HQRawPixelFormat intendedFormat, hquint32 width, hquint32 height)
{
	return CreatePixelBufferImpl(intendedFormat, width, height);//must return HQBaseRawPixelBuffer*
}

HQReturnVal HQBaseTextureManager::AddTexture(const HQRawPixelBuffer* color, bool generateMipmap, HQTexture** pTextureID)
{
	if (dynamic_cast<const HQBaseRawPixelBuffer*> (color) == NULL)
		return HQ_FAILED_INVALID_PARAMETER;

	this->generateMipmap = generateMipmap;

	HQBaseTexture *pNewTex = this->CreateNewTextureObject(HQ_TEXTURE_2D);

	HQReturnVal result=this->CreateTexture(pNewTex , static_cast<const HQBaseRawPixelBuffer*> (color));
	if(result!=HQ_OK)
	{
		//giải phóng bộ nhớ
		delete pNewTex;
		return result;
	}
	if(this->textures.AddItem(pNewTex,pTextureID)==false)
	{
		delete pNewTex;
		return HQ_FAILED_MEM_ALLOC;
	}

	return HQ_OK;
}

HQReturnVal HQBaseTextureManager::AddTexture(HQDataReaderStream* dataStream,
						   hq_float32 maxAlpha,
						   const HQColor *colorKey,
						   hq_uint32 numColorKey,
						   bool generateMipmap,
						   HQTextureType textureType,
						   HQTexture** pTextureID)
{

	this->generateMipmap = generateMipmap;
	HQTexture* texID;
	HQItemManager<HQBaseTexture>::Iterator ite;
	this->textures.GetIterator(ite);
	
	HQBaseTexture *pNewTex = this->CreateNewTextureObject(textureType);
	if (pNewTex == NULL)
        return HQ_FAILED;

	//copy ColorKey & maxAlpha
	if(numColorKey>0 && colorKey!=NULL)
	{
		pNewTex->nColorKey=numColorKey;
		pNewTex->colorKey=new HQColor[numColorKey];
		memcpy(pNewTex->colorKey,colorKey,numColorKey*sizeof(HQColor));
	}
	else
	{
		pNewTex->nColorKey=0;
		pNewTex->colorKey=NULL;
	}
	pNewTex->alpha=maxAlpha;


	HQReturnVal result=this->LoadTextureFromStream(dataStream, pNewTex);
	if(result!=HQ_OK)
	{
		//giải phóng bộ nhớ
		delete pNewTex;
		return result;
	}
	if(this->textures.AddItem(pNewTex,&texID)==false)
	{
		delete pNewTex;
		return HQ_FAILED_MEM_ALLOC;
	}
	

	if(pTextureID)
		*pTextureID = texID;

	bitmap.ClearData();
	return HQ_OK;
}

HQReturnVal HQBaseTextureManager::AddCubeTexture(HQDataReaderStream* dataStreams[6],
						   hq_float32 maxAlpha,
						   const HQColor *colorKey,
						   hq_uint32 numColorKey,
						   bool generateMipmap,
						   HQTexture** pTextureID)
{

	HQTexture* texID;
	this->generateMipmap = generateMipmap;

	
	HQItemManager<HQBaseTexture>::Iterator ite;
	this->textures.GetIterator(ite);
	HQBaseTexture *pNewTex = this->CreateNewTextureObject(HQ_TEXTURE_CUBE);
	if (pNewTex == NULL)
		return HQ_FAILED;

	//copy ColorKey & maxAlpha
	if(numColorKey>0 && colorKey!=NULL)
	{
		pNewTex->nColorKey=numColorKey;
		pNewTex->colorKey=new HQColor[numColorKey];
		memcpy(pNewTex->colorKey,colorKey,numColorKey*sizeof(HQColor));
	}
	else
	{
		pNewTex->nColorKey=0;
		pNewTex->colorKey=NULL;
	}
	pNewTex->alpha=maxAlpha;

	HQReturnVal result=this->LoadCubeTextureFromStreams(dataStreams , pNewTex);
	if(result!=HQ_OK)
	{
		//giải phóng bộ nhớ
		delete pNewTex;
		return result;
	}

	if(this->textures.AddItem(pNewTex,&texID)==false)
	{
		delete pNewTex;
		return HQ_FAILED_MEM_ALLOC;
	}


	if(pTextureID)
		*pTextureID = texID;

	bitmap.ClearData();
	return HQ_OK;
}

bool HQBaseTextureManager::IsPowerOfTwo(const hq_uint32 d,hq_uint32 *pExponent)
{
	hq_uint32 i=d;
	if(i==0)
	{
		if(pExponent)
			*pExponent=0;
		return false;
	}
	hq_uint32 result=31;
	if((i & 0xffffff00) == 0)
	{
		i <<= 24;
		result=7;
	}
	else if ((i & 0xffff0000) == 0)
	{
		i <<=16;
		result=15;
	}
	else if ((i & 0xff000000) == 0)
	{
		i <<=8;
		result=23;
	}

	if((i & 0xf0000000)==0)
	{
		i <<=4;
		result-=4;
	}
	while ((i & 0x80000000) == 0)
	{
		i <<=1;
		result-=1;
	}
	if( i & 0x7fffffff )
	{
		if(pExponent)
			*pExponent=result+1;
		return false;
	}
	if(pExponent)
		*pExponent=result;
	return true;
}


//tính số mipmap level tối đa
hq_uint32 HQBaseTextureManager::CalculateFullNumMipmaps(hq_uint32 width, hq_uint32 height)
{
	hq_uint32 nMips;
	if(IsPowerOfTwo(max(width,height),&nMips))//nMips=1+floor(log2(max(w,h)))
		nMips++;
	return nMips;
}
void HQBaseTextureManager::ChangePixelData( HQBaseTexture *pTex )
{
	if(pTex->colorKey){
		//chuyển color có giá trị RGB như colorkey thành giống giá trị RGBA của colorkey
		for(hq_uint32 i=0;i<pTex->nColorKey;++i)
		{
			if(this->SetAlphaValue(
								(hq_ubyte8)(pTex->colorKey[i].r * 255),
								(hq_ubyte8)(pTex->colorKey[i].g * 255),
								(hq_ubyte8)(pTex->colorKey[i].b * 255),
								(hq_ubyte8)(pTex->colorKey[i].a * 255)
								 )!=HQ_OK)
			{
				Log("Attempt to set alpha value failed");
			}
		}
	}
	//chuyển giá trị alpha lớn nhất thành pTex->alpha
	if(this->SetTransparency(pTex->alpha)!=HQ_OK)
		Log("Attempt to set transparency failed");
}


HQSharedPtr<HQBaseTexture> HQBaseTextureManager::CreateEmptyTexture(HQTextureType textureType, HQTexture** pTextureID)
{
	HQBaseTexture * pNewTex = this->CreateNewTextureObject( textureType);

	HQTexture* newID = 0;
	if(!pNewTex || !this->textures.AddItem(pNewTex , &newID))
	{
		SafeDelete(pNewTex);
		return HQSharedPtr<HQBaseTexture>::null;
	}
	if(pTextureID != NULL)
		*pTextureID = newID;
	return this->textures.GetItemPointerNonCheck(newID);
}


void HQBaseTextureManager::LogTextureCompressionSupportInfo()
{
	char formatStr[16];
	char supportStr[9];

	Log("	2D texture compression support:");
	for (hquint32 format =  0 ; format < HQ_TC_COUNT ; ++format )
	{
		GetTextureCompressionFormatString(formatStr, (HQTextureCompressionFormat)format);
		HQTextureCompressionSupport support = this->IsCompressionSupported(HQ_TEXTURE_2D, (HQTextureCompressionFormat)format);
		GetTextureCompressionSupportString(supportStr, support);

		Log("		%s : %s", formatStr, supportStr);
	}

	Log("	Cube texture compression support:");
	for (hquint32 format =  0 ; format < HQ_TC_COUNT ; ++format )
	{
		GetTextureCompressionFormatString(formatStr, (HQTextureCompressionFormat)format);
		HQTextureCompressionSupport support = this->IsCompressionSupported(HQ_TEXTURE_2D, (HQTextureCompressionFormat)format);
		GetTextureCompressionSupportString(supportStr, support);

		Log("		%s : %s", formatStr, supportStr);
	}
}
