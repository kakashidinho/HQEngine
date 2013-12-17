/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
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
HQReturnVal HQBaseTextureManager::RemoveTexture(hq_uint32 ID)
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
const HQSharedPtr<HQTexture> HQBaseTextureManager::GetTexture(hq_uint32 ID)
{
	return this->textures.GetItemPointer(ID);
}

HQReturnVal HQBaseTextureManager::AddSingleColorTexture(HQColorui color ,  hq_uint32 *pTextureID)
{
	hq_uint32 texID;
	bool exist=false;
	HQItemManager<HQTexture>::Iterator ite;
	this->textures.GetIterator(ite);
	//kiểm tra texture đã có sẵn chưa
	for(;!ite.IsAtEnd();++ite)
	{
		if(ite.GetItemPointerNonCheck() != NULL && ite->fileName != NULL && !memcmp(ite->fileName,&color,sizeof(color)))//kiểm tra cùng giá trị màu
		{
			texID = ite.GetID();
			exist=true;
			Log("Texture has single color %u already exists!",color);
			break;
		}
	}
	if(!exist)
	{
		HQTexture *pNewTex = this->CreateNewTextureObject(HQ_TEXTURE_2D);

		pNewTex->nColorKey=0;
		pNewTex->colorKey=NULL;
		pNewTex->alpha=1.0f;
		//lấy giá trị màu làm tên file để kiểm tra trùng lập sau này
		pNewTex->fileName=new char[sizeof(color)];
		memcpy(pNewTex->fileName , &color,sizeof(color));


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
	}
	if(pTextureID)
		*pTextureID = texID;

	return HQ_OK;
}

#ifndef GLES

HQReturnVal HQBaseTextureManager::AddTextureBuffer(HQTextureBufferFormat format , hq_uint32 size , void *initData ,bool isDynamic , hq_uint32 *pTextureID)
{
	HQTexture *pNewTex = this->CreateNewTextureObject(HQ_TEXTURE_BUFFER);

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

HQReturnVal HQBaseTextureManager::AddTexture(const HQRawPixelBuffer* color , bool generateMipmap, hq_uint32 *pTextureID)
{
	if (dynamic_cast<const HQBaseRawPixelBuffer*> (color) == NULL)
		return HQ_FAILED_INVALID_PARAMETER;

	this->generateMipmap = generateMipmap;

	HQTexture *pNewTex = this->CreateNewTextureObject(HQ_TEXTURE_2D);

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

HQReturnVal HQBaseTextureManager::AddTexture(const char* fileName,
						   hq_float32 maxAlpha,
						   const HQColor *colorKey,
						   hq_uint32 numColorKey,
						   bool generateMipmap,
						   HQTextureType textureType,
						   hq_uint32 *pTextureID)
{

	this->generateMipmap = generateMipmap;
	hq_uint32 texID;
	bool exist=false;
	HQItemManager<HQTexture>::Iterator ite;
	this->textures.GetIterator(ite);
	//kiểm tra texture đã có sẵn chưa
	for(;!ite.IsAtEnd();++ite)
	{
		if(ite.GetItemPointerNonCheck() != NULL && ite->fileName != NULL && !strcmp(fileName,ite->fileName))//cùng load từ 1 file
		{
			texID = ite.GetID();
			exist=true;
			Log("Texture from file %s already exists!",fileName);
			break;
		}
	}
	if(!exist)
	{
		HQTexture *pNewTex = this->CreateNewTextureObject(textureType);
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

		pNewTex->fileName=new char[strlen(fileName)+1];
		strcpy(pNewTex->fileName,fileName);


		HQReturnVal result=this->LoadTextureFromFile(pNewTex);
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
		//Log("Texture from file %s is successfully added!",fileName);
	}

	if(pTextureID)
		*pTextureID = texID;

	bitmap.ClearData();
	return HQ_OK;
}

HQReturnVal HQBaseTextureManager::AddCubeTexture(const char* fileNames[6],
						   hq_float32 maxAlpha,
						   const HQColor *colorKey,
						   hq_uint32 numColorKey,
						   bool generateMipmap,
						   hq_uint32 *pTextureID)
{

	hq_uint32 texID;
	bool exist=false;
	this->generateMipmap = generateMipmap;

	/*---tính tổng độ dài các tên của 6 file----*/
	hq_uint32 totalNameLen = 0;
	for (int i = 0 ; i < 6 ; ++i)
	{
		totalNameLen += strlen(fileNames[i]);
	}
	char *sumfileNames=new char[totalNameLen + 6];
	sumfileNames[0] = '\0';

	for (int i = 0 ; i < 6 ; ++i)
	{
		/*----gộp tên của 6 file------*/
		strcat(sumfileNames , fileNames[i]);
		if(i < 5)
			strcat(sumfileNames , ",");
	}
	HQItemManager<HQTexture>::Iterator ite;
	this->textures.GetIterator(ite);
	//kiểm tra texture đã có sẵn chưa
	for(;!ite.IsAtEnd();++ite)
	{
		if(ite.GetItemPointerNonCheck() != NULL && ite->fileName != NULL && !strcmp(sumfileNames , ite->fileName))//cùng load từ 1 file
		{
			texID = ite.GetID();
			exist=true;
			Log("Cube texture from files %s already exists!",sumfileNames);
			break;
		}
	}
	if(!exist)
	{
		HQTexture *pNewTex = this->CreateNewTextureObject(HQ_TEXTURE_CUBE);
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
		pNewTex->fileName = sumfileNames;

		HQReturnVal result=this->LoadCubeTextureFromFiles(fileNames , pNewTex);
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
		//Log("Texture from file %s is successfully added!",fileName);
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
void HQBaseTextureManager::ChangePixelData( HQTexture *pTex )
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


HQSharedPtr<HQTexture> HQBaseTextureManager::CreateEmptyTexture(HQTextureType textureType , hq_uint32 *pTextureID)
{
	HQTexture * pNewTex = this->CreateNewTextureObject( textureType);

	hq_uint32 newID = 0;
	if(!pNewTex || !this->textures.AddItem(pNewTex , &newID))
	{
		SafeDelete(pNewTex);
		return HQSharedPtr<HQTexture>::null;
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
