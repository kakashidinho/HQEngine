#include "HQDeviceD3D9PCH.h"
#include "HQTextureManagerD3D9.h"
#include "HQDeviceEnumD3D9.h"
#include "HQDeviceD3D9.h"
#include <d3dx9.h>
#pragma comment(lib,"d3dx9.lib")
#include "../../VS/Libs/PVRTexLib/PVRTexLib.h"
#if defined _DEBUG || defined DEBUG || !defined _STATIC_CRT
#pragma comment (lib, "../../VS/Libs/PVRTexLib/Windows_x86_32/Dll/PVRTexLib.lib")
#else
#pragma comment (lib, "../../VS/Libs/PVRTexLib/Windows_x86_32/Lib/PVRTexLib.lib")
#endif
using namespace pvrtexlib;

inline hq_uint32 GetShaderStageIndex(HQShaderType type) 
{
	return type >> 29;
}

struct HQTextureD3D9:public HQTexture
{
	HQTextureD3D9(HQTextureType type = HQ_TEXTURE_2D) : HQTexture()
	{
		this->type = type;
	}
	~HQTextureD3D9()
	{
		if(pData)
		{
			LPDIRECT3DBASETEXTURE9 pTex = (LPDIRECT3DBASETEXTURE9)pData;
			pTex->Release();
		}
	}
};
//************************************************************
//định dạng của texture tương ứng với định dạng của pixel ảnh
//************************************************************
static D3DFORMAT GetTextureFmt(const SurfaceFormat imgFmt)
{
	switch (imgFmt)
	{
	case FMT_R8G8B8: case FMT_B8G8R8: 
		return D3DFMT_X8R8G8B8;
	case FMT_A8R8G8B8:case FMT_A8B8G8R8:
		return D3DFMT_A8R8G8B8;
	case FMT_X8R8G8B8:case FMT_X8B8G8R8:
		return D3DFMT_X8R8G8B8;
	case FMT_R5G6B5: case FMT_B5G6R5:
		return D3DFMT_R5G6B5;
	case FMT_L8:
		return D3DFMT_L8;
	case FMT_A8L8:
		return D3DFMT_A8L8;
	case FMT_A8:
		return D3DFMT_A8;
	case FMT_S3TC_DXT1:
		return D3DFMT_DXT1;
	case FMT_S3TC_DXT3:
		return D3DFMT_DXT3;
	case FMT_S3TC_DXT5:
		return D3DFMT_DXT5;
	case FMT_PVRTC_RGB_4BPP: case FMT_PVRTC_RGB_2BPP:
		return D3DFMT_X8R8G8B8;
	case FMT_PVRTC_RGBA_4BPP: case FMT_PVRTC_RGBA_2BPP:
		return D3DFMT_A8R8G8B8;
	default:
		return D3DFMT_UNKNOWN;
	}
}
/*
//HQTextureManagerD3D9 class
*/
/*
//constructor
*/
HQTextureManagerD3D9::HQTextureManagerD3D9(LPDIRECT3DDEVICE9 pDev, DWORD textureCaps , DWORD s3tc_dxtFlags,
										   DWORD maxVertexShaderSamplers ,DWORD maxPixelShaderSamplers,
										   HQLogStream *logFileStream , bool flushLog)
:HQBaseTextureManager(logFileStream , "D3D9 Texture Manager :" , flushLog)
{ 
	this->pD3DDevice=pDev;
	this->bitmap.SetLoadedOutputRGBLayout(LAYOUT_RGB);
	this->bitmap.SetLoadedOutputRGB16Layout(LAYOUT_RGB);

	this->textureCaps = textureCaps;
	this->s3tc_dxtFlags = s3tc_dxtFlags;
	unsigned int vertexShaderStageIndex = GetShaderStageIndex( HQ_VERTEX_SHADER );
	unsigned int pixelShaderStageIndex = GetShaderStageIndex( HQ_PIXEL_SHADER );

	this->shaderStage[vertexShaderStageIndex].maxSamplers = maxVertexShaderSamplers;
	this->shaderStage[vertexShaderStageIndex].samplerOffset = D3DVERTEXTEXTURESAMPLER0;
	
	this->shaderStage[pixelShaderStageIndex].maxSamplers = maxPixelShaderSamplers;
	this->shaderStage[pixelShaderStageIndex].samplerOffset = 0;

	this->shaderStage[pixelShaderStageIndex].samplerSlots = new HQSharedPtr<HQTexture> [maxPixelShaderSamplers];
	if (maxVertexShaderSamplers > 0)
		this->shaderStage[vertexShaderStageIndex].samplerSlots = new HQSharedPtr<HQTexture> [maxVertexShaderSamplers];
	else 
		this->shaderStage[vertexShaderStageIndex].samplerSlots = NULL;

	Log("Init done!");
	LogTextureCompressionSupportInfo();
}
/*
Destructor
*/
HQTextureManagerD3D9::~HQTextureManagerD3D9()
{
	Log("Released!");
}

/*-------reset device----------*/
void HQTextureManagerD3D9::OnResetDevice()
{
	/*-----reset textures-----*/
	for (hq_uint32 i = 0 ; i < 2 ; ++i)
	{
		for (hq_uint32 samplerIndex = 0 ;samplerIndex < this->shaderStage[i].maxSamplers ; ++samplerIndex)
		{
			if (this->shaderStage[i].samplerSlots[samplerIndex] != NULL)
				pD3DDevice->SetTexture(	this->shaderStage[i].samplerOffset + samplerIndex , 
										(LPDIRECT3DTEXTURE9)this->shaderStage[i].samplerSlots[samplerIndex]->pData);
		}
	
	}
}

/*-------create new texture object---------*/
HQTexture * HQTextureManagerD3D9::CreateNewTextureObject(HQTextureType type)
{
	return new HQTextureD3D9(type);
}
/*-------set texture ---------------------*/
HQReturnVal HQTextureManagerD3D9::SetTexture(hq_uint32 slot , hq_uint32 textureID)
{
	HQSharedPtr<HQTexture> pTexture = this->textures.GetItemPointer(textureID);

	hq_uint32 samplerSlot = slot & 0x0fffffff;
	hq_uint32 shaderStageIndex = GetShaderStageIndex( (HQShaderType)(slot & 0xf0000000) );

#if defined _DEBUG || defined DEBUG
	if (shaderStageIndex >= 2)
	{
		Log("Error : {slot} parameter passing to SetTexture() method didn't bitwise OR with HQ_VERTEX_SHADER/HQ_PIXEL_SHADER!");
		return HQ_FAILED;
	}
	if (samplerSlot >= this->shaderStage[shaderStageIndex].maxSamplers)
	{
		Log("SetTexture() Error : sampler slot is out of range!");
		return HQ_FAILED;
	}
#endif
	
	if (pTexture != this->shaderStage[shaderStageIndex].samplerSlots[samplerSlot] )
	{
		if (pTexture == NULL)
			pD3DDevice->SetTexture(this->shaderStage[shaderStageIndex].samplerOffset + samplerSlot , NULL);
		else
			pD3DDevice->SetTexture(this->shaderStage[shaderStageIndex].samplerOffset + samplerSlot ,  
									(LPDIRECT3DTEXTURE9) pTexture->pData);
		this->shaderStage[shaderStageIndex].samplerSlots[samplerSlot] = pTexture;
	}

	return HQ_OK;
}

HQReturnVal HQTextureManagerD3D9::SetTextureForPixelShader(hq_uint32 samplerSlot , hq_uint32 textureID)
{
	HQSharedPtr<HQTexture> pTexture = this->textures.GetItemPointer(textureID);

	hq_uint32 shaderStageIndex = GetShaderStageIndex( HQ_PIXEL_SHADER );

#if defined _DEBUG || defined DEBUG
	if (samplerSlot >= this->shaderStage[shaderStageIndex].maxSamplers)
	{
		Log("SetTextureForPixelShader() Error : sampler slot is out of range!");
		return HQ_FAILED;
	}
#endif
	
	if (pTexture != this->shaderStage[shaderStageIndex].samplerSlots[samplerSlot] )
	{
		if (pTexture == NULL)
			pD3DDevice->SetTexture(this->shaderStage[shaderStageIndex].samplerOffset + samplerSlot , NULL);
		else
			pD3DDevice->SetTexture(this->shaderStage[shaderStageIndex].samplerOffset + samplerSlot ,  
									(LPDIRECT3DTEXTURE9) pTexture->pData);
		this->shaderStage[shaderStageIndex].samplerSlots[samplerSlot] = pTexture;
	}

	return HQ_OK;
}

/*
Load texture from file
*/
HQReturnVal HQTextureManagerD3D9::LoadTextureFromFile(HQTexture * pTex)
{
	if(!pD3DDevice)
		return HQ_FAILED;


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
		if(result == IMG_FAIL_BAD_FORMAT)
		{
			switch(pTex->type)
			{
			case HQ_TEXTURE_2D:
				if(!FAILED(D3DXCreateTextureFromFileA(pD3DDevice,pTex->fileName,(LPDIRECT3DTEXTURE9*)&pTex->pData)))
				{
					Log("Loaded unmodified texture from file %s",pTex->fileName);
					return HQ_OK;
				}
				break;
			}
		}
		else if (result == IMG_FAIL_NOT_ENOUGH_CUBE_FACES)
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

	if(!g_pD3DDev->IsNpotTextureSupported(pTex->type))//kích thước texture phải là lũy thừa của 2
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

			bitmap.Set(pData,bitmap.GetWidth(),bitmap.GetHeight(),bpp, size, format,origin,complex);
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

	if (((textureCaps & D3DPTEXTURECAPS_SQUAREONLY) != 0) && w!=h)//kích thước texture phải là 1 hình vuông
	{
		Log("Now trying to resize image dimesions to square dimensions");
		unsigned long size=bitmap.GetFirstLevelSize();
		//lấy giá trị lớn nhất trong 2 giá trị width và height
		hq_uint32 maxD=max(w,h);

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
		bitmap.Set(pData,w,h,bpp,size, format,origin,complex);
		//giải nén nếu hình ảnh ở dạng nén 
		if(bitmap.IsCompressed() &&  (result = bitmap.DeCompress())!=IMG_OK)
		{
			if (result == IMG_FAIL_MEM_ALLOC)
				Log("Memory allocation failed when attempt to decompressing compressed data!");
			else
				Log("Couldn't decompress compressed data!");
			return HQ_FAILED;
		}
		//phóng to hình ảnh lên thành kích thước 1 hình vuông
		bitmap.Scalei(maxD,maxD);
		
		//chỉnh lại thông tin cơ sở
		w=maxD;
		h=maxD;
		format=bitmap.GetSurfaceFormat();
		bpp=bitmap.GetBits();
	}//if (square only)
	
	if(
		(
		pTex->type == HQ_TEXTURE_2D && 
			(
				(format ==FMT_S3TC_DXT1 && (this->s3tc_dxtFlags & D3DFMT_DXT1_SUPPORT) == 0) ||
				(format ==FMT_S3TC_DXT3 && (this->s3tc_dxtFlags & D3DFMT_DXT3_SUPPORT) == 0) ||
				(format ==FMT_S3TC_DXT5 && (this->s3tc_dxtFlags & D3DFMT_DXT5_SUPPORT) == 0)
			)
		) ||
		(
		pTex->type == HQ_TEXTURE_CUBE && 
			(
				(format ==FMT_S3TC_DXT1 && (this->s3tc_dxtFlags & D3DFMT_DXT1_CUBE_SUPPORT) == 0) ||
				(format ==FMT_S3TC_DXT3 && (this->s3tc_dxtFlags & D3DFMT_DXT3_CUBE_SUPPORT) == 0) ||
				(format ==FMT_S3TC_DXT5 && (this->s3tc_dxtFlags & D3DFMT_DXT5_CUBE_SUPPORT) == 0)
			)
		)
		)
		//không hỗ trợ dạng nén
	{
		Log("DXT compressed texture is not supported!Now trying to decompress it!");
		
		//giải nén
		if(bitmap.DeCompressDXT()==IMG_FAIL_MEM_ALLOC)
		{
			Log("Memory allocation failed when attempt to decompressing DXTn compressed data!");
			return HQ_FAILED;
		}
		//chỉnh lại thông tin cơ sở
		format=bitmap.GetSurfaceFormat();
		bpp=bitmap.GetBits();

	}//if (not support DXT)

	if (format == FMT_ETC1)
	{
		Log("ETC compressed texture is not supported!Now trying to decompress it!");
		
		//giải nén
		if(bitmap.DeCompressETC()==IMG_FAIL_MEM_ALLOC)
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

		if((textureCaps & D3DPTEXTURECAPS_MIPMAP)==0 )//không hỗ trợ mipmap
		{
			if((complex.dwComplexFlags & SURFACE_COMPLEX_MIPMAP)!=0)//trong file ảnh có nhiều hơn 1 mipmap level
			{
				Log("Can only load first mipmap level");
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
				bitmap.Set(pData,w,h,bpp,size, format,origin,complex);
			}

			nMips=1;//1 mipmap level
		}//if (not support mipmap)
		
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
	
	if(CreateTexture((pTex->alpha<1.0f || (pTex->colorKey!=NULL && pTex->nColorKey>0)),nMips,pTex)!=HQ_OK)
	{
		return HQ_FAILED;
	}

	return HQ_OK;
}

/*
Load cube texture from 6 files
*/
HQReturnVal HQTextureManagerD3D9::LoadCubeTextureFromFiles(const char *fileNames[6] , HQTexture * pTex)
{
	if(!pD3DDevice)
		return HQ_FAILED;


	//các thông tin cơ sở
	hq_uint32 w,h;//width,height
	short bpp;//bits per pixel
	SurfaceFormat format;//định dạng
	SurfaceComplexity complex;//độ phức tạp
	ImgOrigin origin;//vị trí pixel đầu 

	//load bitmap file
	if((textureCaps & D3DPTEXTURECAPS_MIPMAP)==0 && this->generateMipmap )//không hỗ trợ mipmap
		this->generateMipmap = false;

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

	if(textureCaps & D3DPTEXTURECAPS_CUBEMAP_POW2)//kích thước texture phải là lũy thừa của 2
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
	}//if (power of two)

	if((format ==FMT_S3TC_DXT1 && (this->s3tc_dxtFlags & D3DFMT_DXT1_CUBE_SUPPORT) == 0) ||
		(format ==FMT_S3TC_DXT3 && (this->s3tc_dxtFlags & D3DFMT_DXT3_CUBE_SUPPORT) == 0) ||
		(format ==FMT_S3TC_DXT5 && (this->s3tc_dxtFlags & D3DFMT_DXT5_CUBE_SUPPORT) == 0))//không hỗ trợ dạng nén
	{
		Log("DXT compressed texture is not supported!Now trying to decompress it!");
		
		//giải nén
		if(bitmap.DeCompressDXT()==IMG_FAIL_MEM_ALLOC)
		{
			Log("Memory allocation failed when attempt to decompressing DXTn compressed data!");
			return HQ_FAILED;
		}
		//chỉnh lại thông tin cơ sở
		format=bitmap.GetSurfaceFormat();
		bpp=bitmap.GetBits();

	}//if (not support DXT)
	
	if (format == FMT_ETC1)
	{
		Log("ETC compressed texture is not supported!Now trying to decompress it!");
		
		//giải nén
		
		if(bitmap.DeCompressETC()==IMG_FAIL_MEM_ALLOC)
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
		if ((complex.dwComplexFlags & SURFACE_COMPLEX_MIPMAP)!=0 && complex.nMipMap>1)//nếu trong file ảnh có chứa nhiều hơn 1 mipmap level
		{
			nMips=complex.nMipMap;
		}
	}


	if(format==FMT_R8G8B8A8 || format==FMT_B8G8R8A8)
		bitmap.FlipRGBA();//đưa alpha byte lên thành byte có trọng số lớn nhất
	
	
	if(CreateTexture((pTex->alpha<1.0f || (pTex->colorKey!=NULL && pTex->nColorKey>0)),nMips,pTex)!=HQ_OK)
	{
		return HQ_FAILED;
	}

	return HQ_OK;
}

HQReturnVal HQTextureManagerD3D9::CreateSingleColorTexture(HQTexture *pTex,HQColorui color)
{
	if(FAILED(pD3DDevice->CreateTexture(1,1,1,
		0,D3DFMT_X8R8G8B8,D3DPOOL_MANAGED,(LPDIRECT3DTEXTURE9*)&(pTex->pData),0)))
	{
		if(pTex->pData!=NULL)
		{
			((LPDIRECT3DTEXTURE9)(pTex->pData))->Release();
			pTex->pData=NULL;
		}
		Log("Failed to create new texture that has single color %u",color);
		return HQ_FAILED;
	}

	LPDIRECT3DTEXTURE9 temp=(LPDIRECT3DTEXTURE9)(pTex->pData);

	D3DLOCKED_RECT rect;

	hq_ubyte8 *pTexel;
	temp->LockRect(0,&rect,NULL,D3DLOCK_NOSYSLOCK);
	pTexel=(hq_ubyte8*) rect.pBits;

	memcpy(pTexel , &color,sizeof(color)); 
	
	temp->UnlockRect(0);
	

	return HQ_OK;
}
/*
create texture object from pixel data
*/
HQReturnVal HQTextureManagerD3D9::CreateTexture(bool changeAlpha,hq_uint32 numMipmaps,HQTexture * pTex)
{
	SurfaceComplexity complex=bitmap.GetSurfaceComplex();
	SurfaceFormat format=bitmap.GetSurfaceFormat();
	if( bitmap.IsCompressed() && complex.nMipMap < 2 && numMipmaps > 1 )//dữ liệu ảnh dạng nén và chỉ có 1 mipmap level sẵn có,trong khi ta cần nhiều hơn 1 mipmap level
	{
		if (bitmap.DeCompress() == IMG_FAIL_NOT_SUPPORTED)//nếu giải nén vẫn không được
			numMipmaps=1;//cho số mipmap level bằng 1
	}

	if(changeAlpha)//có thay đổi alpha
	{
		int result=IMG_OK;
		if(bitmap.IsCompressed())
			result = bitmap.DeCompress();
		if(result==IMG_OK)
		{
			switch(bitmap.GetSurfaceFormat())
			{
			case FMT_R5G6B5:
				result=bitmap.RGB16ToRGBA();//R5G6B5 => A8R8G8B8
				break;
			case FMT_B5G6R5:
				result=bitmap.RGB16ToRGBA(true);//B5G6R5 => A8R8G8B8
				break;
			case FMT_R8G8B8:
				result=bitmap.RGB24ToRGBA();
				break;
			case FMT_B8G8R8:
				result=bitmap.RGB24ToRGBA(true);//B8G8R8 => A8R8G8B8
				break;
			case FMT_L8:
				result=bitmap.L8ToAL16();//8 bit greyscale => 8 bit greyscale,8 bit alpha
				break;
			}
			if(result==IMG_OK)
				this->ChangePixelData(pTex);
		}

	}//if (alpha)
	switch(pTex->type)
	{
	case HQ_TEXTURE_2D:
		return this->Create2DTexture(numMipmaps , pTex);
	case HQ_TEXTURE_CUBE:
		return this->CreateCubeTexture(numMipmaps , pTex);
	}
	return HQ_FAILED;
}

HQReturnVal  HQTextureManagerD3D9::Create2DTexture(hq_uint32 numMipmaps,HQTexture * pTex)
{
	SurfaceComplexity complex=bitmap.GetSurfaceComplex();
	//định dạng của texture lấy tương ứng theo định dạng của pixel data
	D3DFORMAT textureFmt=GetTextureFmt(bitmap.GetSurfaceFormat());
	
	hq_uint32 w=bitmap.GetWidth();
	hq_uint32 h=bitmap.GetHeight();


	//non power of 2 texture restriction
	hquint32 Exp;

	if(!IsPowerOfTwo(w,&Exp) || !IsPowerOfTwo(h,&Exp))
	{
		if (!g_pD3DDev->IsNpotTextureFullySupported(HQ_TEXTURE_2D))//only 1 mipmap level is allowed
		{
			numMipmaps = 1;
			if (pTex->fileName != NULL)
				Log("warning : only 1 mipmap level is allowed to use in a non power of 2 texture from file %s",pTex->fileName);
			else
				Log("warning : only 1 mipmap level is allowed to use in a non power of 2 texture ");
		}
	}
	
	if(FAILED(pD3DDevice->CreateTexture(w,h,numMipmaps,
		0,textureFmt,D3DPOOL_MANAGED,(LPDIRECT3DTEXTURE9*)&(pTex->pData),0)))
	{
		if(pTex->pData!=NULL)
		{
			((LPDIRECT3DTEXTURE9)(pTex->pData))->Release();
		}
		//thử lại với chỉ 1 mipmap level
		numMipmaps = 1;
		if(FAILED(pD3DDevice->CreateTexture(w,h,numMipmaps,
			0,textureFmt,D3DPOOL_MANAGED,(LPDIRECT3DTEXTURE9*)&(pTex->pData),0)))
		{
			if(pTex->pData!=NULL)
			{
				((LPDIRECT3DTEXTURE9)(pTex->pData))->Release();
				pTex->pData=NULL;
			}
			if (pTex->fileName != NULL)
				Log("Failed to create new texture from file %s",pTex->fileName);
			return HQ_FAILED;
		}
	}

	LPDIRECT3DTEXTURE9 temp=(LPDIRECT3DTEXTURE9)(pTex->pData);

	D3DLOCKED_RECT rect;
	unsigned long lvlSize;//độ lớn của 1 mipmap level trong dữ liệu ảnh
	unsigned long rowSize;//độ lớn 1 hàng của 1 mipmap level trong dữ liệu ảnh
	hq_uint32 numRow;//số hàng (bằng số hàng pixel trong dạng thường và bằng số hàng block trong dạng nén DXT)

	hq_ubyte8 *pRow;//1 hàng pixel data
	hq_ubyte8 *pTexelRow;// 1 hàng texel data
	
	// declare an empty texture to decompress pvr texture into 
	CPVRTexture	sDecompressedTexture;
	if (bitmap.IsPVRTC())//decompress pvrtc
	{
		Log("PVRTC compressed texture is not supported!Now trying to decompress it!");

		PVRTRY 
		{ 
			// get the utilities instance 
			PVRTextureUtilities sPVRU = PVRTextureUtilities(); 

			/*----------original texture-----------*/
			//create pvr header
			PVR_Header header_str;
			bitmap.GetPVRHeader(header_str);

			PixelType pixelType;
			switch (bitmap.GetSurfaceFormat())
			{
			case FMT_PVRTC_RGBA_4BPP : case FMT_PVRTC_RGB_4BPP:
				pixelType = MGLPT_PVRTC4;
				break;
			case FMT_PVRTC_RGBA_2BPP : case FMT_PVRTC_RGB_2BPP:
				pixelType = MGLPT_PVRTC2;
				break;
			}

			CPVRTextureHeader header(header_str.dwWidth, header_str.dwHeight, header_str.dwMipMapCount, 
				header_str.dwNumSurfs, false, false, 
				(bitmap.GetSurfaceComplex().dwComplexFlags & SURFACE_COMPLEX_CUBE) != 0, 
				(bitmap.GetSurfaceComplex().dwComplexFlags & SURFACE_COMPLEX_VOLUME) != 0,
				false , header_str.dwAlphaBitMask != 0, 
				false,//(header_str.dwpfFlags & 0x00010000) != 0, //flipped
				pixelType, 0.0f);
				
			//texture data
			CPVRTextureData textureData(bitmap.GetPixelData(), bitmap.GetImgSize());
			bitmap.ClearData();
			//original texture
			CPVRTexture sOriginalTexture(header, textureData); ; 

			// decompress the compressed texture
			sPVRU.DecompressPVR(sOriginalTexture,sDecompressedTexture); 
			
			bitmap.Set(NULL, w, h, 32, 0, FMT_A8B8G8R8, bitmap.GetPixelOrigin(), bitmap.GetSurfaceComplex());

			pRow = sDecompressedTexture.getData().getData();
		 
		} 
		PVRCATCH(myException) 
		{ 
			// handle any exceptions here 
			Log("Error when trying to decompress PVRT compressed data: %s",myException.what()); 
			return HQ_FAILED;
		}
	}
	else
		pRow = bitmap.GetPixelData();

	for(hq_uint32 level=0;level<numMipmaps;++level)
	{
		lvlSize=bitmap.CalculateSize(w,h);
		rowSize=bitmap.CalculateRowSize(w);
		numRow=lvlSize/rowSize;

		temp->LockRect(level,&rect,NULL,D3DLOCK_NOSYSLOCK);
		pTexelRow=(hq_ubyte8*) rect.pBits;
		for(hq_uint32 row=0;row<numRow;++row)//copy từng hàng
		{
			switch (bitmap.GetSurfaceFormat())
			{
			case FMT_R8G8B8: 
				for(hq_uint32 i=0;i<w; ++i)//từng pixel trên 1 hàng
				{
					hq_ubyte8 *pPixel=pRow + i*3;//pixel data
					hq_ubyte8 *pTexel=pTexelRow + i*4;//texel data
					memcpy(pTexel,pPixel,3);//red green blue
					pTexel[3]=0xff;//alpha=1.0f
				}
				break;
			case FMT_B8G8R8:
				for(hq_uint32 i=0;i<w; ++i)//từng pixel trên 1 hàng
				{
					hq_ubyte8 *pPixel=pRow + i*3;//pixel data
					hq_ubyte8 *pTexel=pTexelRow + i*4;//texel data
					
					pTexel[0] = pPixel[2];//blue
					pTexel[1] = pPixel[1];//green
					pTexel[2] = pPixel[0];//red
					pTexel[3]=0xff;//alpha=1.0f
				}
				break;
			case FMT_X8B8G8R8:
				for(hq_uint32 i=0;i<w; ++i)//từng pixel trên 1 hàng
				{
					hq_ubyte8 *pPixel=pRow + i*4;//pixel data
					hq_ubyte8 *pTexel=pTexelRow + i*4;//texel data
					
					pTexel[0] = pPixel[2];//blue
					pTexel[1] = pPixel[1];//green
					pTexel[2] = pPixel[0];//red
					pTexel[3]=0xff;//alpha=1.0f
				}
				break;
			case FMT_A8B8G8R8:
				for(hq_uint32 i=0;i<w; ++i)//từng pixel trên 1 hàng
				{
					hq_ubyte8 *pPixel=pRow + i*4;//pixel data
					hq_ubyte8 *pTexel=pTexelRow + i*4;//texel data
					
					pTexel[0] = pPixel[2];//blue
					pTexel[1] = pPixel[1];//green
					pTexel[2] = pPixel[0];//red
					pTexel[3] = pPixel[3];//alpha
				}
				break;
			case FMT_B5G6R5:
				for(hq_uint32 i=0;i<w; ++i)//từng pixel trên 1 hàng
				{
					hq_ubyte8 *pPixel=pRow + i*2;//pixel data
					hq_ubyte8 *pTexel=pTexelRow + i*2;//texel data
					
					*((hqushort16*) pTexel) = SwapRGB16 (*((hqushort16*) pPixel));
					
				}
				break;
			default:
				memcpy(pTexelRow,pRow,rowSize);
			}

			pTexelRow += rect.Pitch;//next texel row
			pRow += rowSize;//next pixel data row
		}
		temp->UnlockRect(level);

		if(w > 1) w >>=1; //w/=2
		if(h > 1) h >>=1; //h/=2


		if(complex.nMipMap < 2 && numMipmaps > 1) //nếu trong dữ liệu ảnh chỉ có sẵn 1 mipmap level,tự tạo dữ liệu ảnh ở level thấp hơn bằng cách phóng nhỏ hình ảnh
		{
			bitmap.Scalei(w,h);
			pRow=bitmap.GetPixelData();
		}
	}//for (level)
	

	return HQ_OK;
}

HQReturnVal  HQTextureManagerD3D9::CreateCubeTexture(hq_uint32 numMipmaps,HQTexture * pTex)
{
	SurfaceComplexity complex=bitmap.GetSurfaceComplex();
	if((complex.dwComplexFlags & SURFACE_COMPLEX_CUBE) != 0 && 
		complex.nMipMap < 2 && numMipmaps > 1 )//dữ liệu ảnh dạng cube map và chỉ có 1 mipmap level sẵn có,trong khi ta cần nhiều hơn 1 mipmap level
		numMipmaps = 1;

	//định dạng của texture lấy tương ứng theo định dạng của pixel data
	D3DFORMAT textureFmt=GetTextureFmt(bitmap.GetSurfaceFormat());
	
	hq_uint32 Width=bitmap.GetWidth();

	//non power of 2 texture restriction
	bool onlyFistLevel = false;
	hquint32 Exp;

	if(!IsPowerOfTwo(Width,&Exp))
	{
		if (!g_pD3DDev->IsNpotTextureFullySupported(HQ_TEXTURE_CUBE))//only 1 mipmap level is allowed
		{
			onlyFistLevel = true;
			if (pTex->fileName != NULL)
				Log("warning : only 1 mipmap level is allowed to use in a non power of 2 texture from file %s",pTex->fileName);
			else
				Log("warning : only 1 mipmap level is allowed to use in a non power of 2 texture ");
		}
	}

	if(FAILED(pD3DDevice->CreateCubeTexture(Width,onlyFistLevel? 1: numMipmaps,
		0,textureFmt,D3DPOOL_MANAGED,(LPDIRECT3DCUBETEXTURE9*)&(pTex->pData),0)))
	{
		if(pTex->pData!=NULL)
		{
			((LPDIRECT3DCUBETEXTURE9)(pTex->pData))->Release();
			pTex->pData=NULL;
		}
		if (pTex->fileName != NULL)
			Log("Failed to create new texture from file %s",pTex->fileName);
		return HQ_FAILED;
	}

	LPDIRECT3DCUBETEXTURE9 temp=(LPDIRECT3DCUBETEXTURE9)(pTex->pData);

	D3DLOCKED_RECT rect;
	unsigned long lvlSize;//độ lớn của 1 mipmap level trong dữ liệu ảnh
	unsigned long rowSize;//độ lớn 1 hàng của 1 mipmap level trong dữ liệu ảnh
	hq_uint32 numRow;//số hàng (bằng số hàng pixel trong dạng thường và bằng số hàng block trong dạng nén DXT)

	hq_ubyte8 *pRow;//1 hàng pixel data
	hq_ubyte8 *pTexelRow;// 1 hàng texel data
	// declare an empty texture to decompress pvr texture into 
	CPVRTexture	sDecompressedTexture;
	if (bitmap.IsPVRTC())//decompress pvrtc
	{
		Log("PVRTC compressed texture is not supported!Now trying to decompress it!");

		PVRTRY 
		{ 
			// get the utilities instance 
			PVRTextureUtilities sPVRU = PVRTextureUtilities(); 

			/*----------original texture-----------*/
			//create pvr header
			PVR_Header header_str;
			bitmap.GetPVRHeader(header_str);

			PixelType pixelType;
			switch (bitmap.GetSurfaceFormat())
			{
			case FMT_PVRTC_RGBA_4BPP : case FMT_PVRTC_RGB_4BPP:
				pixelType = MGLPT_PVRTC4;
				break;
			case FMT_PVRTC_RGBA_2BPP : case FMT_PVRTC_RGB_2BPP:
				pixelType = MGLPT_PVRTC2;
				break;
			}

			CPVRTextureHeader header(header_str.dwWidth, header_str.dwHeight, header_str.dwMipMapCount, 
				header_str.dwNumSurfs, false, false, 
				(bitmap.GetSurfaceComplex().dwComplexFlags & SURFACE_COMPLEX_CUBE) != 0, 
				(bitmap.GetSurfaceComplex().dwComplexFlags & SURFACE_COMPLEX_VOLUME) != 0,
				false , header_str.dwAlphaBitMask != 0, 
				false,//(header_str.dwpfFlags & 0x00010000) != 0, //flipped
				pixelType, 0.0f);
				
			//texture data
			CPVRTextureData textureData(bitmap.GetPixelData(), bitmap.GetImgSize());
			bitmap.ClearData();
			//original texture
			CPVRTexture sOriginalTexture(header, textureData); ; 

			// decompress the compressed texture
			sPVRU.DecompressPVR(sOriginalTexture,sDecompressedTexture); 
			
			bitmap.Set(NULL, Width, Width, 32, 0, FMT_A8B8G8R8, bitmap.GetPixelOrigin(), bitmap.GetSurfaceComplex());

			pRow = sDecompressedTexture.getData().getData();
		 
		} 
		PVRCATCH(myException) 
		{ 
			// handle any exceptions here 
			Log("Error when trying to decompress PVRT compressed data: %s",myException.what()); 
			return HQ_FAILED;
		}
	}
	else
		pRow = bitmap.GetPixelData();

	bool loadTextureData;

	for(hq_uint32 face = 0 ; face < 6 ; ++face)
	{
		hq_uint32 w = Width;
		for(hq_uint32 level=0;level<numMipmaps;++level)
		{
			loadTextureData = level == 0 || !onlyFistLevel;//do we really need to load this texture level

			lvlSize=bitmap.CalculateSize(w,w);
			rowSize=bitmap.CalculateRowSize(w);
			numRow=lvlSize/rowSize;

			if (loadTextureData)
			{
				temp->LockRect((D3DCUBEMAP_FACES) face , level,&rect,NULL,D3DLOCK_NOSYSLOCK);
				pTexelRow=(hq_ubyte8*) rect.pBits;
			}
			for(hq_uint32 row=0;row<numRow;++row)//copy từng hàng
			{
				if (loadTextureData)
				{
					switch (bitmap.GetSurfaceFormat())
					{
					case FMT_R8G8B8: 
						for(hq_uint32 i=0;i<w; ++i)//từng pixel trên 1 hàng
						{
							hq_ubyte8 *pPixel=pRow + i*3;//pixel data
							hq_ubyte8 *pTexel=pTexelRow + i*4;//texel data
							memcpy(pTexel,pPixel,3);//red green blue
							pTexel[3]=0xff;//alpha=1.0f
						}
						break;
					case FMT_B8G8R8:
						for(hq_uint32 i=0;i<w; ++i)//từng pixel trên 1 hàng
						{
							hq_ubyte8 *pPixel=pRow + i*3;//pixel data
							hq_ubyte8 *pTexel=pTexelRow + i*4;//texel data
							
							pTexel[0] = pPixel[2];//blue
							pTexel[1] = pPixel[1];//green
							pTexel[2] = pPixel[0];//red
							pTexel[3]=0xff;//alpha=1.0f
						}
						break;
					case FMT_X8B8G8R8:
						for(hq_uint32 i=0;i<w; ++i)//từng pixel trên 1 hàng
						{
							hq_ubyte8 *pPixel=pRow + i*4;//pixel data
							hq_ubyte8 *pTexel=pTexelRow + i*4;//texel data
							
							pTexel[0] = pPixel[2];//blue
							pTexel[1] = pPixel[1];//green
							pTexel[2] = pPixel[0];//red
							pTexel[3]=0xff;//alpha=1.0f
						}
						break;
					case FMT_A8B8G8R8:
						for(hq_uint32 i=0;i<w; ++i)//từng pixel trên 1 hàng
						{
							hq_ubyte8 *pPixel=pRow + i*4;//pixel data
							hq_ubyte8 *pTexel=pTexelRow + i*4;//texel data
							
							pTexel[0] = pPixel[2];//blue
							pTexel[1] = pPixel[1];//green
							pTexel[2] = pPixel[0];//red
							pTexel[3] = pPixel[3];//alpha
						}
						break;
					case FMT_B5G6R5:
						for(hq_uint32 i=0;i<w; ++i)//từng pixel trên 1 hàng
						{
							hq_ubyte8 *pPixel=pRow + i*2;//pixel data
							hq_ubyte8 *pTexel=pTexelRow + i*2;//texel data
							
							*((hqushort16*) pTexel) = SwapRGB16 (*((hqushort16*) pPixel));
							
						}
						break;
					default:
						memcpy(pTexelRow,pRow,rowSize);
					}

					pTexelRow += rect.Pitch;//next texel row
				}//if (loadTextureData)
				
				pRow += rowSize;//next pixel data row
			}
			if (loadTextureData)
				temp->UnlockRect((D3DCUBEMAP_FACES) face , level);

			if(w > 1) w >>=1; //w/=2

		}//for (level)
	}//for (face)	

	return HQ_OK;
}



/*
set alphakey
*/
HQReturnVal HQTextureManagerD3D9::SetAlphaValue(hq_ubyte8 R,hq_ubyte8 G,hq_ubyte8 B,hq_ubyte8 A)
{
	hq_ubyte8* pData=bitmap.GetPixelData();
	SurfaceFormat format=bitmap.GetSurfaceFormat();
	D3DCOLOR color;

	if(pData==NULL)
		return HQ_FAILED;

	switch(format)
	{
	case FMT_A8R8G8B8:
		color=D3DCOLOR_ARGB(A,R,G,B);
		break;
	case FMT_A8B8G8R8:
		color=D3DCOLOR_ARGB(A,B,G,R);
		break;
	case FMT_A8L8:
		color=D3DCOLOR_ARGB(A,R,0,0);
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
				*((hq_ushort16*)pCur)=(hq_ushort16)(color >> 16);//(A,R,0,0) => (A,L)
		}
		else{
			D3DCOLOR * pPixel=(D3DCOLOR*)pCur;
			if(((*pPixel) & 0xffffff )== (color & 0xffffff))//cùng giá trị RGB
				*pPixel=color;
		}
	}

	return HQ_OK;
}

/*
set transparency
*/
HQReturnVal HQTextureManagerD3D9::SetTransparency(hq_float32 alpha)
{
	hq_ubyte8* pData=bitmap.GetPixelData();
	SurfaceFormat format=bitmap.GetSurfaceFormat();

	if(pData==NULL)
		return HQ_FAILED;

	hq_ubyte8 *pCur=pData;
	hq_ubyte8 *pEnd=pData + bitmap.GetImgSize();

	hq_ushort16 pixelSize=bitmap.GetBits()/8;
	hq_ubyte8 A=(hq_ubyte8)(((hq_uint32)(alpha*255)) & 0xff);//alpha range 0->255
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


HQReturnVal HQTextureManagerD3D9::GetTexture2DSize(hq_uint32 textureID, hquint32 &width, hquint32& height)
{
	HQSharedPtr<HQTexture> pTexture = this->textures.GetItemPointer(textureID);
	if (pTexture == NULL)
		return HQ_FAILED_INVALID_ID;

	switch (pTexture->type)
	{
	case HQ_TEXTURE_2D:
		{
			IDirect3DTexture9 *textureD3D = (IDirect3DTexture9 *)pTexture->pData;
			D3DSURFACE_DESC desc;
			textureD3D->GetLevelDesc(0, &desc);
			width = desc.Width;
			height = desc.Height;
		}
		break;
	case HQ_TEXTURE_CUBE:
		{
			IDirect3DCubeTexture9 *textureD3D = (IDirect3DCubeTexture9 *)pTexture->pData;
			D3DSURFACE_DESC desc;
			textureD3D->GetLevelDesc(0, &desc);
			width = desc.Width;
			height = desc.Height;
		}
		break;
	default:
		return HQ_FAILED;
	}
	return HQ_OK;
}

HQTextureCompressionSupport HQTextureManagerD3D9::IsCompressionSupported(HQTextureType textureType, HQTextureCompressionFormat type)
{
	switch (type)
	{
	case HQ_TC_S3TC_DTX1:
		switch (textureType)
		{
		case HQ_TEXTURE_2D:
			if (this->s3tc_dxtFlags & D3DFMT_DXT1_SUPPORT)
				return HQ_TCS_ALL;
		case HQ_TEXTURE_CUBE:
			if (this->s3tc_dxtFlags & D3DFMT_DXT1_CUBE_SUPPORT)
				return HQ_TCS_ALL;
		default:
			return HQ_TCS_SW;
		}
	case HQ_TC_S3TC_DXT3:
		switch (textureType)
		{
		case HQ_TEXTURE_2D:
			if (this->s3tc_dxtFlags & D3DFMT_DXT3_SUPPORT)
				return HQ_TCS_ALL;
		case HQ_TEXTURE_CUBE:
			if (this->s3tc_dxtFlags & D3DFMT_DXT3_CUBE_SUPPORT)
				return HQ_TCS_ALL;
		default:
			return HQ_TCS_SW;
		}
	case HQ_TC_S3TC_DXT5:
		switch (textureType)
		{
		case HQ_TEXTURE_2D:
			if (this->s3tc_dxtFlags & D3DFMT_DXT5_SUPPORT)
				return HQ_TCS_ALL;
		case HQ_TEXTURE_CUBE:
			if (this->s3tc_dxtFlags & D3DFMT_DXT5_CUBE_SUPPORT)
				return HQ_TCS_ALL;
		default:
			return HQ_TCS_SW;
		}
	case HQ_TC_ETC1: 
	case HQ_TC_PVRTC_RGB_2BPP :
	case HQ_TC_PVRTC_RGB_4BPP: 
	case HQ_TC_PVRTC_RGBA_2BPP:
	case HQ_TC_PVRTC_RGBA_4BPP:
		return HQ_TCS_SW;
	default:
		return HQ_TCS_NONE;
	}
}




HQBaseRawPixelBuffer* HQTextureManagerD3D9::CreatePixelBufferImpl(HQRawPixelFormat intendedFormat, hquint32 width, hquint32 height)
{
	switch (intendedFormat)
	{
	case HQ_RPFMT_R8G8B8A8:
	case HQ_RPFMT_B8G8R8A8:
		return HQ_NEW HQBaseRawPixelBuffer(HQ_RPFMT_B8G8R8A8, width, height);
		break;
	default:
		return HQ_NEW HQBaseRawPixelBuffer(intendedFormat, width, height);
	}
}

HQReturnVal HQTextureManagerD3D9::CreateTexture(HQTexture *pTex, const HQBaseRawPixelBuffer* color)
{

	color->MakeWrapperBitmap(bitmap);


	int result;//bimap object operation result

	hquint32 w=bitmap.GetWidth();
	hquint32 h=bitmap.GetHeight();
	hquint32 bpp=bitmap.GetBits();
	SurfaceFormat format=bitmap.GetSurfaceFormat();
	ImgOrigin origin=bitmap.GetPixelOrigin();
	SurfaceComplexity complex=bitmap.GetSurfaceComplex();


	if(!g_pD3DDev->IsNpotTextureSupported(pTex->type))//kích thước texture phải là lũy thừa của 2
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

			bitmap.Set(pData,bitmap.GetWidth(),bitmap.GetHeight(),bpp, size, format,origin,complex);
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

	if (((textureCaps & D3DPTEXTURECAPS_SQUAREONLY) != 0) && w!=h)//kích thước texture phải là 1 hình vuông
	{
		Log("Now trying to resize image dimesions to square dimensions");
		unsigned long size=bitmap.GetFirstLevelSize();
		//lấy giá trị lớn nhất trong 2 giá trị width và height
		hq_uint32 maxD=max(w,h);

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
		bitmap.Set(pData,w,h,bpp,size, format,origin,complex);
		//giải nén nếu hình ảnh ở dạng nén 
		if(bitmap.IsCompressed() &&  (result = bitmap.DeCompress())!=IMG_OK)
		{
			if (result == IMG_FAIL_MEM_ALLOC)
				Log("Memory allocation failed when attempt to decompressing compressed data!");
			else
				Log("Couldn't decompress compressed data!");
			return HQ_FAILED;
		}
		//phóng to hình ảnh lên thành kích thước 1 hình vuông
		bitmap.Scalei(maxD,maxD);
		
		//chỉnh lại thông tin cơ sở
		w=maxD;
		h=maxD;
		format=bitmap.GetSurfaceFormat();
		bpp=bitmap.GetBits();
	}//if (square only)

	hq_uint32 nMips = 1;//số lượng mipmap level

	if(generateMipmap)
	{
		//full range mipmap
		if(IsPowerOfTwo(max(w,h),&nMips))//nMips=1+floor(log2(max(w,h)))
			nMips++;

		if((textureCaps & D3DPTEXTURECAPS_MIPMAP)==0 )//không hỗ trợ mipmap
		{
			if((complex.dwComplexFlags & SURFACE_COMPLEX_MIPMAP)!=0)//trong file ảnh có nhiều hơn 1 mipmap level
			{
				Log("Can only load first mipmap level");
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
				bitmap.Set(pData,w,h,bpp,size, format,origin,complex);
			}

			nMips=1;//1 mipmap level
		}//if (not support mipmap)
		
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