#ifndef _HQ_BASE_IMPL_TEX_MAN_1_H_
#define _HQ_BASE_IMPL_TEX_MAN_1_H_
#include "../HQTextureManager.h"
#include "../HQLoggableObject.h"
#include "HQBaseImplCommon.h"
#include "../HQItemManager.h"


#include "../ImagesLoader/Bitmap.h"
#if defined WIN32 && !(defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#	if defined(_DEBUG)||defined (DEBUG)
#		ifdef _STATIC_CRT
#			pragma comment(lib,"../../VS/Output/Debug static CRT/ImagesLoaderD.lib")
#		else
#			pragma comment(lib,"../../VS/Output/Debug/ImagesLoaderD.lib")
#		endif
#	elif defined (_NOOPTIMIZE)
#		pragma comment(lib,"../../VS/Output/release noptimize/ImagesLoader.lib")
#	elif defined (_STATIC_CRT)
#		pragma comment(lib,"../../VS/Output/Release static CRT/ImagesLoader.lib")
#	else
#		pragma comment(lib,"../../VS/Output/Release/ImagesLoader.lib")
#	endif
#endif

#ifndef max
#define max(a,b) ((a>b) ? a : b)
#endif

///
///base pixel buffer class
///the origin of the pixel buffer is top left
///
class HQBaseRawPixelBuffer: public HQRawPixelBuffer
{
public:
	HQBaseRawPixelBuffer(HQRawPixelFormat format, hquint32 width, hquint32 height);
	~HQBaseRawPixelBuffer();

	virtual hquint32 GetWidth() const {return m_width;}
	virtual hquint32 GetHeight() const {return m_height;}
	///
	///RGB is ignored in A8 pixel buffer. GB is ignored in L8A8 buffer (R is set to luminance value). 
	///A is ignored in R5G6B5 buffer.
	///
	virtual void SetPixelf(hquint32 x, hquint32 y, float r, float g, float b, float a);
	///
	///RGB is ignored in A8 pixel buffer. GB is ignored in L8A8 buffer (R is set to luminance value). 
	///A is ignored in R5G6B5 buffer.
	///
	virtual void SetPixel(hquint32 x, hquint32 y, hqubyte8 r, hqubyte8 g, hqubyte8 b, hqubyte8 a);

	virtual HQColor GetPixelData(int x, int y) const;

	virtual const void* GetPixelData() const {return m_pData;}
	virtual HQRawPixelFormat GetFormat() const {return m_format;}

	hquint32 GetPixelSize() const {return m_pixelSize;}
	hquint32 GetBitsPerPixel() const {return GetPixelSize() * 8;}
	hquint32 GetBufferSize() const {return m_rowSize * m_height;}

	virtual void MakeWrapperBitmap( Bitmap &bitmap) const;//default implementation makes the bitmap which has origin at top left
protected:
	virtual hqubyte8* GetPixel(hquint32 x, hquint32 y);//default implementation: the first byte is pixel (0, 0)
	virtual const hqubyte8* GetPixel(hquint32 x, hquint32 y) const;//default implementation: the first pixel is pixel (0, 0)
	virtual ImgOrigin GetFirstPixelPosition() const;//default implemetation: the first byte is top left

	hqubyte8 *m_pData;
	hquint32 m_width;
	hquint32 m_height;
	HQRawPixelFormat m_format;
	hquint32 m_pixelSize;
	hquint32 m_rowSize;
};

/*----------base texture class---------------*/
typedef struct HQBaseTexture
{
	HQBaseTexture()
	{
		nColorKey = 0;
		colorKey = NULL;
		fileName = NULL;
		alpha = 1.0f;
		pData = NULL;
	}
	virtual ~HQBaseTexture()
	{
		if(colorKey != NULL)
		delete[] colorKey;
		if(fileName!=NULL)
		{
			delete[] fileName;
		}
	}

	HQTextureType type;

	void *pData;//con trỏ đến đối tượng texture của direct3D hoặc openGL
	hq_float32 alpha;//overall/max alpha (range 0.0f->1.0f)
	char * fileName;//tên file ảnh dùng để load texture.Nếu ko phải load từ file ảnh , giá trị này = NULL
	HQColor* colorKey;//colorkey chứa các giá trị pixel đã thay đổi giá trị alpha so với giá trị ban đầu load từ file ảnh
	hq_uint32 nColorKey;//số color key
} HQTexture , _HQTexture;

/*--------base texture manager class---------*/

class HQBaseTextureManager: public HQTextureManager , public HQLoggableObject
{
public:
	HQBaseTextureManager(HQLogStream* logFileStream ,const char *logPrefix , bool flushLog)
		: HQLoggableObject(logFileStream , logPrefix , flushLog)
	{
	}

	HQReturnVal RemoveTexture(hq_uint32 ID);
	void RemoveAllTexture();

	HQReturnVal AddTexture(const char* fileName,
						   hq_float32 maxAlpha,
						   const HQColor *colorKey,
						   hq_uint32 numColorKey,
						   bool generateMipmap,
						   HQTextureType textureType,
						   hq_uint32 *pTextureID);
	HQReturnVal AddCubeTexture(const char * fileNames[6] ,
							   hq_float32 maxAlpha,
							   const HQColor *colorKey,
							   hq_uint32 numColorKey,
							   bool generateMipmap,
							   hq_uint32 *pTextureID);

	//tạo texture mà chỉ chứa 1 màu
	HQReturnVal AddSingleColorTexture(HQColorui color , hq_uint32 *pTextureID);

#ifndef GLES
	HQReturnVal AddTextureBuffer(HQTextureBufferFormat format , hq_uint32 size , void *initData , bool isDynamic ,hq_uint32 *pTextureID);

	HQReturnVal MapTextureBuffer(hq_uint32 textureID , void **ppData) { return HQ_FAILED ;}
	HQReturnVal UnmapTextureBuffer(hq_uint32 textureID) { return HQ_FAILED ;}
#endif
	const HQSharedPtr<HQTexture> GetTexture(hq_uint32 ID);
	HQSharedPtr<HQTexture> CreateEmptyTexture(HQTextureType textureType , hq_uint32 *pTextureID);


	HQRawPixelBuffer* CreatePixelBuffer(HQRawPixelFormat intendedFormat, hquint32 width, hquint32 height);

	HQReturnVal AddTexture(const HQRawPixelBuffer* color , bool generateMipmap, hq_uint32 *pTextureID);


	static bool HasFace(SurfaceComplexity complex , int face);
	//*************************************************************************************************************************
	//kiểm tra số nguyên có phải là 1 lũy thừa cũa 2 không,lưu số mũ của lũy thừa của 2 nhỏ nhất kế tiếp số nguyên cần kiểm tra
	//nếu số nguyên không là lũy thừa của 2, hoặc là log2 của số nguyên đó nếu nó là lũy thừa của 2, vào biến mà pExponent trỏ đến
	//*************************************************************************************************************************
	static bool IsPowerOfTwo(const hq_uint32 d,hq_uint32 *pExponent);
	//tính số mipmap level tối đa
	static hq_uint32 CalculateFullNumMipmaps(hq_uint32 width, hq_uint32 height);
protected:
	~HQBaseTextureManager();

	void LogTextureCompressionSupportInfo();

	virtual void ChangePixelData( HQTexture *pTex );
	virtual HQReturnVal SetAlphaValue(hq_ubyte8 R,hq_ubyte8 G,hq_ubyte8 B,hq_ubyte8 A) = 0;//set giá trị alpha của pixel ảnh vừa load có giá trị RGB như tham số(hoặc R nến định dạng texture chỉ có kênh 8 bit greyscale) thành giá trị <A>.
	virtual HQReturnVal SetTransparency(hq_float32 alpha) = 0;//set giá trị alpha lớn nhất của toàn bộ pixel ảnh vừa load thành <alpha>


	/*--------implement dependent----------*/
	virtual HQTexture * CreateNewTextureObject(HQTextureType type) = 0;

	virtual HQReturnVal LoadTextureFromFile(HQTexture * pTex) = 0;
	virtual HQReturnVal LoadCubeTextureFromFiles(const char *fileNames[6] , HQTexture * pTex) = 0;
	virtual HQReturnVal CreateSingleColorTexture(HQTexture *pTex,HQColorui color) = 0;
#ifndef GLES
	virtual HQReturnVal CreateTextureBuffer(HQTexture *pTex ,HQTextureBufferFormat format , hq_uint32 size  , void *initData,bool isDynamic) { return HQ_FAILED ;}
#endif
	virtual HQBaseRawPixelBuffer* CreatePixelBufferImpl(HQRawPixelFormat intendedFormat, hquint32 width, hquint32 height) = 0;
	virtual HQReturnVal CreateTexture(HQTexture *pTex, const HQBaseRawPixelBuffer* color) = 0;
	/*--------attributes-------------------*/

	HQItemManager<HQTexture> textures;//danh sách texture

	Bitmap bitmap;//dùng để load file ảnh
	bool generateMipmap;
};


#endif
