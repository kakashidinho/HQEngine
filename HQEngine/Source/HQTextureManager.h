/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _HQ_TEXTURE_MANAGER_H_
#define _HQ_TEXTURE_MANAGER_H_

#include "HQRendererCoreType.h"
#include "HQRendererPlatformDef.h"
#include "HQReferenceCountObj.h"
#include "HQReturnVal.h"


///
///the pixel at (0,0) is top left
///
class HQRawPixelBuffer: public HQReferenceCountObj
{
public:
	virtual hquint32 GetWidth() const = 0;
	virtual hquint32 GetHeight() const = 0;
	///
	///RGB is ignored in A8 pixel buffer. GB is ignored in L8A8 buffer (R is set to luminance value). 
	///A is ignored in R5G6B5 buffer. Color channel range is 0.0f-1.0f
	///
	virtual void SetPixelf(hquint32 x, hquint32 y, float r, float g, float b, float a) = 0;
	///
	///RGB is ignored in A8 pixel buffer. GB is ignored in L8A8 buffer (R is set to luminance value). 
	///A is ignored in R5G6B5 buffer. Color channel range is 0-255
	///
	virtual void SetPixel(hquint32 x, hquint32 y, hqubyte8 r, hqubyte8 g, hqubyte8 b, hqubyte8 a) = 0;

	virtual void SetPixelf(hquint32 x, hquint32 y, float r, float g, float b)///alpha is assumed to be 1.0f. Same as SetPixel(x, y, r, g, b, 1)
	{
		SetPixelf(x, y, r, g, b, 1.0f);
	}
	virtual void SetPixel(hquint32 x, hquint32 y, hqubyte8 r, hqubyte8 g, hqubyte8 b)///alpha is assumed to be 255. Same as SetPixel(x, y, r, g, b, 1)
	{
		SetPixel(x, y, r, g, b, 255);
	}
	virtual void SetPixelf(hquint32 x, hquint32 y, float l, float a)///luminance and alpha. Same as SetPixel(x, y, l, l, l, a)
	{
		SetPixelf(x, y, l, l, l, a);
	}
	virtual void SetPixel(hquint32 x, hquint32 y, hqubyte8 l, hqubyte8 a)///luminance and alpha. Same as SetPixel(x, y, l, l, l, a)
	{
		SetPixel(x, y, l, l, l, a);
	}
	virtual void SetPixelf(hquint32 x, hquint32 y, float a)///RGB is assumed to be black. Same as SetPixel(x, y, 0, 0, 0, a)
	{
		SetPixelf(x, y, 0.0f, 0.0f, 0.0f, a);
	}
	virtual void SetPixel(hquint32 x, hquint32 y, hqubyte8 a)///RGB is assumed to be black. Same as SetPixel(x, y, 0, 0, 0, a)
	{
		SetPixel(x, y, 0, 0, 0, a);
	}

	virtual HQColor GetPixelData(int x, int y) const = 0; 
	virtual HQRawPixelFormat GetFormat() const = 0;
protected:
	HQRawPixelBuffer(): HQReferenceCountObj() {}
	virtual ~HQRawPixelBuffer() {}
};

///
///Texture manager
///

class HQTextureManager{
protected:
	virtual ~HQTextureManager(){};
public:
	HQTextureManager(){};
	
	///
	///Direct3d : {slot} = {texture slot} bitwise OR với enum HQShaderType để chỉ  {texture slot} thuộc shader stage nào. 
	///			Ví dụ muốn gắn texture vào texture slot 3 của vertex shader , ta truyền tham số {slot} = (3 | HQ_VERTEX_SHADER). 
	///			Trong direct3d 9 {texture slot} là slot của sampler unit .Shader model 3.0 : pixel shader có 16 sampler. vertex shader có 4 sampler. Các model khác truy vấn bằng method GetMaxShaderStageSamplers() của render device. 
	///			Trong direct3d 10/11 : {texture slot} là slot shader resource view.Mỗi shader stage có 128 slot.  
	///OpenGL : {slot} là slot của sampler unit.{slot} nằm trong khoảng từ 0 đến số trả về từ method GetMaxShaderSamplers() của render device trừ đi 1.Các texture khác loại (ví dụ cube & 2d texture) có thể gắn cùng vào 1 slot.  
	///Lưu ý : -pixel shader dùng trung sampler unit với fixed function.Với openGL , các slot đầu tiên trùng với các slot của fixed function sampler	
	///
	virtual HQReturnVal SetTexture(hq_uint32 slot , hq_uint32 textureID) = 0;
	
	///
	///Direct3d : Trong direct3d 9 {slot} là slot của sampler unit trong pixel shader.Shader model 3.0 : pixel shader có 16 sampler. Các model khác truy vấn bằng method GetMaxShaderStageSamplers(HQ_PIXEL_SHADER) của render device. 
	///			Trong direct3d 10/11 : {slot} là slot pixel shader resource view.Mỗi shader stage có 128 slot.  
	///OpenGL : {slot} là slot của sampler unit.{slot} nằm trong khoảng từ 0 đến số trả về từ method GetMaxShaderSamplers() của render device trừ đi 1.Các texture khác loại (ví dụ cube & 2d texture) có thể gắn cùng vào 1 slot.  
	///Lưu ý : -pixel shader dùng trung sampler unit với fixed function.Với openGL , các slot đầu tiên trùng với các slot của fixed function sampler	
	///
	virtual HQReturnVal SetTextureForPixelShader(hq_uint32 slot , hq_uint32 textureID) = 0;
	
	///
	///2 biến maxAlpha và colorKey sẽ dùng để chuyển tất cả giá trị alpha của texel trong texture này thành (< hay = maxAlpha) 
	///và các texel có giá trị rgb trùng các giá trị rgb trong danh sách colorKey sẽ chuyển giá trị alpha thành alpha tương ứng trong colorKey. 
	///Lưu ý :	-nếu textureType = HQ_TEXTURE_CUBE và file ảnh không chứa đủ 6 mặt của cube map 
	///			thì method này sẽ thất bại, return HQ_FAILED_NOT_ENOUGH_CUBE_FACES
	///
	virtual HQReturnVal AddTexture(const char* fileName,
						   hq_float32 maxAlpha,
						   const HQColor *colorKey,
						   hq_uint32 numColorKey,
						   bool generateMipmap,
						   HQTextureType textureType,
						   hq_uint32 *pTextureID) = 0;

	///
	///tạo cube texture từ 6 file ảnh.Các file ảnh theo thứ tự sẽ dùng tạo các mặt:
	///-positive X,
	///-negative X,
	///-positive Y,
	///-negative Y,
	///-positive Z,
	///-negative Z,
	///Lưu ý :
	///-Mỗi file ảnh ko dc phép có sẵn hơn 1 mipmap level , và ko dc phép chứa sẳn các mặt của cube map.
	///-Các file ảnh phải cùng pixel format.
	///
	virtual HQReturnVal AddCubeTexture(const char * fileNames[6] , 
							   hq_float32 maxAlpha,
							   const HQColor *colorKey,
							   hq_uint32 numColorKey,
							   bool generateMipmap,
							   hq_uint32 *pTextureID) = 0;

	///
	///add thêm 1 texture mà nó chỉ chứa 1 màu {color}
	///
	virtual HQReturnVal AddSingleColorTexture(HQColorui color , hq_uint32 *pTextureID)=0;
#ifndef GLES	
	///
	///tạo texture buffer
	///
	virtual HQReturnVal AddTextureBuffer(HQTextureBufferFormat format , hq_uint32 size , void *initData , bool isDynamic , hq_uint32 *pTextureID)=0;
	
	///
	///Direct3d 10 / 11 : chỉ có thể dùng trên dynamic buffer 
	///
	virtual HQReturnVal MapTextureBuffer(hq_uint32 textureID , void **ppData) = 0;
	///
	///Direct3d 10 / 11 : chỉ có thể dùng trên dynamic buffer
	///
	virtual HQReturnVal UnmapTextureBuffer(hq_uint32 textureID) = 0;
	
#endif
	
	
	
	///
	///Truy vấn kích thước ở level 0 của texture. 
	///return HQ_FAILED nếu texture không phải 2d/cube
	///
	virtual HQReturnVal GetTexture2DSize(hq_uint32 textureID, hquint32 &width, hquint32& height) = 0;
	
	virtual HQTextureCompressionSupport IsCompressionSupported(HQTextureType textureType, HQTextureCompressionFormat compressionType) = 0;
	
	///
	///nếu texture là render target nó sẽ vẫn có thể dùng làm render target nhưng không thể dùng làm texture nữa
	///
	virtual HQReturnVal RemoveTexture(hq_uint32 ID) = 0;
	virtual void RemoveAllTexture() = 0;

	///
	///note that the format of returned buffer isn't alway the same as  {intendedFormat}
	///
	virtual HQRawPixelBuffer* CreatePixelBuffer(HQRawPixelFormat intendedFormat, hquint32 width, hquint32 height) = 0;
	virtual HQReturnVal AddTexture(const HQRawPixelBuffer* color , bool generateMipmap, hq_uint32 *pTextureID)=0;///Add texture from pixel buffer
};

#endif
