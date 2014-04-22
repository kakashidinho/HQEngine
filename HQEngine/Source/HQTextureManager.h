/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _HQ_TEXTURE_MANAGER_H_
#define _HQ_TEXTURE_MANAGER_H_

#include "HQRendererCoreType.h"
#include "HQRendererPlatformDef.h"
#include "HQReferenceCountObj.h"
#include "HQReturnVal.h"
#include "HQDataStream.h"


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
	///			Trong direct3d 10/11 : {texture slot} là slot shader resource view.Mỗi shader stage có 128 slot. Lưu ý: texture sẽ bị unset khỏi mọi UAV slot  
	///OpenGL : {slot} là slot của sampler unit.{slot} nằm trong khoảng từ 0 đến số trả về từ method GetMaxShaderSamplers() của render device trừ đi 1.Các texture khác loại (ví dụ cube & 2d texture) có thể gắn cùng vào 1 slot.  
	///Lưu ý : -pixel shader dùng trung sampler unit với fixed function.Với openGL , các slot đầu tiên trùng với các slot của fixed function sampler. 
	///		   -pass NULL sẽ unset texture ra khỏi slot (Trong OpenGL, tất cả cube/2d/buffer texture đều bị unset)
	///
	virtual HQReturnVal SetTexture(hq_uint32 slot , HQTexture* textureID) = 0;
	
	///
	///Direct3d : Trong direct3d 9 {slot} là slot của sampler unit trong pixel shader.Shader model 3.0 : pixel shader có 16 sampler. Các model khác truy vấn bằng method GetMaxShaderStageSamplers(HQ_PIXEL_SHADER) của render device. 
	///			Trong direct3d 10/11 : {slot} là slot pixel shader resource view.Mỗi shader stage có 128 slot. Lưu ý: texture sẽ bị unset khỏi mọi UAV slot
	///OpenGL : {slot} là slot của sampler unit.{slot} nằm trong khoảng từ 0 đến số trả về từ method GetMaxShaderSamplers() của render device trừ đi 1.Các texture khác loại (ví dụ cube & 2d texture) có thể gắn cùng vào 1 slot.  
	///Lưu ý : -pixel shader dùng trung sampler unit với fixed function.Với openGL , các slot đầu tiên trùng với các slot của fixed function sampler. 
	///		   -pass NULL sẽ unset texture ra khỏi slot (Trong OpenGL, tất cả cube/2d/buffer texture đều bị unset)
	///
	virtual HQReturnVal SetTextureForPixelShader(hq_uint32 slot, HQTexture* textureID) = 0;
	
	///
	///Note: shader stage separate textures are not implemented in OpenGL. Every shader uses same set of texture units. 
	///This method is only relevant in Direct3D
	///
	virtual HQReturnVal SetTexture(HQShaderType shaderStage, hq_uint32 slot, HQTexture* textureID) = 0;

	///
	///Direct3d : {slot} = {texture slot} bitwise OR with enum HQShaderType to specify  {texture slot} belongs to which shader stage. 
	///			Eg. to set texture to texture slot 3 of compute shader , pass {slot} = (3 | HQ_COMPUTE_SHADER). 
	///			Direct3d 11 : {texture slot} refers to UAV slot. Each shader stage has 64 slots.  It should be checked by calling HQRenderDevice::GetMaxShaderStageTextureUAVs(). 
	///						  Note that this will unset the texture from every resource slots as well as unset previously bound UAV buffer from the same slot 
	///OpenGL : {slot} is slot of texture image unit.{slot} is between  0 and HQRenderDevice::GetMaxShaderTextureUAVs() minus 1. 
	///Note : passing NULL will unset the current texture out of the slot
	///
	virtual HQReturnVal SetTextureUAV(hq_uint32 slot, HQTexture* textureID, hq_uint32 mipLevel = 0) = 0;

	///
	///Direct3d 11: {slot}  refers to compute shader's UAV slot. It has 64 slots.  
	///				Note that this will unset the texture from every resource slots  as well as unset previously bound UAV buffer from the same slot.  
	///OpenGL : {slot} is slot of texture image unit.{slot} is between  0 and HQRenderDevice::GetMaxShaderTextureUAVs() minus 1. 
	///Note : passing NULL will unset the current texture out of the slot
	///
	virtual HQReturnVal SetTextureUAVForComputeShader(hq_uint32 slot, HQTexture* textureID, hq_uint32 mipLevel = 0) = 0;


	///
	///2 biến maxAlpha và colorKey sẽ dùng để chuyển tất cả giá trị alpha của texel trong texture này thành (< hay = maxAlpha) 
	///và các texel có giá trị rgb trùng các giá trị rgb trong danh sách colorKey sẽ chuyển giá trị alpha thành alpha tương ứng trong colorKey. 
	///Lưu ý :	-nếu textureType = HQ_TEXTURE_CUBE và file ảnh không chứa đủ 6 mặt của cube map 
	///			thì method này sẽ thất bại, return HQ_FAILED_NOT_ENOUGH_CUBE_FACES
	///
	virtual HQReturnVal AddTexture(HQDataReaderStream* dataStream,
						   hq_float32 maxAlpha,
						   const HQColor *colorKey,
						   hq_uint32 numColorKey,
						   bool generateMipmap,
						   HQTextureType textureType,
						   HQTexture** pTextureID) = 0;

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
	virtual HQReturnVal AddCubeTexture(HQDataReaderStream* dataStreams[6] , 
							   hq_float32 maxAlpha,
							   const HQColor *colorKey,
							   hq_uint32 numColorKey,
							   bool generateMipmap,
							   HQTexture** pTextureID) = 0;

	///
	///add thêm 1 texture mà nó chỉ chứa 1 màu {color}
	///
	virtual HQReturnVal AddSingleColorTexture(HQColorui color, HQTexture** pTextureID) = 0;

	///
	///tạo texture buffer
	///
	virtual HQReturnVal AddTextureBuffer(HQTextureBufferFormat format, hq_uint32 size, void *initData, bool isDynamic, HQTextureBuffer** pTextureID) = 0;
	
	
	///
	///create UAV 2d texture
	///
	virtual HQReturnVal AddTextureUAV(HQTextureUAVFormat format, hquint32 width, hquint32 height, bool hasMipmap, HQTexture ** ppTexture) = 0;

	virtual HQTextureCompressionSupport IsCompressionSupported(HQTextureType textureType, HQTextureCompressionFormat compressionType) = 0;
	
	///
	///nếu texture là render target nó sẽ vẫn có thể dùng làm render target nhưng không thể dùng làm texture nữa
	///
	virtual HQReturnVal RemoveTexture(HQTexture* ID) = 0;
	virtual void RemoveAllTexture() = 0;

	///
	///note that the format of returned buffer isn't alway the same as  {intendedFormat}
	///
	virtual HQRawPixelBuffer* CreatePixelBuffer(HQRawPixelFormat intendedFormat, hquint32 width, hquint32 height) = 0;
	virtual HQReturnVal AddTexture(const HQRawPixelBuffer* color, bool generateMipmap, HQTexture** pTextureID) = 0;///Add texture from pixel buffer
};

#endif
