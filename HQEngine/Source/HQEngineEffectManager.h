/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_ENGINE_EFFECT_MANAGER_H
#define HQ_ENGINE_EFFECT_MANAGER_H

#include "HQEngineCommon.h"
#include "HQEngineResManager.h"
#include "HQRenderDevice.h"

//rendering pass
class HQEngineRenderPass : public virtual HQEngineNamedObj{
public:
	virtual HQReturnVal Apply() = 0;
protected:
	virtual ~HQEngineRenderPass() {}
};

//rendering effect
class HQEngineRenderEffect: public virtual HQEngineNamedObj {
public:
	virtual hquint32 GetNumPasses() const = 0;
	virtual hquint32 GetPassIndexByName(const char* name) = 0;
	virtual HQEngineRenderPass* GetPassByName(const char* name) = 0;
	virtual HQEngineRenderPass* GetPass(hquint32 index) = 0;

protected:
	virtual ~HQEngineRenderEffect() {}
};

//effect loading session
class HQEngineEffectLoadSession {
protected:
	virtual ~HQEngineEffectLoadSession() {}
};

///
///Effect manager. Note: all required resources must be loaded before adding any effect
///
class HQEngineEffectManager {
public:
	virtual HQReturnVal AddEffectsFromFile(const char* fileName) = 0;
	virtual HQEngineEffectLoadSession* BeginAddEffectsFromFile(const char* fileName) = 0;
	virtual bool HasMoreEffects(HQEngineEffectLoadSession* session) = 0;
	virtual HQReturnVal AddNextEffect(HQEngineEffectLoadSession* session) = 0;
	virtual HQReturnVal EndAddEffects(HQEngineEffectLoadSession* session) = 0;///for releasing loading session

	///
	///{vertexShader} is ignored in D3D9 device. if {vertexShader} = NULL, this method will create 
	///input layout for fixed function shader. D3D11 & GL only accepts the following layout: 
	///position (x,y,z); color (r,g,b,a); normal (x,y,z); texcoords (u,v)
	///
	virtual HQReturnVal CreateVertexInputLayout(const HQVertexAttribDesc * vAttribDescs , 
												hq_uint32 numAttrib ,
												HQEngineShaderResource* vertexShader , 
												HQVertexLayout **pInputLayoutID) = 0;

	///
	///Direct3d : {slot} = {texture slot} bitwise OR với enum HQShaderType để chỉ  {texture slot} thuộc shader stage nào. 
	///			Ví dụ muốn gắn texture vào texture slot 3 của vertex shader , ta truyền tham số {slot} = (3 | HQ_VERTEX_SHADER). 
	///			Trong direct3d 9 {texture slot} là slot của sampler unit .Shader model 3.0 : pixel shader có 16 sampler. vertex shader có 4 sampler. Các model khác truy vấn bằng method GetMaxShaderStageSamplers() của render device. 
	///			Trong direct3d 10/11 : {texture slot} là slot shader resource view.Mỗi shader stage có 128 slot.  
	///OpenGL : {slot} là slot của sampler unit.{slot} nằm trong khoảng từ 0 đến số trả về từ method GetMaxShaderSamplers() của render device trừ đi 1.Các texture khác loại (ví dụ cube & 2d texture) có thể gắn cùng vào 1 slot.  
	///Lưu ý : -pixel shader dùng trung sampler unit với fixed function.Với openGL , các slot đầu tiên trùng với các slot của fixed function sampler	
	///
	virtual HQReturnVal SetTexture(hq_uint32 slot, HQEngineTextureResource* texture) = 0;
	
	///
	///Direct3d : Trong direct3d 9 {slot} là slot của sampler unit trong pixel shader.Shader model 3.0 : pixel shader có 16 sampler. Các model khác truy vấn bằng method GetMaxShaderStageSamplers(HQ_PIXEL_SHADER) của render device. 
	///			Trong direct3d 10/11 : {slot} là slot pixel shader resource view.Mỗi shader stage có 128 slot.  
	///OpenGL : {slot} là slot của sampler unit.{slot} nằm trong khoảng từ 0 đến số trả về từ method GetMaxShaderSamplers() của render device trừ đi 1.Các texture khác loại (ví dụ cube & 2d texture) có thể gắn cùng vào 1 slot.  
	///Lưu ý : -pixel shader dùng trung sampler unit với fixed function.Với openGL , các slot đầu tiên trùng với các slot của fixed function sampler	
	///
	virtual HQReturnVal SetTextureForPixelShader(hq_uint32 slot, HQEngineTextureResource* texture) = 0;

	virtual HQEngineRenderEffect * GetEffect(const char *name) = 0;
	virtual HQReturnVal RemoveEffect(HQEngineRenderEffect *effect) = 0;
	virtual void RemoveAllEffects() = 0;
protected:
	virtual ~HQEngineEffectManager() {}
};

#endif