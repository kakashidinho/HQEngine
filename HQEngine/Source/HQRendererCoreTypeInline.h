/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _HQ_TYPE_INL_
#define _HQ_TYPE_INL_
#include "HQRendererCoreType.h"
inline HQColorui HQColoruiRGBA(hq_ubyte8 R,hq_ubyte8 G,hq_ubyte8 B,hq_ubyte8 A,HQColorLayout layout)
{
	if(layout==CL_RGBA)
		return (((A&0xff)<<24) | ((B&0xff)<<16) | ((G&0xff)<<8) | ((R&0xff)) );// R G B A
	return (((A&0xff)<<24) | ((R&0xff)<<16) | ((G&0xff)<<8) | ((B&0xff)) );//B G R A
}

inline HQColor HQColorRGBA(hq_float32 r,hq_float32 g,hq_float32 b,hq_float32 a)
{
	HQColor color={r,g,b,a};
	return color;
}

inline HQColor HQColorRGBAi(hq_ubyte r,hq_ubyte g,hq_ubyte b,hq_ubyte a)
{
	HQColor color={r / 255.f, g / 255.f, b / 255.f, a / 255.f};
	return color;
}

inline HQColor::operator hq_float32 *() 
{
	return c;
}
inline HQColor::operator const hq_float32 *() const 
{
	return c;
}
inline HQColor::operator HQColorui () const
{
	hq_uint32 dwR = r >= 1.0f ? 0xff : r <= 0.0f ? 0x00 : (hq_uint32) (r * 255.0f);
    hq_uint32 dwG = g >= 1.0f ? 0xff : g <= 0.0f ? 0x00 : (hq_uint32) (g * 255.0f);
    hq_uint32 dwB = b >= 1.0f ? 0xff : b <= 0.0f ? 0x00 : (hq_uint32) (b * 255.0f);
    hq_uint32 dwA = a >= 1.0f ? 0xff : a <= 0.0f ? 0x00 : (hq_uint32) (a * 255.0f);

    return (dwA << 24) | (dwB << 16) | (dwG << 8) | dwR ;
}

bool operator ==(const HQColor &color1,const HQColor & color2)
{
	if(color1.a!=color2.a)
		return false;
	if(color1.b!=color2.b)
		return false;
	if(color1.g!=color2.g)
		return false;
	if(color1.r!=color2.r)
		return false;
	return true;
}
bool operator !=(const HQColor &color1,const HQColor & color2)
{
	return !(color1==color2);
}


bool operator ==(const HQColorMaterial &colorMat1,const HQColorMaterial & colorMat2)
{
	if(colorMat1.power!=colorMat2.power)
		return false;
	if(colorMat1.ambient!=colorMat2.ambient)
		return false;
	if(colorMat1.diffuse!=colorMat2.diffuse)
		return false;
	if(colorMat1.emissive!=colorMat2.emissive)
		return false;
	if(colorMat1.specular!=colorMat2.specular)
		return false;
	return true;
}


bool operator !=(const HQColorMaterial &colorMat1,const HQColorMaterial & colorMat2)
{
	return !(colorMat1==colorMat2);
}

inline HQStencilMode::HQStencilMode(HQStencilOp _failOp, 
									HQStencilOp _depthFailOp,
									HQStencilOp _passOp,
									HQStencilFunc _compareFunc)
: failOp(_failOp) , depthFailOp( _depthFailOp), 
passOp (_passOp), compareFunc(_compareFunc)
{
}

inline HQBaseDepthStencilStateDesc::HQBaseDepthStencilStateDesc(HQDepthMode _depthMode ,
																bool _stencilEnable,
																hq_uint32 _readMask,
																hq_uint32 _writeMask ,
																hq_uint32 _refVal )
		:depthMode (_depthMode), stencilEnable(_stencilEnable),
		readMask(_readMask), writeMask(_writeMask),
		refVal(_refVal)
{
}

inline HQDepthStencilStateDesc::HQDepthStencilStateDesc(HQDepthMode _depthMode ,
														bool _stencilEnable ,
														hq_uint32 _readMask ,
														hq_uint32 _writeMask ,
														hq_uint32 _refVal ,
														HQStencilOp _failOp , 
														HQStencilOp _depthFailOp,
														HQStencilOp _passOp ,
														HQStencilFunc _compareFunc
														)
	:HQBaseDepthStencilStateDesc(_depthMode, _stencilEnable, _readMask, _writeMask, _refVal),
	stencilMode(_failOp, _depthFailOp, _passOp, _compareFunc)
{
}

inline HQDepthStencilStateTwoSideDesc::HQDepthStencilStateTwoSideDesc(HQDepthMode _depthMode ,
														bool _stencilEnable ,
														hq_uint32 _readMask ,
														hq_uint32 _writeMask ,
														hq_uint32 _refVal ,
														HQStencilOp _cwFailOp , 
														HQStencilOp _cwDepthFailOp,
														HQStencilOp _cwPassOp ,
														HQStencilFunc _cwCompareFunc,
														HQStencilOp _ccwFailOp , 
														HQStencilOp _ccwDepthFailOp,
														HQStencilOp _ccwPassOp ,
														HQStencilFunc _ccwCompareFunc
														)
	:HQBaseDepthStencilStateDesc(_depthMode, _stencilEnable, _readMask, _writeMask, _refVal),
	cwFaceMode(_cwFailOp, _cwDepthFailOp, _cwPassOp, _cwCompareFunc),
	ccwFaceMode(_ccwFailOp, _ccwDepthFailOp, _ccwPassOp, _ccwCompareFunc)
{
}

inline HQBlendStateDesc::HQBlendStateDesc(HQBlendFactor _srcFactor , HQBlendFactor _destFactor)
: srcFactor(_srcFactor), destFactor(_destFactor)
{
}

inline HQBlendStateExDesc :: HQBlendStateExDesc(HQBlendFactor _srcFactor, 
												HQBlendFactor _destFactor , 
												HQBlendOp _blendOp,
												HQBlendOp _alphaBlendOp) 
	: HQBlendStateDesc(_srcFactor, _destFactor),
	blendOp(_blendOp), alphaBlendOp(_alphaBlendOp)
{
}

inline HQSamplerStateDesc::HQSamplerStateDesc(	HQFilterMode _filterMode ,
												HQTexAddressMode _addressU,
												HQTexAddressMode _addressV ,
												hq_uint32 _maxAnisotropy ,
												const HQColor&	_borderColor )
	: addressU(_addressU), addressV(_addressV), 
	 filterMode (_filterMode), maxAnisotropy(_maxAnisotropy),
	 borderColor (_borderColor)
{
}

inline HQRenderTargetDesc::HQRenderTargetDesc()
: cubeFace (HQ_CTF_POS_X)
{
}


inline HQRenderTargetDesc:: HQRenderTargetDesc(hq_uint32 RTargetID , HQCubeTextureFace face)
: renderTargetID (RTargetID) , cubeFace (face)
{
}
#endif
