/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_VATTR_INL_H
#define HQ_VATTR_INL_H

#include "HQVertexAttribute.h"

/*-----------------------------*/
template <size_t num>
inline void HQVertexAttribDescArray<num>::Set(size_t attribIndex, const HQVertexAttribDesc &desc)
{
	descs[attribIndex] = desc;
}

template <size_t num>
inline void HQVertexAttribDescArray<num>::Set(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType, HQVertAttribUsage usage)
{
	descs[attribIndex].stream = stream;
	descs[attribIndex].offset = offset;
	descs[attribIndex].dataType = dataType;
	descs[attribIndex].usage = usage;
}

template <size_t num>
inline void HQVertexAttribDescArray<num>::SetPosition(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_POSITION);
}

template <size_t num>
inline void HQVertexAttribDescArray<num>::SetColor(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_COLOR);
}

template <size_t num>
inline void HQVertexAttribDescArray<num>::SetNormal(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_NORMAL);
}

template <size_t num>
inline void HQVertexAttribDescArray<num>::SetTangent(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_TANGENT);
}

template <size_t num>
inline void HQVertexAttribDescArray<num>::SetBinormal(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_BINORMAL);
}

template <size_t num>
inline void HQVertexAttribDescArray<num>::SetBlendWeight(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_BLENDWEIGHT);
}

template <size_t num>
inline void HQVertexAttribDescArray<num>::SetBlendIndices(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_BLENDINDICES);
}

template <size_t num>
inline void HQVertexAttribDescArray<num>::SetPointSize(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_PSIZE);
}

template <size_t num>
inline void HQVertexAttribDescArray<num>::SetTexcoord(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType, hquint32 index)
{
	Set(attribIndex, stream, offset, dataType, (HQVertAttribUsage)(HQ_VAU_TEXCOORD0 + index));
}


/*--------------------------*/
inline HQVertexAttribDescArray2::HQVertexAttribDescArray2(size_t num)
: numAttribs(num)
{
	this->descs = HQ_NEW HQVertexAttribDesc[num];
}

inline HQVertexAttribDescArray2::~HQVertexAttribDescArray2()
{
	delete[] this->descs;
}


inline void HQVertexAttribDescArray2::Set(size_t attribIndex, const HQVertexAttribDesc &desc)
{
	descs[attribIndex] = desc;
}


inline void HQVertexAttribDescArray2::Set(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType, HQVertAttribUsage usage)
{
	descs[attribIndex].stream = stream;
	descs[attribIndex].offset = offset;
	descs[attribIndex].dataType = dataType;
	descs[attribIndex].usage = usage;
}


inline void HQVertexAttribDescArray2::SetPosition(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_POSITION);
}


inline void HQVertexAttribDescArray2::SetColor(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_COLOR);
}


inline void HQVertexAttribDescArray2::SetNormal(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_NORMAL);
}


inline void HQVertexAttribDescArray2::SetTangent(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_TANGENT);
}


inline void HQVertexAttribDescArray2::SetBinormal(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_BINORMAL);
}


inline void HQVertexAttribDescArray2::SetBlendWeight(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_BLENDWEIGHT);
}


inline void HQVertexAttribDescArray2::SetBlendIndices(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_BLENDINDICES);
}


inline void HQVertexAttribDescArray2::SetPointSize(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType)
{
	Set(attribIndex, stream, offset, dataType, HQ_VAU_PSIZE);
}


inline void HQVertexAttribDescArray2::SetTexcoord(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType, hquint32 index)
{
	Set(attribIndex, stream, offset, dataType, (HQVertAttribUsage)(HQ_VAU_TEXCOORD0 + index));
}
#endif
