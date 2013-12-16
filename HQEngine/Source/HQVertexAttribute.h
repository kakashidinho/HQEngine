/*********************************************************************
*Copyright 2011 Le Hoang Quyen. All rights reserved.
*********************************************************************/
#ifndef _VERTEX_ATTRIB_
#define _VERTEX_ATTRIB_

/*-------------------------------------------------------------------------
vertex attribute usage dùng để ánh xạ thành phần của dữ liệu vertex với 
biến attribute có sematic tương ứng trong vertex shader - xem thêm bảng ánh xạ 
Shader Semantic phía dưới
-----------------------------------*/

typedef enum HQVertAttribUsage
{
	HQ_VAU_POSITION = 0,
	HQ_VAU_COLOR = 1,
	HQ_VAU_NORMAL = 2,
	HQ_VAU_TANGENT = 3,
	HQ_VAU_BINORMAL = 4,
	HQ_VAU_BLENDWEIGHT = 5,
	HQ_VAU_BLENDINDICES = 6,
	HQ_VAU_TEXCOORD0 = 7,
	HQ_VAU_TEXCOORD1 = 8,//chỉ có thể dùng nếu giá trị trả về từ method GetMaxVertexAttribs() của render device lớn hơn 8
	HQ_VAU_TEXCOORD2 = 9,//chỉ có thể dùng nếu giá trị trả về từ method GetMaxVertexAttribs() của render device lớn hơn 9
	HQ_VAU_TEXCOORD3 = 10,//chỉ có thể dùng nếu giá trị trả về từ method GetMaxVertexAttribs() của render device lớn hơn 10
	HQ_VAU_TEXCOORD4 = 11,//chỉ có thể dùng nếu giá trị trả về từ method GetMaxVertexAttribs() của render device lớn hơn 11
	HQ_VAU_TEXCOORD5 = 12,//chỉ có thể dùng nếu giá trị trả về từ method GetMaxVertexAttribs() của render device lớn hơn 12
	HQ_VAU_TEXCOORD6 = 13,//chỉ có thể dùng nếu giá trị trả về từ method GetMaxVertexAttribs() của render device lớn hơn 13
	HQ_VAU_TEXCOORD7 = 14,//chỉ có thể dùng nếu giá trị trả về từ method GetMaxVertexAttribs() của render device lớn hơn 14
	HQ_VAU_PSIZE = 15,//chỉ có thể dùng nếu giá trị trả về từ method GetMaxVertexAttribs() của render device lớn hơn 15
	HQ_VAU_FORCE_DWORD = 0xffffffff
} _HQVertAttribUsage;

/*-------bảng ánh xạ Shader Semantic---------------*/
/*
Chú thích : 

vertex attribute usage  |		semantic		|
________________________|_______________________|
HQ_VAU_POSITION			|	VPOSITION			|
HQ_VAU_COLOR			|	VCOLOR				|
HQ_VAU_NORMAL			|	VNORMAL				|
HQ_VAU_TEXCOORD0		|	VTEXCOORD0			|
HQ_VAU_TEXCOORD1		|	VTEXCOORD1			|
HQ_VAU_TEXCOORD2		|	VTEXCOORD2			|
HQ_VAU_TEXCOORD3		|	VTEXCOORD3			|
HQ_VAU_TEXCOORD4		|	VTEXCOORD4			|
HQ_VAU_TEXCOORD5		|	VTEXCOORD5			|
HQ_VAU_TEXCOORD6		|	VTEXCOORD6			|
HQ_VAU_TEXCOORD7		|	VTEXCOORD7			|
HQ_VAU_TANGENT			|	VTANGENT			|
HQ_VAU_BINORMAL			|	VBINORMAL			|
HQ_VAU_BLENDWEIGHT		|	VBLENDWEIGHT		|
HQ_VAU_BLENDINDICES		|	VBLENDINDICES		|
HQ_VAU_PSIZE			|	VPSIZE				|
*/

/*--------vertex attribure data type------------------*/
//Note : call IsVertexAttribDataTypeSupported() method of render device to check if data type is supported  
typedef enum HQVertexAttribDataType
{
	HQ_VADT_FLOAT = 0,//dạng dữ liệu là 1 float
	HQ_VADT_FLOAT2 = 1,//dạng dữ liệu là 2 float
	HQ_VADT_FLOAT3 = 2,//dạng dữ liệu là 3 float
	HQ_VADT_FLOAT4 = 3,//dạng dữ liệu là 4 float
	HQ_VADT_UBYTE4 = 4,//dạng dữ liệu là 4 unsigned byte
	HQ_VADT_SHORT = 5,//dạng dữ liệu là 1 short
	HQ_VADT_SHORT2 = 6,//dạng dữ liệu là 2 short
	HQ_VADT_SHORT4 = 7,//dạng dữ liệu là 4 short
	HQ_VADT_USHORT = 8,//dạng dữ liệu là 1 unsigned short
	HQ_VADT_USHORT2 = 9,//dạng dữ liệu là 2 unsigned short
	HQ_VADT_USHORT4 = 10,//dạng dữ liệu là 4 unsigned short
	HQ_VADT_USHORT2N = 11,//dạng dữ liệu là 2 unsigned short .khi vào shader , mỗi thành phần sẽ chia cho 65535.0f
	HQ_VADT_USHORT4N = 12,//dạng dữ liệu là 4 unsigned short .khi vào shader , mỗi thành phần sẽ chia cho 65535.0f
	HQ_VADT_INT = 13,//dạng dữ liệu là 1 int
	HQ_VADT_INT2 = 14,//dạng dữ liệu là 2 int
	HQ_VADT_INT3 = 15,//dạng dữ liệu là 3 int
	HQ_VADT_INT4 = 16,//dạng dữ liệu là 4 int
	HQ_VADT_UINT = 17,//dạng dữ liệu là 1 unsigned int
	HQ_VADT_UINT2 = 18,//dạng dữ liệu là 2 unsigned int
	HQ_VADT_UINT3 = 19,//dạng dữ liệu là 3 unsigned int
	HQ_VADT_UINT4 = 20,//dạng dữ liệu là 4 unsigned int
	HQ_VADT_UBYTE4N = 21,//dạng dữ liệu là 4 unsigned byte .khi vào shader , mỗi thành phần sẽ chia cho 255.0f
	HQ_VADT_FORCE_DWORD = 0xffffffff
}_HQVertexAttribDataType;

/*--------vertex attribute description----------------*/
typedef struct HQVertexAttribDesc
{
	hq_uint32 stream;//thành phần này lấy từ vertex stream nào
	hq_uint32 offset;//offset từ đầu dữ liệu đỉnh đến địa chỉ thành phần này(tính theo byte)
	HQVertexAttribDataType dataType ; //dạng dữ liệu của thành phần này
	HQVertAttribUsage usage;//dùng để ánh xạ vào biến có semantic tương ứng trong vertex shader
}_HQVertexAttribDesc;

//fixed array version
template <size_t numAttribs>
class HQVertexAttribDescArray
{
public:
	size_t GetNumAttribs() const {return numAttribs;}

	operator const HQVertexAttribDesc * () const {return descs;}
	const HQVertexAttribDesc & operator[] (size_t index) const { return descs[index]; }
	HQVertexAttribDesc & operator[] (size_t index) { return descs[index]; }

	inline void Set(size_t attribIndex, const HQVertexAttribDesc &desc);
	inline void Set(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType, HQVertAttribUsage usage);
	inline void SetPosition(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	inline void SetColor(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	inline void SetNormal(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	inline void SetTangent(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	inline void SetBinormal(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	inline void SetBlendWeight(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	inline void SetBlendIndices(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	inline void SetPointSize(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	///{index} must be between 0 and 7
	inline void SetTexcoord(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType, hquint32 index);
private:
	HQVertexAttribDesc descs[numAttribs];
};

//dynamic array version
class HQVertexAttribDescArray2
{
public:
	inline HQVertexAttribDescArray2(size_t numAttribs);
	inline ~HQVertexAttribDescArray2();

	size_t GetNumAttribs() const {return numAttribs;}
	operator const HQVertexAttribDesc * () const {return descs;}
	const HQVertexAttribDesc & operator[] (size_t index) const { return descs[index]; }
	HQVertexAttribDesc & operator[] (size_t index) { return descs[index]; }


	inline void Set(size_t attribIndex, const HQVertexAttribDesc &desc);
	inline void Set(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType, HQVertAttribUsage usage);
	inline void SetPosition(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	inline void SetColor(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	inline void SetNormal(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	inline void SetTangent(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	inline void SetBinormal(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	inline void SetBlendWeight(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	inline void SetBlendIndices(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	inline void SetPointSize(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType);
	///{index} must be between 0 and 7
	inline void SetTexcoord(size_t attribIndex, hquint32 stream, hquint32 offset, HQVertexAttribDataType dataType, hquint32 index);
private:
	HQVertexAttribDesc *descs;
	size_t numAttribs;
};

#include "HQVertexAttributeInline.h"

#endif