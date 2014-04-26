/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _HQ_CORE_TYPE_
#define _HQ_CORE_TYPE_

#include "HQPrimitiveDataType.h"
#include "HQReturnVal.h"
#include "HQReferenceCountObj.h"
#include "HQ2DMath.h"

#ifndef HQ_NOT_AVAIL_ID
#	define HQ_NOT_AVAIL_ID 0xcdcdcdcd
#endif

#ifndef HQ_NULL_ID
#	define HQ_NULL_ID HQ_NOT_AVAIL_ID
#endif


typedef enum HQColorLayout
{
	CL_RGBA = 0,// red ở byte có trọng số nhỏ nhất  , alpha ở byte có trọng số lớn nhất
	CL_BGRA = 1// blue ở byte có trọng số nhỏ nhất  , alpha ở byte có trọng số lớn nhất
}_HQColorLayout;

typedef hq_uint32 HQColorui;//giá trị 32 bit màu biểu thị dưới dạng hq_uint32 data type
inline HQColorui HQColoruiRGBA(hq_ubyte8 R,hq_ubyte8 G,hq_ubyte8 B,hq_ubyte8 A,HQColorLayout layout);

typedef HQRect<hquint32> HQViewPort;//khung nhìn, gốc ở góc trên bên trái

typedef struct HQResolution
{
	hq_uint32 width;
	hq_uint32 height;
} _HQResolution;

typedef struct HQColor
{
	union{
		struct{
			hq_float32 r;//red
			hq_float32 g;//green
			hq_float32 b;//blue
			hq_float32 a;//alpha
		};
		hq_float32 c[4];
	};
	operator hq_float32 *();
	operator const hq_float32 *() const ;
	///return 32 bit value dưới dạng layout RGBA
	operator HQColorui () const;

} _HQColor;
inline HQColor HQColorRGBA(hq_float32 r,hq_float32 g,hq_float32 b,hq_float32 a);
inline HQColor HQColorRGBAi(hq_ubyte r,hq_ubyte g,hq_ubyte b,hq_ubyte a); //color channel value is in [0..255]
inline bool operator ==(const HQColor &color1,const HQColor & color2);
inline bool operator !=(const HQColor &color1,const HQColor & color2);

///vật liệu dùng trong tính toán cường độ sáng
typedef struct HQColorMaterial 
{
	HQColor diffuse;//thành phần tán xạ
	HQColor ambient;//thành phần nền
	HQColor specular;//thành phần phản xạ
	HQColor emissive;//thành phần phát xạ
	hq_float32 power;//số mũ phản xạ
} _HQColorMaterial;

typedef enum HQTextureType
{
	HQ_TEXTURE_2D = 0,
	HQ_TEXTURE_CUBE = 1,
	HQ_TEXTURE_BUFFER = 2,
	HQ_TEXTURE_2D_UAV = 3,///unordered access texture, supports read and write via shader
	HQ_TEXTURE_TYPE_FORCE_DWORD = 0xffffffff
} _HQTextureType;

typedef enum HQCubeTextureFace
{
	HQ_CTF_POS_X = 0,//positive X
	HQ_CTF_NEG_X = 1,//negative X
	HQ_CTF_POS_Y = 2,//positive Y
	HQ_CTF_NEG_Y = 3,//negative Y
	HQ_CTF_POS_Z = 4,//positive Z
	HQ_CTF_NEG_Z = 5,//negative Z
	HQ_CTF_FORCE_DWORD = 0xffffffff
} _HQCubeTextureFace;

inline bool operator ==(const HQColorMaterial &colorMat1,const HQColorMaterial & colorMat2);
inline bool operator !=(const HQColorMaterial &colorMat1,const HQColorMaterial & colorMat2);


typedef enum HQFillMode
{
	HQ_FILL_WIREFRAME = 0,//vẽ dạng khung
	HQ_FILL_SOLID = 1,//vẽ dạng đặc
	HQ_FILL_FORCE_DWORD = 0xffffffff
} _HQFillMode;

typedef enum HQCullMode
{
	HQ_CULL_CW = 0,//loại bỏ mặt có thứ tự đỉnh theo chiều kim đồng hồ
	HQ_CULL_CCW = 1,//loại bỏ mặt có thứ tự đỉnh theo ngược chiều kim đồng hồ
	HQ_CULL_NONE = 2,//không loại bỏ mặt nào
	HQ_CULL_FORCE_DWORD = 0xffffffff
} _HQCullMode;


typedef enum HQDepthMode
{
	HQ_DEPTH_FULL = 0,//ghi và đọc bộ đệm độ sâu
	HQ_DEPTH_READONLY = 1,//chỉ đọc giá trị từ bộ đệm độ sâu để kiểm tra pixel có pass không,không ghi giá trị vào bộ đệm
	HQ_DEPTH_WRITEONLY = 2,//chỉ ghi vào bộ đệm độ sâu.Các pixel được render ra luôn luôn pass phép kiểm tra độ sâu (depth) và ghi giá trị vào bộ đệm độ sâu
	HQ_DEPTH_NONE = 3,//không dùng bộ đệm độ sâu
	HQ_DEPTH_FORCE_DWORD = 0xffffffff
} _HQDepthMode;

typedef enum HQShaderType
{
	HQ_VERTEX_SHADER = 0x10000000,
	HQ_PIXEL_SHADER = 0x20000000,//also known as fragment shader
	HQ_GEOMETRY_SHADER = 0x30000000,
	HQ_COMPUTE_SHADER = 0x40000000
} _HQShaderType;

///
///shader macro - dùng để thêm các định nghĩa preprocessor macro 
///(tương đương với "#define {name} {definition}" trong mã nguồn shader) 
///trước khi compile shader. 
///Lưu ý với glsl nếu {name} = "version" thì {definition} phải là version của glsl .Macro này phải là macro đầu tiên
///Ví dụ {definition] = "120" tương đương "#version 120".
///Lưu ý version macro sẽ bị bỏ qua nếu đã có version định sẵn trong source 
///
struct HQShaderMacro
{
	const char * name;
	const char * definition;
};

typedef enum HQShaderCompileMode
{
	HQ_SCM_CG,//compile source ngôn ngữ CG .Shader manager sẽ compile mà ko thêm thông tin để debug shader
	HQ_SCM_CG_DEBUG,//compile source ngôn ngữ CG.Dùng loại này khi muốn shader manager thêm thông tin debug khi compile shader
	HQ_SCM_GLSL,//compile source ngôn ngữ GLSL hoặc GLSL ES
	HQ_SCM_GLSL_DEBUG,//compile source ngôn ngữ GLSL hoặc GLSL ES.Có thông tin debug
	HQ_SCM_HLSL_10,//compile source directX10/11 HLSL.Không thêm thông tin debug khi compile
	HQ_SCM_HLSL_10_DEBUG//compile source directX10/11 HLSL.Dùng loại này khi muốn shader manager thêm thông tin debug khi compile shader
}_HQShaderCompileMode;

typedef enum HQStencilOp
{
	HQ_SOP_KEEP,//giữ giá trị trên stencil buffer
	HQ_SOP_ZERO,//set giá trị trên stencil buffer thành 0
	HQ_SOP_REPLACE,//thay giá trị trên buffer thành giá rị tham khảo
	HQ_SOP_INCR,//tăng giá trị trên buffer lên 1 , nếu giá trị thành vượt giá trị lớn nhất có thể => giá trị trên buffer thành giá trị lớn nhất
	HQ_SOP_DECR,//giảm giá trị trên buffer xuống 1 , nếu giá trị thành nhỏ hơn giá trị nhỏ nhất có thể => giá trị trên buffer thành giá trị nhỏ nhất
	HQ_SOP_INCR_WRAP,//tăng giá trị trên buffer lên 1 , nếu giá trị thành vượt giá trị lớn nhất có thể => wrap giá trị
	HQ_SOP_DECR_WRAP,//giảm giá trị trên buffer xuống 1 , nếu giá trị thành nhỏ hơn giá trị nhỏ nhất có thể => wrap giá trị
	HQ_SOP_INVERT//đảo các bit của giá trị
}_HQStencilOp;

typedef enum HQStencilFunc//phép toán so sánh 2 giá trị đã bitwise AND với readMask (xem struct HQstencilState phía dưới) của 2 giá trị: tham khảo (op1) và giá trị trên stencil buffer(op2)
{
	HQ_SF_NEVER,//stencil test luôn fail
	HQ_SF_LESS,//stencil pass khi op1 < op2 (tức là (ref value & readMask) < (buffer value & readMask) . ref value là giá trị tham khảo, buffer value là giá trị đang có trên stencil buffer)
	HQ_SF_EQUAL,//pass khi op1 = op2
	HQ_SF_LESS_EQUAL,//pass khi op1 <= op2
	HQ_SF_GREATER ,//pass khi op 1 > op2
	HQ_SF_NOT_EQUAL,//pass khi op1 != op2
	HQ_SF_GREATER_EQUAL,//pass khi op1 >= op2
	HQ_SF_ALWAYS// luôn luôn pass
} _HQStencilFunc;

typedef struct HQStencilMode
{
	HQStencilMode(HQStencilOp failOp = HQ_SOP_KEEP, 
				HQStencilOp depthFailOp = HQ_SOP_KEEP,
				HQStencilOp passOp = HQ_SOP_KEEP,
				HQStencilFunc compareFunc = HQ_SF_ALWAYS
				);

	HQStencilOp failOp;//hành vi phải làm khi stencil test fail (xem HQStencilOp ở trên)
	HQStencilOp depthFailOp;//hành vi phải làm khi stencil test pass nhưng depth test fail
	HQStencilOp passOp;//hành vi phải làm khi cả stencil và depth test đều pass
	HQStencilFunc compareFunc;//phép toán dùng để so sánh giá trị tham khảo (ref value) và giá trị trên stencil buffer (thực chất là so sánh 2 giá trị đã bitwise AND với readMask (xem HQstencilState phía dưới) của 2 giá trị này chứ ko phải trực tiếp 2 giá trị này)
} _HQStencilMode;

typedef struct HQBaseDepthStencilStateDesc
{
	HQBaseDepthStencilStateDesc(HQDepthMode depthMode = HQ_DEPTH_NONE,
								bool stencilEnable = false,
								hq_uint32 readMask = 0xffffffff,
								hq_uint32 writeMask = 0xffffffff,
								hq_uint32 refVal = 0x0);

	HQDepthMode depthMode;//depth buffer mode
	bool stencilEnable;//enable stencil operations?
	hq_uint32 readMask;//giá trị dùng để bitwise AND với giá trị trên stencil buffer và giá trị tham khảo trước khi đem so sánh
	hq_uint32 writeMask;//giá trị dùng để chỉ những bit trên stencil buffer được phép ghi lên .Ví dụ 0xffffffff => tất cả bit được phép ghi
	hq_uint32 refVal;//giá trị tham khảo dùng để so sánh với giá trị trên stencil buffer
}_HQBaseDepthStencilStateDesc;

typedef struct HQDepthStencilStateDesc : public HQBaseDepthStencilStateDesc
{
	HQDepthStencilStateDesc(HQDepthMode depthMode = HQ_DEPTH_NONE,
							bool stencilEnable = false,
							hq_uint32 readMask = 0xffffffff,
							hq_uint32 writeMask = 0xffffffff,
							hq_uint32 refVal = 0x0,
							HQStencilOp failOp = HQ_SOP_KEEP, 
							HQStencilOp depthFailOp = HQ_SOP_KEEP,
							HQStencilOp passOp = HQ_SOP_KEEP,
							HQStencilFunc compareFunc = HQ_SF_ALWAYS
							);

	HQStencilMode  stencilMode;
}_HQDepthStencilStateDesc;

typedef struct HQDepthStencilStateTwoSideDesc : public HQBaseDepthStencilStateDesc //Stencil mode của mặt nào đó không có tác dụng nếu mặt đó bị loại bỏ(tùy vào chế độ loại bỏ mặt HQCullMode)
{
	HQDepthStencilStateTwoSideDesc(HQDepthMode depthMode = HQ_DEPTH_NONE,
							bool stencilEnable = false,
							hq_uint32 readMask = 0xffffffff,
							hq_uint32 writeMask = 0xffffffff,
							hq_uint32 refVal = 0x0,
							HQStencilOp cwFailOp = HQ_SOP_KEEP, 
							HQStencilOp cwDepthFailOp = HQ_SOP_KEEP,
							HQStencilOp cwPassOp = HQ_SOP_KEEP,
							HQStencilFunc cwCompareFunc = HQ_SF_ALWAYS,
							HQStencilOp ccwFailOp = HQ_SOP_KEEP, 
							HQStencilOp ccwDepthFailOp = HQ_SOP_KEEP,
							HQStencilOp ccwPassOp = HQ_SOP_KEEP,
							HQStencilFunc ccwCompareFunc = HQ_SF_ALWAYS
							);

	HQStencilMode  cwFaceMode;//stencil mode dành cho các pixel thuộc đa giác có các đỉnh thứ tự hướng theo chiều kim đồng hồ
	HQStencilMode  ccwFaceMode;//stencil mode dành cho các pixel thuộc đa giác có các đỉnh thứ tự hướng theo ngược chiều kim đồng hồ .
} _HQDepthStencilStateTwoSideDesc;

typedef enum HQBlendFactor
{
	HQ_BF_ONE = 0,//blend factor = (1 , 1 , 1 , 1)
	HQ_BF_ZERO = 1,//blend factor = (0 , 0 , 0 , 0)
	HQ_BF_SRC_COLOR = 2,//blend factor = (Rs , Gs , Bs , As) . Rs, Gs ,Bs ,As = red ,green ,blue ,alpha of source color in range [0 - 1]
	HQ_BF_ONE_MINUS_SRC_COLOR = 3,//blend factor = (1 - Rs , 1 - Gs , 1 - Bs , 1 - As) . Rs, Gs ,Bs ,As = red ,green ,blue ,alpha of source color in range [0 - 1]
	HQ_BF_SRC_ALPHA = 4,//blend factor = (As , As , As , As) . As = alpha value of source color  in range [0 - 1]
	HQ_BF_ONE_MINUS_SRC_ALPHA = 5,//blend factor = (1 - As , 1 - As , 1 - As , 1 - As) . As = alpha value of source color in range [0 - 1]
	HQ_BF_FORCE_DWORD = 0xffffffff
} _HQBlendFactor;

//in most case , source color = color of pixel / fragment that is being computed
//destination color = color of pixel currently on back buffer
typedef enum HQBlendOp
{
	HQ_BO_ADD = 0,//output color = source color * source blend factor + destination color * destination blend factor
	HQ_BO_SUBTRACT = 1,//output color = source color * source blend factor - destination color * destination blend factor
	HQ_BO_REVSUBTRACT = 2, //output color = destination color * destination blend factor - source color * source blend factor
	HQ_BO_FORCE_DWORD = 0xffffffff
} _HQBlendOp;

typedef struct HQBlendStateDesc//blend operation is HQ_BO_ADD
{
	HQBlendStateDesc(HQBlendFactor srcFactor = HQ_BF_ONE, HQBlendFactor destFactor = HQ_BF_ZERO);

	HQBlendFactor srcFactor; /// blend factor for  source color
	HQBlendFactor destFactor ; /// blend factor for destination color
}_HQBlendStateDesc;

//description for creating extended blend state
//supports blend operation other than HQ_BO_ADD and separate blend settings for alpha channel
typedef struct HQBlendStateExDesc : public HQBlendStateDesc
{
	HQBlendStateExDesc(	HQBlendFactor srcFactor = HQ_BF_ONE, 
						HQBlendFactor destFactor = HQ_BF_ZERO, 
						HQBlendOp blendOp = HQ_BO_ADD,
						HQBlendOp alphaBlendOp = HQ_BO_ADD);

	HQBlendOp blendOp;//blend operation for color channel
	HQBlendFactor srcAlphaFactor; // blend factor for  source alpha
	HQBlendFactor destAlphaFactor ; // blend factor for destination alpha
	HQBlendOp alphaBlendOp;//blend operation for alpha channel
} _HQBlendStateExDesc;


typedef enum HQFilterMode
{
	HQ_FM_MIN_MAG_POINT , //point sampling for minification, magnification sampling, and mip-map is disabled. Note: not supported in D3D11 device feature level 9
	HQ_FM_MIN_POINT_MAG_LINEAR,//point sampling for minification ,linear interpolation for magnification sampling , and mip-map is disabled. Note: not supported in D3D11 device feature level 9
	HQ_FM_MIN_LINEAR_MAG_POINT,//point sampling for magnification ,linear interpolation for minification sampling , and mip-map is disabled. Note: not supported in D3D11 device feature level 9
	HQ_FM_MIN_MAG_LINEAR , //linear interpolation for minification, magnification sampling, and mip-map is disabled. Note: not supported in D3D11 device feature level 9
	HQ_FM_MIN_MAG_ANISOTROPIC,  //anisotropic interpolation for minification, magnification sampling , and mip-map is disabled. Note: not supported in D3D11 device feature level 9
	HQ_FM_MIN_MAG_MIP_POINT,//point sampling for minification, magnification, and mip-level sampling
	HQ_FM_MIN_MAG_POINT_MIP_LINEAR,//point sampling for minification, magnification, and linear interpolation for mip-level sampling
	HQ_FM_MIN_POINT_MAG_LINEAR_MIP_POINT,//point sampling for minification ,linear interpolation for magnification and point sampling for mip-level sampling
	HQ_FM_MIN_POINT_MAG_MIP_LINEAR,//point sampling for minification;linear interpolation for magnification and mip-level sampling
	HQ_FM_MIN_LINEAR_MAG_MIP_POINT,//linear interpolation for minification; point sampling for magnification and mip-level sampling
	HQ_FM_MIN_LINEAR_MAG_POINT_MIP_LINEAR,//linear interpolation for minification; point sampling for magnification;linear interpolation for mip-level samplin
	HQ_FM_MIN_MAG_LINEAR_MIP_POINT,//linear interpolation for minification and magnification; point sampling for mip-level sampling
	HQ_FM_MIN_MAG_MIP_LINEAR,//linear interpolation for minification, magnification, and mip-level sampling
	HQ_FM_MIN_MAG_MIP_ANISOTROPIC,  //anisotropic interpolation for minification, magnification, and mip-level sampling
} _HQFilterMode;

typedef enum HQTexAddressMode
{
	HQ_TAM_WRAP,//texel có tọa độ t ngoài [0,1] sẽ có màu của texel tại frac(t) với frac(t) là phần thập phân của t
	HQ_TAM_CLAMP,//texel có tọa độ ngoài [0,1] sẽ có màu sắc của texel tại 0 hoặc 1
	HQ_TAM_BORDER,//texel có tọa độ ngoài [0,1] sẽ có màu sắc của đường viền. OpenGL ES chưa hỗ trợ
	HQ_TAM_MIRROR//texel có tọa độ t ngoài [0,1] sẽ có màu sắc của texel tại 1 - frac(t) (frac(t) là phần thập phân của t) nếu phần nguyên của t là số lẻ .Hoặc bằng frac(t) nếu phần nguyên chẵn
} _HQTexAddressMode;

typedef struct HQSamplerStateDesc
{
	HQSamplerStateDesc(	HQFilterMode filterMode = HQ_FM_MIN_MAG_LINEAR,
						HQTexAddressMode addressU = HQ_TAM_WRAP,
						HQTexAddressMode addressV = HQ_TAM_WRAP,
						hq_uint32 maxAnisotropy = 1,
						const HQColor&	borderColor = HQColorRGBA(0.f, 0.f, 0.f, 0.f));

	HQFilterMode filterMode;
	HQTexAddressMode addressU;
	HQTexAddressMode addressV;
	hq_uint32 maxAnisotropy;
	HQColor	borderColor;//màu đường viền nếu chế độ HQ_TAM_BORDER được set ở addressU hoặc addressV hoặc addressW

} _HQSamplerStateDesc;

class HQRenderTargetView;

typedef struct HQRenderTargetDesc
{
	HQRenderTargetDesc();
	HQRenderTargetDesc(HQRenderTargetView* renderTargetID, HQCubeTextureFace cubeFace = HQ_CTF_POS_X);

	HQRenderTargetView* renderTargetID;//id of render target
	HQCubeTextureFace cubeFace;//if render target is cube texture , this attribute will indicate which face will be used as render target
}_HQRenderTargetDesc;

//format của render target
typedef enum HQRenderTargetFormat
{
	HQ_RTFMT_R_FLOAT32 = 0,//chứa 1 channel (red) với dữ liệu dạng 32 bit float
	HQ_RTFMT_R_FLOAT16 = 1,//chứa 1 channel (red) với dữ liệu dạng 16 bit float
	HQ_RTFMT_RGBA_32 = 2,//chứa 4 channel (red , green , blue ,alpha) mỗi channel có dữ liệu 8 bit
	HQ_RTFMT_A_UINT8 = 3,//chứa 1 channel (alpha) với dữ liệu dạng 8 bit unsigned int
	HQ_RTFMT_R_UINT8 = 4,//chứa 1 channel (red) với dữ liệu dạng 8 bit unsigned int,
	HQ_RTFMT_RGBA_FLOAT64 = 5,//chứa 4 channel, mỗi channel 16 bit float.
	HQ_RTFMT_RG_FLOAT32 = 6,//chứa 2 channel, mỗi channel 16 bit float.
	HQ_RTFMT_RGBA_FLOAT128 = 7,//chứa 4 channel, mỗi channel 32 bit float.
	HQ_RTFMT_RG_FLOAT64 = 8,//chứa 2 channel, mỗi channel 32 bit float.
	HQ_RTFMT_FORCE_DWORD = 0xffffffff
} _HQRenderTargetFormat;

//format của depth stencil buffer
typedef enum HQDepthStencilFormat
{
	HQ_DSFMT_DEPTH_16 = 0,//depth stencil buffer chỉ có 16 bit depth channel
	HQ_DSFMT_DEPTH_24 = 1,//depth stencil buffer chỉ có 24 bit depth channel
	HQ_DSFMT_DEPTH_32 = 2,//depth stencil buffer chỉ có 32 bit depth channel
	HQ_DSFMT_STENCIL_8 = 3,//depth stencil buffer chỉ có 8 bit stencil channel
	HQ_DSFMT_DEPTH_24_STENCIL_8 = 4,//depth stencil buffer có 24 bit depth channel và 8 bit stencil channel
	HQ_DSFMT_FORCE_DWORD = 0xffffffff
} _HQDepthStencilFormat;

//kiểu siêu lấy mẫu
typedef enum HQMultiSampleType
{
	HQ_MST_NONE = 0 ,//không dùng multisample
	HQ_MST_2_SAMPLES = 2,//2 mẫu
	HQ_MST_4_SAMPLES = 4,//4 mẫu
	HQ_MST_8_SAMPLES = 8,//8 mẫu
	HQ_MST_FORCE_DWORD = 0xffffffff //force compiler to compile this enum to 32 bits in size
} _HQMultiSampleType;


typedef enum HQIndexDataType
{
	HQ_IDT_USHORT = 0,//mỗi thành phần trong index buffer là 16 bit unsigned short
	HQ_IDT_UINT = 1,//mỗi thành phần trong index buffer là 32 bit unsigned int
	HQ_IDT_FORCE_DWORD = 0xffffffff
}_HQIndexDataType;

typedef enum HQMapType
{
	HQ_MAP_NORMAL = 0,
	HQ_MAP_DISCARD = 1,//dicard old buffer
	HQ_MAP_NOOVERWRITE = 2//tell driver that we will not overwrite already initialized data in buffer
} _HQMapType;

typedef enum HQPrimitiveMode
{
	HQ_PRI_TRIANGLES = 0,
	HQ_PRI_TRIANGLE_STRIP = 1,
	HQ_PRI_LINES = 2,
	HQ_PRI_LINE_STRIP = 3,
	HQ_PRI_POINT_SPRITES = 4,
	HQ_PRI_NUM_PRIMITIVE_MODE = 5,//don't use this value
	HQ_PRI_FORCE_DWORD = 0xffffffff//don't use this value
} _HQPrimitiveMode;

typedef enum HQTextureBufferFormat
{
	HQ_TBFMT_R16_FLOAT ,//chứa 1 channel (red) với dữ liệu dạng 16 bit float
	HQ_TBFMT_R16G16B16A16_FLOAT ,//chứa 4 channel (red , green , blue , alpha) với dữ liệu dạng 16 bit float
	HQ_TBFMT_R32_FLOAT ,//chứa 1 channel (red) với dữ liệu dạng 32 bit float
	HQ_TBFMT_R32G32B32_FLOAT ,//chứa 3 channel (red , green , blue) với dữ liệu dạng 32 bit float
	HQ_TBFMT_R32G32B32A32_FLOAT ,//chứa 4 channel (red , green , blue , alpha) với dữ liệu dạng 32 bit float
	HQ_TBFMT_R8_INT ,//chứa 1 channel (red) với dữ liệu dạng 8 bit int
	HQ_TBFMT_R8G8B8A8_INT ,//chứa 4 channel (red , green , blue ,alpha) với dữ liệu dạng 8 bit int
	HQ_TBFMT_R8_UINT ,//chứa 1 channel (re) với dữ liệu dạng 8 bit unsigned int
	HQ_TBFMT_R8G8B8A8_UINT ,//chứa 4 channel (red , green , blue ,alpha) với dữ liệu dạng 8 bit unsigned int
	HQ_TBFMT_R8_UNORM ,//chứa 1 channel (red) với dữ liệu dạng 8 bit unsigned int , trong shader dữ liệu này sẽ có giá trị normalized trong khoảng [0..1]
	HQ_TBFMT_R8G8B8A8_UNORM ,//chứa 4 channel (red , green , blue , alpha) với dữ liệu dạng 8 bit unsigned int , trong shader dữ liệu này sẽ có giá trị normalized trong khoảng [0..1]
	HQ_TBFMT_R16_INT ,//chứa 1 channel (red) với dữ liệu dạng 16 bit int
	HQ_TBFMT_R16G16B16A16_INT ,//chứa 4 channel (red , green , blue ,alpha) với dữ liệu dạng 16 bit int
	HQ_TBFMT_R16_UINT ,//chứa 1 channel (re) với dữ liệu dạng 16 bit unsigned int
	HQ_TBFMT_R16G16B16A16_UINT ,//chứa 4 channel (red , green , blue ,alpha) với dữ liệu dạng 16 bit unsigned int
	HQ_TBFMT_R16_UNORM ,//chứa 1 channel (red) với dữ liệu dạng 16 bit unsigned int , trong shader dữ liệu này sẽ có giá trị normalized trong khoảng [0..1]
	HQ_TBFMT_R16G16B16A16_UNORM ,//chứa 4 channel (red , green , blue , alpha) với dữ liệu dạng 16 bit unsigned int , trong shader dữ liệu này sẽ có giá trị normalized trong khoảng [0..1]
	HQ_TBFMT_R32_INT ,//chứa 1 channel (red) với dữ liệu dạng 32 bit int
	HQ_TBFMT_R32G32B32A32_INT ,//chứa 4 channel (red , green , blue ,alpha) với dữ liệu dạng 32 bit int
	HQ_TBFMT_R32_UINT ,//chứa 1 channel (re) với dữ liệu dạng 32 bit unsigned int
	HQ_TBFMT_R32G32B32A32_UINT ,//chứa 4 channel (red , green , blue ,alpha) với dữ liệu dạng 32 bit unsigned int
	HQ_TBFMT_FORCE_DWORD = 0xffffffff
} _HQTextureBufferFormat;

typedef enum _HQTextureCompressionFormat
{
	HQ_TC_S3TC_DTX1 = 0,
	HQ_TC_S3TC_DXT3 = 1,
	HQ_TC_S3TC_DXT5 = 2,
	HQ_TC_ETC1  = 3,
	HQ_TC_PVRTC_RGB_2BPP = 4,
	HQ_TC_PVRTC_RGB_4BPP = 5,
	HQ_TC_PVRTC_RGBA_2BPP = 6,
	HQ_TC_PVRTC_RGBA_4BPP = 7,
	HQ_TC_COUNT = 8,//don't use this value
	HQ_TC_FORCE_DWORD = 0xffffffff
}HQTextureCompressionFormat;


typedef enum _HQRawPixelFormat
{
	HQ_RPFMT_R8G8B8A8 = 0, //each pixel: first byte is red, last byte is alpha
	HQ_RPFMT_B8G8R8A8 = 1, //each pixel: first byte is blue, last byte is alpha
	HQ_RPFMT_R5G6B5 = 2, //each pixel: 16 bits
	HQ_RPFMT_L8A8 = 3, //each pixel: first byte is luminance value, last byte is alpha
	HQ_RPFMT_A8 = 4, //each pixel is alpha
	HQ_RPFMT_R32_FLOAT = 8,//32bit floating point pixel data
	HQ_RPFMT_R32G32_FLOAT = 9,//32bit floating point pixel data
	HQ_RPFMT_R32G32B32A32_FLOAT = 10,//32bit floating point pixel data
	HQ_RPFMT_FORCE_DWORD = 0xffffffff
}HQRawPixelFormat;

typedef enum _HQTextureCompressionSupport
{
	HQ_TCS_ALL,//hardware support and software decompression on load support
	HQ_TCS_HW,//hardware support, but no software decompression, so you can't resize or change alpha channel
	HQ_TCS_SW,//software decompression on load support
	HQ_TCS_NONE,//no hardware/software support
	HQ_TCS_FORCE_DWORD = 0xffffffff
}HQTextureCompressionSupport;

///
///format for UAV texture
///
typedef enum _HQTextureUAVFormat {
	HQ_UAVTFMT_R16_FLOAT,
	HQ_UAVTFMT_R16G16_FLOAT,
	HQ_UAVTFMT_R16G16B16A16_FLOAT,
	HQ_UAVTFMT_R32_FLOAT,
	HQ_UAVTFMT_R32G32_FLOAT,
	HQ_UAVTFMT_R32G32B32A32_FLOAT,
	HQ_UAVTFMT_R32_INT,
	HQ_UAVTFMT_R32G32_INT,
	HQ_UAVTFMT_R32G32B32A32_INT,
	HQ_UAVTFMT_R32_UINT,
	HQ_UAVTFMT_R32G32_UINT,
	HQ_UAVTFMT_R32G32B32A32_UINT,
	HQ_UAVTFMT_R8G8B8A8_UNORM, //8 bit per channel, when use as input, each channel will be converted to floating point between [0..1] in shader. while in Direct3d11, when use as UAV, it will be uint format instead of float4
	HQ_UAVTFMT_FORCE_DWORD = 0xffffffff
} HQTextureUAVFormat;

///
///the pixel at (0,0) is top left
///
class HQRawPixelBuffer : public HQReferenceCountObj
{
public:
	virtual hquint32 GetWidth() const = 0;
	virtual hquint32 GetHeight() const = 0;
	///
	///RGB is ignored in A8 pixel buffer. GB is ignored in L8A8/R32_FLOAT buffer (R is set to luminance value). 
	///A is ignored in R5G6B5/R32/R32G32_FLOAT buffer. Except floating point format, color channel range is 0.0f-1.0f
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
	HQRawPixelBuffer() : HQReferenceCountObj() {}
	virtual ~HQRawPixelBuffer() {}
};

class HQRestrictedObj {
protected:
	virtual ~HQRestrictedObj() {};
};

class HQGraphicsResourceRawRetrievable : public HQRestrictedObj{
public:
	///
	///D3D: return IDirect3DResource9 or ID3D11Resource pointer. 
	///GL: return OpenGL handle (GLuint).
	///Use with caution
	///
	virtual void* GetRawHandle() = 0;
};

class HQMappableResource : public virtual HQRestrictedObj {
public:
	template <class T>
	inline HQReturnVal Map(T** ppData, HQMapType mapType = HQ_MAP_DISCARD, hquint32 offset = 0, hquint32 size = 0) {
		return GenericMap((void**)ppData, mapType, offset, size);
	}
	virtual hquint32 GetSize() const = 0;///mappable size
	inline HQReturnVal Update(const void * pData)///update entire resource
	{
		return Update(0, GetSize(), pData);
	}
	virtual HQReturnVal Update(hq_uint32 offset, hq_uint32 size, const void * pData) = 0;
	virtual HQReturnVal Unmap() = 0;
protected:
	virtual HQReturnVal GenericMap(void ** ppData, HQMapType mapType = HQ_MAP_DISCARD, hquint32 offset = 0, hquint32 size = 0) = 0;
};

class HQTexture : public virtual HQGraphicsResourceRawRetrievable {
public:
	virtual hquint32 GetResourceIndex() const = 0;///return assigned index for resource

	virtual HQTextureType GetType() const = 0;
	///
	///get first level's size
	///
	virtual hquint32 GetWidth() const =  0;
	///
	///get first level's size
	///
	virtual hquint32 GetHeight() const = 0;
protected:
	virtual ~HQTexture() {}
};

class HQUpdatableTexture : public HQTexture, public HQMappableResource {

};

class HQTextureBuffer : public HQUpdatableTexture {
};

class HQGraphicsBufferRawRetrievable : public virtual HQMappableResource, public virtual HQGraphicsResourceRawRetrievable{

};

typedef HQGraphicsBufferRawRetrievable HQVertexBuffer;
typedef HQGraphicsBufferRawRetrievable HQIndexBuffer;
typedef HQGraphicsBufferRawRetrievable HQBufferUAV;///unordered access buffer, supports read and write via shader
typedef HQGraphicsBufferRawRetrievable HQVertexBufferUAV;///unordered access buffer, supports read and write via shader
typedef HQGraphicsBufferRawRetrievable HQIndexBufferUAV;///unordered access buffer, supports read and write via shader
typedef HQGraphicsBufferRawRetrievable HQComputeIndirectArgsBuffer;///unordered access buffer, supports read and write via shader
typedef HQGraphicsBufferRawRetrievable HQDrawIndirectArgsBuffer; ///unordered access buffer, supports read and write via shader
typedef HQGraphicsBufferRawRetrievable HQDrawIndexedIndirectArgsBuffer;///unordered access buffer, supports read and write via shader


class HQUniformBuffer : public virtual HQMappableResource {

};

class HQRenderTargetView : public virtual HQRestrictedObj {

};

class HQDepthStencilBufferView : public virtual HQRestrictedObj {

};

class HQRenderTargetGroup : public virtual HQRestrictedObj {

};

class HQVertexLayout : public virtual HQRestrictedObj {

};

class HQShaderObject : public virtual HQRestrictedObj {
public:
	virtual HQShaderType GetType() const = 0;
};

class HQShaderProgram : public virtual HQRestrictedObj {
public:
	virtual bool HasShaderStage(HQShaderType type) {
		return GetShader(type) == NULL;
	}
	virtual HQShaderObject * GetShader(HQShaderType type) = 0;
};

#include "HQRendererCoreTypeInline.h"
#include "HQRendererCoreTypeFixedFunction.h"
#include "HQVertexAttribute.h"

#endif
