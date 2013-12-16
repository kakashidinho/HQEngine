#ifndef BASEIMPLSHADERSTRING_HQCLEARVIEWPORTSHADERCODED3D1X_H_INCLUDED
#define BASEIMPLSHADERSTRING_HQCLEARVIEWPORTSHADERCODED3D1X_H_INCLUDED
const char HQClearViewportShaderCodeD3D1x[] = "\
cbuffer Parameters : register(b0)\r\n\
{\r\n\
	float4 COLOR;\r\n\
	float DEPTH;\r\n\
};\r\n\
\r\n\
struct VS_OUT\r\n\
{\r\n\
	float4 position : SV_Position;\r\n\
	float4 color : COLOR;\r\n\
};\r\n\
\r\n\
VS_OUT VS(in float2 position : POSITION)\r\n\
{\r\n\
	VS_OUT vout;\r\n\
	vout.position = float4(position.x, position.y, DEPTH, 1.0f);\r\n\
	vout.color = COLOR;\r\n\
	return vout;\r\n\
}\r\n\
\r\n\
float4 PS(in VS_OUT pIn) :SV_Target\r\n\
{\r\n\
	return pIn.color;\r\n\
}\n";
#endif