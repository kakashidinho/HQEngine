#include "../common_shader_code.h"

/*--------vertex shader------------*/
struct VIN {
	float2 pos : VPOSITION;
};

struct VOUT {
	float4 pos : SV_POSITION;
	float2 texcoords: TEXCOORDS;
};

VOUT VS(in VIN input)
{
	VOUT output;

	output.pos.xy = input.pos;
	output.pos.zw = float2(0.0, 1.0);
	output.texcoords = scaleToTexcoord(input.pos);
	return output;
}

/*--------pixel shader------------*/
Texture2D<float4> gbuffer_direct_flux : TEXUNIT0;//direct illumination flux
uniform SamplerState gbuffer_direct_flux_sampler : TEXUNIT0;
Texture2D<float4> final_interpolated_illum : TEXUNIT1;//final interpolated indirect illumination
uniform SamplerState final_interpolated_illum_sampler : TEXUNIT1;

float4 PS(in VOUT input) : SV_Target0
{
	float4 color = gbuffer_direct_flux.Sample(gbuffer_direct_flux_sampler, input.texcoords);
	color += final_interpolated_illum.Sample(final_interpolated_illum_sampler, input.texcoords);

	return color;
}