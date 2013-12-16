

#define MAX_LIGHTS 4


/* Vertex Attributes */

//uniform vertex input structure
struct Vinput
{
	float3 position	: VPOSITION;
	float4 color	: VCOLOR;
	float3 normal	: VNORMAL;
	float2 texcoords: VTEXCOORD0;
};

cbuffer Parameters : register (b0)
{
	/* Matrix Uniforms */

	float4x4 uMvpMatrix;
	float4x4 uWorldMatrix;

	/* Light Uniforms */
	float4  uLightPosition    [MAX_LIGHTS];
	float4  uLightAmbient     [MAX_LIGHTS];
	float4  uLightDiffuse     [MAX_LIGHTS];

	float4  uLightSpecular    [MAX_LIGHTS];

	float3  uLightAttenuation [MAX_LIGHTS];// C struct need to add padding float at the end of each element

	/* Global ambient color */
	float4  uAmbientColor;

	/* Material Uniforms */
	float4  uMaterialAmbient;
	float4  uMaterialEmission;
	float4  uMaterialDiffuse;
	float4  uMaterialSpecular;
	float uMaterialShininess;
	
	/* Eye position */
	float3 uEyePos;
	
	/* For light 0 - 3 */
	int4  uUseLight;
	
	/* Normalize normal? */
	int uNormalize;
	


};//cbuffer Parameters

//uniform vertex output structure
struct Voutput
{
	float4 position	: SV_Position;
	float4 color	: COLOR;
	float2 texcoords: TEXCOORD0;
};

float4 lightEquation(int lidx, in float3 gWorldNormal, in float3 gWorldPos)
{		
	float4 color = float4(0.0, 0.0, 0.0, 0.0);
	
	float att = 1.0;
	float3 lightDir;
	
	
	if( uLightPosition[lidx].w == 0.0) // Directional light
	{
		lightDir = -uLightPosition[lidx].xyz;
	}
	else
	{
		lightDir = uLightPosition[lidx].xyz - gWorldPos.xyz;
		float dir = length(lightDir);
		att = 1.0 / (uLightAttenuation[lidx].x + uLightAttenuation[lidx].y * dir + uLightAttenuation[lidx].z * dir * dir);
		lightDir = normalize(lightDir);
	}
	
	if(att >= 0.0 )
	{
		color += uLightAmbient[lidx] * uMaterialAmbient;
		
		//Compute cos(Light, Normal)
		float NdotL = max(dot(normalize(gWorldNormal), lightDir), 0.0);
		color += NdotL * uLightDiffuse[lidx] * uMaterialDiffuse;
		
		//Compute cos(hvec, Normal)
#ifdef USE_SPECULAR
		{
			float3 hvec = normalize(lightDir + normalize(uEyePos - gWorldPos));
			float NdotH = dot(gWorldNormal, hvec);
			if(NdotH > 0.0)
			{
				color += pow(NdotH, uMaterialShininess) * uLightSpecular[lidx] * uMaterialSpecular;
			}
		}
#endif//#ifdef USE_SPECULAR
		color *= att;
	}
	return color;
}

float4 computeLighting( in float3 gWorldNormal, in float3 gWorldPos)
{
	float4 color = uMaterialEmission + uMaterialAmbient * uAmbientColor;

	for ( int i = 0 ; i < MAX_LIGHTS; ++i)
	{
		if ( uUseLight[i] )
		{
			color += lightEquation(i, gWorldNormal, gWorldPos) ;
		}
	}
	color.a = uMaterialDiffuse.a;
	return color;
}

/*---------------vertex shader----------*/
Voutput VS(in Vinput vin)
{
	Voutput vout;
	
	float3 gWorldNormal;
	float3 gWorldPos;
	float4 gColor;

	float4 vposition = float4(vin.position, 1.0);

	vout.position = mul( uMvpMatrix, vposition  ) ;
	
	gWorldPos = mul( uWorldMatrix, vposition  ).xyz;
	
	gColor = vin.color;
	
#ifdef USE_LIGHTING
	{
		gWorldNormal = mul( uWorldMatrix , float4(vin.normal, 0.0)).xyz;
		if(uNormalize)
			gWorldNormal = normalize(gWorldNormal);
	
		vout.color = computeLighting(gWorldNormal, gWorldPos) * gColor;
	}
#else
		vout.color = gColor;
#endif //#ifdef USE_LIGHTING
	
	
#ifdef USE_TEXTURE
	{
		vout.texcoords = vin.texcoords;
	}
#endif
	
	vout.color.rgb  = clamp(vout.color.rgb, 0.0, 1.0);
	vout.color.a = gColor.a;
	
	return vout;
}


/*---------------pixel shader----------*/

Texture2D texture0 : register (t0);
SamplerState sampler0 : register (s0);

float4 PS (in Voutput pin) :SV_Target
{
	float4 color = pin.color;
#ifdef USE_TEXTURE
	{
		float4 texColor = (texture0.Sample ( sampler0 , pin.texcoords));
		color.xyz *=  texColor.xyz;
		color.a = texColor.a;
	}
#endif
		
	return color;
}