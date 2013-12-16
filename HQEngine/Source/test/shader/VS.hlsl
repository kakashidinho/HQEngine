//#include "vertexType.hlsl"

/*---------------------------------------Vertex types-----------------*/
struct VS_P
{
	float3 posL: VPOSITION;
};

struct VS_PC
{
	float3 posL: VPOSITION;
	float4 color : VCOLOR;
};

struct VS_PT
{
	float3 posL: VPOSITION;
	float2 texcoord0 : VTEXCOORD0;
};

struct VS_PCT
{
	float3 posL: VPOSITION;
	float4 color : VCOLOR;
	float2 texcoord0 : VTEXCOORD0;
};

struct VS_PNT
{
	float3 posL: VPOSITION;
	float3 normalL : VNORMAL;
	float2 texcoord0 : VTEXCOORD0;
};

struct VS_PN2T
{
	float3 posL: VPOSITION;
	float3 normalL : VNORMAL;
	float2 texcoord0 : VTEXCOORD0;
	float2 texcoord1 : VTEXCOORD1;
};

struct VS_PN3T
{
	float3 posL: VPOSITION;
	float3 normalL : VNORMAL;
	float2 texcoord0 : VTEXCOORD0;
	float2 texcoord1 : VTEXCOORD1;
	float2 texcoord2 : VTEXCOORD2;
};

struct VS_PNTTB
{
	float3 posL: VPOSITION;
	float3 normalL : VNORMAL;
	float2 texcoord0 : VTEXCOORD0;
	float3 texcoord1 : VTEXCOORD1;//tangent
	float3 texcoord2 : VTEXCOORD2;//binormal
};

struct VS_Bone
{
	float3 posL: VPOSITION;
	float3 normalL : VNORMAL;
	float2 texcoord0 : VTEXCOORD0;
	float4 texcoord1 : VTEXCOORD1;//bone weight
	int4 texcoord2 : VTEXCOORD2;//bone index
};

struct VS_BoneTB
{
	float3 posL: VPOSITION;
	float3 normalL : VNORMAL;
	float2 texcoord0 : VTEXCOORD0;
	float3 texcoord1 : VTEXCOORD1;//tangent
	float3 texcoord2 : VTEXCOORD2;//binormal
	float4 texcoord3 : VTEXCOORD3;//bone weight
	int4 texcoord4 : VTEXCOORD4;//bone index
};
/*-------------------------------------------------------------------------*/

struct Light
{
	float3 pos;
	float3 dir;
	float4 ambient;
	float4 diffuse;
	float4 specular;
	float3 att; // attenuation parameters (a0, a1, a2)
	float spotPower;
	float spotCutOff;
	float range;
};

struct LightInfo
{	
	Light light;
	int type;
};

struct Material
{
	float4 diffuse;
	float4 ambient;
	float4 specular;
	float4 emissive;
	float power;
};

struct VS_OUT
{
	float4 posH: SV_POSITION;
	float3 posW: POSITION;
	float2 texcoord[3] : TEXCOORD;
	float4 color : DIFFUSE;
};

cbuffer camera
{	
	float4x4 viewProj;
	float3 eyePos;
};

cbuffer transform
{
	float4x4 world;
};

cbuffer lightingState
{
	int lightingEnable;
	int specularHighlight;
}
cbuffer lightSource
{
	LightInfo lightInfo[4];
};

cbuffer lightEnable
{
	vector<int , 4> lightEnable; 
};

cbuffer material
{	
	Material mat;
};

float4 directionalLightColor(float3 posW , float3 normalW,int index)
{	
	float4 litColor = float4(0.0f, 0.0f, 0.0f ,1.0f);
	float3 lightVec = -lightInfo[index].light.dir;
	litColor += mat.ambient * lightInfo[index].light.ambient ;

	float diffuseFactor = dot(lightVec , normalW);
	[branch]
	if( diffuseFactor > 0.0f )
	{
		litColor += diffuseFactor * mat.diffuse * lightInfo[index].light.diffuse;
		
		[branch]
		if(specularHighlight != 0)
		{
			float specPower = max(mat.power , 1.0f);
			float3 toEye = normalize(eyePos - posW);
			float3 r = reflect(-lightVec, normalW);
			float specFactor = pow(max(dot(r, toEye), 0.0f),specPower);
			litColor += specFactor * mat.specular * lightInfo[index].light.specular;
		}
	}
	return litColor;
}

float4 sumLitColor(float3 posW , float3 normalW)
{
	float4 litColor = float4(1.0f, 1.0f, 1.0f ,1.0f);
	[branch]
	if(lightingEnable == 1)
	{
		litColor = mat.emissive;
		for(int i = 0; i < 4 ; ++i)
		{
			[branch]
			if(lightEnable[i] == 1)//enable
			{
				[branch]
				if(lightInfo[i].type== 0)//direcional light
					litColor += directionalLightColor(posW, normalW , i );
			}
		} 
	}
	return litColor;
}

/*-----------------------------------------------P--------------------------------------------------------------------------------------*/

VS_OUT lightVS_P(VS_P vIn)
{
	VS_OUT vOut;
	
	vOut.posW = mul(float4(vIn.posL , 1.0f) , world).xyz;
	vOut.posH = mul(float4(vOut.posW,1.0f) , viewProj);
	vOut.texcoord[0] =float2(0,0);
	vOut.texcoord[1] = float2(0,0);
	vOut.texcoord[2] = float2(0,0);

	vOut.color =float4(1.0f , 1.0f , 1.0f , 1.0f);
	
	return vOut;
}

/*-----------------------------------------------PC--------------------------------------------------------------------------------------*/

VS_OUT lightVS_PC(VS_PC vIn)
{
	VS_OUT vOut;
	
	vOut.posW = mul(float4(vIn.posL , 1.0f) , world).xyz;
	vOut.posH = mul(float4(vOut.posW,1.0f) , viewProj);
	vOut.texcoord[0] = float2(0,0);
	vOut.texcoord[1] = float2(0,0);
	vOut.texcoord[2] = float2(0,0);

	vOut.color =vIn.color;
	return vOut;
}

/*-----------------------------------------------PT--------------------------------------------------------------------------------------*/

VS_OUT lightVS_PT(VS_PT vIn)
{
	VS_OUT vOut;
	
	vOut.posW = mul(float4(vIn.posL , 1.0f) , world).xyz;
	vOut.posH = mul(float4(vOut.posW,1.0f) , viewProj);
	vOut.texcoord[0] = vIn.texcoord0;
	vOut.texcoord[1] = vIn.texcoord0;
	vOut.texcoord[2] = vIn.texcoord0;

	vOut.color =float4(1.0f , 1.0f , 1.0f , 1.0f);
	return vOut;
}

/*-----------------------------------------------PCT--------------------------------------------------------------------------------------*/

VS_OUT lightVS_PCT(VS_PCT vIn)
{
	VS_OUT vOut;
	
	vOut.posW = mul(float4(vIn.posL , 1.0f) , world).xyz;
	vOut.posH = mul(float4(vOut.posW,1.0f) , viewProj);
	vOut.texcoord[0] = vIn.texcoord0;
	vOut.texcoord[1] = vIn.texcoord0;
	vOut.texcoord[2] = vIn.texcoord0;

	vOut.color =vIn.color;

	return vOut;
}

/*-----------------------------------------------PNT--------------------------------------------------------------------------------------*/

VS_OUT lightVS_PNT(VS_PNT vIn)
{
	VS_OUT vOut;
	vOut.posW = mul(float4(vIn.posL , 1.0f) , world).xyz;
	vOut.posH = mul(float4(vOut.posW,1.0f) , viewProj);
	float3 normalW =  normalize (mul(float4(vIn.normalL , 0.0f) , world).xyz );

	vOut.texcoord[0] = vIn.texcoord0;
	vOut.texcoord[1] = vIn.texcoord0;
	vOut.texcoord[2] = vIn.texcoord0;

	vOut.color = sumLitColor(vOut.posW , normalW);
	return vOut;
}

/*-----------------------------------------------PN2T--------------------------------------------------------------------------------------*/

VS_OUT lightVS_PN2T(VS_PN2T vIn)
{
	VS_OUT vOut;
	vOut.posW = mul(float4(vIn.posL , 1.0f) , world).xyz;
	vOut.posH = mul(float4(vOut.posW,1.0f) , viewProj);
	float3 normalW =  normalize (mul(float4(vIn.normalL , 0.0f) , world).xyz );
	vOut.texcoord[0] = vIn.texcoord0;
	vOut.texcoord[1] = vIn.texcoord1;
	vOut.texcoord[2] = vIn.texcoord0;

	vOut.color = sumLitColor(vOut.posW , normalW);
	return vOut;
}

/*-----------------------------------------------PN3T--------------------------------------------------------------------------------------*/

VS_OUT lightVS_PN3T(VS_PN3T vIn)
{
	VS_OUT vOut;
	vOut.posW = mul(float4(vIn.posL , 1.0f) , world).xyz;
	vOut.posH = mul(float4(vOut.posW,1.0f) , viewProj);
	float3 normalW =  normalize (mul(float4(vIn.normalL , 0.0f) , world).xyz );
	
	vOut.texcoord[0] = vIn.texcoord0;
	vOut.texcoord[1] = vIn.texcoord1;
	vOut.texcoord[2] = vIn.texcoord2;

	vOut.color = sumLitColor(vOut.posW , normalW);
	return vOut;
}

/*-----------------------------------------------PNTTB--------------------------------------------------------------------------------------*/

VS_OUT lightVS_PNTTB(VS_PNTTB vIn)
{
	VS_OUT vOut;
	vOut.posW = mul(float4(vIn.posL , 1.0f) , world).xyz;
	vOut.posH = mul(float4(vOut.posW,1.0f) , viewProj);
	float3 normalW =  normalize (mul(float4(vIn.normalL , 0.0f) , world).xyz );
	
	vOut.texcoord[0] = vIn.texcoord0;
	vOut.texcoord[1] = vIn.texcoord0;
	vOut.texcoord[2] = vIn.texcoord0;

	vOut.color = sumLitColor(vOut.posW , normalW);
	return vOut;
}

/*-----------------------------------------------Bone--------------------------------------------------------------------------------------*/

VS_OUT lightVS_Bone(VS_Bone vIn)
{
	VS_OUT vOut;
	vOut.posW = mul(float4(vIn.posL , 1.0f) , world).xyz;
	vOut.posH = mul(float4(vOut.posW,1.0f) , viewProj);
	float3 normalW =  normalize (mul(float4(vIn.normalL , 0.0f) , world).xyz );
	
	vOut.texcoord[0] = vIn.texcoord0;
	vOut.texcoord[1] = vIn.texcoord0;
	vOut.texcoord[2] = vIn.texcoord0;

	vOut.color = sumLitColor(vOut.posW , normalW);
	return vOut;
}
/*-----------------------------------------------BoneTB--------------------------------------------------------------------------------------*/

VS_OUT lightVS_BoneTB(VS_BoneTB vIn)
{
	VS_OUT vOut;
	vOut.posW = mul(float4(vIn.posL , 1.0f) , world).xyz;
	vOut.posH = mul(float4(vOut.posW,1.0f) , viewProj);
	float3 normalW =  normalize (mul(float4(vIn.normalL , 0.0f) , world).xyz );
	
	vOut.texcoord[0] = vIn.texcoord0;
	vOut.texcoord[1] = vIn.texcoord0;
	vOut.texcoord[2] = vIn.texcoord0;

	vOut.color = sumLitColor(vOut.posW , normalW);
	return vOut;
}

/*--------------clear VS --------------------------------------------*/
struct CLEARVS_OUT
{
	float4 posH : SV_VPOSITION ;
	float4 oColor : DIFFUSE	;	
};
CLEARVS_OUT clearVS(float3 pos: VPOSITION , float4 color : VCOLOR)
{
	CLEARVS_OUT vOut;
	vOut.posH = float4(pos, 1.0f);
	vOut.oColor = color;
	return vOut;
}