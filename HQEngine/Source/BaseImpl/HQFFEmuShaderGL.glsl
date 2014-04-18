

#define MAX_LIGHTS 4

#ifdef VERTEX_SHADER
/* Vertex Attributes */

//uniform vertex input structure
#if __VERSION__ >= 130
in vec3 aPosition;
in vec4 aColor;
in vec3 aNormal;
in vec2 aTexcoords;
#else
attribute vec3 aPosition;
attribute vec4 aColor;
attribute vec3 aNormal;
attribute vec2 aTexcoords;
#endif

#ifdef HQEXT_GLSL_SEPARATE_SHADER
out gl_PerVertex{
	vec4 gl_Position;
};
#endif

/* Matrix Uniforms */

uniform mat4 uMvpMatrix;
uniform mat4 uWorldMatrix;

/* Light Uniforms */
uniform vec4  uLightPosition    [MAX_LIGHTS];
uniform vec4  uLightAmbient     [MAX_LIGHTS];
uniform vec4  uLightDiffuse     [MAX_LIGHTS];

uniform vec4  uLightSpecular    [MAX_LIGHTS];

uniform vec4  uLightAttenuation [MAX_LIGHTS];// w is ignored. C struct need to add padding float at the end of each element

/* Global ambient color */
uniform vec4  uAmbientColor;

/* Material Uniforms */
uniform vec4  uMaterialAmbient;
uniform vec4  uMaterialEmission;
uniform vec4  uMaterialDiffuse;
uniform vec4  uMaterialSpecular;
uniform float uMaterialShininess;

/* Eye position */
uniform vec3 uEyePos;

/* For light 0 - 3 */
uniform vec4  uUseLight;

/* Normalize normal? */
uniform float uNormalize;
	

//uniform vertex output structure
#if __VERSION__ >= 130
out lowp vec4 vColor;
out mediump	vec2 vTexcoords;
#else
varying lowp vec4 vColor;
varying mediump	vec2 vTexcoords;
#endif

vec4 lightEquation(int lidx, vec3 gWorldNormal, vec3 gWorldPos)
{		
	vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
	
	float att = 1.0;
	vec3 lightDir;
	
	
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
			vec3 hvec = normalize(lightDir + normalize(uEyePos - gWorldPos));
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

vec4 computeLighting( vec3 gWorldNormal, vec3 gWorldPos)
{
	vec4 color = uMaterialEmission + uMaterialAmbient * uAmbientColor;

	for ( int i = 0 ; i < MAX_LIGHTS; ++i)
	{
		if ( uUseLight[i] == 1.0)
		{
			color += lightEquation(i, gWorldNormal, gWorldPos) ;
		}
	}
	color.w = uMaterialDiffuse.w;
	return color;
}

/*---------------vertex shader----------*/
void main()
{	
	vec3 gWorldNormal;
	vec3 gWorldPos;
	vec4 gColor;

	vec4 vposition = vec4(aPosition, 1.0);

	gl_Position = vposition *  uMvpMatrix;
	
	gWorldPos = ( vposition * uWorldMatrix).xyz;
	
	gColor = aColor;
	
#ifdef USE_LIGHTING
	{
		gWorldNormal = ( vec4(aNormal, 0.0) * uWorldMatrix  ).xyz;
		if(uNormalize == 1.0)
			gWorldNormal = normalize(gWorldNormal);
	
		vColor = computeLighting(gWorldNormal, gWorldPos) * gColor;
	}
#else
		vColor = gColor;
#endif //#ifdef USE_LIGHTING
	
	
#ifdef USE_TEXTURE
	{
		vTexcoords = aTexcoords;
	}
#endif
	
	vColor.xyz  = clamp(vColor.xyz, 0.0, 1.0);
	vColor.w = gColor.w;
}

#elif defined FRAGMENT_SHADER


/*---------------pixel shader----------*/
#if __VERSION__ >= 130
in lowp vec4 vColor;
in mediump	vec2 vTexcoords;
out lowp vec4 color0;
#define gl_FragColor color0
#define texture2D texture
#else
varying lowp vec4 vColor;
varying mediump	vec2 vTexcoords;
#endif

uniform sampler2D texture0 ;

void main ()
{
	gl_FragColor = vColor;
#ifdef USE_TEXTURE
	
		lowp vec4 texColor = texture2D ( texture0 , vTexcoords);
		gl_FragColor.xyz *=  texColor.xyz;
		gl_FragColor.w = texColor.w;
	
#endif
}

#endif//#ifdef VERTEX_SHADER