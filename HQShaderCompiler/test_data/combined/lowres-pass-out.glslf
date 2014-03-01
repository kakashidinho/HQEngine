#version 120
#extension GL_ARB_shader_texture_lod : enable
uniform sampler2D randomMap TEXUNIT0;
uniform sampler2D normalMap TEXUNIT2;
uniform sampler2D fluxMap TEXUNIT3;
uniform sampler2D posMap TEXUNIT1;
varying vec3 xlv_TEXCOORD0;
varying vec3 xlv_TEXCOORD1;
varying vec4 xlv_TEXCOORD2;
void main ()
{
  vec3 posW_1;
  posW_1 = xlv_TEXCOORD0;
  vec3 normalW_2;
  normalW_2 = xlv_TEXCOORD1;
  vec2 shadowMapUV_4;
  vec4 tmpvar_5;
  vec4 tmpvar_6;
  tmpvar_6.w = 1.0;
  tmpvar_6.xyz = ((xlv_TEXCOORD0 + vec3(10.0, 10.0, 10.0)) / 20.0);
  vec4 tmpvar_7;
  tmpvar_7.w = 0.0;
  tmpvar_7.xyz = ((xlv_TEXCOORD1 + vec3(1.0, 1.0, 1.0)) / 2.0);
  shadowMapUV_4 = (((xlv_TEXCOORD2.xy / xlv_TEXCOORD2.w) * vec2(0.5, 0.5)) + vec2(0.5, 0.5));
  tmpvar_5 = vec4(0.0, 0.0, 0.0, 1.0);
  for (float u_3 = 0.025; u_3 < 1.0; u_3 += 0.05) {
    for (float v_8 = 0.025; v_8 < 1.0; v_8 += 0.05) {
      vec4 random_9;
      vec4 tmpvar_10;
      tmpvar_10.zw = vec2(0.0, 0.0);
      tmpvar_10.x = u_3;
      tmpvar_10.y = v_8;
      vec4 tmpvar_11;
      tmpvar_11 = texture2DLod (randomMap, tmpvar_10.xy, 0.0);
      random_9.xw = tmpvar_11.xw;
      random_9.yz = ((2.0 * tmpvar_11.yz) - 1.0);
      vec2 tmpvar_12;
      tmpvar_12.x = ((tmpvar_11.x * 0.3) * random_9.y);
      tmpvar_12.y = ((tmpvar_11.x * 0.3) * random_9.z);
      vec4 tmpvar_13;
      tmpvar_13.zw = vec2(0.0, 0.0);
      tmpvar_13.xy = (shadowMapUV_4 + tmpvar_12);
      vec3 tmpvar_14;
      tmpvar_14 = normalize(((
        (20.0 * texture2DLod (posMap, tmpvar_13.xy, 0.0).xyz)
       - 10.0) - posW_1));
      tmpvar_5.xyz = (tmpvar_5.xyz + ((
        ((texture2DLod (fluxMap, tmpvar_13.xy, 0.0).xyz * max (dot (
          ((2.0 * texture2DLod (normalMap, tmpvar_13.xy, 0.0).xyz) - 1.0)
        , 
          -(tmpvar_14)
        ), 0.0)) * max (dot (normalW_2, tmpvar_14), 0.0))
       / 
        (1.0 + pow (sqrt(dot (tmpvar_14, tmpvar_14)), 4.0))
      ) / (5.0 + tmpvar_11.w)));
    };
  };
  gl_FragData[0] = tmpvar_5;
  gl_FragData[1] = tmpvar_6;
  gl_FragData[2] = tmpvar_7;
}

