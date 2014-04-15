uniform vec4 worldMat[3];
uniform mat4 viewMat;
uniform mat4 projMat;
uniform mat4 lightViewMat;
uniform mat4 lightProjMat;
attribute vec3 xlat_attrib_VPOSITION VPOSITION;
attribute vec3 xlat_attrib_VNORMAL VNORMAL;
varying vec3 xlv_TEXCOORD0;
varying vec3 xlv_TEXCOORD1;
varying vec4 xlv_TEXCOORD2;
void main ()
{
  vec3 vec_1;
  vec3 vec_2;
  vec4 tmpvar_3;
  tmpvar_3.w = 1.0;
  tmpvar_3.xyz = xlat_attrib_VPOSITION;
  vec_2.x = dot (worldMat[0], tmpvar_3);
  vec_2.y = dot (worldMat[1], tmpvar_3);
  vec_2.z = dot (worldMat[2], tmpvar_3);
  vec4 tmpvar_4;
  tmpvar_4.w = 0.0;
  tmpvar_4.xyz = xlat_attrib_VNORMAL;
  vec_1.x = dot (worldMat[0], tmpvar_4);
  vec_1.y = dot (worldMat[1], tmpvar_4);
  vec_1.z = dot (worldMat[2], tmpvar_4);
  vec4 tmpvar_5;
  tmpvar_5.w = 1.0;
  tmpvar_5.xyz = vec_2;
  vec4 tmpvar_6;
  tmpvar_6.w = 1.0;
  tmpvar_6.xyz = vec_2;
  gl_Position = ((tmpvar_5 * viewMat) * projMat);
  xlv_TEXCOORD0 = vec_2;
  xlv_TEXCOORD1 = normalize(vec_1);
  xlv_TEXCOORD2 = ((tmpvar_6 * lightViewMat) * lightProjMat);
}

