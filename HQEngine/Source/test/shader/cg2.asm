vs_3_0
// cgc version 3.0.0007, build date Jul 22 2010
// command line args: -DTEXTURE_COLOR -DVPOSITION=POSITION -DVCOLOR=COLOR -DVNORMAL=NORMAL -DVTEXCOORD0=TEXCOORD0 -DVTEXCOORD1=TEXCOORD1 -DVTEXCOORD2=TEXCOORD2 -DVTEXCOORD3=TEXCOORD3 -DVTEXCOORD4=TEXCOORD4 -DVTEXCOORD5=TEXCOORD5 -DVTEXCOORD6=TEXCOORD6 -DVTEXCOORD7=TEXCOORD7 -DVTANGENT=TANGENT -DVBINORMAL=BINORMAL -DVBLENDWEIGHT=BLENDWEIGHT -DVBLENDINDICES=BLENDINDICES -DVPSIZE=PSIZE -profile vs_3_0
// source file: H:\My Document\Visual Studio 2008\Projects\HQEngine\test\shader\cg2.txt
//vendor NVIDIA Corporation
//version 3.0.0.07
//profile vs_3_0
//program VS
//semantic VS.texture0 : TEXUNIT3
//semantic viewProj
//semantic cbuffer : BUFFER[14]
//semantic texture1 : TEXUNIT15
//var sampler2D texture0 : TEXUNIT3 : texunit 3 : 0 : 1
//var float3 input.position : $vin.POSITION0 : ATTR0 : 1 : 1
//var float2 input.texcoord0 : $vin.TEXCOORD6 : ATTR1 : 1 : 1
//var float4 input.color0 : $vin.COLOR0 :  : 1 : 0
//var float input.psize : $vin.PSIZE : ATTR2 : 1 : 1
//var float4x4 viewProj :  : , 4 : -1 : 0
//var float3x4 cbuffer.rotation :  : c[0], 3 : -1 : 1
//var float4x4 cbuffer.viewProj :  : c[3], 4 : -1 : 1
//var sampler2D texture1 : TEXUNIT15 :  : -1 : 0
//var float4 VS.position : $vout.POSITION : ATTR0 : -1 : 1
//var float2 VS.texcoord : $vout.TEXCOORD0 : ATTR1 : -1 : 1
//var float4 VS.color : $vout.COLOR0 : ATTR2 : -1 : 1
//var float VS.psize : $vout.PSIZE : ATTR3 : -1 : 1
//const c[7] = 0 1
dcl_position o0
dcl_texcoord0 o1
dcl_color0 o2
dcl_psize o3
def c7, 0.00000000, 1.00000000, 0, 0
dcl_position0 v0
dcl_texcoord6 v1
dcl_psize v2
dcl_2d s3
mov r0.xyz, v0
mov r0.w, c7.y
dp4 r1.x, r0, c1
dp4 r2.x, r0, c0
dp4 r0.x, r0, c2
mul r1, r1.x, c4
mad r1, r2.x, c3, r1
mad r1, r0.x, c5, r1
mov r0.z, c7.x
mov r0.xy, v1
texldl r0, r0.xyzz, s3
add o0, r1, c6
mov o2, r0
mov o1.xy, v1
mov o3, v2.x
