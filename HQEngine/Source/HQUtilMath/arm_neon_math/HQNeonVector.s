;Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

;This program is free software; you can redistribute it and/or
;modify it under the terms of the GNU General Public License as
;published by the Free Software Foundation; either version 2 of the
;License, or (at your option) any later version.  See the file
;COPYING.txt included with this distribution for more information.

	AREA     HQNeonVector, CODE, READONLY, THUMB
	THUMB
	
	MACRO	;function definition macro
	FUNCTION_BEGIN $name
	EXPORT $name
$name
$name.Label	ROUT	;new local label scope
	MEND

	
	MACRO	;function return macro
	FUNCTION_RETURN
	bx	lr
	MEND

	
	
	MACRO	;vector in q0
	HQ_VECTOR_NORMALIZE_ASM_BLOCK
	vmul.f32		q1 , q0 , q0			 ;/*x^2  y^2 z^2 _*/
	vpadd.f32		d2 , d2 , d2			 ;/*x^2 + y^2		x^2 + y^2*/
	vmov.f32		s5 , s6					 ;/*d2 = x^2 + y^2			z^2*/
	vpadd.f32		d2 , d2 ,d2				 ;/*d2 = x^2 + y^2 + z^2        x^2 + y^2 + z^2*/
	
	vrsqrte.f32	d3 , d2					;/*d3 ~= 1 / sqrt(x^2 + y^2 + z^2)*/
	;/*Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2));y = 1/sqrt(x) ; reduce approximate error .*/
	;/*here d2 = x ; d3 = y(n)*/
	vmul.f32		d4 , d3 , d3			;/*y(n) ^ 2*/
	vrsqrts.f32	d4 , d2 , d4			;/*1/2  * (3 - x*y(n)^2))*/
	vmul.f32		d3 , d3 , d4			;/*1/2 *(y(n)*(3-x*y(n)^2))*/

	vmul.f32		q0 , q0 , d3[0]			;/*(x  , y  ,  z , w)/ length*/
	MEND	
	
	
	FUNCTION_BEGIN HQNeonVector4Dot	;float HQNeonVector4Dot(const float* v1 , const float* v2)
	

	vld1.32   {d0 , d1}, [r0	@128]	; v1.x v1.y v1.z v1.w
	vld1.32   {d2 , d3},[r1	@128]	; v2.x v2.y v2.z v2.w
					 
	vmul.f32 d0 , d0  , d2	; v1.x * v2.x		v1.y * v2.y	
	vadd.f32 s0 , s0 , s1	;(v1.x * v2.x + v1.y * v2.y)
	vmla.f32 s0 , s2 , s6	;(v1.x * v2.x + v1.y * v2.y) + (v1.z * v2.z) ;returned result in s0

	;old gcc inline asm
	;:=r (f)
	;:r (v1) , r (v2)  
	;:q0 , q1


	FUNCTION_RETURN	;HQNeonVector4Dot

	;=====================================
			
	FUNCTION_BEGIN HQNeonVector4Length	;float HQNeonVector4Length(const float* v)
	

	vld1.32   {d0 , d1} , [r0	@128]				; v1.x v1.y v1.z v1.w
	;add		r0, r0, #8
	;vld1.32		{d1[0]}   , [r0 @32]		; v1.z
				 
	vmul.f32		d0 , d0  , d0			; v1.x^2		v1.y^2	
	vmul.f32		d1 , d1  , d1			; v1.z^2		v1.w^2
	vpadd.f32		d0 , d0 , d0			;(v1.x^2 + v1.y^2)
	vadd.f32		d0 ,  d0 ,  d1			;(v1.x^2 + v1.y^2) + (v1.z^2)
	vsqrt.f32		s0 , s0					;length				;returned result in s0
	
	;old gcc inline asm			
	;:=r (f) 
	;:r (v)  
	;:d0 , d1
	;);

		
	FUNCTION_RETURN	;HQNeonVector4Length

	;=======================================
		
	FUNCTION_BEGIN	HQNeonVector4Cross	; void HQNeonVector4Cross(const float* v1 , const float* v2 , float *cross)


	vmov.i64	d5 , 0x0000000000000000			;d5 = 0 0
	vld1.32		{d1[1]}  , [r0 @32]				;x1
	add		r0 , r0 , #4		
	vld1.32	{d0}  , [r0]		;y1 z1
	vmov.f32  s2 , s1					;q0 = y1 z1 z1 x1
					 
	vld1.32		{d2[1]}  , [r1 @32]				;x2
	add		r1 , r1 , #4		
	vld1.32	{d3}  , [r1]		;y2 z2
	vmov.f32  s4 , s7					;q1 = z2 x2 y2 z2
					 
	vmul.f32	d4   , d0  , d2				;d4 = y1z2  z1x2
	vmls.f32  d4   , d1  , d3				;d4 = y1z2 - z1y2			z1x2 - x1z1
					 
	vmul.f32  s10  , s3  , s6				;x1y2
	vmls.f32  s10  , s0  , s5				;x1y2  - y1x2
					 
	vst1.32  {d4 , d5}   , [r2	@128]			
				
	;old gcc inline asm	 
	;:+r (v1) , +r (v2) 
	;: r (cross)   
	;:q0  , q1 , d4 , s10 , memory
	;);
		

	FUNCTION_RETURN	;HQNeonVector4Cross

	;=======================================
	
	FUNCTION_BEGIN HQNeonVector4Normalize	;void HQNeonVector4Normalize(const float* v , float *normalizedVec)


	vmov.i64		d1  , #0x0000000000000000					;d1 = 0	0
	vld1.32		{d0} , [r0	@64]			 	;load x y
	add			r0, r0, #8							
	vld1.32		{d1[0]}	,  [r0 @32]				;load z = > q0 = x y z 0
					 
	HQ_VECTOR_NORMALIZE_ASM_BLOCK
					 
	vst1.32		{d0 , d1} , [r1	@128]			;store x y z 0
	
	;old gcc inline asm				 
	;: 
	;:r  (v) , r (normalizedVec) 
	;:q0  , q1 , d4 , memory
	;);

	
	FUNCTION_RETURN	;HQNeonVector4Normalize	

	END
