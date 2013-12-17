;Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

;This program is free software; you can redistribute it and/or
;modify it under the terms of the GNU General Public License as
;published by the Free Software Foundation; either version 2 of the
;License, or (at your option) any later version.  See the file
;COPYING.txt included with this distribution for more information.

	AREA     HQNeonQuaternion, CODE, READONLY,  THUMB
	THUMB
	
	MACRO	;function definition macro
	;===================

	FUNCTION_BEGIN $name
	EXPORT $name
$name
$name.Label	ROUT	;new local label scope
	MEND

	
	MACRO	;function return macro
	FUNCTION_RETURN
	bx	lr
	MEND



	;===================

	FUNCTION_BEGIN HQNeonQuatMagnitude	;	float HQNeonQuatMagnitude(float * quat)
	

	vld1.32		{d0 , d1}  , [r0	@128]				
	vmul.f32		q0 , q0 , q0						;x^2 y^2 z^2 w^2
	vpadd.f32		d0 , d0 , d1						;x^2 + y^2			z^2 + w^2
	vadd.f32		s0 , s0 , s1						;x^2 + y^2 + z^2 + w^2
	vsqrt.f32		s0 , s0							;returned value in s0						
	
	;old gcc inline asm
	;:=r (re)
	;:r (quat)
	;:q0
	;);

		
	FUNCTION_RETURN	;HQNeonQuatMagnitude
	
	;===================

	FUNCTION_BEGIN HQNeonQuatSumSquares; float HQNeonQuatSumSquares(float * quat)
	

	vld1.32		{d0 , d1}  , [r0	@128]				
	vmul.f32		q0 , q0 , q0						;x^2 y^2 z^2 w^2
	vpadd.f32		d0 , d0 , d1						;x^2 + y^2			z^2 + w^2
	vadd.f32		s0 , s0 , s1						;x^2 + y^2 + z^2 + w^2   
	;returned value in s0						
	
	;old gcc inline asm
	;:=r (re)
	;:r (quat)
	;:q0
	;);
		

	FUNCTION_RETURN	;HQNeonQuatSumSquares

	;===================

	FUNCTION_BEGIN HQNeonQuatDot	;float HQNeonQuatDot(const float * quat1 ,const float *quat2)
	

	vld1.32		{d0 , d1}  , [r0	@128]				
	vld1.32		{d2 , d3}  , [r1	@128]				
	vmul.f32		q0 , q0 , q1						;x1.x2 y1.y2 z1.z2 w1.w2
	vpadd.f32		d0 , d0 , d1						;x1x2 + y1y2			z1z2 + w1w2
	vadd.f32		s0 , s0 , s1						;x1x2 + y1y2 + z1z2 + w1w2
	;returned value in s0							
	
	;old gcc inline asm
	;:=r (re)
	;:r (quat1) , r (quat2)
	;:q0 , q1
	;);
	
		
	FUNCTION_RETURN		;HQNeonQuatDot
	
	;===================

	FUNCTION_BEGIN HQNeonQuatNormalize	;void HQNeonQuatNormalize(const float* q , float *normalizedQuat)
	

	vld1.32		{d0 , d1} , [r0	@128]			 	;x y z w
					 
	vmul.f32		q1 , q0 , q0				 	;x^2  y^2 z^2 w^2
	vpadd.f32		d2 , d2 , d3				 	;x^2 + y^2		z^2 + w^2
	vpadd.f32		d2 , d2 ,d2					 	;d2 = x^2 + y^2 + z^2 + w^2     x^2 + y^2 + z^2 + w^2
					 
	vrsqrte.f32	d3 , d2							;d3 ~= 1 / sqrt(x^2 + y^2 + z^2 + w^2)
		;Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2));y = 1/sqrt(x) ; reduce approximate error .
		;here d2 = x ; d3 = y(n)
	vmul.f32		d4 , d3 , d3					;y(n) ^ 2
	vrsqrts.f32		d4 , d2 , d4				;1/2  * (3 - x*y(n)^2))
	vmul.f32		d3 , d3 , d4					;1/2 *(y(n)*(3-x*y(n)^2))
					 
	vmul.f32		q0 , q0 , d3[0]					;(x  , y  ,  z , w)/ length
					 
	vst1.32		{d0 , d1} , [r1	@128]			
	
	;old gcc inline asm				 
	;:
	;:r  (q) ,r (normalizedQuat) 
	;:q0  , q1 , d4 , memory
	;);

		
	FUNCTION_RETURN		;HQNeonQuatNormalize
	
	;===================

	FUNCTION_BEGIN HQNeonQuatInverse	;void HQNeonQuatInverse(const float* q , float *result)
	

	vld1.32		{d0 , d1} , [r0	@128]		 	;x y z w
	vneg.f32		d0 , d0							;-x -y
	vneg.f32		s2 , s2							;-z
					 
	vmul.f32		q1 , q0 , q0				 	;x^2  y^2 z^2 w^2
	vpadd.f32		d2 , d2 , d3				 	;x^2 + y^2		z^2 + w^2
	vpadd.f32		d2 , d2 ,d2					 	;d2 = x^2 + y^2 + z^2 + w^2     x^2 + y^2 + z^2 + w^2
					 
	vrecpe.f32	d3 , d2							;d3 ~= 1 / (x^2 + y^2 + z^2 + w^2)
		;Newton_Raphson iteration Y(n+1)=2*Y(n)-x*Y(n)^2 =  Y(n) * (2 - x * Y (n)) ; reduce estimate error .
		;here d2 = x ; d3 = y(n)
	vrecps.f32	d4 , d2 , d3					;2 - x * Y (n)
	vmul.f32		d3 , d3 , d4					;Y(n) * (2 - x * Y (n))
					 
	vmul.f32		q0 , q0 , d3[0]					;(-x  , -y  ,  -z , w) * d3[0]
					 
	vst1.32		{d0 , d1} , [r1	@128]			
	
	;old gcc inline asm				 
	;:
	;:r  (q) ,r (result) 
	;:q0  , q1 , d4 , memory
	;);

		
	FUNCTION_RETURN		;HQNeonQuatInverse
	
	;===================

	FUNCTION_BEGIN HQNeonQuatMultiply	;void HQNeonQuatMultiply(const float * quat1 ,const  float *quat2 , float* result)
	
	vpush {q4-q5}	;save registers

	vld1.32		{d2 , d3} , [r1	@128]			 	;q2 = x2 y1 z2 w2
	vld1.32		{d0 , d1} , [r0	@128]			 	;q1 = x1 y1 z1 w1
					 
	vmov			d5 , d2						
	vmov			d4 , d3							;q2 = z2 w2 x2 y2
	vrev64.32		q3 , q2							;q3 = w2 z2 y2 x2
					 
					 
	vmul.f32		q5 , q3 , d0[0]					;q5 = x1w2    x1z2    x1y2    x1x2
	vmul.f32		q4 , q1 , d1[1]					;q4 = w1x2	w1y2	w2z2	w1w2
	vswp			d6 , d7							;q3 = y2 x2 w2 z2
	vmul.f32		q3 , q3 , d1[0]					;q3 = z1y2	z1x2	z1w2	z1z2
	vmla.f32		d8 , d4 , d0[1]					;d8 = w1x1 + y1z2     w1x2 + y1w2
	vmls.f32		d9 , d5 , d0[1]					;d9 = w1z2 - y1x2		w1w2 - y1y2
	vneg.f32		s21 , s21						;q5 = x1w2    -x1z2    x1y2    x1x2
	vneg.f32		s23 , s23						;q5 = x1w2    -x1z2    x1y2    -x1x2
	vadd.f32		q4 , q4 , q5				
	vneg.f32		s12 , s12						;q3 = -z1y2	z1x2	z1w2	z1z2
	vneg.f32		s15 , s15						;q3 = -z1y2	z1x2	z1w2	-z1z2
	vadd.f32		q4 , q4 , q3				
					 
	vst1.32		{d8 , d9} , [r2	@128]			
	
	;old gcc inline asm
	;:
	;:r(quat1) , r (quat2) , r(result)
	;:q0 , q1 , q2 , q3 , q4 , q5 , memory				 
	;);
	
	vpop {q4-q5}	;restore registers

	FUNCTION_RETURN		;HQNeonQuatMultiply
	
	;===================

	IMPORT HQNeonVector4Normalize

	FUNCTION_BEGIN HQNeonQuatUnitToRotAxis	;void HQNeonQuatUnitToRotAxis(const float* q , float *axisVector)
	
	push lr;
	bl		HQNeonVector4Normalize	;	call HQNeonVector4Normalize(q, axisVector);
	pop lr;

	FUNCTION_RETURN	;HQNeonQuatUnitToRotAxis
	
	;===================

	FUNCTION_BEGIN HQNeonQuatUnitToMatrix3x4c	;void HQNeonQuatUnitToMatrix3x4c(const float* q , float *matrix)
	
	vpush {q4}	;save registers
	vpush {d10}	;save registers

	vld1.32		{d0 , d1} , [r0	@128]				 	;x y z w
	vmov.i64		d7 , #0x0000000000000000			;d7 = 0 0
	vmov.f32		s13 , #1.0							;s13 = 1
					 
	vadd.f32		q1 , q0 , q0						;q1 = d2{2x 2y} d3{2z 2w}
	vrev64.32		d8 , d0								;d8 = y x
					 
	vmul.f32		d4 , d8 , d2[1]						;d4 = 2yy  2xy 
	vmul.f32		d9 , d8 , d3[1]						;d9 = 2yw  2xw
	vmul.f32		d10 , d0 , d2						;d10 = 2xx 2yy
	vmul.f32		d5 , d0 , d3[0]						;d5 = 2xz  2yz ; q2 = d4{2yy  2xy}	d5{2xz  2yz}
	vsub.f32		s14 , s13 , s20						;s14 = 1 - 2xx
	vmul.f32		d8 , d1 , d3[0]						;d8 = 2zz  2zw ; q4 = d8{2zz	2zw}	d9{2yw	2xw}
					 
	vsub.f32		s14 , s14 , s21						;s14 = 1 - 2xx - 2yy ; d7 = 1-2xx-2yy	0.0
					 
		;first row
	vneg.f32		d5 , d5							;q2 = 2yy 2xy -2xz -2yz
	vneg.f32		d8 , d8							;q4 = -2zz -2zw 2yw 2xw
	vneg.f32		s9 , s9							;q2 = 2yy -2xy -2xz -2yz
	vadd.f32		s16 , s16 , s13					;q4 = 1-2zz	-2zw	2yw		2xw
	vsub.f32		q0 , q4 , q2					;q0 = -2zz-2yy+1 -2zw+2xy 2yw+2xz 2xw+2yz 	;first row (except last element)
					 
	vst1.32		{d0} , [r1	@64]!					;11 12
	vst1.32		{d1[0]}  , [r1 @32]	!					;13
	;fsts			s15 , [r1 , #4]					;store 0.0 to 14   pre UAL code
	vst1.32		{d7[1]}, [r1 @32]	!					;store 0.0 to 14
					 
					 
		;second row
	vmov.f32			s8 , s20						;q2 = 2xx -2xy -2xz -2yz
	vneg.f32		d9 , d9							;q4 = -2zz+1, -2zw, -2yw, -2xw
	vneg.f32		s17 , s17						;q4 = -2zz+1, 2zw, -2yw, -2xw
	vsub.f32		q4 , q4 , q2					;q4 = -2zz-2xx+1 2xy+2zw -2yw+2xz -2xw+2yz
	vrev64.32		q1 , q4							;q1 = 2xy+2zw -2zz-2xx+1 -2xw+2yz -2yw+2xz	;second row (except last element)
					 
	vst1.32		{d2} , [r1	@64]!					;21	22
	;fsts			s6  , [r1]						;23 pre UAL
	;fsts			s15 , [r1 , #4]					;store 0.0 to 24 pre UAL
	vst1.32			{d3[0]}, [r1 @32] !				;23
	vst1.32			{d7[1]}, [r1 @32] !			;store 0.0 to 24
					 
		;third row
	vtrn.32		d3 , d1							;d1 =	-2yw+2xz	2xw+2yz ; d7 = 1-2xx-2yy	0.0
	vst1.32		{d1 }	,[r1	@64]!					;31 32
	vst1.32		{d7}	,[r1	@64]					;33 34
	
	;old gcc inline asm				 
	;:+r (matrix)
	;:r (q)
	;:q0 , q1 , q2 , s13 , d7 , q4 , d10 , memory
	;);
	
	vpop {d10}	;restore registers
	vpop {q4}	;restore registers

	FUNCTION_RETURN	;HQNeonQuatUnitToMatrix3x4c
	
	
	;===================

	FUNCTION_BEGIN HQNeonQuatUnitToMatrix4r	;void HQNeonQuatUnitToMatrix4r(const float* q , float *matrix)
	
	vpush {q4-q5}	;save registers

	vld1.32		{d0 , d1} , [r0	@128]				 	;x y z w
	vmov.i64		d7 , #0x0000000000000000			;d7 = 0 0
	vmov.i64		d9 , #0x0000000000000000			;d9 = 0 0
	vmov.f32		s19 , #1.0							;s19 = 1 ; d9 = 0 1
					 
	vadd.f32		q1 , q0 , q0						;q1 = d2{2x 2y} d3{2z 2w}
	vrev64.32		d10 , d0							;d10 = y x
					 
	vmul.f32		d4 , d10 , d2[1]					;d4 = 2yy  2xy 
	vmul.f32		d11 , d10, d3[1]					;d11 = 2yw  2xw
	vmul.f32		d8 , d0 , d2						;d8 = 2xx 2yy
	vmul.f32		d5 , d0 , d3[0]						;d5 = 2xz  2yz ; q2 = d4{2yy  2xy}	d5{2xz  2yz}
	vmul.f32		d10 , d1 , d3[0]					;d10 = 2zz  2zw ; q5 = d10{2zz	2zw}	d11{2yw	2xw}
					 
	vsub.f32		s14 , s19 , s16						;s14 = 1 - 2xx
	vsub.f32		s14 , s14 , s17						;s14 = 1 - 2xx - 2yy ; d7 = 1-2xx-2yy	0.0
					 
		;first column
	vneg.f32		d5 , d5							;q2 = 2yy 2xy -2xz -2yz
	vneg.f32		d10 , d10						;q5 = -2zz -2zw 2yw 2xw
	vneg.f32		s9 , s9							;q2 = 2yy -2xy -2xz -2yz
	vadd.f32		s20 , s20 , s19					;q5 = 1-2zz	-2zw	2yw		2xw
	vsub.f32		q1 , q5 , q2					;q1 = -2zz-2yy+1 -2zw+2xy 2yw+2xz 2xw+2yz 	;first column (except last element)
					 
		;second column
	vmov.f32			s8 , s16						;q2 = 2xx -2xy -2xz -2yz
	vmov.i64		d8 , #0x0000000000000000		;q4 = 0 0 0 1
	vneg.f32		d11 , d11						;q5 = -2zz+1, -2zw, -2yw, -2xw
	vneg.f32		s21 , s21						;q5 = -2zz+1, 2zw, -2yw, -2xw
	vsub.f32		q5 , q5 , q2					;q5 = -2zz-2xx+1 2xy+2zw -2yw+2xz -2xw+2yz
	vrev64.32		q2 , q5							;q2 = 2xy+2zw -2zz-2xx+1 -2xw+2yz -2yw+2xz	;second column (except last element)
					 
		;third column
	vmov			d6 , d3							;d6 = 2yw+2xz 2xw+2yz
	vtrn.32		d5 , d6							;q3 : d6 =	-2yw+2xz	2xw+2yz ; d7 = 1-2xx-2yy	0.0
					 
	vmov.f32			s7 , s15						;q1 =  -2zz-2yy+1 -2zw+2xy 2yw+2xz 0
	vmov.f32			s11 , s15						;q2 =  2xy+2zw -2zz-2xx+1 -2xw+2yz 0
					 
		;store transpose of {q1 , q2 , q3 , q4}
	vst4.32		{d2 , d4 , d6 , d8}	,[r1	@128]!			
	vst4.32		{d3 , d5 , d7 , d9}	,[r1	@128]			
	
	;old gcc inline asm				 
	;:+r (matrix)
	;:r (q)
	;:q0 , q1 , q2 , q3 , q4 , q5 , memory
	;);
	
	vpop {q4-q5}	;restore registers

	FUNCTION_RETURN	;HQNeonQuatUnitToMatrix4r

	;===================
	
	;static const hq_uint32 SIMD_DW_mat2quatShuffle0  = (12<<0)|(8<<8)|(4<<16)|(0<<24) ;
	;static const hq_uint32 SIMD_DW_mat2quatShuffle1  = (0<<0)|(4<<8)|(8<<16)|(12<<24) ;
	;static const hq_uint32 SIMD_DW_mat2quatShuffle2  = (4<<0)|(0<<8)|(12<<16)|(8<<24) ; 
	;static const hq_uint32 SIMD_DW_mat2quatShuffle3  = (8<<0)|(12<<8)|(0<<16)|(4<<24) ;
	;static const hq_uint32 NegSignMask = 0x80000000;

	GBLA	SIMD_low_mat2quatShuffle0;
	GBLA	SIMD_high_mat2quatShuffle0;
	GBLA	SIMD_low_mat2quatShuffle1;
	GBLA	SIMD_high_mat2quatShuffle1;
	GBLA	SIMD_low_mat2quatShuffle2;
	GBLA	SIMD_high_mat2quatShuffle2;
	GBLA	SIMD_low_mat2quatShuffle3;
	GBLA	SIMD_high_mat2quatShuffle3;

SIMD_low_mat2quatShuffle0 SETA	((12<<0):OR:(8<<8))	
SIMD_high_mat2quatShuffle0 SETA	((4<<0):OR:(0<<8))	

SIMD_low_mat2quatShuffle1 SETA	((0<<0):OR:(4<<8))	
SIMD_high_mat2quatShuffle1 SETA	((8<<0):OR:(12<<8))
	
SIMD_low_mat2quatShuffle2 SETA	((4<<0):OR:(0<<8))	
SIMD_high_mat2quatShuffle2 SETA	((12<<0):OR:(8<<8))	

SIMD_low_mat2quatShuffle3 SETA	((8<<0):OR:(12<<8))	
SIMD_high_mat2quatShuffle3 SETA	((0<<0):OR:(4<<8))	

	
	;===================

	FUNCTION_BEGIN HQNeonQuatFromMatrix3x4c	;void HQNeonQuatFromMatrix3x4c(const float *matrix , float * quaternion)
	
	push {r4}			;save registers
	push {r5}			;save registers
	vpush {q4-q7}	;save registers

	vmov.i64		d6 , #0x0000000000000000								;d6 = 0 0
	vld1.f32		{d0 , d1} , [r0	@128] !										;11 12 13 14
	vld1.f32		{d2 , d3} , [r0	@128] !										;21 22 23 24
	vld1.f32		{d4 , d5} , [r0	@128]										;31 32 33 34
	vrev64.32		d2 , d2													;d2 = 22 21
					 
	vcge.f32		d7 , d0 , d2											;d7[0] = 11 >= 22 ?
	vcge.f32		d8 , d0 , d5											;d8[0] = 11 >= 33 ?
	vadd.f32		s12 , s0 , s4											;s12 = 11 + 22
	vadd.f32		s12 , s12 , s10											;s12 = 11 + 22 + 33
	vcge.f32		d6 , d6 , #0											;d6[0] = (11 + 22 + 33) >= 0?
	vand			d8 , d7 , d8											;d8[0] = 11 is max?
					 
	vcge.f32		d9 , d2 , d5											;d9[0] = 22 >= 33 ?
	vbic			d7 , d8 , d6											;d7[0] = (max == 11 && !((11 + 22 + 33) >= 0))
	vorr			d10 , d8 , d6											;d10[0] = ((11 + 22 + 33) >= 0 || max = 11)?
					 
	vbic			d8 , d9 , d10											;d8[0] =!((11 + 22 + 33) >= 0 || max = 11) && 22 >= 33 ?
	vorr			d10 , d9 , d10											;d10[0] =((11 + 22 + 33) >= 0 || max = 11) || 22 >= 33
	vmvn			d9 , d10												;d9[0] = !d10[0]
	
	;load SIMD_DW_mat2quatShuffle0
	mov				r3,	  #(SIMD_low_mat2quatShuffle0)						;low order part
	mov				r4,	  #(SIMD_high_mat2quatShuffle0)						;high order part
	lsl				r4,	  r4, #16											;r4 = r4 << 16
	orr				r3,	  r4												;r3 = r3 | r4
	vmov			s22,  r3

	;load SIMD_DW_mat2quatShuffle1
	mov				r3,	  #(SIMD_low_mat2quatShuffle1)						;low order part
	mov				r4,	  #(SIMD_high_mat2quatShuffle1)						;high order part
	lsl				r4,	  r4, #16											;r4 = r4 << 16
	orr				r3,	  r4												;r3 = r3 | r4
	vmov			s24,  r3

	;load SIMD_DW_mat2quatShuffle2
	mov				r3,	  #(SIMD_low_mat2quatShuffle2)						;low order part
	mov				r4,	  #(SIMD_high_mat2quatShuffle2)						;high order part
	lsl				r4,	  r4, #16											;r4 = r4 << 16
	orr				r3,	  r4												;r3 = r3 | r4
	vmov			s26,  r3

	;load SIMD_DW_mat2quatShuffle3
	mov				r3,	  #(SIMD_low_mat2quatShuffle3)						;low order part
	mov				r4,	  #(SIMD_high_mat2quatShuffle3)						;high order part
	lsl				r4,	  r4, #16											;r4 = r4 << 16
	orr				r3,	  r4												;r3 = r3 | r4
	vmov			s28,  r3
		
	;doesn't work because intermediate value cannot be too large			 
	;vmov			s22 , #(SIMD_DW_mat2quatShuffle0)					
	;vmov			s24	, #(SIMD_DW_mat2quatShuffle1)					
	;vmov			s26	, #(SIMD_DW_mat2quatShuffle2)					
	;vmov			s28	, #(SIMD_DW_mat2quatShuffle3)					
					 
	vand			d6 , d6 , d11										
	vand			d10 , d7 , d12										
	vorr			d6 , d6 , d10										
	vand			d10 , d8 , d13										
	vorr			d6 , d6 , d10										
	vand			d10 , d9 , d14										
	vorr			d10 , d6 , d10										
					 
	vmov			r4 , s20											
	
	; load sign mask (0x80000000)
	mov				r3	, #1												;0x1
	lsl				r3	, r3 , #31											;r3 = 0x1 << 31 = 0x80000000							 
	vmov			s22 , r3								
	vorr			d6 , d8 , d9											;(max = 22 || max = 33) && sum < 0
	vorr			d8 , d7 , d8											;(max = 11 || max = 22) && sum < 0
	vorr			d7 , d7 , d9											;(max = 11 || max = 33) && sum < 0
					 
	vand			d6 , d6 , d11											;d6[0] = s0
	vand			d7 , d7 , d11											;d7[0] = s1
	vand			d8 , d8 , d11											;d8[0] = s2
	vdup.32		d6 , d6[0]											
	vdup.32		d7 , d7[0]											
	vdup.32		d8 , d8[0]											
					 
	veor			d9 , d6 , d0											;s0 ^ 11 (^ is xor)
	veor			d10 , d7 , d2											;s1 ^ 22 (^ is xor)
	veor			d11 , d8 , d5											;s2 ^ 33 (^ is xor)
	vmov.f32			s19 , s20												;d9[1] = d10[0]
					 
	vmov.f32		s23 , #1.0												;d11[1] = 1
	vpadd.f32		d9 , d9 , d11											;d9 = s0 ^ 11 + s1 ^ 22			s2 ^ 33 + 1.0f
	vpadd.f32		d9 , d9 , d9											;d9 = s0 ^ 11 + s1 ^ 22 + s2 ^ 33 + 1.0f = t
					 
	vmov.f32		s24 , #0.5											
					 
	vrsqrte.f32	d10 , d9											;/*d10 ~= 1 / sqrt(t)*/
	;/*Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2));y = 1/sqrt(x) ; reduce approximate error .*/
	;/*here d9 = x ; d10 = y(n)*/
	vmul.f32		d11 , d10 , d10										;/*y(n) ^ 2*/
	vrsqrts.f32	d11 , d9 , d11										;/*1/2  * (3 - x*y(n)^2))*/
	vmul.f32		d10 , d10 , d11										;/*1/2 *(y(n)*(3-x*y(n)^2))*/
					 
	vmul.f32		d10 , d10 , d12[0]										;d10 = 0.5 / sqrt(t) = s
	vmul.f32		d11 , d9 , d10											;d11 = s * t
					 
		;q[k0]
	and			r5 , r4 , #0xff											;r5 = k0 * 4 = r4 & 0xff
	add			r5 , r5 , r1								;r5 = &q[k0]
	;fsts			s22	, [r5]												;q[k0] = s * t preUAL
	vst1.32			{d11[0]}, [r5 @32]											;q[k0] = s * t 
					 
					 
		;q[k1]
	veor			d9 , d0 , d8											;d9[1] = s2 ^ 12
	vsub.f32		d9 , d2 , d9											;d9[1] = 21 - s2 ^ 12
	vmul.f32		d9 , d9 , d10											;d9[1] = (21 - s2 ^ 12) * s
					 
	and			r5 , r4 , #0xff00										;r5 = r4 & 0xff00
	add			r5 , r1	, r5, LSR #8						;r5 = &q[k1] = q + (r5 >> 8)
	;fsts			s19	, [r5]												;q[k1] = (21 - s2 ^ 12) * s preUAL
	vst1.32			{d9[1]}	, [r5 @32]												;q[k1] = (21 - s2 ^ 12) * s
					 
		;q[k2]
	veor			d9 , d4 , d7											;d9[0] = s1 ^ 31
	vsub.f32		d9 , d1 , d9											;d9[0] = 13 - s1 ^ 31
	vmul.f32		d9 , d9 , d10											;d9[0] = (13 - s1 ^ 31) * s
					 
	and			r5 , r4 , #0xff0000										;r5 = (r4) & 0xff0000
	add			r5 , r1	, r5 ,LSR #16						;r5 = &q[k2] = q + (r5 >> 16)
	;fsts			s18	, [r5]												;q[k2] = (13 - s1 ^ 31) * s pre UAL
	vst1.32			{d9[0]}	, [r5 @32]												;q[k2] = (13 - s1 ^ 31) * s
					 
		;q[k3]
	veor			d9 , d3 , d6											;d9[0] = s0 ^ 23
	vsub.f32		s18 , s9 , s18											;d9[0] = 32 - s0 ^ 23
	vmul.f32		d9 , d9 , d10											;d9[0] = (32 - s0 ^ 23) * s
					 
	add			r5 , r1, r4 , LSR #24						;r5 = &q[k3] = q + (r4 >> 24)
	;fsts			s18	, [r5]												;q[k3] = (32 - s0 ^ 23) * s ; pre UAL
	vst1.32			{d9[0]}	, [r5 @32]												;q[k3] = (32 - s0 ^ 23) * s
	
	;old gcc inline asm				 
	;:	+r(matrix)
	;:	[SIMD_DW_mat2quatShuffle0]r (SIMD_DW_mat2quatShuffle0),
	;[SIMD_DW_mat2quatShuffle1]r (SIMD_DW_mat2quatShuffle1),
	;[SIMD_DW_mat2quatShuffle2]r (SIMD_DW_mat2quatShuffle2),
	;[SIMD_DW_mat2quatShuffle3]r (SIMD_DW_mat2quatShuffle3),
	;[NegSignMask]r (NegSignMask),
	;[quatPointer]r (quaternion)
	;:r4 , r5 ,q0 , q1 , q2 , q3 , q4 , q5 , q6 , q7, memory
	;);
	

	vpop {q4-q7}	;restore registers
	pop {r5}			;restore registers
	pop {r4}			;restore registers

	FUNCTION_RETURN	;HQNeonQuatFromMatrix3x4c


	;===================

	FUNCTION_BEGIN HQNeonQuatFromMatrix4r	;void HQNeonQuatFromMatrix4r(const float *matrix , float * quaternion)
	
	push {r4}			;save registers
	push {r5}			;save registers
	vpush {q4-q7}	;save registers

	vmov.i64		d6 , #0x0000000000000000								;d6 = 0 0
	vmov.i64		d11 , #0xffffffffffffffff								;
	vld1.f32		{d0 , d1} , [r0	@128] !								;11 12 13 14
	vld1.f32		{d2 , d3} , [r0	@128] !								;21 22 23 24
	vld1.f32		{d4 , d5} , [r0	@128]									;31 32 33 34
	vrev64.32		d2 , d2													;d2 = 22 21
					 
	vcge.f32		d7 , d0 , d2											;d7[0] = 11 >= 22 ?
	vcge.f32		d8 , d0 , d5											;d8[0] = 11 >= 33 ?
	vadd.f32		s12 , s0 , s4											;s12 = 11 + 22
	vadd.f32		s12 , s12 , s10											;s12 = 11 + 22 + 33
	vcge.f32		d6 , d6 , #0											;d6[0] = (11 + 22 + 33) >= 0?
	vand			d8 , d7 , d8											;d8[0] = 11 is max?
					 
	vcge.f32		d9 , d2 , d5											;d9[0] = 22 >= 33 ?
	vbic			d7 , d8 , d6											;d7[0] = (max == 11 && !((11 + 22 + 33) >= 0))
	vorr			d10 , d8 , d6											;d10[0] = ((11 + 22 + 33) >= 0 || max = 11)?
					 
	vbic			d8 , d9 , d10											;d8[0] =!((11 + 22 + 33) >= 0 || max = 11) && 22 >= 33 ?
	vorr			d9 , d9 , d10											;d9[0] =((11 + 22 + 33) >= 0 || max = 11) || 22 >= 33
	veor			d9 , d9 , d11											;d10[0] = !d9[0]
					 
	;load SIMD_DW_mat2quatShuffle0
	mov				r3,	  #(SIMD_low_mat2quatShuffle0)						;low order part
	mov				r4,	  #(SIMD_high_mat2quatShuffle0)						;high order part
	lsl				r4,	  r4, #16											;r4 = r4 << 16
	orr				r3,	  r4												;r3 = r3 | r4
	vmov			s22,  r3

	;load SIMD_DW_mat2quatShuffle1
	mov				r3,	  #(SIMD_low_mat2quatShuffle1)						;low order part
	mov				r4,	  #(SIMD_high_mat2quatShuffle1)						;high order part
	lsl				r4,	  r4, #16											;r4 = r4 << 16
	orr				r3,	  r4												;r3 = r3 | r4
	vmov			s24,  r3

	;load SIMD_DW_mat2quatShuffle2
	mov				r3,	  #(SIMD_low_mat2quatShuffle2)						;low order part
	mov				r4,	  #(SIMD_high_mat2quatShuffle2)						;high order part
	lsl				r4,	  r4, #16											;r4 = r4 << 16
	orr				r3,	  r4												;r3 = r3 | r4
	vmov			s26,  r3

	;load SIMD_DW_mat2quatShuffle3
	mov				r3,	  #(SIMD_low_mat2quatShuffle3)						;low order part
	mov				r4,	  #(SIMD_high_mat2quatShuffle3)						;high order part
	lsl				r4,	  r4, #16											;r4 = r4 << 16
	orr				r3,	  r4												;r3 = r3 | r4
	vmov			s28,  r3
		
	;doesn't work because intermediate value cannot be too large			 
	;vmov			s22 , #(SIMD_DW_mat2quatShuffle0)					
	;vmov			s24	, #(SIMD_DW_mat2quatShuffle1)					
	;vmov			s26	, #(SIMD_DW_mat2quatShuffle2)					
	;vmov			s28	, #(SIMD_DW_mat2quatShuffle3)					
					 
	vand			d6 , d6 , d11										
	vand			d10 , d7 , d12										
	vorr			d6 , d6 , d10										
	vand			d10 , d8 , d13										
	vorr			d6 , d6 , d10										
	vand			d10 , d9 , d14										
	vorr			d10 , d6 , d10										
					 
	vmov			r4 , s20											
	
	; load sign mask (0x80000000)
	mov				r3	, #1												;0x1
	lsl				r3	, r3 , #31											;r3 = 0x1 << 31 = 0x80000000							 
	vmov			s22 , r3											
	vorr			d6 , d8 , d9											;(max = 22 || max = 33) && sum < 0
	vorr			d8 , d7 , d8											;(max = 11 || max = 22) && sum < 0
	vorr			d7 , d7 , d9											;(max = 11 || max = 33) && sum < 0
					 
	vand			d6 , d6 , d11											;d6[0] = s0
	vand			d7 , d7 , d11											;d7[0] = s1
	vand			d8 , d8 , d11											;d8[0] = s2
	vdup.32		d6 , d6[0]											
	vdup.32		d7 , d7[0]											
	vdup.32		d8 , d8[0]											
					 
	veor			d9 , d6 , d0											;s0 ^ 11 (^ is xor)
	veor			d10 , d7 , d2											;s1 ^ 22 (^ is xor)
	veor			d11 , d8 , d5											;s2 ^ 33 (^ is xor)
	vmov.f32			s19 , s20												;d9[1] = d10[0]
					 
	vmov.f32		s23 , #1.0												;d11[1] = 1
	vpadd.f32		d9 , d9 , d11											;d9 = s0 ^ 11 + s1 ^ 22			s2 ^ 33 + 1.0f
	vpadd.f32		d9 , d9 , d9											;d9 = s0 ^ 11 + s1 ^ 22 + s2 ^ 33 + 1.0f = t
					 
	vmov.f32		s24 , #0.5											
					 
	vrsqrte.f32	d10 , d9											;/*d10 ~= 1 / sqrt(t)*/
	;/*Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2));y = 1/sqrt(x) ; reduce approximate error .*/
	;/*here d9 = x ; d10 = y(n)*/
	vmul.f32		d11 , d10 , d10										;/*y(n) ^ 2*/
	vrsqrts.f32	d11 , d9 , d11										;/*1/2  * (3 - x*y(n)^2))*/
	vmul.f32		d10 , d10 , d11										;/*1/2 *(y(n)*(3-x*y(n)^2))*/
					 
	vmul.f32		d10 , d10 , d12[0]										;d10 = 0.5 / sqrt(t) = s
	vmul.f32		d11 , d9 , d10											;d11 = s * t
					 
		;q[k0]
	and			r5 , r4 , #0xff											;r5 = k0 * 4
	add			r5 , r5 , r1								;r5 = &q[k0]
	;fsts			s22	, [r5]												;q[k0] = s * t preUAL
	vst1.32			{d11[0]}	, [r5 @32]												;q[k0] = s * t
					 
					 
		;q[k1]
	veor			d9 , d2 , d8											;d9[1] = s2 ^ 21
	vsub.f32		d9 , d0 , d9											;d9[1] = 12 - s2 ^ 21
	vmul.f32		d9 , d9 , d10											;d9[1] = (12 - s2 ^ 21) * s
					 
	and			r5 , r4 , #0xff00										;r5 = r4 & 0xff00
	add			r5 , r1	, r5, LSR #8						;r5 = &q[k1] = q + (r5 >> 8)
	;fsts			s19	, [r5]												;q[k1] = (12 - s2 ^ 21) * s pre UAL
	vst1.32			{d9[1]}	, [r5 @32]												;q[k1] = (12 - s2 ^ 21) * s
					 
		;q[k2]
	veor			d9 , d1 , d7											;d9[0] = s1 ^ 13
	vsub.f32		d9 , d4 , d9											;d9[0] = 31 - s1 ^ 13
	vmul.f32		d9 , d9 , d10											;d9[0] = (31 - s1 ^ 13) * s
					 
	and			r5 , r4 , #0xff0000										;r5 = (r4) & 0xff0000
	add			r5 , r1	, r5 ,LSR #16						;r5 = &q[k2] = q + (r5 >> 16)
	;fsts			s18	, [r5]												;q[k2] = (31 - s1 ^ 13) * s pre UAL
	vst1.32			{d9[0]}	, [r5 @32]												;q[k2] = (31 - s1 ^ 13) * s
					 
		;q[k3]
	veor			d9 , d4 , d6											;d9[1] = s0 ^ 32
	vsub.f32		s19 , s6 , s19											;d9[1] = 23 - s0 ^ 32
	vmul.f32		d9 , d9 , d10											;d9[1] = (23 - s0 ^ 32) * s
					 
	add			r5 , r1, r4 , LSR #24						;r5 = &q[k3] = q + (r4 >> 24)
	;fsts			s19	, [r5]												;q[k3] = (23 - s0 ^ 32) * s pre UAL
	vst1.32			{d9[1]}	, [r5 @32]												;q[k3] = (23 - s0 ^ 32) * s
		
	;old gcc inline asm			 
	;:	+r(matrix)
	;:	[SIMD_DW_mat2quatShuffle0]r (SIMD_DW_mat2quatShuffle0),
	;[SIMD_DW_mat2quatShuffle1]r (SIMD_DW_mat2quatShuffle1),
	;[SIMD_DW_mat2quatShuffle2]r (SIMD_DW_mat2quatShuffle2),
	;[SIMD_DW_mat2quatShuffle3]r (SIMD_DW_mat2quatShuffle3),
	;[NegSignMask]r (NegSignMask),
	;[quatPointer]r (quaternion)
	;:r4 , r5 ,q0 , q1 , q2 , q3 , q4 , q5 , q6 , q7, memory
	;	);
	

	vpop {q4-q7}	;restore registers
	pop {r5}			;restore registers
	pop {r4}			;restore registers

	FUNCTION_RETURN	;HQNeonQuatFromMatrix4r
	
	;===================

	FUNCTION_BEGIN HQNeonQuatAddImpl		;void HQNeonQuatAdd(const hqfloat32 *quat1, const hqfloat32 *quat2, hqfloat32 *result)
	

	vld1.32	{d0 , d1} , [r0	@128]			
	vld1.32	{d2 , d3} , [r1	@128]			
	vadd.f32	q2 , q0 , q1				
	vst1.32	{d4 , d5} , [r2	@128]			

	;old gcc inline asm
	;:
	;:r(quat1) , r (quat2), r (result)
	;:q0 , q1 , q2 , memory
	;);


	FUNCTION_RETURN ;HQNeonQuatAddImpl

	;==========================================

	FUNCTION_BEGIN HQNeonQuatSubImpl	;void HQNeonQuatSub(const hqfloat32 *quat1, const hqfloat32 *quat2, hqfloat32 *result)
	

	vld1.32	{d0 , d1} , [r0	@128]			
	vld1.32	{d2 , d3} , [r1	@128]			
	vsub.f32	q2 , q0 , q1				
	vst1.32	{d4 , d5} , [r2	@128]			
				 
	;old gcc inline asm
	;:
	;:r(quat1) , r (quat2), r (result)
	;:q0 , q1 , q2 , memory
	;	);
	

	FUNCTION_RETURN ;HQNeonQuatSubImpl

	;==============================================

	FUNCTION_BEGIN HQNeonQuatMultiplyScalarImpl	;void HQNeonQuatMultiplyScalar(const hqfloat32 *quat1, hqfloat32 f, hqfloat32 *result)
	;r0 = quat1 , r1 = quat2, s0 = d0[0] = f


	vld1.32	{d2 , d3} , [r0	@128]			
	vmul.f32	q2 , q1 , d0[0]					
	vst1.32	{d4 , d5} , [r1	@128]			
				 
	;old gcc inline asm
	;:
	;:r(quat1) , r (f), r (result)
	;:q0 , q1 , q2 , memory
	;);


	FUNCTION_RETURN	;HQNeonQuatMultiplyScalarImpl

	END

