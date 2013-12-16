	AREA     HQNeonMatrix, CODE, READONLY, THUMB
	THUMB
	
	MACRO	;function definition macro
	FUNCTION_BEGIN $name
	EXPORT $name
$name
;$name.Label	ROUT	;new local label scope
	MEND

	
	MACRO	;function return macro
	FUNCTION_RETURN
	bx	lr
	MEND

	MACRO	;matrix1 (q4 - q7)  matrix2 (q8 - q11)
	NEON_MATRIX_MUL_ASM_BLOCK1 
	vmul.f32		q0  , q8 , d8[0]			
	vmul.f32		q1  , q8 , d10[0]			
	vmul.f32		q2  , q8 , d12[0]			
	vmul.f32		q3  , q8 , d14[0]			
	
	vmla.f32		q0 ,  q9 , d8[1]			
	vmla.f32		q1 ,  q9 , d10[1]			
	vmla.f32		q2 ,  q9 , d12[1]			
	vmla.f32		q3 ,  q9 , d14[1]			
	
	vmla.f32		q0 ,  q10 , d9[0]			
	vmla.f32		q1 ,  q10 , d11[0]			
	vmla.f32		q2 ,  q10 , d13[0]			
	vmla.f32		q3 ,  q10 , d15[0]			
	
	vmla.f32		q0 ,  q11 , d9[1]			
	vmla.f32		q1 ,  q11 , d11[1]			
	vmla.f32		q2 ,  q11 , d13[1]			
	vmla.f32		q3 ,  q11 , d15[1]			
	MEND
	
	MACRO	;matrix1 (q0 - q3)  matrix2 (q8 - q11)
	NEON_MATRIX_MUL_ASM_BLOCK1_REV 
	vmul.f32		q4  , q8 , d0[0]			
	vmul.f32		q5  , q8 , d2[0]			
	vmul.f32		q6  , q8 , d4[0]			
	vmul.f32		q7  , q8 , d6[0]			
	
	vmla.f32		q4 ,  q9 , d0[1]			
	vmla.f32		q5 ,  q9 , d2[1]			
	vmla.f32		q6 ,  q9 , d4[1]			
	vmla.f32		q7 ,  q9 , d6[1]			
	
	vmla.f32		q4 ,  q10 , d1[0]			
	vmla.f32		q5 ,  q10 , d3[0]			
	vmla.f32		q6 ,  q10 , d5[0]			
	vmla.f32		q7 ,  q10 , d7[0]			
	
	vmla.f32		q4 ,  q11 , d1[1]			
	vmla.f32		q5 ,  q11 , d3[1]			
	vmla.f32		q6 ,  q11 , d5[1]			
	vmla.f32		q7 ,  q11 , d7[1]			
	MEND
	
	MACRO	;matrix1 (q3 - q5)  matrix2 (q6 - q8) .matrices is 3x4
	NEON_MATRIX_MUL_ASM_BLOCK2 
	vmul.f32		q0  , q6 , d6[0]			
	vmul.f32		q1  , q6 , d8[0]			
	vmul.f32		q2  , q6 , d10[0]			
	
	vadd.f32		s3 , s3, s15				
	vadd.f32		s7 , s7, s19				
	vadd.f32		s11 , s11, s23				
	
	vmla.f32		q0 ,  q7 , d6[1]			
	vmla.f32		q1 ,  q7 , d8[1]			
	vmla.f32		q2 ,  q7 , d10[1]			
	
	vmla.f32		q0 ,  q8 , d7[0]			
	vmla.f32		q1 ,  q8 , d9[0]			
	vmla.f32		q2 ,  q8 , d11[0]			
	MEND


	MACRO	;matrix1 (q0 - q2)  matrix2 (q6 - q8) .matrices is 3x4
	NEON_MATRIX_MUL_ASM_BLOCK2_REV 
	vmul.f32		q3  , q6 , d0[0]			
	vmul.f32		q4  , q6 , d2[0]			
	vmul.f32		q5  , q6 , d4[0]			
	
	vadd.f32		s15 , s3, s15				
	vadd.f32		s19 , s7, s19				
	vadd.f32		s23 , s11, s23				
	
	vmla.f32		q3 ,  q7 , d0[1]			
	vmla.f32		q4 ,  q7 , d2[1]			
	vmla.f32		q5 ,  q7 , d4[1]			
	
	vmla.f32		q3 ,  q8 , d1[0]			
	vmla.f32		q4 ,  q8 , d3[0]			
	vmla.f32		q5 ,  q8 , d5[0]			
	MEND

	MACRO	;matrix1 (q4 - q7)  matrix2 (q8 - q10) .matrices1 is 4x4 , matrix2 is 3x4
	NEON_MATRIX_MUL_ASM_BLOCK3 
	vmul.f32		q0  , q8 , d8[0]			
	vmul.f32		q1  , q8 , d10[0]			
	vmul.f32		q2  , q8 , d12[0]			
	vmul.f32		q3  , q8 , d14[0]			
	
	vadd.f32		s3 , s3, s19				
	vadd.f32		s7 , s7, s23				
	vadd.f32		s11 , s11, s27				
	vadd.f32		s15 , s15, s31				
	
	vmla.f32		q0 ,  q9 , d8[1]			
	vmla.f32		q1 ,  q9 , d10[1]			
	vmla.f32		q2 ,  q9 , d12[1]			
	vmla.f32		q3 ,  q9 , d14[1]			
	
	vmla.f32		q0 ,  q10 , d9[0]			
	vmla.f32		q1 ,  q10 , d11[0]			
	vmla.f32		q2 ,  q10 , d13[0]			
	vmla.f32		q3 ,  q10 , d15[0]			
	MEND		
				
	MACRO	;matrix (q1 - q4)  vector (q0)
	NEON_MATRIX_MUL_VEC_ASM_BLOCK1 
	vmul.f32		q5  , q1 , d0[0]			
	
	vmla.f32		q5 ,  q2 , d0[1]			
	vmla.f32		q5 ,  q3 , d1[0]			
	vmla.f32		q5 ,  q4 , d1[1]			
	MEND

	MACRO	;matrix (q1 - q4)  vector (q0) . transformNormal
	NEON_MATRIX_MUL_VEC_ASM_BLOCK2 
	vmul.f32		q5  , q1 , d0[0]			
	
	vmla.f32		q5 ,  q2 , d0[1]			
	vmla.f32		q5 ,  q3 , d1[0]			
	MEND

	MACRO	;matrix (q1 - q4)  vector (q0) . transformCoord
	NEON_MATRIX_MUL_VEC_ASM_BLOCK3 
	vmul.f32		q5  , q1 , d0[0]			
	
	vadd.f32		q5 ,  q5 , q4				
	vmla.f32		q5 ,  q2 , d0[1]			
	vmla.f32		q5 ,  q3 , d1[0]			
	MEND

	MACRO	;matrix3x4 (d2 - d5  , q3)  vector (q0)
	NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK1 
	vmul.f32		q3  , q3 , q0				;/*q3 = x * 31      y * 32		z * 33		w * 34*/
	vpadd.f32		d9  , d6 , d7				;/*d9 = x * 31 + y * 32		z * 33 + w * 34*/
	vadd.f32		s18 , s18 ,s19				;/*s18 = x * 31 + y * 32 + z * 33 + w * 34*/
	
	vmul.f32		d8  , d2 , d0[0]			;/*x * 11   x * 21*/
	vmla.f32		d8 ,  d3 , d0[1]			;/*x * 11 + y * 12			x * 21 + y * 22*/
	vmla.f32		d8 ,  d4 , d1[0]			;/*x * 11 + y * 12 + z * 13			x * 21 + y * 22 + z * 23*/
	vmla.f32		d8 ,  d5 , d1[1]			;/*d8 = x * 11 + y * 12 + z * 13 + w * 14			x * 21 + y * 22 + z * 23 + w * 24*/
	MEND

	MACRO	;matrix3x4 (d2 - d5  , q3)  vector (q0) transformNormal
	NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK2 
	vmul.f32		q3  , q3 , q0				;/*q3 = x * 31      y * 32		z * 33		w * 34*/
	vadd.f32		s18  , s12 , s13			;/*s18 = x * 31 + y * 32*/
	vadd.f32		s18 , s18 ,s14				;/*s18 = x * 31 + y * 32 + z * 33*/
	
	vmul.f32		d8  , d2 , d0[0]			;/*x * 11   x * 21*/
	vmla.f32		d8 ,  d3 , d0[1]			;/*x * 11 + y * 12			x * 21 + y * 22*/
	vmla.f32		d8 ,  d4 , d1[0]			;/*x * 11 + y * 12 + z * 13			x * 21 + y * 22 + z * 23*/
	MEND

	MACRO	;matrix3x4 (d2 - d5  , q3)  vector (q0) transformCoord
	NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK3 
	vmul.f32		q3  , q3 , q0				;/*q3 = x * 31      y * 32		z * 33		(w = 1) * 34*/
	vpadd.f32		d9  , d6 , d7				;/*d9 = x * 31 + y * 32		z * 33 + (w = 1) * 34*/
	vadd.f32		s18 , s18 ,s19				;/*s18 = x * 31 + y * 32 + z * 33 + (w = 1) * 34*/
	
	vmul.f32		d8  , d2 , d0[0]			;/*x * 11   x * 21*/
	vadd.f32		d8 ,  d8 , d5				;/*d8 = x * 11 + 14			x * 21 + 24*/
	vmla.f32		d8 ,  d3 , d0[1]			;/*x * 11 + y * 12 + 14			x * 21 + y * 22 + 24*/
	vmla.f32		d8 ,  d4 , d1[0]			;/*x * 11 + y * 12 + z * 13 + 14			x * 21 + y * 22 + z * 23 + 24*/
	MEND
	
	;===============================================================
	FUNCTION_BEGIN HQNeonMatrix4Inverse	;void HQNeonMatrix4Inverse(const float *matrix1 , float *result , float * pDeterminant)
	
	vpush	{q4-q7}	;save registers

	vld4.32	{d0 , d2 , d4 , d6} , [r0	@128]!				;first 2 transposed rows
	vld4.32	{d1 , d3 , d5 , d7} , [r0	@128]					;last 2 transposed rows
	;****************************
	; m11 m21 m31 m41		row0
	; m12 m22 m32 m42		row1
	; m13 m23 m33 m43		row2
	; m14 m24 m34 m44		row3
	;****************************
	vswp		d2 , d3										;m12 m22 m32 m42			=>			m32 m42 m12 m22
	vswp		d6 , d7										;m14 m24 m34 m44			=>			m34 m44 m14 m24
					
	;**************************************
	;transposed matrix 
	;11 12 13 14<====row0	
	;23 24 21 22<====row1
	;31 32 33 34<====row2
	;43 44 41 42<====row3
	;**************************************
					 
					
					
	vmul.f32	q4 , q2 , q3								;q4 = 31.43, 32.44 ,33.41 ,34.42
	vrev64.32	q4 , q4										;q4 = 32.44, 31.43, 34.42, 33.41
	vmul.f32	q6 , q1 , q4								;23.32.44, 24.31.43, 21.33.42, 22.33.41
	vmul.f32	q7 , q0 , q4								;11.32.44, 12.31.43, 13.33.42, 14.33.41
	vswp		d8 , d9										;q4 = 34.42, 33.41, 32.44, 31.43
	vmul.f32	q8 , q1 , q4								;23.34.42, 24.33.41, 21.32.43, 22.31.43
	vmul.f32	q9 , q0 , q4								;11.34.42, 12.33.41, 13.32.43, 14.31.43
	;q6 = 23.34.42 - 23.32.44, 24.33.41 - 24.31.43, 21.32.44 - 21.34.42, 22.31.43 - 22.33.41 = minor0
	vsub.f32	q6 , q8 , q6								
	;q7 = 11.34.42 - 11.32.44, 12.33.41 - 12.31.43, 13.32.44 - 13.34.42, 14.31.43 - 14.33.41 = minor1
	vsub.f32	q7 , q9 , q7								
	;q7 = 13.32.44 - 13.34.42, 14.31.43 - 14.33.41, 11.34.42 - 11.32.44, 12.33.41 - 12.31.43 = minor1
	vswp		d14 , d15									
	;	-----------------------------------------------
					
	;**************************************
	;transposed matrix 
	;11 12 13 14<====row0	
	;23 24 21 22<====row1
	;31 32 33 34<====row2
	;43 44 41 42<====row3
	;**************************************
	vmul.f32	q4 , q1 , q2								;q4 = 23.31, 24.32, 21.33, 22.34
	vrev64.32	q4 , q4										;q4 = 24.32, 23.31, 22.34, 21.33
	vmul.f32	q8 , q3 , q4								;q8 = 43.24.32, 44.23.31 , 41.22.34 , 42.21.33
	vmul.f32	q9 , q0 , q4								;q9 = 11.24.32, 12.23.31 , 13.22.34 , 14.21.33 = minor3
	vswp		d8 , d9										;q4 = 22.34 , 21.33 , 24.32 , 23.31
	vadd.f32	q6 , q8 , q6								;q6 = minor0
	vmul.f32  q10 , q3 , q4								;q10 = 43.22.34, 44.21.33 , 41.24.32 , 42.23.31
	vmul.f32	q11 , q0 , q4								;q11 = 11.22.34, 12.21.33 , 13.24.32 , 14.23.31
	vsub.f32	q6 , q6 , q10								;q6 = minor0
	vsub.f32	q9 , q11 , q9								;q9 = minor3
	vswp		d18 , d19									
	;	-----------------------------------------------
					
	;**************************************
	;transposed matrix 
	;11 12 13 14<====row0	
	;23 24 21 22<====row1
	;31 32 33 34<====row2
	;43 44 41 42<====row3
	;**************************************
	vmov		d20 , d3									
	vmov		d21 , d2									;q10 = swap (q1)
	vmul.f32	q4 , q10 , q3								
	vrev64.32	q4 , q4										
	vswp		d4 , d5										;swap(q2)
	vmul.f32	q11 , q2 , q4								
	vmul.f32	q8 , q0 , q4								;q8 = minor2
	vadd.f32	q6 , q6 , q11								;q6 = minor0
	vswp		d8 , d9										;swap(q4)
	vmul.f32	q10 , q2 , q4								
	vmul.f32	q11 , q0 , q4								
	vsub.f32	q6 , q6 , q10								
	vsub.f32	q8 , q11 , q8								
	vswp		d16 , d17									
	;	-----------------------------------------------
					
	;**************************************
	;transposed matrix 
	;11 12 13 14<====row0	
	;23 24 21 22<====row1
	;31 32 33 34<====row2
	;43 44 41 42<====row3
	;**************************************
	vmul.f32	q4 , q0 , q1								
	vrev64.32	q4 , q4										
	vmul.f32	q10 , q3 , q4								
	vmul.f32	q11 , q2 , q4								
	vadd.f32	q8 , q10 , q8								
	vsub.f32	q9 , q11 , q9								
	vswp		d8 , d9										
	vmul.f32	q10 , q3 , q4								
	vmul.f32	q11 , q2 , q4								
	vsub.f32	q8 , q10 , q8								
	vsub.f32	q9 , q9 , q11								
	; -----------------------------------------------
					
	;**************************************
	;transposed matrix 
	;11 12 13 14<====row0	
	;23 24 21 22<====row1
	;31 32 33 34<====row2
	;43 44 41 42<====row3
	;**************************************
	vmul.f32	q4 ,q0 , q3									
	vrev64.32	q4 , q4										
	vmul.f32	q10 , q2 , q4								
	vmul.f32	q11 , q1 , q4								
	vsub.f32	q7 , q7 , q10								
	vadd.f32	q8 , q8 , q11								
	vswp		d8 , d9										
	vmul.f32	q10 , q2 , q4								
	vmul.f32	q11 , q1 , q4								
	vadd.f32	q7 , q7 , q10								
	vsub.f32	q8 , q8 , q11								
	; -----------------------------------------------
					
	;**************************************
	;transposed matrix 
	;11 12 13 14<====row0	
	;23 24 21 22<====row1
	;31 32 33 34<====row2
	;43 44 41 42<====row3
	;**************************************
	vmul.f32	q4 , q0 , q2								
	vrev64.32	q4 , q4										
	vmul.f32	q10 , q3 , q4								
	vmul.f32	q11 , q1 , q4								
	vadd.f32	q7 , q7 , q10								
	vsub.f32	q9 , q9 , q11								
	vswp		d8 , d9										
	vmul.f32	q10 , q3 , q4								
	vmul.f32	q11 , q1 , q4								
	vsub.f32	q7 , q7 , q10								
	vadd.f32	q9 , q9 , q11								
					
	; determinant
	vmul.f32	q5 , q0 , q6								
	vpadd.f32	d10 , d10 , d11								
	vpadd.f32	d10 , d10 , d10								;d10 = det	,	det
					
	;store determinant
	cmp		r2 , #0										;pDeterminant == NULL?
	beq		HQNeonMatrix4InverseLabel0					;skip if Null			
	vst1.32	{d10[0]} , [r2]									;store if not NULL
					 
					 
HQNeonMatrix4InverseLabel0  
	vrecpe.f32	d8 , d10				;d8 ~= 1/det
					 
	;Newton_Raphson iteration Y(n+1)=2*Y(n)-x*Y(n)^2 =  Y(n) * (2 - x * Y (n)) - reduce estimate error
	;Y(n) = d8
	;x = d10
	vrecps.f32	d9 , d8 , d10							;d9 = (2 - d8 * d10)
	vmul.f32		d8 , d9 , d8							;d8 ~= 1/det
					
	; multiply with 1/det, store result
	vmul.f32		q6 , q6 , d8[0]							
	vmul.f32		q7 , q7 , d8[0]							
	vmul.f32		q8 , q8 , d8[0]							
	vmul.f32		q9 , q9 , d8[0]							
					 
	vst1.32		{d12 , d13} , [r1	@128] !					
	vst1.32		{d14 , d15} , [r1	@128] !					
	vst1.32		{d16 , d17} , [r1	@128] !					
	vst1.32		{d18 , d19} , [r1	@128]					
		
	;old gcc inline asm syntax			 
	;: +r (result) , +r (matrix1)
	;: r (pDeterminant)
	;: q0 , q1 , q2 , q3 , q4  , q5 , q6 ,
	;q7 , q8 , q9 , q10 , q11 , memory
	
	vpop	{q4-q7}	;restore registers

	FUNCTION_RETURN ;end HQNeonMatrix4Inverse
	
	;============================================================


	FUNCTION_BEGIN HQNeonMatrix3x4InverseToMatrix4 ;void HQNeonMatrix3x4InverseToMatrix4(const float *matrix1 , float *result , float * pDeterminant)
	
	vpush	{q4-q7}	;save registers

	vld4.32	{d0 , d2 , d4 , d6} , [r0	@128] !				;first 2 transposed rows
	vld1.32	{d1 } , [r0	@64] !							;d1 = m31 m32
	vld1.32	{d5 } , [r0	@64] 								;d5 = m33 m34
	vmov.i64	d7  , #0x0000000000000000						;d7 = 0	0
	vmov.i64	d3  , #0x0000000000000000						;d3 = 0	0
	vmov.f32	s15	, #1.0										;d7 = 0 	1
	vtrn.32	d1 , d3											;d1 = m31 0  ; d3 = m32 0
	vtrn.32	d5 , d7											;d5 = m33 0  ; d7 = m34 1
	;****************************
	; m11 m21 m31 0		row0	d0  d1
	; m12 m22 m32 0		row1	d2	d3
	; m13 m23 m33 0		row2	d4	d5
	; m14 m24 m34 1		row3	d6	d7
	;****************************
	vswp		d2 , d3										;m12 m22 m32 m42			=>			m32 m42 m12 m22
	vswp		d6 , d7										;m14 m24 m34 m44			=>			m34 m44 m14 m24
					
	;**************************************
	;transposed matrix 
	;11 12 13 14<====row0	
	;23 24 21 22<====row1
	;31 32 33 34<====row2
	;43 44 41 42<====row3
	;**************************************
					 
					
					
	vmul.f32	q4 , q2 , q3								;q4 = 31.43, 32.44 ,33.41 ,34.42
	vrev64.32	q4 , q4										;q4 = 32.44, 31.43, 34.42, 33.41
	vmul.f32	q6 , q1 , q4								;23.32.44, 24.31.43, 21.33.42, 22.33.41
	vmul.f32	q7 , q0 , q4								;11.32.44, 12.31.43, 13.33.42, 14.33.41
	vswp		d8 , d9										;q4 = 34.42, 33.41, 32.44, 31.43
	vmul.f32	q8 , q1 , q4								;23.34.42, 24.33.41, 21.32.43, 22.31.43
	vmul.f32	q9 , q0 , q4								;11.34.42, 12.33.41, 13.32.43, 14.31.43
	;q6 = 23.34.42 - 23.32.44, 24.33.41 - 24.31.43, 21.32.44 - 21.34.42, 22.31.43 - 22.33.41 = minor0
	vsub.f32	q6 , q8 , q6								
	;q7 = 11.34.42 - 11.32.44, 12.33.41 - 12.31.43, 13.32.44 - 13.34.42, 14.31.43 - 14.33.41 = minor1
	vsub.f32	q7 , q9 , q7								
	;q7 = 13.32.44 - 13.34.42, 14.31.43 - 14.33.41, 11.34.42 - 11.32.44, 12.33.41 - 12.31.43 = minor1
	vswp		d14 , d15									
	;	-----------------------------------------------
					
	;**************************************
	;transposed matrix 
	;11 12 13 14<====row0	
	;23 24 21 22<====row1
	;31 32 33 34<====row2
	;43 44 41 42<====row3
	;**************************************
	vmul.f32	q4 , q1 , q2								;q4 = 23.31, 24.32, 21.33, 22.34
	vrev64.32	q4 , q4										;q4 = 24.32, 23.31, 22.34, 21.33
	vmul.f32	q8 , q3 , q4								;q8 = 43.24.32, 44.23.31 , 41.22.34 , 42.21.33
	vmul.f32	q9 , q0 , q4								;q9 = 11.24.32, 12.23.31 , 13.22.34 , 14.21.33 = minor3
	vswp		d8 , d9										;q4 = 22.34 , 21.33 , 24.32 , 23.31
	vadd.f32	q6 , q8 , q6								;q6 = minor0
	vmul.f32  q10 , q3 , q4								;q10 = 43.22.34, 44.21.33 , 41.24.32 , 42.23.31
	vmul.f32	q11 , q0 , q4								;q11 = 11.22.34, 12.21.33 , 13.24.32 , 14.23.31
	vsub.f32	q6 , q6 , q10								;q6 = minor0
	vsub.f32	q9 , q11 , q9								;q9 = minor3
	vswp		d18 , d19									
	;	-----------------------------------------------
					
	;**************************************
	;transposed matrix 
	;11 12 13 14<====row0	
	;23 24 21 22<====row1
	;31 32 33 34<====row2
	;43 44 41 42<====row3
	;**************************************
	vmov		d20 , d3									
	vmov		d21 , d2									;q10 = swap (q1)
	vmul.f32	q4 , q10 , q3								
	vrev64.32	q4 , q4										
	vswp		d4 , d5										;swap(q2)
	vmul.f32	q11 , q2 , q4								
	vmul.f32	q8 , q0 , q4								;q8 = minor2
	vadd.f32	q6 , q6 , q11								;q6 = minor0
	vswp		d8 , d9										;swap(q4)
	vmul.f32	q10 , q2 , q4								
	vmul.f32	q11 , q0 , q4								
	vsub.f32	q6 , q6 , q10								
	vsub.f32	q8 , q11 , q8								
	vswp		d16 , d17									
	;	-----------------------------------------------
					
	;**************************************
	;transposed matrix 
	;11 12 13 14<====row0	
	;23 24 21 22<====row1
	;31 32 33 34<====row2
	;43 44 41 42<====row3
	;**************************************
	vmul.f32	q4 , q0 , q1								
	vrev64.32	q4 , q4										
	vmul.f32	q10 , q3 , q4								
	vmul.f32	q11 , q2 , q4								
	vadd.f32	q8 , q10 , q8								
	vsub.f32	q9 , q11 , q9								
	vswp		d8 , d9										
	vmul.f32	q10 , q3 , q4								
	vmul.f32	q11 , q2 , q4								
	vsub.f32	q8 , q10 , q8								
	vsub.f32	q9 , q9 , q11								
	; -----------------------------------------------
					
	;**************************************
	;transposed matrix 
	;11 12 13 14<====row0	
	;23 24 21 22<====row1
	;31 32 33 34<====row2
	;43 44 41 42<====row3
	;**************************************
	vmul.f32	q4 ,q0 , q3									
	vrev64.32	q4 , q4										
	vmul.f32	q10 , q2 , q4								
	vmul.f32	q11 , q1 , q4								
	vsub.f32	q7 , q7 , q10								
	vadd.f32	q8 , q8 , q11								
	vswp		d8 , d9										
	vmul.f32	q10 , q2 , q4								
	vmul.f32	q11 , q1 , q4								
	vadd.f32	q7 , q7 , q10								
	vsub.f32	q8 , q8 , q11								
	; -----------------------------------------------
					
	;**************************************
	;transposed matrix 
	;11 12 13 14<====row0	
	;23 24 21 22<====row1
	;31 32 33 34<====row2
	;43 44 41 42<====row3
	;**************************************
	vmul.f32	q4 , q0 , q2								
	vrev64.32	q4 , q4										
	vmul.f32	q10 , q3 , q4								
	vmul.f32	q11 , q1 , q4								
	vadd.f32	q7 , q7 , q10								
	vsub.f32	q9 , q9 , q11								
	vswp		d8 , d9										
	vmul.f32	q10 , q3 , q4								
	vmul.f32	q11 , q1 , q4								
	vsub.f32	q7 , q7 , q10								
	vadd.f32	q9 , q9 , q11								
					
	; determinant
	vmul.f32	q5 , q0 , q6								
	vpadd.f32	d10 , d10 , d11								
	vpadd.f32	d10 , d10 , d10								;d10 = det	,	det
					
	;store determinant
	cmp		r2 , #0										;pDeterminant == NULL?
	beq		HQNeonMatrix3x4InverseToMatrix4Label0			; skip if NULL					
	vst1.32	{d10[0]} , [r2]									;store if not NULL
					 
					 
HQNeonMatrix3x4InverseToMatrix4Label0  
	vrecpe.f32	d8 , d10				;d8 ~= 1/det
					 
	;Newton_Raphson iteration Y(n+1)=2*Y(n)-x*Y(n)^2 =  Y(n) * (2 - x * Y (n)) - reduce estimate error
	;Y(n) = d8
	;x = d10
	vrecps.f32	d9 , d8 , d10							;d9 = (2 - d8 * d10)
	vmul.f32		d8 , d9 , d8							;d8 ~= 1/det
					
	; multiply with 1/det, store result
	vmul.f32		q6 , q6 , d8[0]							
	vmul.f32		q7 , q7 , d8[0]							
	vmul.f32		q8 , q8 , d8[0]							
	vmul.f32		q9 , q9 , d8[0]							
					 
	vst1.32		{d12 , d13} , [r1	@128] !					
	vst1.32		{d14 , d15} , [r1	@128] !					
	vst1.32		{d16 , d17} , [r1	@128] !					
	vst1.32		{d18 , d19} , [r1	@128]					
		
	;old gcc inline asm syntax			 
	;: +r (result) , +r (matrix1)
	;: r (pDeterminant)
	;: q0 , q1 , q2 , q3 , q4  , q5 , q6 ,
	;q7 , q8 , q9 , q10 , q11 , memory
	
	
	vpop	{q4-q7}	;restore registers

	FUNCTION_RETURN	;HQNeonMatrix3x4InverseToMatrix4
	
	;=======================================
	
	FUNCTION_BEGIN HQNeonMatrix4Multiply	; void HQNeonMatrix4Multiply(const float * matrix1 , const float *matrix2 , float * result )
	
	vpush	{q4-q7}	;save registers
		
	vld1.32		{d8 , d9} , [r0	@128]	!			;1st row of matrix1
	vld1.32		{d10 , d11} , [r0	@128]	!		;2nd row of matrix1
	vld1.32		{d12 , d13} , [r0	@128]	!		;3rd row of matrix1
	vld1.32		{d14 , d15} , [r0	@128]			;4th row of matrix1
					 
	vld1.32		{d16 , d17} , [r1	@128]	!		;1st row of matrix2
	vld1.32		{d18 , d19} , [r1	@128]	!		;2nd row of matrix2
	vld1.32		{d20 , d21} , [r1	@128]	!		;3rd row of matrix2
	vld1.32		{d22 , d23} , [r1	@128]			;4th row of matrix2
					 
	NEON_MATRIX_MUL_ASM_BLOCK1
					 
	vst1.32		{d0 , d1}  , [r2	@128]	!		;store first row	
	vst1.32		{d2 , d3}  , [r2	@128]	!		;store second row	
	vst1.32		{d4 , d5}  , [r2	@128]	!		;store third row
	vst1.32		{d6 , d7}  , [r2	@128]			;store fourth row
					 
	;old gcc inline asm
	;:+r (matrix1) , +r (matrix2) , +r (result)
	;::q0 , q1 , q2 , q3 , q4 , q5 , q6 , q7 , q8 , q9,
	;q10 , q11 , memory
	
	vpop	{q4-q7}	;restore registers
		 
	FUNCTION_RETURN	;HQNeonMatrix4Multiply

	;===========================================
	
	FUNCTION_BEGIN HQNeonMultiMatrix4Multiply	;void HQNeonMultiMatrix4Multiply(const float * matrix , hq_uint32 numMat , float * result )
	;r0 = matrix  , r1 = numMat; r2 = result
	push {r5}			;save register
	vpush	{q4-q7}	;save registers

	and			r5  ,  r1 , #0x1				
					 
	cmp			r5 , #0								;if r5 = 0 , num matrices is divisible by 2.If it is , number of matrix pairs is odd
	bne			HQNEONMULTIMATRIX4MULTIPLYLABEL0						;jump if r5 != 0 (we have even number of pairs)
					 
	;multiply first pair of matrices
	vld1.32		{d8 , d9} , [r0	@128]	!			;1st row of matrix1
	vld1.32		{d10 , d11} , [r0	@128]	!		;2nd row of matrix1
	vld1.32		{d12 , d13} , [r0	@128]	!		;3rd row of matrix1
	vld1.32		{d14 , d15} , [r0	@128]	!		;4th row of matrix1
					 
	vld1.32		{d16 , d17} , [r0	@128]	!		;1st row of matrix2
	vld1.32		{d18 , d19} , [r0	@128]	!		;2nd row of matrix2
	vld1.32		{d20 , d21} , [r0	@128]	!		;3rd row of matrix2
	vld1.32		{d22 , d23} , [r0	@128]	!		;4th row of matrix2
					 
	NEON_MATRIX_MUL_ASM_BLOCK1							
					 
	mov			r5 , #2								;start loop at matrix index 2
	b				HQNEONMULTIMATRIX4MULTIPLYLABEL1						
					 
HQNEONMULTIMATRIX4MULTIPLYLABEL0
	;we have even pairs , so just load the first matrix
	vld1.32		{d0 , d1} , [r0	@128]	!		;1st row of matrix1
	vld1.32		{d2 , d3} , [r0	@128]	!		;2nd row of matrix1
	vld1.32		{d4 , d5} , [r0	@128]	!		;3rd row of matrix1
	vld1.32		{d6 , d7} , [r0	@128]	!		;4th row of matrix1
					 
HQNEONMULTIMATRIX4MULTIPLYLABEL1	;loop:	multiply 2 pairs of matrices each iteration
					 
	cmp			r5 , r1				
	bhs			HQNEONMULTIMATRIX4MULTIPLYLABEL2					;end loop if r5 >= numMat
					 
	vld1.32		{d16 , d17} , [r0	@128]	!		;1st row of matrix2
	vld1.32		{d18 , d19} , [r0	@128]	!		;2nd row of matrix2
	vld1.32		{d20 , d21} , [r0	@128]	!		;3rd row of matrix2
	vld1.32		{d22 , d23} , [r0	@128]	!		;4th row of matrix2
					 
	NEON_MATRIX_MUL_ASM_BLOCK1_REV
					 
	vld1.32		{d16 , d17} , [r0	@128]	!		;1st row of matrix2
	vld1.32		{d18 , d19} , [r0	@128]	!		;2nd row of matrix2
	vld1.32		{d20 , d21} , [r0	@128]	!		;3rd row of matrix2
	vld1.32		{d22 , d23} , [r0	@128]	!		;4th row of matrix2
					 
	NEON_MATRIX_MUL_ASM_BLOCK1
					 
	add			r5 , r5 , #2						;r5 += 2
	b				HQNEONMULTIMATRIX4MULTIPLYLABEL1						
					 
HQNEONMULTIMATRIX4MULTIPLYLABEL2
					 
	vst1.32		{d0 , d1}  , [r2	@128]	!		;store first row	
	vst1.32		{d2 , d3}  , [r2	@128]	!		;store second row	
	vst1.32		{d4 , d5}  , [r2	@128]	!		;store third row
	vst1.32		{d6 , d7}  , [r2	@128]			;store fourth row
			
	;old gcc		 
	;:+r (matrix) , +r (result)
	;:[numMat]r(numMat)
	;:q0 , q1 , q2 , q3 , q4 , q5 , q6 , q7 , q8 , q9,
	;q10 , q11 , memory , r5
	
	vpop	{q4-q7}	;restore registers
	pop		{r5}		;restore register
		
	FUNCTION_RETURN	;HQNeonMultiMatrix4Multiply

	;=======================================
	
	FUNCTION_BEGIN HQNeonVector4MultiplyMatrix4	; void HQNeonVector4MultiplyMatrix4(const float *vector ,const float * matrix ,  float * resultVector )
	
	vpush {q4-q5}	;save registers

	vld1.32		{d0 , d1} , [r1	@128]				;vector
					 
	vld1.32		{d2 , d3} , [r0	@128]	!			;1st row of matrix
	vld1.32		{d4 , d5} , [r0	@128]	!			;2nd row of matrix
	vld1.32		{d6 , d7} , [r0	@128]	!			;3rd row of matrix
	vld1.32		{d8 , d9} , [r0	@128]				;4th row of matrix
					 
	NEON_MATRIX_MUL_VEC_ASM_BLOCK1
					 
	vst1.32		{d10 , d11}  , [r2	@128]			
			
	;old gcc inline asm		 
	;:+r (matrix)
	;:r (vector) , r(resultVector)
	;:q0 , q1 , q2 , q3 , q4 , q5 , memory
	;);
		
	vpop {q4-q5}	;restore registers
	
	FUNCTION_RETURN ; HQNeonVector4MultiplyMatrix4

	;=======================================
	
	FUNCTION_BEGIN HQNeonMultiVector4MultiplyMatrix4	;void HQNeonMultiVector4MultiplyMatrix4(const float *vector ,hq_uint32 numVec ,const float * matrix ,  float * resultVector )
	
	;r0 = vector, r1 = numVec, r2 = matrix, r3 = resultVector

	push {r5}			;save register
	vpush {q4-q5}	;save registers

	vld1.32		{d2 , d3} , [r2	@128]	!			;1st row of matrix
	vld1.32		{d4 , d5} , [r2	@128]	!			;2nd row of matrix
	vld1.32		{d6 , d7} , [r2	@128]	!			;3rd row of matrix
	vld1.32		{d8 , d9} , [r2	@128]				;4th row of matrix
					 
					 
	mov			r5	, #0						;r5 = 0
					 
HQNEONMULTIVECTOR4MULTIPLYMATRIX4LABEL0
	cmp			r5 , r1					
	bhs			HQNEONMULTIVECTOR4MULTIPLYMATRIX4LABEL1						;end loop if r5 >= numVec
					 
	vld1.32		{d0 , d1} , [r0	@128]	!			;vector
					 
	NEON_MATRIX_MUL_VEC_ASM_BLOCK1
					 
	vst1.32		{d10 , d11}  , [r3	@128]!			
					 
	add			r5 , r5 , #1					;increase by 1
	b				HQNEONMULTIVECTOR4MULTIPLYMATRIX4LABEL0					;loop 
					 
HQNEONMULTIVECTOR4MULTIPLYMATRIX4LABEL1
		
	;old gcc inline asm			 
	;:+r (matrix), +r (vector) , +r(resultVector)
	;:[numVec] r (numVec)
	;:q0 , q1 , q2 , q3 , q4 , q5 , memory , r5
	;);
		
	vpop {q4-q5}	;restore registers
	pop	 {r5}			;restore register

	FUNCTION_RETURN	;HQNeonMultiVector4MultiplyMatrix4
	
	;=============================================

	FUNCTION_BEGIN HQNeonVector4TransformCoord	;void HQNeonVector4TransformCoord(const float *vector ,const float * matrix ,  float * resultVector )
	
	vpush {q4-q5}	;save registers

	vld1.32		{d0 , d1} , [r0	@128]				;vector
					 
	vld1.32		{d2 , d3} , [r1	@128]	!			;1st row of matrix
	vld1.32		{d4 , d5} , [r1	@128]	!			;2nd row of matrix
	vld1.32		{d6 , d7} , [r1	@128]	!			;3rd row of matrix
	vld1.32		{d8 , d9} , [r1	@128]				;4th row of matrix
					 
	NEON_MATRIX_MUL_VEC_ASM_BLOCK3
					 
	vst1.32		{d10 , d11}  , [r2	@128]			
	
	;old gcc inline asm				 
	;:+r (matrix)
	;:r (vector) , r(resultVector)
	;:q0 , q1 , q2 , q3 , q4 , q5 , memory
	;);

	vpop {q4-q5}	;restore registers
		
	FUNCTION_RETURN	;HQNeonVector4TransformCoord
	

	;===========================================

	FUNCTION_BEGIN HQNeonMultiVector4TransformCoord		;void HQNeonMultiVector4TransformCoord(const float *vector ,hq_uint32 numVec ,const float * matrix ,  float * resultVector )
	;r0 = vector, r1 = numVec, r2 = matrix, r3 = resultVector

	push {r5}			;save register
	vpush {q4-q5}	;save registers

	vld1.32		{d2 , d3} , [r2	@128]	!			;1st row of matrix
	vld1.32		{d4 , d5} , [r2	@128]	!			;2nd row of matrix
	vld1.32		{d6 , d7} , [r2	@128]	!			;3rd row of matrix
	vld1.32		{d8 , d9} , [r2	@128]				;4th row of matrix
					 
					 
	mov			r5	, #0							;r5 = count
					 
HQNEONMULTIVECTOR4TRANSFORMCOORDLABEL0
					 
	cmp			r5 , r1					
	bhs			HQNEONMULTIVECTOR4TRANSFORMCOORDLABEL1						;end loop if r5 >= numVec
					 
	vld1.32		{d0 , d1} , [r0	@128]	!			;vector
					 
	NEON_MATRIX_MUL_VEC_ASM_BLOCK3
					 
	vst1.32		{d10 , d11}  , [r3	@128]!			
					 
	add			r5 , r5 , #1						;increase by 1
	b			HQNEONMULTIVECTOR4TRANSFORMCOORDLABEL0								;loop 
					 
HQNEONMULTIVECTOR4TRANSFORMCOORDLABEL1
		
	;old gcc inline asm			 
	;:+r (matrix), +r (vector) , +r(resultVector)
	;:[numVec] r (numVec)
	;:q0 , q1 , q2 , q3 , q4 , q5 , memory , r5
	;);
		
	vpop {q4-q5}	;restore registers
	pop	 {r5}			;restore register

	FUNCTION_RETURN	;HQNeonMultiVector4TransformCoord
	
	;=====================================================

	FUNCTION_BEGIN HQNeonVector4TransformNormal	;void HQNeonVector4TransformNormal(const float *vector ,const float * matrix ,  float * resultVector )
	
	vpush {q4-q5}	;save registers

	vld1.32		{d0 , d1} , [r0	@128]				;vector
					 
	vld1.32		{d2 , d3} , [r1	@128]	!			;1st row of matrix
	vld1.32		{d4 , d5} , [r1	@128]	!			;2nd row of matrix
	vld1.32		{d6 , d7} , [r1	@128]	!			;3rd row of matrix
	vld1.32		{d8 , d9} , [r1	@128]				;4th row of matrix
					 
	NEON_MATRIX_MUL_VEC_ASM_BLOCK2
					 
	vst1.32		{d10 , d11}  , [r2	@128]			
	
	;old gcc inline asm				 
	;:+r (matrix)
	;:r (vector) , r(resultVector)
	;:q0 , q1 , q2 , q3 , q4 , q5 , memory
	;);

	vpop {q4-q5}	;restore registers
		
	FUNCTION_RETURN	;HQNeonVector4TransformNormal


	;=========================================
	
	FUNCTION_BEGIN HQNeonMultiVector4TransformNormal	;void HQNeonMultiVector4TransformNormal(const float *vector ,hq_uint32 numVec ,const float * matrix ,  float * resultVector )
	
	push {r5}			;save register
	vpush {q4-q5}	;save registers

	vld1.32		{d2 , d3} , [r2	@128]	!			;1st row of matrix
	vld1.32		{d4 , d5} , [r2	@128]	!			;2nd row of matrix
	vld1.32		{d6 , d7} , [r2	@128]	!			;3rd row of matrix
	vld1.32		{d8 , d9} , [r2	@128]				;4th row of matrix
					 
					 
	mov			r5	, #0							;r5 = count
					 
HQNEONMULTIVECTOR4TRANSFORMNORMALLABEL0
					 
	cmp			r5 , r1					
	bhs			HQNEONMULTIVECTOR4TRANSFORMNORMALLABEL1						;end loop if r5 >= numVec
					 
	vld1.32		{d0 , d1} , [r0	@128]	!			;vector
					 
	NEON_MATRIX_MUL_VEC_ASM_BLOCK2
					 
	vst1.32		{d10 , d11}  , [r3	@128]!			
					 
	add			r5 , r5 , #1						;increase by 1
	b			HQNEONMULTIVECTOR4TRANSFORMNORMALLABEL0								;loop 
					 
HQNEONMULTIVECTOR4TRANSFORMNORMALLABEL1
		
	;old gcc inline asm			 
	;:+r (matrix), +r (vector) , +r(resultVector)
	;:[numVec] r (numVec)
	;:q0 , q1 , q2 , q3 , q4 , q5 , memory , r5
	;);
		
	vpop {q4-q5}	;restore registers
	pop	 {r5}			;restore register
		
	FUNCTION_RETURN	;HQNeonMultiVector4TransformNormal

	;=================================================
	FUNCTION_BEGIN HQNeonMatrix4MultiplyVector4	;void HQNeonMatrix4MultiplyVector4(const float * matrix , const float *vector , float * resultVector )
	
	vpush {q4-q5}	;save registers

	vld1.32		{d0 , d1} , [r1	@128]				;vector
					 
	vld4.32		{d2 , d4  ,d6 , d8} , [r0	@128]!	;2 rows of transposed matrix
	vld4.32		{d3 , d5  ,d7 , d9} , [r0	@128]	;last 2 rows of transposed matrix
					 
	NEON_MATRIX_MUL_VEC_ASM_BLOCK1
					 
	vst1.32		{d10 , d11}  , [r2	@128]			
	
	;old gcc inline asm					 
	;:+r (matrix)
	;:r (vector) , r (resultVector) 
	;:q0 , q1 , q2 , q3 , q4 , q5 , memory
	;);

	vpop {q4-q5}	;restore registers
		
	FUNCTION_RETURN	;HQNeonMatrix4MultiplyVector4

	;==================================
	
	
	FUNCTION_BEGIN HQNeonMatrix3x4Multiply	;void HQNeonMatrix3x4Multiply(const float * matrix1 , const float *matrix2 , float * result )
	
	vpush {q4-q7}	;save registers

	vld1.32		{d6 , d7} , [r0	@128]	!				;1st row of matrix1
	vld1.32		{d8 , d9} , [r0	@128]	!				;2nd row of matrix1
	vld1.32		{d10 , d11} , [r0	@128]				;3rd row of matrix1
					 
	vld1.32		{d12 , d13} , [r1	@128]	!			;1st row of matrix2
	vld1.32		{d14 , d15} , [r1	@128]	!			;2nd row of matrix2
	vld1.32		{d16 , d17} , [r1	@128]				;3rd row of matrix2
					 
	NEON_MATRIX_MUL_ASM_BLOCK2
					 
	vst1.32		{d0 , d1}  , [r2	@128]	!		;store first row	
	vst1.32		{d2 , d3}  , [r2	@128]	!		;store second row	
	vst1.32		{d4 , d5}  , [r2	@128]			;store third row
	
	;old gcc inline asm				 
	;:+r (matrix1) , +r (matrix2) , +r (result)
	;::q0 , q1 , q2 , q3 , q4 , q5 , q6 , q7 , q8, memory
	;);
	
	vpop {q4-q7}	;restore registers
		
	FUNCTION_RETURN	;HQNeonMatrix3x4Multiply

	;========================================================
	
	FUNCTION_BEGIN HQNeonMatrix4MultiplyMatrix3x4	;void HQNeonMatrix4MultiplyMatrix3x4(const float * matrix1 , const float *matrix2 , float * result )
	
	vpush {q4-q7}	;save registers

	vld1.32		{d8 , d9} , [r0	@128]	!				;1st row of matrix1
	vld1.32		{d10 , d11} , [r0	@128]	!			;2nd row of matrix1
	vld1.32		{d12 , d13} , [r0	@128]	!			;3rd row of matrix1
	vld1.32		{d14 , d15} , [r0	@128]				;4th row of matrix1
					 
	vld1.32		{d16 , d17} , [r1	@128]	!			;1st row of matrix2
	vld1.32		{d18 , d19} , [r1	@128]	!			;2nd row of matrix2
	vld1.32		{d20 , d21} , [r1	@128]				;3rd row of matrix2
					 
	NEON_MATRIX_MUL_ASM_BLOCK3
					 
	vst1.32		{d0 , d1}  , [r2	@128]	!		;store first row	
	vst1.32		{d2 , d3}  , [r2	@128]	!		;store second row	
	vst1.32		{d4 , d5}  , [r2	@128]	!		;store third row
	vst1.32		{d6 , d7}  , [r2	@128]			;store fourth row
	
	;old gcc inline asm				 
	;:+r (matrix1) , +r (matrix2) , +r (result)
	;::q0 , q1 , q2 , q3 , q4 , q5 , q6 , q7 , q8, q9 , q10 , memory
	;);
	
	vpop {q4-q7}	;restore registers
		
	FUNCTION_RETURN	;HQNeonMatrix3x4Multiply
	
	;===================================================================

	FUNCTION_BEGIN HQNeonMultiMatrix3x4Multiply	;void HQNeonMultiMatrix3x4Multiply(const float * matrix , hq_uint32 numMat , float * result )
	
	push {r5}			;save register
	vpush {q4-q7}	;save registers

	and			r5  ,  r1 , #0x1				
					 
	cmp			r5 , #0								;if r5 = 0 , num matrices is divisible by 2.If it is , number of matrix pairs is odd
	bne			HQNEONMULTIMATRIX3X4MULTIPLYLABEL0						;jump if r5 != 0 (we have even number of pairs)
					 
	;multiply first pair of matrices
	vld1.32		{d6 , d7} , [r0	@128]	!				;1st row of matrix1
	vld1.32		{d8 , d9} , [r0	@128]	!				;2nd row of matrix1
	vld1.32		{d10 , d11} , [r0	@128]	!			;3rd row of matrix1
					 
	vld1.32		{d12 , d13} , [r0	@128]	!			;1st row of matrix2
	vld1.32		{d14 , d15} , [r0	@128]	!			;2nd row of matrix2
	vld1.32		{d16 , d17} , [r0	@128]	!			;3rd row of matrix2
					 
	NEON_MATRIX_MUL_ASM_BLOCK2					
					 
	mov			r5 , #2									;start loop at matrix index 2
	b				HQNEONMULTIMATRIX3X4MULTIPLYLABEL1							
					 
HQNEONMULTIMATRIX3X4MULTIPLYLABEL0
	;we have even pairs , so just load the first matrix
	vld1.32		{d0 , d1} , [r0	@128]	!				;1st row of matrix1
	vld1.32		{d2 , d3} , [r0	@128]	!				;2nd row of matrix1
	vld1.32		{d4 , d5} , [r0	@128]	!				;3rd row of matrix1
					 
					 
HQNEONMULTIMATRIX3X4MULTIPLYLABEL1	;multiply 2 pairs of matrices each iteration
					 
	cmp			r5 , r1					
	bhs			HQNEONMULTIMATRIX3X4MULTIPLYLABEL2						;end loop if r5 >= numMat
					 
	vld1.32		{d12 , d13} , [r0	@128]	!			;1st row of matrix2
	vld1.32		{d14 , d15} , [r0	@128]	!			;2nd row of matrix2
	vld1.32		{d16 , d17} , [r0	@128]	!			;3rd row of matrix2
					 
	NEON_MATRIX_MUL_ASM_BLOCK2_REV
					 
	vld1.32		{d12 , d13} , [r0	@128]	!			;1st row of matrix2
	vld1.32		{d14 , d15} , [r0	@128]	!			;2nd row of matrix2
	vld1.32		{d16 , d17} , [r0	@128]	!			;3rd row of matrix2
					 
	NEON_MATRIX_MUL_ASM_BLOCK2
					 
	add			r5 , r5 , #2					;r5 += 2
	b				HQNEONMULTIMATRIX3X4MULTIPLYLABEL1					
					 
HQNEONMULTIMATRIX3X4MULTIPLYLABEL2	;end loop									
					 
	vst1.32		{d0 , d1}  , [r2	@128]	!			;store first row	
	vst1.32		{d2 , d3}  , [r2	@128]	!			;store second row	
	vst1.32		{d4 , d5}  , [r2	@128]				;store third row
	
	;old gcc inline asm					 
	;:+r (matrix), +r (result)
	;:[numMat] r (numMat)
	;:q0 , q1 , q2 , q3 , q4 , q5 , q6 , q7 , q8, memory , r5
	;);
		
		
	vpop {q4-q7}	;restore registers
	pop {r5}			;restore register

	FUNCTION_RETURN	;HQNeonMultiMatrix3x4Multiply

	;=========================================
	
	FUNCTION_BEGIN HQNeonMatrix3x4MultiplyVector4	;void HQNeonMatrix3x4MultiplyVector4(const float * matrix , const float *vector , float * resultVector )
	
	vpush {q4}	;save registers

	vld1.32		{d0 , d1} , [r1	@128]						;vector
					 
	vld4.32		{d2 , d3  ,d4 , d5} , [r0	@128]!				;2 rows of transposed matrix
	vld1.32		{d6 , d7} , [r0	@128]						;last row of matrix
					 
	NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK1
					 
	vmov.f32			s19   ,  s3								;copy w
	vst1.32		{d8 , d9}  , [r2	@128]						;store
		
	;old gcc inline asm			 
	;:+r (matrix)
	;:r (vector) , r (resultVector)
	;:q0 , q1 , q2 , q3 , q4, memory
	;);

	vpop {q4}	;restore registers
		
	FUNCTION_RETURN	;HQNeonMatrix3x4MultiplyVector4

	;===========================================
	
	FUNCTION_BEGIN HQNeonMatrix3x4MultiplyMultiVector4	;void HQNeonMatrix3x4MultiplyMultiVector4(const float * matrix , const float *vector ,hq_uint32 numVec , float * resultVector )
	
	;r0 = matrix, r1 = vector, r2 = numVec, r3 = resultVector

	push {r5}			;save register
	vpush {q4}	;save registers

	vld4.32		{d2 , d3  ,d4 , d5} , [r0	@128]!			;2 rows of transposed matrix
	vld1.32		{d6 , d7} , [r0	@128]					;last row of matrix
					 
	mov			r5	, #0							;r5 = 0
					 
HQNEONMATRIX3X4MULTIPLYMULTIVECTOR4LABEL0
					 
	cmp			r5 , r2						
	bhs			HQNEONMATRIX3X4MULTIPLYMULTIVECTOR4LABEL1							;end loop if r5 >= numVec
					 
	vld1.32		{d0 , d1} , [r1	@128]	!				;vector
					 
	NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK1
					 
	vmov.f32			s19   ,  s3							;copy w
	vst1.32		{d8 , d9}  , [r3	@128]!					
					 
	add			r5 , r5 , #1						;increase by 1
	b				HQNEONMATRIX3X4MULTIPLYMULTIVECTOR4LABEL0							;loop 
					 
HQNEONMATRIX3X4MULTIPLYMULTIVECTOR4LABEL1										
			
	;old gcc inline asm		 
	;:+r (matrix) , +r (vector) , +r (resultVector) 
	;:[numVec]r (numVec)
	;:q0 , q1 , q2 , q3 , q4, memory , r5
	;);

	vpop {q4}	;restore registers
	pop		{r5}		;restore register
		
	FUNCTION_RETURN	;HQNeonMatrix3x4MultiplyMultiVector4

	;=================================================
	
	FUNCTION_BEGIN HQNeonVector4TransformCoordMatrix3x4	;void HQNeonVector4TransformCoordMatrix3x4(const float *vector ,const float * matrix ,  float * resultVector )
	
	vpush {q4}	;save registers

	vld1.32		{d0} , [r0	@64]!						;vector d0 =  x y
	vld1.32			{d1[0]}  , [r0 @32]							;q0 = x y z _
	vmov.f32		s3	, #1.0							;q0 = x y z 1
					 
	vld4.32		{d2 , d3  ,d4 , d5} , [r1	@128]!			;2 rows of transposed matrix
	vld1.32		{d6 , d7} , [r1	@128]					;last row of matrix
					 
	NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK3
					 
	vst1.32		{d8}  , [r2	@64 ]!						;store x y
	vst1.32			{d9[0]}   , [r2 @32]	!					;store z
	vst1.32			{d1[1]}	  , [r2 @32]					;store 1
		
	;old gcc inline asm			 
	;:+r (matrix) , +r (resultVector) ,+r (vector)
	;:
	;:q0 , q1 , q2 , q3 , q4, memory
	;);

	vpop {q4}	;restore registers
		
	FUNCTION_RETURN	;HQNeonVector4TransformCoordMatrix3x4
	;================================================================

	FUNCTION_BEGIN HQNeonMultiVector4TransformCoordMatrix3x4	;void HQNeonMultiVector4TransformCoordMatrix3x4(const float *vector ,hq_uint32 numVec , const float * matrix , float * resultVector )
	;r0 = vector, r1 = numVec, r2 = matrix, r3 = resultVector

	push {r5}			;save register
	vpush {q4}	;save registers

	vld4.32		{d2 , d3  ,d4 , d5} , [r2	@128]!			;2 rows of transposed matrix
	vld1.32		{d6 , d7} , [r2	@128]					;last row of matrix
					 
	vmov.f32		s3	, #1.0							;q0 = _ _ _ 1
					 
	mov			r5	, #0							;r5 = 0
					 
HQNEONMULTIVECTOR4TRANSFORMCOORDMATRIX3X4LABEL0	;loop
					 
	cmp			r5 , r1						
	bhs			HQNEONMULTIVECTOR4TRANSFORMCOORDMATRIX3X4LABEL1							;end loop if r5 >= numVec
					 
	vld1.32		{d0} , [r0	@64]	!						;x y
	vld1.32			{d1[0]}   , [r0 @32]							;z
					 
	NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK3
					 
	vst1.32		{d8}  , [r3	@64]!						;store x y
	vst1.32			{d9[0]}   , [r3 @32]	!					;store z
	vst1.32			{d1[1]}	  , [r3 @32]	!				;store 1
					 
	add			r0 , r0 , #8						;next source vector
					 
	add			r5 , r5 , #1						;increase by 1
	b				HQNEONMULTIVECTOR4TRANSFORMCOORDMATRIX3X4LABEL0							;loop
					 
HQNEONMULTIVECTOR4TRANSFORMCOORDMATRIX3X4LABEL1		;end loop									
		
	;old gcc inline asm			 
	;:+r (matrix) , +r (vector) , +r (resultVector) 
	;:[numVec]r (numVec)
	;:q0 , q1 , q2 , q3 , q4, memory , r5
	;);
		
	vpop {q4}	;restore registers
	pop		{r5}		;restore register

	FUNCTION_RETURN	;HQNeonMultiVector4TransformCoordMatrix3x4

	;==========================================================
	
	FUNCTION_BEGIN HQNeonVector4TransformNormalMatrix3x4	;void HQNeonVector4TransformNormalMatrix3x4(const float *vector ,const float * matrix ,  float * resultVector )
	
	vpush {q4}	;save registers

	vld1.32		{d0 , d1} , [r0	@128]				;vector
	vmov.i64		d9 , 0x0000000000000000				;d9 = 0 0
					 
	vld4.32		{d2 , d3  ,d4 , d5} , [r1	@128]!	;2 rows of transposed matrix
	vld1.32		{d6 , d7} , [r1	@128]				;last row of matrix
					 
	NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK2
					 
	vst1.32		{d8 , d9}  , [r2	@128 ]				;store x y z 0
	
	;old gcc inline asm				 
	;:+r (matrix) 
	;:r (resultVector)  , r (vector)
	;:q0 , q1 , q2 , q3 , q4, memory
	;);

	vpop {q4}	;restore registers
		
	FUNCTION_RETURN	;HQNeonVector4TransformNormalMatrix3x4

	;======================================================
	
	FUNCTION_BEGIN HQNeonMultiVector4TransformNormalMatrix3x4	;void HQNeonMultiVector4TransformNormalMatrix3x4(const float *vector ,hq_uint32 numVec , const float * matrix ,  float * resultVector )
	;r0 = vector, r1 = numVec, r2 = matrix, r3 = resultVector

	push {r5}			;save register
	vpush {q4}	;save registers

	vld4.32		{d2 , d3  ,d4 , d5} , [r2	@128]!	;2 rows of transposed matrix
	vld1.32		{d6 , d7} , [r2	@128]				;last row of matrix
	vmov.i64		d9 , 0x0000000000000000				;d9 = 0 0
					 
	mov			r5	, #0							;r5 = 0
					 
HQNEONMULTIVECTOR4TRANSFORMNORMALMATRIX3X4LABEL0	;loop
					 
	cmp			r5 , r1						
	bhs			HQNEONMULTIVECTOR4TRANSFORMNORMALMATRIX3X4LABEL1							;end loop if r5 >= numVec
					 
	vld1.32		{d0 , d1} , [r0	@128]!					;vector
					 
	NEON_MATRIX3X4_MUL_VEC_ASM_BLOCK2
					 
	vst1.32		{d8 , d9}  , [r3	@128]!					;store x y z 0
					 
	add			r5 , r5 , #1						;increase by 1
	b				HQNEONMULTIVECTOR4TRANSFORMNORMALMATRIX3X4LABEL0							;loop
					 
HQNEONMULTIVECTOR4TRANSFORMNORMALMATRIX3X4LABEL1										
		
	;old gcc inline asm			 
	;:+r (matrix)   , +r (vector),+r (resultVector)
	;:[numVec] r (numVec)
	;:q0 , q1 , q2 , q3 , q4, memory , r5
	;);
		
	vpop {q4}	;restore registers
	pop		{r5}		;restore register

	FUNCTION_RETURN	;HQNeonMultiVector4TransformNormalMatrix3x4
	
	;=======================================

	FUNCTION_BEGIN HQNeonMatrix4TransposeImpl	;void HQNeonMatrix4Transpose(const float * matrix , float * result)
	

	vld4.32	{d0 , d2 , d4 , d6}	, [r0	@128]!						
	vld4.32	{d1 , d3 , d5 , d7}	, [r0	@128]						
					 
					 
	vst1.32	{d0 , d1} , [r1	@128] !								
	vst1.32	{d2 , d3} , [r1	@128] !								
	vst1.32	{d4 , d5} , [r1	@128] !								
	vst1.32	{d6 , d7} , [r1	@128]								
	
	;old gcc inline asm	
	;:"+r" (matrix) , "+r" (result)
	;:
	;:"q0" , "q1" , "q2" , "q3" ,"memory"
	;);


	FUNCTION_RETURN ;HQNeonMatrix4TransposeImpl

	END

