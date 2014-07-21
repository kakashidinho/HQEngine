/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQUtilMathPCH.h"
#include "../HQ3DMath.h"
#ifdef HQ_NEON_MATH
#include "arm_neon_math/HQNeonVector.h"
#elif defined HQ_DIRECTX_MATH
#include "directx_math/HQDXVector.h"
#endif

//*********************************************************************
//chuẩn hóa mặt phẳng,pháp vector sẽ là vector đơn vị
//*********************************************************************

HQPlane& HQPlane::Normalize()
{
#if defined HQ_CMATH || defined HQ_NEON_ASM
	N.w=0.0f;
	hq_float32 _1_over_length;
	_1_over_length = 1.0f / sqrtf(N.x * N.x + N.y * N.y + N.z * N.z);
	N *= _1_over_length;
#elif defined HQ_NEON_MATH
	
	register hq_float32 _1_over_length;
	
	//normalize N
	asm volatile(
				 "vmov.i64		d1  , #0x0000000000000000				\n\t"//d1 = 0	0
				 "vld1.32		{d0} , [%1, : 64]			\n\t" //load x y
				 "add			%1, %1, #8					\n\t"
				 "vld1.32		{d1[0]}	,  [%1 , :32]			\n\t"//load z = > q0 = x y z 0
				 
				 HQ_VECTOR_NORMALIZE_ASM_BLOCK//1/length in d3
				 
				 "vst1.32		{d0 , d1} , [%1, : 128]		\n\t"//store x y z 0
				 
				 "vmov			%0 , s6					\n\t"
				 
				 :"=r" (_1_over_length) 
				 :"r"  (N.v) 
				 :"q0"  , "q1" , "d4" , "memory"
					 );
	
#elif defined HQ_DIRECTX_MATH
	N.w=0.0f;

	//normalize the normal and get the 1/(normal's length)
	register hq_float32 _1_over_length;

	XMVECTOR simdNormal = XMLoadFloat4A((XMFLOAT4A*) N.v);

	XMVECTOR recpLen = XMVector3ReciprocalLength(simdNormal);

	simdNormal = XMVectorMultiply(simdNormal, recpLen);// *= (1 / length)

	XMStoreFloat4A((XMFLOAT4A*) N.v, simdNormal);

	_1_over_length = XMVectorGetX(recpLen);

#else/*SSE*/
	hq_float32 _1_over_length;
	
	hq_sse_float4 m0,m1,m2,m3;
	m0=_mm_load_ps(N.v);//copy vector data to 16 bytes(128 bit) aligned memory
	m2=m0;//copy m0 vào m2
	m0=_mm_mul_ps(m0,m0);//nhân vector với chính nó x^2 y^2 z^2 w^2

	m1=_mm_shuffle_ps(m0,m0,SSE_SHUFFLEL(1,0,0,0));//	y^2 x^2 x^2 x^2
	m3=_mm_shuffle_ps(m0,m0,SSE_SHUFFLEL(2,2,1,0));//	z^2 z^2 y^2 x^2

	m0=_mm_add_ps(m1,m0);//x^2+y^2		y^2+x^2		z^2+x^2		x^2+x^2
	m0=_mm_add_ps(m3,m0);//x^2+y^2+z^2		x^2+y^2+z^2		x^2+y^2+z^2		x^2+x^2+x^2

	hq_sse_float4 temp=_mm_rsqrt_ps(m0);//	tính gần đúng 1/căn ,nhanh hơn tính căn rồi lấy 1 chia cho căn

	
	//Newton Raphson iteration y(n+1)=1/2 *(y(n)*(3-x*y(n)^2)); //phương pháp giảm sai số
	m0=_mm_mul_ps(_mm_mul_ps(_3Halves_1Zero,temp),_mm_sub_ps(_4Threes,_mm_mul_ps(m0,_mm_mul_ps(temp,temp))));

	m2=_mm_mul_ps(m2,m0);// nhân 1/độ dài => chia cho độ dài
	_mm_store_ps(N.v,m2);
	
	_mm_store_ss(&_1_over_length,m0);
#endif

	D*=_1_over_length;

	return *this;
}
//
//trasform the plane
//
void HQPlane::Transform(const HQPlane& source, const HQMatrix4& mat)
{
	//find inverse transpose matrix
	HQ_DECL_STACK_MATRIX4_CTOR_PARAMS(invTrans, (NULL));
	HQMatrix4Inverse(&mat, &invTrans);

	invTrans.Transpose();

	source.ToFloat4(this->N);

	HQVector4Transform(&this->N, &invTrans, &this->N);

	this->D = this->N.w;
	this->N.w = 0.f;
}

void HQPlane::Transform(const HQPlane& source, const HQMatrix3x4& mat)
{
	//find inverse transpose matrix
	HQ_DECL_STACK_MATRIX4_CTOR_PARAMS(invTrans, (NULL));
	HQMatrix3x4Inverse(&mat, &invTrans);

	source.ToFloat4(this->N);

	HQVector4Transform(&this->N, &invTrans, &this->N);

	this->D = this->N.w;
	this->N.w = 0.f;
}

void HQPlane::TransformInvTrans(const HQPlane& source, const HQMatrix4& mat)
{
	source.ToFloat4(this->N);

	HQVector4Transform(&this->N, &mat, &this->N);

	this->D = this->N.w;
	this->N.w = 0.f;
}

//*********************************************************************
//kiềm tra đa giác có cắt hay nằm ở 1 phía so với mặt phẳng
//*********************************************************************
HQPlane::Side HQPlane::CheckSide(const HQPolygon3D &Poly)const{
	hq_int32 nFront,nBack,nIn;//số điểm nằm ở trước,sau và nằm trên mặt phẳng
	nFront=nBack=nIn=0;
	Side side;
	HQPolygon3D* pPoly=(HQPolygon3D*)&Poly;//loại bỏ ràng buộc const
	hq_int32 numP=pPoly->GetNumPoints();

	for(hq_int32 i=0;i<numP;++i){
		side=this->CheckSide(pPoly->pPoints[i]);
		if(side==FRONT_PLANE) nFront++;
		else if(side==BACK_PLANE) nBack++;
		else {//nếu nằm trên mặt phẳng,coi như cả 2 phía đều có điểm này
			nFront++;
			nBack++;
			nIn++;
		}
	}
	if(nIn==numP) return IN_PLANE;
	else if (nFront==numP) return FRONT_PLANE;
	else if (nBack==numP) return BACK_PLANE;
	else return BOTH_SIDE;
}
//*********************************************************************
//kiểm tra tam giác cắt mặt phẳng
//*********************************************************************
bool HQPlane::Intersect(const HQVector4 &p0,const HQVector4& p1,const HQVector4& p2)const{
	hq_int32 n=this->CheckSide(p0);
	if(n==CheckSide(p1)&&n==CheckSide(p2))
		return false;
	return true;
}

//********************************************************************
//kiểm tra 2 mặt phẳng cắt nhau
//********************************************************************
bool HQPlane::Intersect(const HQPlane &plane, HQRay3D *rIntersection)const{
	HQ_DECL_STACK_VECTOR4( cross );
	HQVector4 *pN2=(HQVector4*)&plane.N;
	HQVector4Cross(&N,pN2,&cross);

	hq_float32 sqrlengthCross= cross.LengthSqr();

	if(sqrlengthCross<0.00000001f)
		return false;

//tính đường thẳng giao của 2 mặt phẳng 
//phương trình đường thằng :P(t)=P0 + t*D
//****************************************
//tọa độ điểm nằm trên đường thẳng giao 2 mặt (N1,D1) và (N2,D2):
//		(D1*N2-D2*N1)x(N1 x N2)
//P0=	-----------------------
//			|N1 x N2|^2
//
//|N1 x N2|^2 = |N1|^2 * |N2|^2 * (1 - (cos P)^2) (P góc giữa 2 vector N1 ,N2)
//= d0*d1-d^2 (với d0=|N1|^2  ; d1=|N2|^2 ;  d= N1*N2 = |N1| * |N2| * cos P)

// từ a x (b x c)= (a*c)*b - (a*b)*c =>
// (D1*N2-D2*N1)x(N1 x N2) = ((D1*N2-D2*N1)* N2) * N1 - ((D1*N2-D2*N1)* N1) * N2
//=(D1*|N2|^2-D2*N1*N2) * N1 - (D1*N1*N2-D2*|N1|^2)*N2
//=(d1*D1-d*D2)*N1 + (d0*D2-d*D1)*N2
//
//Đặt C0=(d1*D1-d*D2)/(d0*d1-d^2)  và C1=(d0*D2-d*D1)/(d0*d1-d^2)
//=>P0= C0*N1 + C1*N2
//****************************************
	if(rIntersection){
		hq_float32 d=N*plane.N;
		hq_float32 d0=N.LengthSqr();
		hq_float32 d1=pN2->LengthSqr();
		hq_float32 det=d0*d1-sqr(d);

		if(fabs(det)<0.00000001f)
			return false;
		hq_float32 invDet=1.0f/det;

		hq_float32 C0=(d1*D-d*plane.D)*invDet;
		hq_float32 C1=(d0*plane.D-d*D)*invDet;

		rIntersection->D=cross;

		//rIntersection->O=C0*N+C1*plane.N;
		HQ_DECL_STACK_VECTOR4_CTOR_PARAMS(temp, (NULL));
		HQVector4Mul(C0, &N, &rIntersection->O);
		HQVector4Mul(C1, &plane.N, &temp);
		rIntersection->O += temp;
	}
	return true;
}

//*****************************************************************
//kiểm tra mặt phẳng cắt hình hộp HQAABB
//*****************************************************************

bool HQPlane::Intersect(const HQAABB &aabb)const{
	HQ_DECL_STACK_2VECTOR4( vMin,vMax );

	if(N.x>=0.0f)
	{
		vMin.x=aabb.vMin.x;
		vMax.x=aabb.vMax.x;
	}
	else{
		vMax.x=aabb.vMin.x;
		vMin.x=aabb.vMax.x;
	}

	if(N.y>=0.0f)
	{
		vMin.y=aabb.vMin.y;
		vMax.y=aabb.vMax.y;
	}
	else{
		vMax.y=aabb.vMin.y;
		vMin.y=aabb.vMax.y;
	}

	if(N.z>=0.0f)
	{
		vMin.z=aabb.vMin.z;
		vMax.z=aabb.vMax.z;
	}
	else{
		vMax.z=aabb.vMin.z;
		vMin.z=aabb.vMax.z;
	}
	if (N*vMin+D>0.0f) //điểm tọa độ vMin ko được nằm ở mặt trên mặt phẳng
		return false;
	if(N*vMax+D<0.0f)//điểm tọa độ vMax ko được nằm ở mặt dưới mặt phẳng
		return false;
	return true;
}

//********************************************************
//kiểm tra mặt phẳng cắt hình hộp HQOBB
//********************************************************
bool HQPlane::Intersect(const HQOBB &obb) const{
//chiếu hình hộp lên pháp vector của mặt phẳng
	hq_float32 radius=fabs(obb.fA[0]*(N*obb.vA[0]))+fabs(obb.fA[1]*(N*obb.vA[1]))+fabs(obb.fA[2]*(N*obb.vA[2]));
	hq_float32 distance=Distance(obb.vCenter);
	return (distance<=radius);
}
