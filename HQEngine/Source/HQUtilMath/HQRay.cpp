/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQUtilMathPCH.h"
#include "../HQ3DMath.h"
//*************************************************
//biến đổi tia với ma trận biến đổi mat
//*************************************************
void HQRay3D::Transform(const HQMatrix4& mat){
	HQVector4TransformCoord(&O,&mat,&O);
	HQVector4TransformNormal(&D,&mat,&D);
}

void HQRay3D::Transform(const HQMatrix3x4& mat){
	HQVector4TransformCoord(&O,&mat,&O);
	HQVector4TransformNormal(&D,&mat,&D);
}

void HQRayTransform(const HQRay3D* pRIn,const HQMatrix4 *pMat,HQRay3D* pROut){
	HQVector4TransformCoord(&pRIn->O,pMat,&pROut->O);
	HQVector4TransformNormal(&pRIn->D,pMat,&pROut->D);

}

void HQRayTransform(const HQRay3D* pRIn,const HQMatrix3x4 *pMat,HQRay3D* pROut){
	HQVector4TransformCoord(&pRIn->O,pMat,&pROut->O);
	HQVector4TransformNormal(&pRIn->D,pMat,&pROut->D);

}
//**************************************************
//kiểm tra tia đi qua tam giác 
//**************************************************
bool HQRay3D::Intersect(const HQVector4 &V0, const HQVector4 &V1, const HQVector4 &V2,hq_float32* pU,hq_float32* pV, hq_float32 *pT,bool Cull) const{
//**************************************************************************
//giải thuật Moller - Trumbore
//
// phương trình 3 ẩn (t,u,v) tìm tọa độ điểm giao của tia(gốc O hướng D) và tam giác (3 điểm V0,V1,V2)
// O + t*D= V0 + u*(V1-V0) + v*(V2-V0)
//=>-D*t + (V1-V0)*u + (V2-V0)*v=O - V0
//
//|t|					|Q*E2|
//|u|		=	1/det*	|P*T |
//|v|					|Q*D |
//E1=V1-V0 ,E2=V2-V0, P=D cross E2,det = P*E1 , T=O-V0, Q=T cross E1
//**************************************************************************
	

	HQ_DECL_STACK_VECTOR4_ARRAY(vArray , 5);
	HQVector4 &E1 = vArray[0];
	HQVector4 &E2 = vArray[1];
	HQVector4 &P = vArray[2];
	HQVector4 &T = vArray[3];
	HQVector4 &Q = vArray[4];

	HQVector4Sub(&V1, &V0, &E1);
	HQVector4Sub(&V2, &V0, &E2);
	P.Cross(D,E2);
	
	hq_float32 det=E1*P;

	//kiểm tra tia song song với mặt phẳng chứa tam giác
	if(Cull && det <EPSILON) //backface
		return false;
	else if (det < EPSILON && det > -EPSILON) 
		return false;
	
	//u
	HQVector4Sub(&O, &V0, &T);
	hq_float32 u=P*T;
	if(u<0.0f||u>det)
		return false;

	//v
	
	Q.Cross(T,E1);
	hq_float32 v=Q*D;
	if(v<0.0f||u+v>det)
		return false;

	hq_float32 invDet=1.0f/det;

	if(pU)
		*pU=u * invDet;
	if(pV)
		*pV=v * invDet;
	//t
	if(pT){
		hq_float32 t=Q*E2;
		
		t*=invDet;
		*pT=t;
	}
	return true;
}

//**************************************************
//kiểm tra tia đi qua tam giác ,có kiểm tra thời điểm cắt tối đa maxT (điểm cắt không vượt ra ngoài O+maxT*D)
//**************************************************

bool HQRay3D::Intersect(const HQVector4 &V0, const HQVector4 &V1, const HQVector4 &V2,hq_float32* pU,hq_float32* pV, hq_float32 *pT,hq_float32 maxT,bool Cull) const{
//**************************************************************************
//giải thuật Moller - Trumbore
//
// phương trình 3 ẩn (t,u,v) tìm tọa độ điểm giao của tia(gốc O hướng D) và tam giác (3 điểm V0,V1,V2)
// O + t*D= V0 + u*(V1-V0) + v*(V2-V0)
//=>-D*t + (V1-V0)*u + (V2-V0)*v=O - V0
//
//|t|					|Q*E2|
//|u|		=	1/det*	|P*T |
//|v|					|Q*D |
//E1=V1-V0 ,E2=V2-V0, P=D cross E2,det = P*E1 , T=O-V0, Q=T cross E1
//**************************************************************************
	HQ_DECL_STACK_5VECTOR4(E1 , E2 , P , T , Q);
	HQVector4Sub(&V1, &V0, &E1);
	HQVector4Sub(&V2, &V0, &E2);
	P.Cross(D,E2);
	
	hq_float32 det=E1*P;
	//kiểm tra tia song song với mặt phẳng chứa tam giác
	if(Cull && det <EPSILON) //backface
		return false;
	else if (det < EPSILON && det > -EPSILON) 
		return false;
	
	//u
	HQVector4Sub(&O, &V0, &T);
	hq_float32 u=P*T;
	if(u<0.0f||u>det)
		return false;

	//v
	
	Q.Cross(T,E1);
	hq_float32 v=Q*D;
	if(v<0.0f||u+v>det)
		return false;

	//t
	hq_float32 t=Q*E2;
	hq_float32 invDet=1.0f/det;
	t*=invDet;
	if(t>maxT)
		return false;

	if(pU)
		*pU=u * invDet;
	if(pV)
		*pV=v * invDet;
	if(pT)
		*pT=t;
	return true;
}
//*************************************************************************
//kiểm tra tia cắt đa giác
//*************************************************************************
bool HQRay3D::Intersect(const HQPolygon3D &poly, hq_float32 *pT) const{
	hq_int32 numI=poly.GetNumIndices();
	//kiểm tra từng tam giác của đa giác
	for (hq_int32 i=0;i<numI;i+=3){
		if(this->Intersect(poly.pPoints[poly.pIndices[i]],
									poly.pPoints[poly.pIndices[i+1]],
									poly.pPoints[poly.pIndices[i+2]],
									0,0,pT,false))
			return true;
	}
	return false;
}
bool HQRay3D::Intersect(const HQPolygon3D &poly, hq_float32 *pT,hq_float32 maxT)const{
	hq_int32 numI=poly.GetNumIndices();
	//kiểm tra từng tam giác của đa giác
	for (hq_int32 i=0;i<numI;i+=3){
		if(this->Intersect(poly.pPoints[poly.pIndices[i]],
									poly.pPoints[poly.pIndices[i+1]],
									poly.pPoints[poly.pIndices[i+2]],
									0,0,pT,maxT,false))
			return true;
	}
	return false;
}
//*************************************************************************
//kiểm tra tia cắt mặt phẳng
//*************************************************************************

bool HQRay3D::Intersect(const HQPlane &plane, hq_float32 *pT,bool Cull) const{
//**************************************************************************
//phương trình tia :P(t) = O + t*D
//phương trình mặt phẳng :pN*P + pD = 0
//pN*(O + t*D) + pD=0
//=>pN*O + pD=-pN*t*D
//=>t= -(pN*O + pD)/pN*D
//**************************************************************************
	hq_float32 vD=plane.N*D;
	if (vD<EPSILON&&vD>-EPSILON) //tia song song mặt phằng
		return false;
	if (Cull&&vD>0) //backface
		return false;
	hq_float32 V0=-((plane.N*O)+plane.D);

	hq_float32 t=V0/vD;

	if(t<0.0f)
		return false;

	if(pT)
		*pT=t;
	return true;

}

//*************************************************************************
//kiểm tra tia cắt mặt phẳng ,có kiểm tra thời điểm cắt tối đa maxT (điểm cắt không vượt ra ngoài O+maxT*D)
//*************************************************************************

bool HQRay3D::Intersect(const HQPlane &plane, hq_float32 *pT, hq_float32 maxT,bool Cull) const{
//**************************************************************************
//phương trình tia :P(t) = O + t*D
//phương trình mặt phẳng :pN*P + pD = 0
//pN*(O + t*D) + pD=0
//=>pN*O + pD=-pN*t*D
//=>t= -(pN*O + pD)/pN*D
//**************************************************************************
	hq_float32 vD=plane.N*D;
	if (vD<EPSILON&&vD>-EPSILON) //tia song song mặt phằng
		return false;
	if (Cull&&vD>0) //backface
		return false;
	hq_float32 V0=-((plane.N*O)+plane.D);

	hq_float32 t=V0/vD;

	if(t<0.0f||(t>maxT))
		return false;

	if(pT)
		*pT=t;
	return true;

}

//*************************************************************************
//kiểm tra tia cắt hình hộp HQAABB
//*************************************************************************

bool HQRay3D::Intersect(const HQAABB &aabb, hq_float32 *pT) const{
//**********************************************************
//với mỗi cặp mặt phẳng song song theo hướng trục Ox,Oy,Oz của hình hộp
//tính thời điểm giao mặt phẳng gần (tNear) và thời điểm giao mặt phẳng xa (tFar)
//nếu tNear lớn nhấn lớn hơn tFar nhỏ nhất => ko cắt
//**********************************************************
	hq_float32 tNear=-999999.9f;
	hq_float32 tFar=999999.9f;
	hq_float32 t0,t1;
	//cặp mặt song song theo hướng Ox
	if(fabs(D.x)<EPSILON){
		if(O.x<aabb.vMin.x||O.x>aabb.vMax.x)
			return false;
	}
	t0=(aabb.vMin.x-O.x)/D.x;
	t1=(aabb.vMax.x-O.x)/D.x;
	if(t0>t1) swapf(t0,t1);
	if(t0>tNear) tNear=t0;
	if(t1<tFar) tFar=t1;
	if(tNear>tFar) return false;
	if(tFar<0) return false;

	//cặp mặt song song theo hướng Oy
	if(fabs(D.y)<EPSILON){
		if(O.y<aabb.vMin.y||O.y>aabb.vMax.y)
			return false;
	}
	t0=(aabb.vMin.y-O.y)/D.y;
	t1=(aabb.vMax.y-O.y)/D.y;
	if(t0>t1) swapf(t0,t1);
	if(t0>tNear) tNear=t0;
	if(t1<tFar) tFar=t1;
	if(tNear>tFar) return false;
	if(tFar<0) return false;

	//cặp mặt song song theo hướng Oz
	if(fabs(D.z)<EPSILON){
		if(O.z<aabb.vMin.z||O.z>aabb.vMax.z)
			return false;
	}
	t0=(aabb.vMin.z-O.z)/D.z;
	t1=(aabb.vMax.z-O.z)/D.z;
	if(t0>t1) swapf(t0,t1);
	if(t0>tNear) tNear=t0;
	if(t1<tFar) tFar=t1;
	if(tNear>tFar) return false;
	if(tFar<0) return false;
	
	//trả về thời điểm cắt
	if(tNear>0) 
	{
		if (pT)
			*pT=tNear;
	}

	else 
	{
		if (pT)
			*pT=tFar;
	}

	return true;
}

//*************************************************************************
//kiểm tra tia cắt hình hộp HQAABB,có kiểm tra thời điểm cắt tối đa maxT
//*************************************************************************

bool HQRay3D::Intersect(const HQAABB &aabb, hq_float32 *pT,hq_float32 maxT) const{
//**********************************************************
//với mỗi cặp mặt phẳng song song theo hướng trục Ox,Oy,Oz của hình hộp
//tính thời điểm giao mặt phẳng gần (tNear) và thời điểm giao mặt phẳng xa (tFar)
//nếu tNear lớn nhấn lớn hơn tFar nhỏ nhất => ko cắt
//**********************************************************
	hq_float32 tNear=-999999.9f;
	hq_float32 tFar=999999.9f;
	hq_float32 t0,t1;
	//cặp mặt song song theo hướng Ox
	if(fabs(D.x)<EPSILON){
		if(O.x<aabb.vMin.x||O.x>aabb.vMax.x)
			return false;
	}
	t0=(aabb.vMin.x-O.x)/D.x;
	t1=(aabb.vMax.x-O.x)/D.x;
	if(t0>t1) swapf(t0,t1);
	if(t0>tNear) tNear=t0;
	if(t1<tFar) tFar=t1;
	if(tNear>tFar) return false;
	if(tFar<0) return false;

	//cặp mặt song song theo hướng Oy
	if(fabs(D.y)<EPSILON){
		if(O.y<aabb.vMin.y||O.y>aabb.vMax.y)
			return false;
	}
	t0=(aabb.vMin.y-O.y)/D.y;
	t1=(aabb.vMax.y-O.y)/D.y;
	if(t0>t1) swapf(t0,t1);
	if(t0>tNear) tNear=t0;
	if(t1<tFar) tFar=t1;
	if(tNear>tFar) return false;
	if(tFar<0) return false;

	//cặp mặt song song theo hướng Oz
	if(fabs(D.z)<EPSILON){
		if(O.z<aabb.vMin.z||O.z>aabb.vMax.z)
			return false;
	}
	t0=(aabb.vMin.z-O.z)/D.z;
	t1=(aabb.vMax.z-O.z)/D.z;
	if(t0>t1) swapf(t0,t1);
	if(t0>tNear) tNear=t0;
	if(t1<tFar) tFar=t1;
	if(tNear>tFar) return false;
	if(tFar<0) return false;
	
	//trả về thời điểm cắt
	hq_float32 tFinal;
	if(tNear>0) 
	{
		tFinal=tNear;
	}

	else 
	{
		tFinal=tFar;
	}
	if(tFinal>maxT)
		return false;
	if(pT)
		*pT=tFinal;

	return true;
}
//*************************************************************************
//kiểm tra tia cắt hình hộp HQOBB
//*************************************************************************

bool HQRay3D::Intersect(const HQOBB &obb, hq_float32 *pT) const{
//**********************************************************
//tương tự kiểm tra cắt HQAABB,chỉ khác là ta cần chuyển tia từ hệ trục thế giới 
//sang hệ trục tọa độ địa phương của hình hộp
//**********************************************************
	hq_float32 tNear=-999999.9f;
	hq_float32 tFar=999999.9f;
	hq_float32 t0,t1,o,f;
	
	HQ_DECL_STACK_VECTOR4(diff);
	HQVector4Sub(&O, &obb.vCenter, &diff);
	//cặp mặt song song theo hướng trục cơ sở thứ 1

	o=obb.vA[0]*diff;
	f=obb.vA[0]*D;
	if(fabs(f)<EPSILON){
		if(o<-obb.fA[0]||o>obb.fA[0])
			return false;
	}
	t0=(-obb.fA[0]-o)/f;
	t1=(obb.fA[0]-o)/f;
	if(t0>t1) swapf(t0,t1);
	if(t0>tNear) tNear=t0;
	if(t1<tFar) tFar=t1;
	if(tNear>tFar) return false;
	if(tFar<0) return false;

	//cặp mặt song song theo hướng trục cơ sở thứ 2
	o=obb.vA[1]*diff;
	f=obb.vA[1]*D;
	if(fabs(f)<EPSILON){
		if(o<-obb.fA[1]||o>obb.fA[1])
			return false;
	}
	t0=(-obb.fA[1]-o)/f;
	t1=(obb.fA[1]-o)/f;
	if(t0>t1) swapf(t0,t1);
	if(t0>tNear) tNear=t0;
	if(t1<tFar) tFar=t1;
	if(tNear>tFar) return false;
	if(tFar<0) return false;

	//cặp mặt song song theo hướng trục cơ sở thứ 3
	o=obb.vA[2]*diff;
	f=obb.vA[2]*D;
	if(fabs(f)<EPSILON){
		if(o<-obb.fA[2]||o>obb.fA[2])
			return false;
	}
	t0=(-obb.fA[2]-o)/f;
	t1=(obb.fA[2]-o)/f;
	if(t0>t1) swapf(t0,t1);
	if(t0>tNear) tNear=t0;
	if(t1<tFar) tFar=t1;
	if(tNear>tFar) return false;
	if(tFar<0) return false;
	
	//trả về thời điểm cắt
	if(tNear>0) 
	{
		if (pT)
			*pT=tNear;
	}

	else 
	{
		if (pT)
			*pT=tFar;
	}

	return true;
}

//*************************************************************************
//kiểm tra tia cắt hình hộp HQOBB,có kiểm tra thời điểm cắt tối đa maxT
//*************************************************************************

bool HQRay3D::Intersect(const HQOBB &obb, hq_float32 *pT,hq_float32 maxT) const{
//**********************************************************
//tương tự kiểm tra cắt HQAABB,chỉ khác là ta cần chuyển tia từ hệ trục thế giới 
//sang hệ trục tọa độ địa phương của hình hộp
//**********************************************************
	hq_float32 tNear=-999999.9f;
	hq_float32 tFar=999999.9f;
	hq_float32 t0,t1,o,f;
	
	HQ_DECL_STACK_VECTOR4(diff);
	HQVector4Sub(&O, &obb.vCenter, &diff);
	//cặp mặt song song theo hướng trục cơ sở thứ 1

	o=obb.vA[0]*diff;
	f=obb.vA[0]*D;
	if(fabs(f)<EPSILON){
		if(o<-obb.fA[0]||o>obb.fA[0])
			return false;
	}
	t0=(-obb.fA[0]-o)/f;
	t1=(obb.fA[0]-o)/f;
	if(t0>t1) swapf(t0,t1);
	if(t0>tNear) tNear=t0;
	if(t1<tFar) tFar=t1;
	if(tNear>tFar) return false;
	if(tFar<0) return false;

	//cặp mặt song song theo hướng trục cơ sở thứ 2
	o=obb.vA[1]*diff;
	f=obb.vA[1]*D;
	if(fabs(f)<EPSILON){
		if(o<-obb.fA[1]||o>obb.fA[1])
			return false;
	}
	t0=(-obb.fA[1]-o)/f;
	t1=(obb.fA[1]-o)/f;
	if(t0>t1) swapf(t0,t1);
	if(t0>tNear) tNear=t0;
	if(t1<tFar) tFar=t1;
	if(tNear>tFar) return false;
	if(tFar<0) return false;

	//cặp mặt song song theo hướng trục cơ sở thứ 3
	o=obb.vA[2]*diff;
	f=obb.vA[2]*D;
	if(fabs(f)<EPSILON){
		if(o<-obb.fA[2]||o>obb.fA[2])
			return false;
	}
	t0=(-obb.fA[2]-o)/f;
	t1=(obb.fA[2]-o)/f;
	if(t0>t1) swapf(t0,t1);
	if(t0>tNear) tNear=t0;
	if(t1<tFar) tFar=t1;
	if(tNear>tFar) return false;
	if(tFar<0) return false;
	
	//trả về thời điểm cắt
	hq_float32 tFinal;
	if(tNear>0) 
	{
		tFinal=tNear;
	}

	else 
	{
		tFinal=tFar;
	}
	if(tFinal>maxT)
		return false;
	if(pT)
		*pT=tFinal;

	return true;
}

//*************************************************************************
//kiểm tra tia cắt hình cầu
//*************************************************************************

bool HQRay3D::Intersect(const HQSphere& sphere,hq_float32 *pT1,hq_float32 *pT2) const
{
/*
dịch chuyển hình cầu về gốc tọa độ
tia : o + td
hình cầu : p * p = r^2
điểm vừa nằm trên hình cầu vửa nằm trên tia 
<=> (o + td) * (o + td) = r^2
<=> d^2 * t^2 + 2*o*d*t + o^2 = r^2
<=> A * t^2 + B * t + C = 0
A = d^2
B = 2 * o * d
C = o ^ 2 - r ^ 2

t1 = (-B - sqrt(B ^ 2 - 4 * A * C)) / (2 * A)
t2 = (-B + sqrt(B ^ 2 - 4 * A * C)) / (2 * A)
*/

	HQ_DECL_STACK_VECTOR4(newOrigin);
	
	HQVector4Sub(&this->O, &sphere.center, &newOrigin);

	float A = this->D.LengthSqr();
	float B = (newOrigin * this->D) * 2;
	float C = newOrigin.LengthSqr() - sqr(sphere.radius);

	float t1 , t2;
	float sqrtD;

	float D = sqr(B) - 4 * A * C;
	if (D < EPSILON)//D < 0
		return false;
	
	sqrtD = sqrtf(D);
	A = 1.0f / (2 * A);
	t1 = (-B - sqrtD) * A;	
	t2 = (-B + sqrtD) * A;	

	if (t2 < 0)
		return false;

	if (pT1 != NULL)
		*pT1 = t1;
	if (pT2 != NULL)
		*pT2 = t2;

	

	return true;
}

//*************************************************************************
//kiểm tra tia cắt hình cầu,có kiểm tra thời điểm cắt tối đa maxT
//*************************************************************************
bool HQRay3D::Intersect(const HQSphere& sphere,hq_float32 *pT1,hq_float32 *pT2,hq_float32 maxT) const
{
/*
dịch chuyển hình cầu về gốc tọa độ
tia : o + td
hình cầu : p * p = r^2
điểm vừa nằm trên hình cầu vửa nằm trên tia 
<=> (o + td) * (o + td) = r^2
<=> d^2 * t^2 + 2*o*d*t + o^2 = r^2
<=> A * t^2 + B * t + C = 0
A = d^2
B = 2 * o * d
C = o ^ 2 - r ^ 2

t1 = (-B - sqrt(B ^ 2 - 4 * A * C)) / (2 * A)
t2 = (-B + sqrt(B ^ 2 - 4 * A * C)) / (2 * A)
*/

	HQ_DECL_STACK_VECTOR4(newOrigin);
	
	HQVector4Sub(&this->O, &sphere.center, &newOrigin);

	float A = this->D.LengthSqr();
	float B = (newOrigin * this->D) * 2;
	float C = newOrigin.LengthSqr() - sqr(sphere.radius);

	float t1 , t2;
	float sqrtD;

	float D = sqr(B) - 4 * A * C;
	if (D < EPSILON)//D < 0
		return false;
	
	sqrtD = sqrtf(D);
	A = 1.0f / (2 * A);
	t1 = (-B - sqrtD) * A;	
	t2 = (-B + sqrtD) * A;	

	if (t2 < 0 || t1 > maxT)
		return false;

	if (pT1 != NULL)
		*pT1 = t1;
	if (pT2 != NULL)
		*pT2 = t2;

	

	return true;
}
