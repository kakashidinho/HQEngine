/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQUtilMathPCH.h"
#include "../HQ3DMath.h"

HQVisibility HQSphere::Cull(const HQPlane* planes,const hq_int32 numPlanes) const
{
	float d;
	bool centerOutSide = false;
	for (hq_int32 i = 0 ; i < numPlanes ; ++i)
	{
		d = planes[i].N * this->center + planes[i].D;
		if (fabs(d) < this->radius)
			return HQ_CLIPPED;
		if (d > 0)
			centerOutSide = true;
	}
	if (centerOutSide)
		return HQ_CULLED;
	return HQ_VISIBLE;
}
bool HQSphere::Intersect(const HQVector4 & p0,const HQVector4 & p1,const HQVector4 & p2) const
{
#ifdef HQ_EXPLICIT_ALIGN
	HQA16ByteStorage<sizeof(HQPlane) + 2 * sizeof(HQRay3D) + sizeof(HQVector4)> storage;
	void *pBuffer = storage();
	HQPlane &plane = *(HQPlane::PlNew(pBuffer, p0 , p1 , p2 , true));
	HQRay3D &ray1 = *(HQRay3D::PlNew((char*)pBuffer + sizeof(HQPlane)));
	HQRay3D &ray2 = *(HQRay3D::PlNew((char*)pBuffer + sizeof(HQPlane) + sizeof(HQRay3D)));
	HQVector4 &temp = *(HQVector4::PlNew((char*)pBuffer + sizeof(HQPlane) + 2 *sizeof(HQRay3D)));
#else

	HQPlane plane(p0 , p1 ,p2 , true);
	HQRay3D ray1 , ray2;
	HQVector4 temp;
#endif
	
	float d = plane.N * this->center + plane.D;
	if (fabs(d) > this->radius)//không cắt mặt phẳng
		return false;
	//hình chiếu của tâm hình cầu lên mặt phẳng
	HQVector4Mul(d, &plane.N, &temp);
	HQVector4Sub(&this->center, &temp, &ray1.O);//p
	

/*------------------------
Kiểm tra điểm nằm trong tam giác

v1 = (p1 - p0) , v2 = (p2 - p0)  

p = p0 + s * v1 + t * v2
s + t <= 1 && 0 <= s && 0<= t

s * v1 + t * v2 = p - p0

v3 = (p - p0)

s * v1 + t * v2 = v3
v1 * v1 * s + v1 * v2 * t = v1 * v3
v1 * v2 * s + v2 * v2 * t = v2 * v3

A = v1 ^ 2     B = v2 ^ 2    C = v1 * v2    D = v1 * v3   E = v2 * v3

A * s + C * t = D
C * s + B * t = E

s = (BD - CE) / (AB - C^2) 
	(v2 * v2) * (v1 * v3) - (v1 * v2) * (v2 * v3)
=  -------------------------------------------------
	(v1 * v1) * (v2 * v2) - (v1 * v2) * (v1 * v2)

t = (AE - DC) / (AB - C^2)
	(v1 * v1) * (v2 * v3) - (v1 * v3) * (v1 * v2)
=  -------------------------------------------------
	(v1 * v1) * (v2 * v2) - (v1 * v2) * (v1 * v2)

-------------------------*/
	HQVector4Sub(&p1, &p0, &ray1.D);//v1
	HQVector4Sub(&p2, &p0, &ray2.D);//v2
	HQVector4Sub(&ray1.O, &p0, &ray2.O);//v3

	float A = ray1.D.LengthSqr();//v1 * v1
	float B = ray2.D.LengthSqr();//v2 * v2
	float C = ray1.D * ray2.D;//v1 * v2
	float D = ray1.D * ray2.O;//v1 * v3
	float E = ray2.D * ray2.O;//v2 * v3
	
	float AB = A * B;
	float C2 = C * C;

	float s = (B * D - C * E) / (AB - C2);
	float t = (A * E - D * C) / (AB - C2);

	if ((s >= 0) && (t >= 0) && (s + t <= 1))
		return true;//hình chiếu tâm nằm trong tam giác

	//kiểm tra có cắt cạnh nào không
	ray1.O = p0 ; 
	ray2.O = p0 ; 

	if (ray1.Intersect(*this , NULL , NULL ,1.0f) || 
		ray2.Intersect(*this , NULL , NULL ,1.0f) )
		return true;
	ray1.O = p1 ; 
	HQVector4Sub(&p2, &p1, &ray1.D);

	return ray1.Intersect(*this , NULL , NULL , 1.0f);

}
