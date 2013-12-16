#include "HQUtilMathPCH.h"
#include "../HQ3DMath.h"

//*****************************************
//tạo hình hộp aabb bao ngoài obb
//*****************************************
HQ_FORCE_INLINE void ConstructAABBFromOBB(const HQOBB *obb , HQAABB *aabbOut)
{
	HQ_DECL_STACK_3VECTOR4( vA0,vA1,vA2);

	HQVector4Mul(obb->fA[0], &obb->vA[0], &vA0);
	HQVector4Mul(obb->fA[1], &obb->vA[1], &vA1);
	HQVector4Mul(obb->fA[2], &obb->vA[2], &vA2);

	if (vA0.x>vA1.x){
		if(vA0.x>vA2.x)
		{
			aabbOut->vMin.x=-vA0.x;aabbOut->vMax.x=vA0.x;
		}
		else{
			aabbOut->vMin.x=-vA2.x;aabbOut->vMax.x=vA2.x;
		}
	}
	else {
		if(vA1.x>vA2.x)
		{
			aabbOut->vMin.x=-vA1.x;aabbOut->vMax.x=vA1.x;
		}
		else{
			aabbOut->vMin.x=-vA2.x;aabbOut->vMax.x=vA2.x;
		}
	}

	if (vA0.y>vA1.y){
		if(vA0.y>vA2.y)
		{
			aabbOut->vMin.y=-vA0.y;aabbOut->vMax.y=vA0.y;
		}
		else{
			aabbOut->vMin.y=-vA2.y;aabbOut->vMax.y=vA2.y;
		}
	}
	else {
		if(vA1.y>vA2.y)
		{
			aabbOut->vMin.y=-vA1.y;aabbOut->vMax.y=vA1.y;
		}
		else{
			aabbOut->vMin.y=-vA2.y;aabbOut->vMax.y=vA2.y;
		}
	}

	if (vA0.z>vA1.z){
		if(vA0.z>vA2.z)
		{
			aabbOut->vMin.z=-vA0.z;aabbOut->vMax.z=vA0.z;
		}
		else{
			aabbOut->vMin.z=-vA2.z;aabbOut->vMax.z=vA2.z;
		}
	}
	else {
		if(vA1.z>vA2.z)
		{
			aabbOut->vMin.z=-vA1.z;aabbOut->vMax.z=vA1.z;
		}
		else{
			aabbOut->vMin.z=-vA2.z;aabbOut->vMax.z=vA2.z;
		}
	}

	aabbOut->vMin+=obb->vCenter;
	aabbOut->vMax+=obb->vCenter;
}

void HQAABB::Construct(const HQOBB &obb){
	ConstructAABBFromOBB(&obb , this);
}

bool HQAABB::ContainsOBB(const HQOBB &obb) const//kiểm tra hình hộp obb có nằm trong hình hộp aabb hay không
{
	HQ_DECL_STACK_VAR(HQAABB , obbBoundingBox);
	
	ConstructAABBFromOBB(&obb , &obbBoundingBox);
	
	return this->ContainsAABB(obbBoundingBox);
}

//****************************************************************************
//kiểm tra hình hộp bị cắt hay nằm ngoài thể tính nhìn (hợp bởi các mặt phẳng)
//****************************************************************************
HQVisibility HQAABB::Cull(const HQPlane *planes, const hq_int32 numPlanes){
	HQ_DECL_STACK_2VECTOR4 (_vMin,_vMax);
	bool Intersect=false;

	for(hq_int32 i=0;i<numPlanes;++i){
		if(planes[i].N.x>=0.0f)
		{
			_vMin.x=this->vMin.x;
			_vMax.x=this->vMax.x;
		}
		else{
			_vMax.x=this->vMin.x;
			_vMin.x=this->vMax.x;
		}

		if(planes[i].N.y>=0.0f)
		{
			_vMin.y=this->vMin.y;
			_vMax.y=this->vMax.y;
		}
		else{
			_vMax.y=this->vMin.y;
			_vMin.y=this->vMax.y;
		}

		if(planes[i].N.z>=0.0f)
		{
			_vMin.z=this->vMin.z;
			_vMax.z=this->vMax.z;
		}
		else{
			_vMax.z=this->vMin.z;
			_vMin.z=this->vMax.z;
		}
		if(planes[i].N*_vMin+planes[i].D>0.0f)
			return HQ_CULLED;
		if(planes[i].N*_vMax+planes[i].D>=0.0f)
			Intersect=true;
	}
	if(Intersect)
		return HQ_CLIPPED;
	return HQ_VISIBLE;
}

//************************************
//truy vấn các mặt phẳng của hình hộp
//************************************
void HQAABB::GetPlanes(HQPlane *planesOut)const{
	//Ox+
	planesOut[0].N.Set(1,0,0);
	planesOut[0].D = -vMax.x;
	//Ox-
	planesOut[1].N.Set(-1,0,0);
	planesOut[1].D = vMin.x;
	//Oy+
	planesOut[2].N.Set(0,1,0);
	planesOut[2].D = -vMax.y;
	//Oy-
	planesOut[3].N.Set(0,-1,0);
	planesOut[3].D = vMin.y;
	//Oz+
	planesOut[4].N.Set(0,0,1);
	planesOut[4].D = -vMax.z;
	//Oz-
	planesOut[5].N.Set(0,0,-1);
	planesOut[5].D = vMin.z;
}

//******************************************
//kiểm tra đoạn thẳng nằm trong
//******************************************
bool HQAABB::ContainsSegment(const HQVector4 &p0, const HQVector4 &p1)const{
	return (ContainsPoint(p0)&&ContainsPoint(p1));
}
bool HQAABB::ContainsSegment(const HQRay3D& ray,const hq_float32 t)const{
	HQ_DECL_STACK_VECTOR4(endP);
	//endP = ray.O+t*ray.D;
	HQVector4Mul(t, &ray.D, &endP);
	endP += ray.O;

	return (ContainsPoint(ray.O)&&ContainsPoint(endP));
}


//***************************************
//kiểm tra hình hộp cắt hình cầu
//***************************************

bool HQAABB::Intersect(const HQSphere &sphere) const
{
	float dMin = 0.0f;
	
	float r2 = sqr(sphere.radius);

	if (sphere.center.x < this->vMin.x)
		dMin += sqr(sphere.center.x - this->vMin.x);
	else if (sphere.center.x > this->vMax.x)
		dMin += sqr(sphere.center.x - this->vMax.x);

	if (sphere.center.y < this->vMin.y)
		dMin += sqr(sphere.center.y - this->vMin.y);
	else if (sphere.center.y > this->vMax.y)
		dMin += sqr(sphere.center.y - this->vMax.y);

	if (sphere.center.z < this->vMin.z)
		dMin += sqr(sphere.center.z - this->vMin.z);
	else if (sphere.center.z > this->vMax.z)
		dMin += sqr(sphere.center.z - this->vMax.z);

	if (dMin <= r2)
		return true;
	return false;
}
