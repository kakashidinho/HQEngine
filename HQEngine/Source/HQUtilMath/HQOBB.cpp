#include "HQUtilMathPCH.h"
#include "../HQ3DMath.h"


//***************************************************************************************
//biến đổi hình hộp với ma trận biến đổi, không có kiểm tra lại các vector 4D là điểm hay vector
//***************************************************************************************
void HQOBB::Transform(const HQOBB &source, const HQMatrix4 &mat){
	fA[0]=source.fA[0];
	fA[1]=source.fA[1];
	fA[2]=source.fA[2];

	HQVector4TransformNormal(&source.vA[0],&mat,&vA[0]);
	HQVector4TransformNormal(&source.vA[1],&mat,&vA[2]);
	HQVector4TransformNormal(&source.vA[2],&mat,&vA[2]);
	HQVector4TransformCoord(&source.vCenter,&mat,&vCenter);
}

void HQOBB::Transform(const HQOBB &source, const HQMatrix3x4 &mat){
	fA[0]=source.fA[0];
	fA[1]=source.fA[1];
	fA[2]=source.fA[2];

	HQVector4TransformNormal(&source.vA[0],&mat,&vA[0]);
	HQVector4TransformNormal(&source.vA[1],&mat,&vA[2]);
	HQVector4TransformNormal(&source.vA[2],&mat,&vA[2]);
	HQVector4TransformCoord(&source.vCenter,&mat,&vCenter);
}
//********************************************************************************
//kiểm tra hình hộp nằm ngoài hay cắt 1 phần thể tính nhìn (tạo bởi các mặt phẳng)
//********************************************************************************
HQ_FORCE_INLINE HQVisibility IsOBBCulled(const HQOBB *obb , const HQPlane *planes, const hq_int32 numPlanes)
{
	HQVisibility re = HQ_VISIBLE;
	hq_float32 radius,test;

	for(hq_int32 i = 0 ; i < numPlanes ; ++i){
		const HQVector4 &vN = planes[i].N;
		//chiếu các đường chéo lên pháp vector của mặt phẳng .lấy giá trị tuyệt đối lớn nhất
		radius=	fabs(obb->fA[0] * (vN * obb->vA[0]))+
				fabs(obb->fA[1] * (vN * obb->vA[1]))+
				fabs(obb->fA[2] * (vN * obb->vA[2]));

		test= vN * obb->vCenter + planes[i].D;

		if(test > radius)
			return HQ_CULLED;
		else if(test > -radius) re = HQ_CLIPPED;
	}
	return re;
}

HQVisibility HQOBB::Cull(const HQPlane *planes, const hq_int32 numPlanes) const{
	return IsOBBCulled(this , planes , numPlanes);
}

//*************************************************************************
//hàm chiếu hình hộp lên 1 vector ,lưu giá trị tọa độ điểm lớn nhất
//và điểm nhỏ nhất của hình chiếu trên đường thẳng đi qua vector 
//*************************************************************************
void ObbProj(const HQOBB& obb,const HQVector4& v,hq_float32* pMin,hq_float32 *pMax){
	hq_float32 c=v*obb.vCenter;

	hq_float32 r=fabs(obb.fA[0]*(v*obb.vA[0]))+fabs(obb.fA[1]*(v*obb.vA[1]))+fabs(obb.fA[2]*(v*obb.vA[2]));

	*pMin=c-r;
	*pMax=c+r;
}
//*********************************************************************
//hàm chiếu hình tam giác lên 1 vector,lưu giá trị tọa độ điểm lớn nhất
//và điểm nhỏ nhất của hình chiếu trên đường thẳng đi qua vector 
//*********************************************************************
void TriProj(const HQVector4 &p0,const HQVector4&p1 ,const HQVector4& p2,const HQVector4& v,hq_float32* pMin,hq_float32 *pMax){
	hq_float32 pr=v*p0;
	*pMin=pr;
	*pMax=pr;

	pr=v*p1;
	if(pr<*pMin) *pMin=pr;
	else if (pr>*pMax) *pMax=pr;

	pr=v*p2;
	if(pr<*pMin) *pMin=pr;
	else if (pr>*pMax) *pMax=pr;
}

//************************************
//kiểm tra hình hộp cắt tam giác
//************************************
bool HQOBB::Intersect(const HQVector4 &p0, const HQVector4 &p1, const HQVector4 &p2) const{
//*************************************************************************************
//Kỹ thuật separate axes theorem (SAT)
//kiểm tra chiếu hình hộp và tam giác lên 13 trục ,nếu hình chiếu của cả 2 trên bất kỳ 
//trục nào không đè lên nhau thì => không cắt.
//13 trục là :
//N- pháp tuyến của mặt chứa tam giác
//
//A[0] -trục cơ sở thứ 1 của hình hộp
//A[1]
//A[2]

//A[0] x E[0]
//A[0] x E[1]
//A[0] x E[2]

//A[1] x E[0]
//A[1] x E[1]
//A[1] x E[2]

//A[2] x E[0]
//A[2] x E[1]
//A[2] x E[2]

//E[3] là 3 cạnh của tam giác
//E[0] = p[1] - p[0]
//E[1] = p[2] - p[1]
//E[2] = p[0] - p[2]
//************************************************************************************

	hq_float32 min0,max0,min1,max1;
	hq_float32 o;
	HQ_DECL_STACK_VECTOR4_ARRAY( E , 4 );
	HQVector4 &N = E[4];
	
	HQVector4Sub(&p1, &p0, &E[0]);
	HQVector4Sub(&p2, &p1, &E[1]);

	//N
	N.Cross(E[0],E[1]);

	min0=N*p0;
	max0=min0; //hình chiếu của tam giác lên N là 1 điểm

	ObbProj(*this,N,&min1,&max1);
	if (min0>max1||min1>max0)
		return false;//hình chiếu không đè lên nhau

	//trục cơ sở 1
	o=this->vA[0]*this->vCenter;

	TriProj(p0,p1,p2,vA[0],&min0,&max0);
	min1=o-fA[0];
	max1=o+fA[0];
	if (min0>max1||min1>max0)
		return false;//hình chiếu không đè lên nhau

	//trục cơ sở 2
	o=this->vA[1]*this->vCenter;

	TriProj(p0,p1,p2,vA[1],&min0,&max0);
	min1=o-fA[1];
	max1=o+fA[1];
	if (min0>max1||min1>max0)
		return false;//hình chiếu không đè lên nhau

	//trục cơ sở 3
	o=this->vA[2]*this->vCenter;

	TriProj(p0,p1,p2,vA[2],&min0,&max0);
	min1=o-fA[2];
	max1=o+fA[2];
	if (min0>max1||min1>max0)
		return false;//hình chiếu không đè lên nhau

	//Các trục tích vô hướng của cạnh tam giác & trục cơ sở của hình hộp
	HQVector4Sub(&p0, &p2, &E[2]);

	for(hq_int32 i=0;i<3;++i)
		for(hq_int32 j=0;j<3;++j){
			N.Cross(vA[i],E[j]);
			TriProj(p0,p1,p2,N,&min0,&max0);
			ObbProj(*this,N,&min1,&max1);
			if (min0>max1||min1>max0)
				return false;//hình chiếu không đè lên nhau
		}
	return true;
}
//******************************
//Kiểm tra 2 hình hộp cắt nhau
//******************************
bool HQOBB::Intersect(const HQOBB &obb) const{
//*******************************************************************
//tương tự kiểm tra cắt tam giác,nhưng lần này ta dùng 15 trục:
//3 trục cơ sở của hình hộp 1,3 trục cơ sở của hình hộp 2 và các 
//tích có hướng của các trục cơ sở của 2 hình hộp 
//
//kiểm tra độ dài hình chiếu khoảng cách giữa 2 tâm 2 hình hộp lên 1 trục 
//nếu lớn hơn tổng độ dài hình chiếu đường chéo lớn nhất của 2 hình thì
//suy ra không cắt
//
//Giả sử v là vector chỉ phương của 1 trục được chọn để chiếu lên
//tọa độ vector v này là trong không gian địa phương của hình 1
//
//T là vector hiệu tọa độ 2 tâm trong không gian địa phương của hình 1
//2 vector đường chéo khi chiếu lên trục có độ dài lớn nhất của hình 1 và 2 lần lượt là a,b
//2 vector a,b lấy trong không gian địa phương của hình hộp chứa chúng
//
//R là ma trận chứa theo cột 3 trục cơ sở của hình hộp 2 trong không gian 
//địa phương của hình hộp 1
//
//Ta kiểm tra |v*T| > |v*a| + |R^T * v *b| (R^T là ma trận đảo của R)
//R^T * v là tọa độ vector v trong không gian hình hộp 2
//|v*T| là độ dài hình chiếu khoảng cách 2 tâm lên v
//|v*a| là độ dài hình chiếu của a lên v
//|R^T * v *b | là độ dài hình chiếu của b lên v
//*******************************************************************
	HQ_DECL_STACK_VECTOR4(vDiff);
	HQVector4Sub(&obb.vCenter, &vCenter, &vDiff);//vector hiệu vị trí 2 tâm của 2 hình hộp

	hq_float32 T[3]; //chứa tọa độ vector hiệu trên trong không gian tọa độ địa phương của hình hộp 1
	hq_float32 t;//t chứa độ dài hình chiếu khoảng cách giữa 2 tâm hình hộp
	hq_float32 R[3][3];//ma trận 3 chiều R chứa theo cột 3 trục cơ sở của hình hộp 2 
	//trong không gian địa phương của hình hộp 1
	//R[i][j]=trục cơ sở i cùa hình hộp 1 nhân vô hướng trục j của hình hộp 2
	hq_float32 AbsR[3][3];//AbsR[i][j] chứa giá trị tuyệt đối của R[i][j]
	hq_float32 ra,rb;//độ dài lớn nhất hình chiếu của đường chéo 2 hình hộp lên 1 trục

	//trục 1 của hình hộp 1
	//thêm 1 chút sai số để tránh trường hợp có 2 trục của 2 hình hộp gần song song 
	//dẫn đến trục tính vô hướng gần bằng 0 => phép so sánh 0>0
	//thêm sai số thì vế trái sẽ lớn hơn 1 chút,giảm sai xót
	R[0][0]=vA[0]*obb.vA[0];	AbsR[0][0]=fabs(R[0][0])+EPSILON;
	R[0][1]=vA[0]*obb.vA[1];	AbsR[0][1]=fabs(R[0][1])+EPSILON;
	R[0][2]=vA[0]*obb.vA[2];	AbsR[0][2]=fabs(R[0][2])+EPSILON;
	ra=fA[0];
	rb=obb.fA[0]*AbsR[0][0]+obb.fA[1]*AbsR[0][1]+obb.fA[2]*AbsR[0][2];
	T[0]=vA[0]*vDiff;
	t=fabs(T[0]);
	if(t>(ra+rb))
		return false;
	//trục 2 của hình hộp 1
	R[1][0]=vA[1]*obb.vA[0];	AbsR[1][0]=fabs(R[1][0])+EPSILON;
	R[1][1]=vA[1]*obb.vA[1];	AbsR[1][1]=fabs(R[1][1])+EPSILON;
	R[1][2]=vA[1]*obb.vA[2];	AbsR[1][2]=fabs(R[1][2])+EPSILON;
	ra=fA[1];
	rb=obb.fA[0]*AbsR[1][0]+obb.fA[1]*AbsR[1][1]+obb.fA[2]*AbsR[1][2];
	T[1]=vA[1]*vDiff;
	t=fabs(T[1]);
	if(t>(ra+rb))
		return false;
	//trục 3 của hình hộp 1
	R[2][0]=vA[2]*obb.vA[0];	AbsR[2][0]=fabs(R[2][0])+EPSILON;
	R[2][1]=vA[2]*obb.vA[1];	AbsR[2][1]=fabs(R[2][1])+EPSILON;
	R[2][2]=vA[2]*obb.vA[2];	AbsR[2][2]=fabs(R[2][2])+EPSILON;
	ra=fA[2];
	rb=obb.fA[0]*AbsR[2][0]+obb.fA[1]*AbsR[2][1]+obb.fA[2]*AbsR[2][2];
	T[2]=vA[2]*vDiff;
	t=fabs(T[2]);
	if(t>(ra+rb))
		return false;

	//trục 1 của hình hộp 2
	ra=fA[0]*AbsR[0][0]+fA[1]*AbsR[1][0]+fA[2]*AbsR[2][0];
	rb=obb.fA[0];
	t=fabs(obb.vA[0]*vDiff);
	if(t>(ra+rb))
		return false;
	//trục 2 của hình hộp 2
	ra=fA[0]*AbsR[0][1]+fA[1]*AbsR[1][1]+fA[2]*AbsR[2][1];
	rb=obb.fA[1];
	t=fabs(obb.vA[1]*vDiff);
	if(t>(ra+rb))
		return false;
	//trục 3 của hình hộp 2
	ra=fA[0]*AbsR[0][2]+fA[1]*AbsR[1][2]+fA[2]*AbsR[2][2];
	rb=obb.fA[2];
	t=fabs(obb.vA[2]*vDiff);
	if(t>(ra+rb))
		return false;

	//Các trục tính vô hướng của các trục cơ sở 2 hình hộp
	
	//v=A0xB0
	ra=fA[1]*AbsR[2][0]+fA[2]*AbsR[1][0];
	rb=obb.fA[1]*AbsR[0][2]+obb.fA[2]*AbsR[0][1];
	if(fabs(T[2]*R[1][0]-T[1]*R[2][0])>ra+rb)return false;
	//v=A0xB1
	ra=fA[1]*AbsR[2][1]+fA[2]*AbsR[1][1];
	rb=obb.fA[0]*AbsR[0][2]+obb.fA[2]*AbsR[0][0];
	if(fabs(T[2]*R[1][1]-T[1]*R[2][1])>ra+rb)return false;
	//v=A0xB2
	ra=fA[1]*AbsR[2][2]+fA[2]*AbsR[1][2];
	rb=obb.fA[0]*AbsR[0][1]+obb.fA[1]*AbsR[0][0];
	if(fabs(T[2]*R[1][2]-T[1]*R[2][2])>ra+rb)return false;
	//v=A1xB0
	ra=fA[0]*AbsR[2][0]+fA[2]*AbsR[0][0];
	rb=obb.fA[1]*AbsR[1][2]+obb.fA[2]*AbsR[1][1];
	if(fabs(T[0]*R[2][0]-T[2]*R[0][0])>ra+rb)return false;
	//v=A1xB1
	ra=fA[0]*AbsR[2][1]+fA[2]*AbsR[0][1];
	rb=obb.fA[0]*AbsR[1][2]+obb.fA[2]*AbsR[1][0];
	if(fabs(T[0]*R[2][1]-T[2]*R[0][1])>ra+rb)return false;
	//v=A1xB2
	ra=fA[0]*AbsR[2][2]+fA[2]*AbsR[0][2];
	rb=obb.fA[0]*AbsR[1][1]+obb.fA[1]*AbsR[1][0];
	if(fabs(T[0]*R[2][2]-T[2]*R[0][2])>ra+rb)return false;
	//v=A2xB0
	ra=fA[0]*AbsR[1][0]+fA[1]*AbsR[0][0];
	rb=obb.fA[1]*AbsR[2][2]+obb.fA[2]*AbsR[2][1];
	if(fabs(T[1]*R[0][0]-T[0]*R[1][0])>ra+rb)return false;
	//v=A2xB1
	ra=fA[0]*AbsR[1][1]+fA[1]*AbsR[0][1];
	rb=obb.fA[0]*AbsR[2][2]+obb.fA[2]*AbsR[2][0];
	if(fabs(T[1]*R[0][1]-T[0]*R[1][1])>ra+rb)return false;
	//v=A2xB2
	ra=fA[0]*AbsR[1][2]+fA[1]*AbsR[0][2];
	rb=obb.fA[0]*AbsR[2][1]+obb.fA[1]*AbsR[2][0];
	if(fabs(T[1]*R[0][2]-T[0]*R[1][2])>ra+rb)return false;
	return true;
}

bool HQOBB::Intersect(const HQAABB& aabb) const
{
	HQ_DECL_STACK_VAR_ARRAY(HQPlane , boxPlanes , 6);
	aabb.GetPlanes(boxPlanes);
	
	float radius , test;
	//right plane (normal = 1 , 0, 0)
	//chiếu các đường chéo lên pháp vector của mặt phẳng .lấy giá trị tuyệt đối lớn nhất
	radius=	fabs(this->fA[0] * this->vA[0].x)+
			fabs(this->fA[1] * this->vA[1].x)+
			fabs(this->fA[2] * this->vA[2].x);

	test= vCenter.x + boxPlanes[0].D;

	if(test > radius)
		return false;

	//left plane (normal = -1 , 0, 0)
	//chiếu các đường chéo lên pháp vector của mặt phẳng .lấy giá trị tuyệt đối lớn nhất

	test= -vCenter.x + boxPlanes[1].D;

	if(test > radius)
		return false;

	//up plane (normal = 0 , 1, 0)
	//chiếu các đường chéo lên pháp vector của mặt phẳng .lấy giá trị tuyệt đối lớn nhất
	radius=	fabs(this->fA[0] * this->vA[0].y)+
			fabs(this->fA[1] * this->vA[1].y)+
			fabs(this->fA[2] * this->vA[2].y);

	test= vCenter.y + boxPlanes[2].D;

	if(test > radius)
		return false;

	//bottom plane (normal = 0 , -1, 0)
	//chiếu các đường chéo lên pháp vector của mặt phẳng .lấy giá trị tuyệt đối lớn nhất

	test= -vCenter.y + boxPlanes[3].D;

	if(test > radius)
		return false;

	//near plane (normal = 0 , 0, 1)
	//chiếu các đường chéo lên pháp vector của mặt phẳng .lấy giá trị tuyệt đối lớn nhất
	radius=	fabs(this->fA[0] * this->vA[0].z)+
			fabs(this->fA[1] * this->vA[1].z)+
			fabs(this->fA[2] * this->vA[2].z);

	test= vCenter.z + boxPlanes[4].D;

	if(test > radius)
		return false;

	//far plane (normal = 0 , 0, -1)
	//chiếu các đường chéo lên pháp vector của mặt phẳng .lấy giá trị tuyệt đối lớn nhất

	test= -vCenter.z + boxPlanes[5].D;

	if(test > radius)
		return false;
	
	
	return true;

/*seperate axis version
//*******************************************************************
//tương tự kiểm tra cắt hình hộp OBB, coi hình hộp AABB là 1 hình hộp OBB , 
//	dùng các trục sau :
//3 trục cơ sở của hình hộp 1,3 trục cơ sở của hình hộp 2 và các 
//tích có hướng của các trục cơ sở của 2 hình hộp 
//
//kiểm tra độ dài hình chiếu khoảng cách giữa 2 tâm 2 hình hộp lên 1 trục 
//nếu lớn hơn tổng độ dài hình chiếu đường chéo lớn nhất của 2 hình thì
//suy ra không cắt
//
//Giả sử v là vector chỉ phương của 1 trục được chọn để chiếu lên
//tọa độ vector v này là trong không gian địa phương của hình 1
//
//T là vector hiệu tọa độ 2 tâm trong không gian địa phương của hình 1
//2 vector đường chéo khi chiếu lên trục có độ dài lớn nhất của hình 1 và 2 lần lượt là a,b
//2 vector a,b lấy trong không gian địa phương của hình hộp chứa chúng
//
//R là ma trận chứa theo cột 3 trục cơ sở của hình hộp 2 trong không gian 
//địa phương của hình hộp 1
//
//Ta kiểm tra |v*T| > |v*a| + |R^T * v *b| (R^T là ma trận đảo của R)
//R^T * v là tọa độ vector v trong không gian hình hộp 2
//|v*T| là độ dài hình chiếu khoảng cách 2 tâm lên v
//|v*a| là độ dài hình chiếu của a lên v
//|R^T * v *b | là độ dài hình chiếu của b lên v
//*******************************************************************
	HQ_DECL_STACK_VECTOR4(vDiff);
	vDiff = (aabb.vMax + aabb.vMin) * 0.5f;
	vDiff -= this->vCenter;//vector hiệu vị trí 2 tâm của 2 hình hộp

	hq_float32 T[3]; //chứa tọa độ vector hiệu trên trong không gian tọa độ địa phương của hình hộp 1
	hq_float32 t;//t chứa độ dài hình chiếu khoảng cách giữa 2 tâm hình hộp
	
	//R[3][3] ma trận 3 chiều R chứa theo cột 3 trục cơ sở của hình hộp 2 trong không gian địa phương của hình hộp 1
	//R[i][0] = this->vA[i].x ; R[i][1] = this->vA[i].y ; R[i][2] = this->vA[i].z
	hq_float32 AbsR[3][3];//AbsR[i][j] chứa giá trị tuyệt đối của R[i][j]
	hq_float32 ra,rb;//độ dài lớn nhất hình chiếu của đường chéo 2 hình hộp lên 1 trục
	hq_float32 aabbFA[3];//nửa độ dài các cạnh của hình hộp AABB
	
	aabbFA[0] = (aabb.vMax.x - aabb.vMin.x) * 0.5f;
	aabbFA[1] = (aabb.vMax.y - aabb.vMin.y) * 0.5f;
	aabbFA[2] = (aabb.vMax.z - aabb.vMin.z) * 0.5f;

	//trục 1 của hình hộp 1
	//thêm 1 chút sai số để tránh trường hợp có 2 trục của 2 hình hộp gần song song 
	//dẫn đến trục tính vô hướng gần bằng 0 => phép so sánh 0>0
	//thêm sai số thì vế trái sẽ lớn hơn 1 chút,giảm sai xót
	AbsR[0][0]=fabs(vA[0].x)+EPSILON;
	AbsR[0][1]=fabs(vA[0].x)+EPSILON;
	AbsR[0][2]=fabs(vA[0].z)+EPSILON;
	ra=fA[0];
	rb=aabbFA[0]*AbsR[0][0]+aabbFA[1]*AbsR[0][1]+aabbFA[2]*AbsR[0][2];
	T[0]=vA[0]*vDiff;
	t=fabs(T[0]);
	if(t>(ra+rb))
		return false;
	//trục 2 của hình hộp 1
	AbsR[1][0]=fabs(vA[1].x)+EPSILON;
	AbsR[1][1]=fabs(vA[1].x)+EPSILON;
	AbsR[1][2]=fabs(vA[1].z)+EPSILON;
	ra=fA[1];
	rb=aabbFA[0]*AbsR[1][0]+aabbFA[1]*AbsR[1][1]+aabbFA[2]*AbsR[1][2];
	T[1]=vA[1]*vDiff;
	t=fabs(T[1]);
	if(t>(ra+rb))
		return false;
	//trục 3 của hình hộp 1
	AbsR[2][0]=fabs(vA[2].x)+EPSILON;
	AbsR[2][1]=fabs(vA[2].x)+EPSILON;
	AbsR[2][2]=fabs(vA[2].z)+EPSILON;
	ra=fA[2];
	rb=aabbFA[0]*AbsR[2][0]+aabbFA[1]*AbsR[2][1]+aabbFA[2]*AbsR[2][2];
	T[2]=vA[2]*vDiff;
	t=fabs(T[2]);
	if(t>(ra+rb))
		return false;

	//trục 1 của hình hộp 2
	ra=fA[0]*AbsR[0][0]+fA[1]*AbsR[1][0]+fA[2]*AbsR[2][0];
	rb=aabbFA[0];
	t=fabs(vDiff.x);
	if(t>(ra+rb))
		return false;
	//trục 2 của hình hộp 2
	ra=fA[0]*AbsR[0][1]+fA[1]*AbsR[1][1]+fA[2]*AbsR[2][1];
	rb=aabbFA[1];
	t=fabs(vDiff.y);
	if(t>(ra+rb))
		return false;
	//trục 3 của hình hộp 2
	ra=fA[0]*AbsR[0][2]+fA[1]*AbsR[1][2]+fA[2]*AbsR[2][2];
	rb=aabbFA[2];
	t=fabs(vDiff.z);
	if(t>(ra+rb))
		return false;

	//Các trục tính vô hướng của các trục cơ sở 2 hình hộp
	
	//v=A0xB0
	ra=fA[1]*AbsR[2][0]+fA[2]*AbsR[1][0];
	rb=aabbFA[1]*AbsR[0][2]+aabbFA[2]*AbsR[0][1];
	if(fabs(T[2]*vA[1].x-T[1]*vA[2].x)>ra+rb)return false;
	//v=A0xB1
	ra=fA[1]*AbsR[2][1]+fA[2]*AbsR[1][1];
	rb=aabbFA[0]*AbsR[0][2]+aabbFA[2]*AbsR[0][0];
	if(fabs(T[2]*vA[1].y-T[1]*vA[2].y)>ra+rb)return false;
	//v=A0xB2
	ra=fA[1]*AbsR[2][2]+fA[2]*AbsR[1][2];
	rb=aabbFA[0]*AbsR[0][1]+aabbFA[1]*AbsR[0][0];
	if(fabs(T[2]*vA[1].z-T[1]*vA[2].z)>ra+rb)return false;
	//v=A1xB0
	ra=fA[0]*AbsR[2][0]+fA[2]*AbsR[0][0];
	rb=aabbFA[1]*AbsR[1][2]+aabbFA[2]*AbsR[1][1];
	if(fabs(T[0]*vA[2].x-T[2]*vA[0].x)>ra+rb)return false;
	//v=A1xB1
	ra=fA[0]*AbsR[2][1]+fA[2]*AbsR[0][1];
	rb=aabbFA[0]*AbsR[1][2]+aabbFA[2]*AbsR[1][0];
	if(fabs(T[0]*vA[2].y-T[2]*vA[0].y)>ra+rb)return false;
	//v=A1xB2
	ra=fA[0]*AbsR[2][2]+fA[2]*AbsR[0][2];
	rb=aabbFA[0]*AbsR[1][1]+aabbFA[1]*AbsR[1][0];
	if(fabs(T[0]*vA[2].z-T[2]*vA[0].z)>ra+rb)return false;
	//v=A2xB0
	ra=fA[0]*AbsR[1][0]+fA[1]*AbsR[0][0];
	rb=aabbFA[1]*AbsR[2][2]+aabbFA[2]*AbsR[2][1];
	if(fabs(T[1]*vA[0].x-T[0]*vA[1].x)>ra+rb)return false;
	//v=A2xB1
	ra=fA[0]*AbsR[1][1]+fA[1]*AbsR[0][1];
	rb=aabbFA[0]*AbsR[2][2]+aabbFA[2]*AbsR[2][0];
	if(fabs(T[1]*vA[0].y-T[0]*vA[1].y)>ra+rb)return false;
	//v=A2xB2
	ra=fA[0]*AbsR[1][2]+fA[1]*AbsR[0][2];
	rb=aabbFA[0]*AbsR[2][1]+aabbFA[1]*AbsR[2][0];
	if(fabs(T[1]*vA[0].z-T[0]*vA[1].z)>ra+rb)return false;
	return true;
	*/
}

bool HQOBB::Intersect(const HQSphere &sphere) const
{
	HQ_DECL_STACK_VECTOR4(transformedCenter);
	//transform sphere to obb local space
	HQVector4TransformCoord(&sphere.center , this , &transformedCenter);
	//now just check intersection between AABB and sphere
	float dMin = 0.0f;
	
	float r2 = sqr(sphere.radius);

	if (transformedCenter.x < -this->fA[0])
		dMin += sqr(transformedCenter.x + this->fA[0]);
	else if (transformedCenter.x > this->fA[0])
		dMin += sqr(transformedCenter.x - this->fA[0]);

	if (transformedCenter.y < -this->fA[1])
		dMin += sqr(transformedCenter.y + this->fA[1]);
	else if (transformedCenter.y > this->fA[1])
		dMin += sqr(transformedCenter.y - this->fA[1]);

	if (transformedCenter.z < -this->fA[2])
		dMin += sqr(transformedCenter.z + this->fA[2]);
	else if (transformedCenter.z > this->fA[2])
		dMin += sqr(transformedCenter.z - this->fA[2]);

	if (dMin <= r2)
		return true;
	return false;
}
