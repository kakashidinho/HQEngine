/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQUtilMathPCH.h"
#include "../HQ3DMath.h"
#ifdef LINUX
#include <string.h>//for memset
#endif

/*----------------------------------
HQPolygon3D
-----------------------------------*/

HQPolygon3D::~HQPolygon3D() {
	SafeDeleteArray(pPoints);
	SafeDeleteArray(pIndices);
}

//*********************************
//tính hình hộp bao ngoài
//*********************************
void HQPolygon3D::CalBoundingBox(){
	aabb.vMax=aabb.vMin=pPoints[0];
	for (hq_int32 i=1;i<numP;++i){
		if(pPoints[i].x>aabb.vMax.x)
			aabb.vMax.x=pPoints[i].x;
		else if(pPoints[i].x<aabb.vMin.x)
			aabb.vMin.x=pPoints[i].x;
		if(pPoints[i].y>aabb.vMax.y)
			aabb.vMax.y=pPoints[i].y;
		else if(pPoints[i].y<aabb.vMin.y)
			aabb.vMin.y=pPoints[i].y;
		if(pPoints[i].z>aabb.vMax.z)
			aabb.vMax.z=pPoints[i].z;
		else if(pPoints[i].z<aabb.vMin.z)
			aabb.vMin.z=pPoints[i].z;
	}
}
//**************************************************
//khởi tạo danh sách điểm và chỉ mục
//**************************************************
void HQPolygon3D::Set(const HQVector4 *_pPoints, hq_int32 _numP, const hq_uint32 *_pIndices, hq_int32 _numI){
	HQ_DECL_STACK_2VECTOR4(e0,e1);

	SafeDeleteArray(pPoints);
	SafeDeleteArray(pIndices);
	numP=_numP;
	numI=_numI;

	pPoints=HQVector4::NewArray(numP);
	pIndices=new hq_uint32[numI];

	memcpy(pPoints,_pPoints,numP*sizeof(HQVector4));
	memcpy(pIndices,_pIndices,numI*sizeof(hq_uint32));

	HQVector4Sub(&pPoints[pIndices[1]], &pPoints[pIndices[0]], &e0);

	for(hq_int32 i=2;i<numI;++i){
		HQVector4Sub(&pPoints[pIndices[i]], &pPoints[pIndices[0]], &e1);

		hq_float32 angle=fabs(e1.AngleWith(e0));

		if(angle>0.0f+EPSILON&&angle<HQPiFamily::PI-EPSILON)//kiểm tra vector song song
			break;
	}
	plane.N.Cross(e0,e1);
	plane.N.Normalize();
	plane.D=-(plane.N * pPoints[pIndices[0]]);
	this->CalBoundingBox();
}

//**************************************
//Đảo hướng mặt trước mặt sau
//**************************************
void HQPolygon3D::SwapFace(){
	hq_int32 mid=(numI)/2;
	for(hq_int32 i=1;i<mid+1;++i)//đảo thứ tự chỉ mục
		swapui(pIndices[i],pIndices[numI-i-1]);
	plane.N*=-1.0f;
	plane.D*=-1.0f;
}
//************************************************************
//kiểm tra đa giác nằm ngoài,bị cắt xén,hay nằm trong hình hộp
//************************************************************
HQVisibility HQPolygon3D::Cull(const HQAABB& _aabb)const{

	if(!this->aabb.Intersect(_aabb))//hình hộp bao ngoài của đa giác không cắt hình hộp cần xét
		return HQ_CULLED;//đa giác nằm ngoài hình hộp đang xét
	//kiểm tra số điểm của đa giác nằm trong hình hộp
	hq_int32 nIn=0;//số điểm nằm trong hình hộp
	for(hq_int32 i=0;i<numP;++i){
		if(_aabb.ContainsPoint(pPoints[i]))
		{
			nIn++;
		}
	}
	if(nIn==numP)
		return HQ_VISIBLE;//nằm hoàn toàn trong hình hộp
	if(nIn)
		return HQ_CLIPPED;//bị cắt

	//Kiểm tra các cạnh của đa giác có cắt hình hộp không
	hq_int32 nCurrent;
	HQ_DECL_STACK_VAR( HQRay3D  , ray);//cạnh kiểm tra
	for(hq_int32 i=1;i<=numP;++i){
		if(i==numP) nCurrent=0;
		else nCurrent=i;
		ray.O=pPoints[i-1];
		HQVector4Sub(&pPoints[nCurrent], &pPoints[i-1], &ray.D);
		if(ray.Intersect(_aabb,0,1.0f))//cạnh cắt hình hộp
			return HQ_CLIPPED;
	}
	return HQ_CULLED;//không có điểm nào trong hình hộp và không có cạnh nào cắt
}
//************************************
//cắt đa giác làm 2 phần bởi mặt phẳng
//************************************
void HQPolygon3D::Clip(const HQPlane &plane, HQPolygon3D *pFront, HQPolygon3D *pBack)const{
	if(!pFront&&!pBack) return;
	
	HQA16ByteStorage<3 * sizeof(HQVector4) + sizeof(HQRay3D) > buffer;
	HQVector4 *pVecPool = (HQVector4*)buffer();
	HQVector4 &vA = pVecPool[0];
	HQVector4 &vB = pVecPool[1];
	HQVector4 &vHit = pVecPool[2];//vHit - điểm cắt giữa đa giác và mặt phẳng (nếu có)
	HQRay3D& ray = *((HQRay3D*)&pVecPool[3]);//lưu đoạn thẳng nối 2 điểm để kiểm tra có cắt mặt phẳng không
	hq_float32 tHit;//lưu thời điểm cắt của ray và mặt phẳng

	HQVector4* vFront= HQVector4::NewArray(numP);//danh sách điểm nằm ở trước mặt phẳng
	HQVector4* vBack= HQVector4::NewArray(numP);//danh sách điểm nằm ở sau mặt phẳng

	hq_int32 nFront=0;//số điểm nằm mặt trước
	hq_int32 nBack=0;//số điểm nằm mặt sau
	hq_int32 nLoop,nCurrent;

	HQPlane* pPlane=(HQPlane*)&plane;//loại bỏ ràng buộc const
	switch(pPlane->CheckSide(pPoints[0]))//kiểm tra điểm thứ nhất
	{
	case HQPlane::FRONT_PLANE:
		vFront[nFront++]=pPoints[0];
		break;
	case HQPlane::BACK_PLANE:
		vBack[nBack++]=pPoints[0];
		break;
	case HQPlane::IN_PLANE://nằm trên mặt phẳng => cho vào cả 2 danh sách phía trước và sau mặt phẳng
		vFront[nFront++]=pPoints[0];
		vBack[nBack++]=pPoints[0];
		break;
	}

	for(nLoop=1;nLoop<(numP+1);++nLoop){
		if(nLoop==numP) nCurrent=0;
		else nCurrent=nLoop;

		vA=pPoints[nLoop-1];//điểm kế cận thứ tự trước điểm đang xét
		vB=pPoints[nCurrent];//điểm đang xét

		HQPlane::Side sideA=pPlane->CheckSide(vA);
		HQPlane::Side sideB=pPlane->CheckSide(vB);

		if(sideB==HQPlane::IN_PLANE){//điểm đang xét nằm trên mặt phẳng
			vFront[nFront++]=vB;
			vBack[nBack++]=vB;
		}
		else{//kiểm tra cạnh nối vA và vB có cắt mặt phẳng không
			ray.O=vA;
			HQVector4Sub(&vB, &vA, &ray.D);
			if(ray.Intersect(plane,&tHit,1.0f,false)&&sideA!=HQPlane::IN_PLANE)//cắt
			{
				//vHit=ray.O+tHit*ray.D;
				HQVector4Mul(tHit, &ray.D, &vHit);
				vHit += ray.O;
				//thêm điểm cắt vào cả 2 phía
				vFront[nFront++]=vHit;
				vBack[nBack++]=vHit;
			}
			if(nCurrent==0) continue;//điểm thứ 0 đã xét ở đầu

			if (sideB==HQPlane::FRONT_PLANE)//điểm đang xét ở mặt trước
				vFront[nFront++]=vB;
			else if (sideB==HQPlane::BACK_PLANE)//ở mặt sau
				vBack[nBack++]=vB;
		}//else
	}//for (nLoop)
	
	//tính toán chỉ số đỉnh kế cận tạo thành tam giác
	hq_uint32 i1,i2;//chỉ số đỉnh 2,3 ,đỉnh thứ 1 của tam giác luôn có chỉ số 0
	//       0        đa giác có 4 đỉnh 0,1,2,3
	//      /|\       danh sách kế cận là {0,1,2   ,  0,2,3}
	//     / | \
	//    /  |  3
	//   /   | /
	//  /____|/
	// 1     2
	hq_uint32 *iFront=0;
	hq_uint32 *iBack=0;

	hq_int32 numiF=(nFront-2)*3;//số chỉ số của đa giác ở phía trước
	hq_int32 numiB=(nBack-2)*3;//số chỉ số của đa giác ở phía sau
	if(nFront>2) //tạo chỉ số đỉnh kế cận tạo thành các tam giác khi số đỉnh ở phía này lớn hơn 2
	{
		iFront=new hq_uint32[numiF];
		i1=0;i2=1;
		for(nLoop=0;nLoop<numiF;nLoop+=3){
			i1++;
			i2++;
			iFront[nLoop]=0;
			iFront[nLoop+1]=i1;
			iFront[nLoop+2]=i2;
		}
	}
	if(nBack>2) //tạo chỉ số đỉnh kế cận tạo thành các tam giác khi số đỉnh ở phía này lớn hơn 2
	{
		iBack=new hq_uint32[numiB];
		i1=0;i2=1;
		for(nLoop=0;nLoop<numiB;nLoop+=3){
			i1++;
			i2++;
			iBack[nLoop]=0;
			iBack[nLoop+1]=i1;
			iBack[nLoop+2]=i2;
		}
	}
	//khởi tạo đa giác trước mặt phẳng nếu có
	if(pFront&&iFront){
		pFront->Set(vFront,nFront,iFront,numiF);
		//kiểm tra đa giác này cùng hướng với đa giác gốc
		if (plane.N*pFront->GetPlane().N<0)
			pFront->SwapFace();
	}
	if(pBack&&iBack){
		pBack->Set(vBack,nBack,iBack,numiB);
		//kiểm tra đa giác này cùng hướng với đa giác gốc
		if (plane.N*pBack->GetPlane().N<0)
			pBack->SwapFace();
	}
	SafeDeleteArray(vFront);
	SafeDeleteArray(vBack);
	SafeDeleteArray(iFront);
	SafeDeleteArray(iBack);
}
//************************************
//cắt đa giác bởi hình hộp
//************************************
void HQPolygon3D::Clip(const HQAABB& aabb){
	HQPlane planes[6];//6 mặt phẳng của hình hộp
	HQPolygon3D backPoly,clippedPoly;

	
	bool clipped=false;
	//các mặt phẳng của hình hộp pháp vector hướng ra ngoài hình hộp
	aabb.GetPlanes(planes);

	clippedPoly.Copy(*this);
	for(hq_int32 i = 0 ; i < 6 ; ++i){
		if(planes[i].CheckSide(clippedPoly) == HQPlane::BOTH_SIDE){
			clippedPoly.Clip(planes[i],0,&backPoly);
			clippedPoly.Copy(backPoly);
			clipped=true;
		}
	}
	if(clipped)
		this->Copy(clippedPoly);

}
//************************************
//copy
//************************************
HQPolygon3D::HQPolygon3D(const HQPolygon3D &source){
	this->pPoints = NULL;
	this->pIndices = NULL;
	this->flag = 0;

	Set(source.GetPoints(),source.GetNumPoints(),source.GetIndices(),source.GetNumIndices());
}
void HQPolygon3D::Copy(const HQPolygon3D &source){
	Set(source.GetPoints(),source.GetNumPoints(),source.GetIndices(),source.GetNumIndices());
}


/*-------------------------------------
HQPolygonList
--------------------------------------*/
HQPolygonList::HQPolygonList()
{
	this->numPoly = 0;
	this->headNode = NULL;
	this->pLastPolyNode = NULL;
}
HQPolygonList::~HQPolygonList()
{
	ClearPolyList();
}


void HQPolygonList::AddPolygon(const HQPolygon3D *poly)
{
	if(headNode == NULL)
	{
		headNode = HQ_NEW HQPolyListNode(poly);
		pLastPolyNode = headNode;
	}
	else
	{
		pLastPolyNode->pNext = HQ_NEW HQPolyListNode(poly);
		pLastPolyNode = pLastPolyNode->pNext;
	}

	this->numPoly ++;
}
void HQPolygonList::ClearPolyList()
{
	HQPolyListNode *pNode = headNode;
	while(pNode != NULL)
	{
		headNode = headNode->pNext;
		delete pNode;
		pNode = headNode;
	}

	headNode = NULL;
	pLastPolyNode = NULL;
	this->numPoly = 0;
}
