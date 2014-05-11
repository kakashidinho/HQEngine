#include "Light.h"

#include <sstream>


static hquint32 g_lightCounter = 0;

//calculate perpendicular vector
static void GetPerpendicularVector(hqfloat32 originalX, hqfloat32 originalY, hqfloat32 originalZ, HQFloat4& out){
	if (originalX == 0.f && originalY == 0.0f)
		out.Set(0.0f, -originalZ, 0.0f);
	else
		out.Set(originalY, -originalX, 0.0f);
}

/*--------diffuse Spot Light----------*/
DiffuseSpotLight::DiffuseSpotLight(
	const HQColor &ambient,
	const HQColor& diffuse,
	hqfloat32 posX, hqfloat32 posY, hqfloat32 posZ,
	hqfloat32 dirX, hqfloat32 dirY, hqfloat32 dirZ,
	hqfloat32 _angle,
	hqfloat32 _theta,//inner cone angle in radian
	hqfloat32 _falloff,
	hqfloat32 _maxRange,
	HQRenderAPI renderApi
	)
	: BaseLight(ambient, diffuse, posX, posY, posZ),
	angle(_angle), theta(_theta), falloff(_falloff), maxRange(_maxRange),
	_direction(HQVector4::New())
{
	HQFloat4 upVec;//up vector for light camera
	GetPerpendicularVector(dirX, dirY, dirZ, upVec);

	std::stringstream lightName; 
	lightName << "light" << (g_lightCounter++);

	_lightCam = new HQCamera(
		lightName.str().c_str(),
		posX, posY, posZ,
		upVec.x, upVec.y, upVec.z,
		dirX, dirY, dirZ,
		angle,
		1.0f,
		1.0f,
		maxRange,
		renderApi
		);
	
	//register light camera's update listener
	_lightCam->SetListener(this);

	//get world position
	_lightCam->GetWorldPositionVec(*_position);
	//get world direction
	_lightCam->GetWorldDirectionVec(*_direction);
}

DiffuseSpotLight::~DiffuseSpotLight()
{
	delete _lightCam;
	delete _direction;
}

void DiffuseSpotLight::OnUpdated(const HQSceneNode * node, hqfloat32 dt)
{
	//get world position
	_lightCam->GetWorldPositionVec(*_position);
	//get world direction
	_lightCam->GetWorldDirectionVec(*_direction);
}

/*----------specular spot light ---------------*/
SpecularSpotLight::SpecularSpotLight(
	const HQColor &ambient,
	const HQColor& diffuse,
	const HQColor& specular,
	hqfloat32 posX, hqfloat32 posY, hqfloat32 posZ,
	hqfloat32 dirX, hqfloat32 dirY, hqfloat32 dirZ,
	hqfloat32 angle,//cone angle in radian
	hqfloat32 theta,//inner cone angle in radian
	hqfloat32 falloff,
	hqfloat32 maxRange,
	HQRenderAPI renderApi
	)
	: DiffuseSpotLight(
		ambient,
		diffuse,
		posX, posY, posZ,
		dirX, dirY, dirZ,
		angle, theta, 
		falloff,
		maxRange,
		renderApi
	),
	specularColor(specular)
{

}