/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.security.interfaces.RSAPrivateCrtKey
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_SECURITY_INTERFACES_RSAPRIVATECRTKEY_HPP_DECL
#define J2CPP_JAVA_SECURITY_INTERFACES_RSAPRIVATECRTKEY_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace math { class BigInteger; } } }
namespace j2cpp { namespace java { namespace security { namespace interfaces { class RSAPrivateKey; } } } }


#include <java/lang/Object.hpp>
#include <java/math/BigInteger.hpp>
#include <java/security/interfaces/RSAPrivateKey.hpp>


namespace j2cpp {

namespace java { namespace security { namespace interfaces {

	class RSAPrivateCrtKey;
	class RSAPrivateCrtKey
		: public object<RSAPrivateCrtKey>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)
		J2CPP_DECLARE_METHOD(4)
		J2CPP_DECLARE_METHOD(5)
		J2CPP_DECLARE_FIELD(0)

		explicit RSAPrivateCrtKey(jobject jobj)
		: object<RSAPrivateCrtKey>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<java::security::interfaces::RSAPrivateKey>() const;


		local_ref< java::math::BigInteger > getCrtCoefficient();
		local_ref< java::math::BigInteger > getPrimeP();
		local_ref< java::math::BigInteger > getPrimeQ();
		local_ref< java::math::BigInteger > getPrimeExponentP();
		local_ref< java::math::BigInteger > getPrimeExponentQ();
		local_ref< java::math::BigInteger > getPublicExponent();

		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(0), J2CPP_FIELD_SIGNATURE(0), jlong > serialVersionUID;
	}; //class RSAPrivateCrtKey

} //namespace interfaces
} //namespace security
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_SECURITY_INTERFACES_RSAPRIVATECRTKEY_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_SECURITY_INTERFACES_RSAPRIVATECRTKEY_HPP_IMPL
#define J2CPP_JAVA_SECURITY_INTERFACES_RSAPRIVATECRTKEY_HPP_IMPL

namespace j2cpp {



java::security::interfaces::RSAPrivateCrtKey::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

java::security::interfaces::RSAPrivateCrtKey::operator local_ref<java::security::interfaces::RSAPrivateKey>() const
{
	return local_ref<java::security::interfaces::RSAPrivateKey>(get_jobject());
}

local_ref< java::math::BigInteger > java::security::interfaces::RSAPrivateCrtKey::getCrtCoefficient()
{
	return call_method<
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_CLASS_NAME,
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_METHOD_NAME(0),
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_METHOD_SIGNATURE(0), 
		local_ref< java::math::BigInteger >
	>(get_jobject());
}

local_ref< java::math::BigInteger > java::security::interfaces::RSAPrivateCrtKey::getPrimeP()
{
	return call_method<
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_CLASS_NAME,
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_METHOD_NAME(1),
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_METHOD_SIGNATURE(1), 
		local_ref< java::math::BigInteger >
	>(get_jobject());
}

local_ref< java::math::BigInteger > java::security::interfaces::RSAPrivateCrtKey::getPrimeQ()
{
	return call_method<
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_CLASS_NAME,
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_METHOD_NAME(2),
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< java::math::BigInteger >
	>(get_jobject());
}

local_ref< java::math::BigInteger > java::security::interfaces::RSAPrivateCrtKey::getPrimeExponentP()
{
	return call_method<
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_CLASS_NAME,
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_METHOD_NAME(3),
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_METHOD_SIGNATURE(3), 
		local_ref< java::math::BigInteger >
	>(get_jobject());
}

local_ref< java::math::BigInteger > java::security::interfaces::RSAPrivateCrtKey::getPrimeExponentQ()
{
	return call_method<
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_CLASS_NAME,
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_METHOD_NAME(4),
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_METHOD_SIGNATURE(4), 
		local_ref< java::math::BigInteger >
	>(get_jobject());
}

local_ref< java::math::BigInteger > java::security::interfaces::RSAPrivateCrtKey::getPublicExponent()
{
	return call_method<
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_CLASS_NAME,
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_METHOD_NAME(5),
		java::security::interfaces::RSAPrivateCrtKey::J2CPP_METHOD_SIGNATURE(5), 
		local_ref< java::math::BigInteger >
	>(get_jobject());
}


static_field<
	java::security::interfaces::RSAPrivateCrtKey::J2CPP_CLASS_NAME,
	java::security::interfaces::RSAPrivateCrtKey::J2CPP_FIELD_NAME(0),
	java::security::interfaces::RSAPrivateCrtKey::J2CPP_FIELD_SIGNATURE(0),
	jlong
> java::security::interfaces::RSAPrivateCrtKey::serialVersionUID;


J2CPP_DEFINE_CLASS(java::security::interfaces::RSAPrivateCrtKey,"java/security/interfaces/RSAPrivateCrtKey")
J2CPP_DEFINE_METHOD(java::security::interfaces::RSAPrivateCrtKey,0,"getCrtCoefficient","()Ljava/math/BigInteger;")
J2CPP_DEFINE_METHOD(java::security::interfaces::RSAPrivateCrtKey,1,"getPrimeP","()Ljava/math/BigInteger;")
J2CPP_DEFINE_METHOD(java::security::interfaces::RSAPrivateCrtKey,2,"getPrimeQ","()Ljava/math/BigInteger;")
J2CPP_DEFINE_METHOD(java::security::interfaces::RSAPrivateCrtKey,3,"getPrimeExponentP","()Ljava/math/BigInteger;")
J2CPP_DEFINE_METHOD(java::security::interfaces::RSAPrivateCrtKey,4,"getPrimeExponentQ","()Ljava/math/BigInteger;")
J2CPP_DEFINE_METHOD(java::security::interfaces::RSAPrivateCrtKey,5,"getPublicExponent","()Ljava/math/BigInteger;")
J2CPP_DEFINE_FIELD(java::security::interfaces::RSAPrivateCrtKey,0,"serialVersionUID","J")

} //namespace j2cpp

#endif //J2CPP_JAVA_SECURITY_INTERFACES_RSAPRIVATECRTKEY_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
