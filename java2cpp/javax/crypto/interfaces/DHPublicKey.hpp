/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: javax.crypto.interfaces.DHPublicKey
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVAX_CRYPTO_INTERFACES_DHPUBLICKEY_HPP_DECL
#define J2CPP_JAVAX_CRYPTO_INTERFACES_DHPUBLICKEY_HPP_DECL


namespace j2cpp { namespace javax { namespace crypto { namespace interfaces { class DHKey; } } } }
namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace math { class BigInteger; } } }
namespace j2cpp { namespace java { namespace security { class PublicKey; } } }


#include <java/lang/Object.hpp>
#include <java/math/BigInteger.hpp>
#include <java/security/PublicKey.hpp>
#include <javax/crypto/interfaces/DHKey.hpp>


namespace j2cpp {

namespace javax { namespace crypto { namespace interfaces {

	class DHPublicKey;
	class DHPublicKey
		: public object<DHPublicKey>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_FIELD(0)

		explicit DHPublicKey(jobject jobj)
		: object<DHPublicKey>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<javax::crypto::interfaces::DHKey>() const;
		operator local_ref<java::security::PublicKey>() const;


		local_ref< java::math::BigInteger > getY();

		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(0), J2CPP_FIELD_SIGNATURE(0), jlong > serialVersionUID;
	}; //class DHPublicKey

} //namespace interfaces
} //namespace crypto
} //namespace javax

} //namespace j2cpp

#endif //J2CPP_JAVAX_CRYPTO_INTERFACES_DHPUBLICKEY_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVAX_CRYPTO_INTERFACES_DHPUBLICKEY_HPP_IMPL
#define J2CPP_JAVAX_CRYPTO_INTERFACES_DHPUBLICKEY_HPP_IMPL

namespace j2cpp {



javax::crypto::interfaces::DHPublicKey::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

javax::crypto::interfaces::DHPublicKey::operator local_ref<javax::crypto::interfaces::DHKey>() const
{
	return local_ref<javax::crypto::interfaces::DHKey>(get_jobject());
}

javax::crypto::interfaces::DHPublicKey::operator local_ref<java::security::PublicKey>() const
{
	return local_ref<java::security::PublicKey>(get_jobject());
}

local_ref< java::math::BigInteger > javax::crypto::interfaces::DHPublicKey::getY()
{
	return call_method<
		javax::crypto::interfaces::DHPublicKey::J2CPP_CLASS_NAME,
		javax::crypto::interfaces::DHPublicKey::J2CPP_METHOD_NAME(0),
		javax::crypto::interfaces::DHPublicKey::J2CPP_METHOD_SIGNATURE(0), 
		local_ref< java::math::BigInteger >
	>(get_jobject());
}


static_field<
	javax::crypto::interfaces::DHPublicKey::J2CPP_CLASS_NAME,
	javax::crypto::interfaces::DHPublicKey::J2CPP_FIELD_NAME(0),
	javax::crypto::interfaces::DHPublicKey::J2CPP_FIELD_SIGNATURE(0),
	jlong
> javax::crypto::interfaces::DHPublicKey::serialVersionUID;


J2CPP_DEFINE_CLASS(javax::crypto::interfaces::DHPublicKey,"javax/crypto/interfaces/DHPublicKey")
J2CPP_DEFINE_METHOD(javax::crypto::interfaces::DHPublicKey,0,"getY","()Ljava/math/BigInteger;")
J2CPP_DEFINE_FIELD(javax::crypto::interfaces::DHPublicKey,0,"serialVersionUID","J")

} //namespace j2cpp

#endif //J2CPP_JAVAX_CRYPTO_INTERFACES_DHPUBLICKEY_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
