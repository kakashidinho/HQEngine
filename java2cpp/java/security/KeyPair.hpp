/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.security.KeyPair
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_SECURITY_KEYPAIR_HPP_DECL
#define J2CPP_JAVA_SECURITY_KEYPAIR_HPP_DECL


namespace j2cpp { namespace java { namespace io { class Serializable; } } }
namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace security { class PublicKey; } } }
namespace j2cpp { namespace java { namespace security { class PrivateKey; } } }


#include <java/io/Serializable.hpp>
#include <java/lang/Object.hpp>
#include <java/security/PrivateKey.hpp>
#include <java/security/PublicKey.hpp>


namespace j2cpp {

namespace java { namespace security {

	class KeyPair;
	class KeyPair
		: public object<KeyPair>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)

		explicit KeyPair(jobject jobj)
		: object<KeyPair>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<java::io::Serializable>() const;


		KeyPair(local_ref< java::security::PublicKey > const&, local_ref< java::security::PrivateKey > const&);
		local_ref< java::security::PrivateKey > getPrivate();
		local_ref< java::security::PublicKey > getPublic();
	}; //class KeyPair

} //namespace security
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_SECURITY_KEYPAIR_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_SECURITY_KEYPAIR_HPP_IMPL
#define J2CPP_JAVA_SECURITY_KEYPAIR_HPP_IMPL

namespace j2cpp {



java::security::KeyPair::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

java::security::KeyPair::operator local_ref<java::io::Serializable>() const
{
	return local_ref<java::io::Serializable>(get_jobject());
}


java::security::KeyPair::KeyPair(local_ref< java::security::PublicKey > const &a0, local_ref< java::security::PrivateKey > const &a1)
: object<java::security::KeyPair>(
	call_new_object<
		java::security::KeyPair::J2CPP_CLASS_NAME,
		java::security::KeyPair::J2CPP_METHOD_NAME(0),
		java::security::KeyPair::J2CPP_METHOD_SIGNATURE(0)
	>(a0, a1)
)
{
}


local_ref< java::security::PrivateKey > java::security::KeyPair::getPrivate()
{
	return call_method<
		java::security::KeyPair::J2CPP_CLASS_NAME,
		java::security::KeyPair::J2CPP_METHOD_NAME(1),
		java::security::KeyPair::J2CPP_METHOD_SIGNATURE(1), 
		local_ref< java::security::PrivateKey >
	>(get_jobject());
}

local_ref< java::security::PublicKey > java::security::KeyPair::getPublic()
{
	return call_method<
		java::security::KeyPair::J2CPP_CLASS_NAME,
		java::security::KeyPair::J2CPP_METHOD_NAME(2),
		java::security::KeyPair::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< java::security::PublicKey >
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(java::security::KeyPair,"java/security/KeyPair")
J2CPP_DEFINE_METHOD(java::security::KeyPair,0,"<init>","(Ljava/security/PublicKey;Ljava/security/PrivateKey;)V")
J2CPP_DEFINE_METHOD(java::security::KeyPair,1,"getPrivate","()Ljava/security/PrivateKey;")
J2CPP_DEFINE_METHOD(java::security::KeyPair,2,"getPublic","()Ljava/security/PublicKey;")

} //namespace j2cpp

#endif //J2CPP_JAVA_SECURITY_KEYPAIR_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
