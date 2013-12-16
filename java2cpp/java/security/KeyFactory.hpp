/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.security.KeyFactory
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_SECURITY_KEYFACTORY_HPP_DECL
#define J2CPP_JAVA_SECURITY_KEYFACTORY_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class Class; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace security { class Provider; } } }
namespace j2cpp { namespace java { namespace security { class Key; } } }
namespace j2cpp { namespace java { namespace security { class PublicKey; } } }
namespace j2cpp { namespace java { namespace security { class PrivateKey; } } }
namespace j2cpp { namespace java { namespace security { namespace spec { class KeySpec; } } } }


#include <java/lang/Class.hpp>
#include <java/lang/Object.hpp>
#include <java/lang/String.hpp>
#include <java/security/Key.hpp>
#include <java/security/PrivateKey.hpp>
#include <java/security/Provider.hpp>
#include <java/security/PublicKey.hpp>
#include <java/security/spec/KeySpec.hpp>


namespace j2cpp {

namespace java { namespace security {

	class KeyFactory;
	class KeyFactory
		: public object<KeyFactory>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)
		J2CPP_DECLARE_METHOD(4)
		J2CPP_DECLARE_METHOD(5)
		J2CPP_DECLARE_METHOD(6)
		J2CPP_DECLARE_METHOD(7)
		J2CPP_DECLARE_METHOD(8)
		J2CPP_DECLARE_METHOD(9)

		explicit KeyFactory(jobject jobj)
		: object<KeyFactory>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;


		static local_ref< java::security::KeyFactory > getInstance(local_ref< java::lang::String >  const&);
		static local_ref< java::security::KeyFactory > getInstance(local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&);
		static local_ref< java::security::KeyFactory > getInstance(local_ref< java::lang::String >  const&, local_ref< java::security::Provider >  const&);
		local_ref< java::security::Provider > getProvider();
		local_ref< java::lang::String > getAlgorithm();
		local_ref< java::security::PublicKey > generatePublic(local_ref< java::security::spec::KeySpec >  const&);
		local_ref< java::security::PrivateKey > generatePrivate(local_ref< java::security::spec::KeySpec >  const&);
		local_ref< java::security::spec::KeySpec > getKeySpec(local_ref< java::security::Key >  const&, local_ref< java::lang::Class >  const&);
		local_ref< java::security::Key > translateKey(local_ref< java::security::Key >  const&);
	}; //class KeyFactory

} //namespace security
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_SECURITY_KEYFACTORY_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_SECURITY_KEYFACTORY_HPP_IMPL
#define J2CPP_JAVA_SECURITY_KEYFACTORY_HPP_IMPL

namespace j2cpp {



java::security::KeyFactory::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}


local_ref< java::security::KeyFactory > java::security::KeyFactory::getInstance(local_ref< java::lang::String > const &a0)
{
	return call_static_method<
		java::security::KeyFactory::J2CPP_CLASS_NAME,
		java::security::KeyFactory::J2CPP_METHOD_NAME(1),
		java::security::KeyFactory::J2CPP_METHOD_SIGNATURE(1), 
		local_ref< java::security::KeyFactory >
	>(a0);
}

local_ref< java::security::KeyFactory > java::security::KeyFactory::getInstance(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1)
{
	return call_static_method<
		java::security::KeyFactory::J2CPP_CLASS_NAME,
		java::security::KeyFactory::J2CPP_METHOD_NAME(2),
		java::security::KeyFactory::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< java::security::KeyFactory >
	>(a0, a1);
}

local_ref< java::security::KeyFactory > java::security::KeyFactory::getInstance(local_ref< java::lang::String > const &a0, local_ref< java::security::Provider > const &a1)
{
	return call_static_method<
		java::security::KeyFactory::J2CPP_CLASS_NAME,
		java::security::KeyFactory::J2CPP_METHOD_NAME(3),
		java::security::KeyFactory::J2CPP_METHOD_SIGNATURE(3), 
		local_ref< java::security::KeyFactory >
	>(a0, a1);
}

local_ref< java::security::Provider > java::security::KeyFactory::getProvider()
{
	return call_method<
		java::security::KeyFactory::J2CPP_CLASS_NAME,
		java::security::KeyFactory::J2CPP_METHOD_NAME(4),
		java::security::KeyFactory::J2CPP_METHOD_SIGNATURE(4), 
		local_ref< java::security::Provider >
	>(get_jobject());
}

local_ref< java::lang::String > java::security::KeyFactory::getAlgorithm()
{
	return call_method<
		java::security::KeyFactory::J2CPP_CLASS_NAME,
		java::security::KeyFactory::J2CPP_METHOD_NAME(5),
		java::security::KeyFactory::J2CPP_METHOD_SIGNATURE(5), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::security::PublicKey > java::security::KeyFactory::generatePublic(local_ref< java::security::spec::KeySpec > const &a0)
{
	return call_method<
		java::security::KeyFactory::J2CPP_CLASS_NAME,
		java::security::KeyFactory::J2CPP_METHOD_NAME(6),
		java::security::KeyFactory::J2CPP_METHOD_SIGNATURE(6), 
		local_ref< java::security::PublicKey >
	>(get_jobject(), a0);
}

local_ref< java::security::PrivateKey > java::security::KeyFactory::generatePrivate(local_ref< java::security::spec::KeySpec > const &a0)
{
	return call_method<
		java::security::KeyFactory::J2CPP_CLASS_NAME,
		java::security::KeyFactory::J2CPP_METHOD_NAME(7),
		java::security::KeyFactory::J2CPP_METHOD_SIGNATURE(7), 
		local_ref< java::security::PrivateKey >
	>(get_jobject(), a0);
}

local_ref< java::security::spec::KeySpec > java::security::KeyFactory::getKeySpec(local_ref< java::security::Key > const &a0, local_ref< java::lang::Class > const &a1)
{
	return call_method<
		java::security::KeyFactory::J2CPP_CLASS_NAME,
		java::security::KeyFactory::J2CPP_METHOD_NAME(8),
		java::security::KeyFactory::J2CPP_METHOD_SIGNATURE(8), 
		local_ref< java::security::spec::KeySpec >
	>(get_jobject(), a0, a1);
}

local_ref< java::security::Key > java::security::KeyFactory::translateKey(local_ref< java::security::Key > const &a0)
{
	return call_method<
		java::security::KeyFactory::J2CPP_CLASS_NAME,
		java::security::KeyFactory::J2CPP_METHOD_NAME(9),
		java::security::KeyFactory::J2CPP_METHOD_SIGNATURE(9), 
		local_ref< java::security::Key >
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(java::security::KeyFactory,"java/security/KeyFactory")
J2CPP_DEFINE_METHOD(java::security::KeyFactory,0,"<init>","(Ljava/security/KeyFactorySpi;Ljava/security/Provider;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(java::security::KeyFactory,1,"getInstance","(Ljava/lang/String;)Ljava/security/KeyFactory;")
J2CPP_DEFINE_METHOD(java::security::KeyFactory,2,"getInstance","(Ljava/lang/String;Ljava/lang/String;)Ljava/security/KeyFactory;")
J2CPP_DEFINE_METHOD(java::security::KeyFactory,3,"getInstance","(Ljava/lang/String;Ljava/security/Provider;)Ljava/security/KeyFactory;")
J2CPP_DEFINE_METHOD(java::security::KeyFactory,4,"getProvider","()Ljava/security/Provider;")
J2CPP_DEFINE_METHOD(java::security::KeyFactory,5,"getAlgorithm","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::security::KeyFactory,6,"generatePublic","(Ljava/security/spec/KeySpec;)Ljava/security/PublicKey;")
J2CPP_DEFINE_METHOD(java::security::KeyFactory,7,"generatePrivate","(Ljava/security/spec/KeySpec;)Ljava/security/PrivateKey;")
J2CPP_DEFINE_METHOD(java::security::KeyFactory,8,"getKeySpec","(Ljava/security/Key;Ljava/lang/Class;)Ljava/security/spec/KeySpec;")
J2CPP_DEFINE_METHOD(java::security::KeyFactory,9,"translateKey","(Ljava/security/Key;)Ljava/security/Key;")

} //namespace j2cpp

#endif //J2CPP_JAVA_SECURITY_KEYFACTORY_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
