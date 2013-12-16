/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.security.GuardedObject
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_SECURITY_GUARDEDOBJECT_HPP_DECL
#define J2CPP_JAVA_SECURITY_GUARDEDOBJECT_HPP_DECL


namespace j2cpp { namespace java { namespace io { class Serializable; } } }
namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace security { class Guard; } } }


#include <java/io/Serializable.hpp>
#include <java/lang/Object.hpp>
#include <java/security/Guard.hpp>


namespace j2cpp {

namespace java { namespace security {

	class GuardedObject;
	class GuardedObject
		: public object<GuardedObject>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)

		explicit GuardedObject(jobject jobj)
		: object<GuardedObject>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<java::io::Serializable>() const;


		GuardedObject(local_ref< java::lang::Object > const&, local_ref< java::security::Guard > const&);
		local_ref< java::lang::Object > getObject();
	}; //class GuardedObject

} //namespace security
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_SECURITY_GUARDEDOBJECT_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_SECURITY_GUARDEDOBJECT_HPP_IMPL
#define J2CPP_JAVA_SECURITY_GUARDEDOBJECT_HPP_IMPL

namespace j2cpp {



java::security::GuardedObject::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

java::security::GuardedObject::operator local_ref<java::io::Serializable>() const
{
	return local_ref<java::io::Serializable>(get_jobject());
}


java::security::GuardedObject::GuardedObject(local_ref< java::lang::Object > const &a0, local_ref< java::security::Guard > const &a1)
: object<java::security::GuardedObject>(
	call_new_object<
		java::security::GuardedObject::J2CPP_CLASS_NAME,
		java::security::GuardedObject::J2CPP_METHOD_NAME(0),
		java::security::GuardedObject::J2CPP_METHOD_SIGNATURE(0)
	>(a0, a1)
)
{
}


local_ref< java::lang::Object > java::security::GuardedObject::getObject()
{
	return call_method<
		java::security::GuardedObject::J2CPP_CLASS_NAME,
		java::security::GuardedObject::J2CPP_METHOD_NAME(1),
		java::security::GuardedObject::J2CPP_METHOD_SIGNATURE(1), 
		local_ref< java::lang::Object >
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(java::security::GuardedObject,"java/security/GuardedObject")
J2CPP_DEFINE_METHOD(java::security::GuardedObject,0,"<init>","(Ljava/lang/Object;Ljava/security/Guard;)V")
J2CPP_DEFINE_METHOD(java::security::GuardedObject,1,"getObject","()Ljava/lang/Object;")

} //namespace j2cpp

#endif //J2CPP_JAVA_SECURITY_GUARDEDOBJECT_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
