/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.util.UUID
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_UUID_HPP_DECL
#define J2CPP_JAVA_UTIL_UUID_HPP_DECL


namespace j2cpp { namespace java { namespace io { class Serializable; } } }
namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class Comparable; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }


#include <java/io/Serializable.hpp>
#include <java/lang/Comparable.hpp>
#include <java/lang/Object.hpp>
#include <java/lang/String.hpp>


namespace j2cpp {

namespace java { namespace util {

	class UUID;
	class UUID
		: public object<UUID>
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
		J2CPP_DECLARE_METHOD(10)
		J2CPP_DECLARE_METHOD(11)
		J2CPP_DECLARE_METHOD(12)
		J2CPP_DECLARE_METHOD(13)
		J2CPP_DECLARE_METHOD(14)
		J2CPP_DECLARE_METHOD(15)

		explicit UUID(jobject jobj)
		: object<UUID>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<java::io::Serializable>() const;
		operator local_ref<java::lang::Comparable>() const;


		UUID(jlong, jlong);
		static local_ref< java::util::UUID > randomUUID();
		static local_ref< java::util::UUID > nameUUIDFromBytes(local_ref< array<jbyte,1> >  const&);
		static local_ref< java::util::UUID > fromString(local_ref< java::lang::String >  const&);
		jlong getLeastSignificantBits();
		jlong getMostSignificantBits();
		jint version();
		jint variant();
		jlong timestamp();
		jint clockSequence();
		jlong node();
		jint compareTo(local_ref< java::util::UUID >  const&);
		jboolean equals(local_ref< java::lang::Object >  const&);
		jint hashCode();
		local_ref< java::lang::String > toString();
		jint compareTo(local_ref< java::lang::Object >  const&);
	}; //class UUID

} //namespace util
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_UUID_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_UUID_HPP_IMPL
#define J2CPP_JAVA_UTIL_UUID_HPP_IMPL

namespace j2cpp {



java::util::UUID::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

java::util::UUID::operator local_ref<java::io::Serializable>() const
{
	return local_ref<java::io::Serializable>(get_jobject());
}

java::util::UUID::operator local_ref<java::lang::Comparable>() const
{
	return local_ref<java::lang::Comparable>(get_jobject());
}


java::util::UUID::UUID(jlong a0, jlong a1)
: object<java::util::UUID>(
	call_new_object<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(0),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(0)
	>(a0, a1)
)
{
}


local_ref< java::util::UUID > java::util::UUID::randomUUID()
{
	return call_static_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(1),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(1), 
		local_ref< java::util::UUID >
	>();
}

local_ref< java::util::UUID > java::util::UUID::nameUUIDFromBytes(local_ref< array<jbyte,1> > const &a0)
{
	return call_static_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(2),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< java::util::UUID >
	>(a0);
}

local_ref< java::util::UUID > java::util::UUID::fromString(local_ref< java::lang::String > const &a0)
{
	return call_static_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(3),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(3), 
		local_ref< java::util::UUID >
	>(a0);
}

jlong java::util::UUID::getLeastSignificantBits()
{
	return call_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(4),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(4), 
		jlong
	>(get_jobject());
}

jlong java::util::UUID::getMostSignificantBits()
{
	return call_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(5),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(5), 
		jlong
	>(get_jobject());
}

jint java::util::UUID::version()
{
	return call_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(6),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(6), 
		jint
	>(get_jobject());
}

jint java::util::UUID::variant()
{
	return call_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(7),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(7), 
		jint
	>(get_jobject());
}

jlong java::util::UUID::timestamp()
{
	return call_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(8),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(8), 
		jlong
	>(get_jobject());
}

jint java::util::UUID::clockSequence()
{
	return call_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(9),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(9), 
		jint
	>(get_jobject());
}

jlong java::util::UUID::node()
{
	return call_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(10),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(10), 
		jlong
	>(get_jobject());
}

jint java::util::UUID::compareTo(local_ref< java::util::UUID > const &a0)
{
	return call_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(11),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(11), 
		jint
	>(get_jobject(), a0);
}

jboolean java::util::UUID::equals(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(12),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(12), 
		jboolean
	>(get_jobject(), a0);
}

jint java::util::UUID::hashCode()
{
	return call_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(13),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(13), 
		jint
	>(get_jobject());
}

local_ref< java::lang::String > java::util::UUID::toString()
{
	return call_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(14),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(14), 
		local_ref< java::lang::String >
	>(get_jobject());
}

jint java::util::UUID::compareTo(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::UUID::J2CPP_CLASS_NAME,
		java::util::UUID::J2CPP_METHOD_NAME(15),
		java::util::UUID::J2CPP_METHOD_SIGNATURE(15), 
		jint
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(java::util::UUID,"java/util/UUID")
J2CPP_DEFINE_METHOD(java::util::UUID,0,"<init>","(JJ)V")
J2CPP_DEFINE_METHOD(java::util::UUID,1,"randomUUID","()Ljava/util/UUID;")
J2CPP_DEFINE_METHOD(java::util::UUID,2,"nameUUIDFromBytes","([B)Ljava/util/UUID;")
J2CPP_DEFINE_METHOD(java::util::UUID,3,"fromString","(Ljava/lang/String;)Ljava/util/UUID;")
J2CPP_DEFINE_METHOD(java::util::UUID,4,"getLeastSignificantBits","()J")
J2CPP_DEFINE_METHOD(java::util::UUID,5,"getMostSignificantBits","()J")
J2CPP_DEFINE_METHOD(java::util::UUID,6,"version","()I")
J2CPP_DEFINE_METHOD(java::util::UUID,7,"variant","()I")
J2CPP_DEFINE_METHOD(java::util::UUID,8,"timestamp","()J")
J2CPP_DEFINE_METHOD(java::util::UUID,9,"clockSequence","()I")
J2CPP_DEFINE_METHOD(java::util::UUID,10,"node","()J")
J2CPP_DEFINE_METHOD(java::util::UUID,11,"compareTo","(Ljava/util/UUID;)I")
J2CPP_DEFINE_METHOD(java::util::UUID,12,"equals","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::UUID,13,"hashCode","()I")
J2CPP_DEFINE_METHOD(java::util::UUID,14,"toString","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::util::UUID,15,"compareTo","(Ljava/lang/Object;)I")

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_UUID_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
