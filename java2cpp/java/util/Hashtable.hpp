/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.util.Hashtable
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_HASHTABLE_HPP_DECL
#define J2CPP_JAVA_UTIL_HASHTABLE_HPP_DECL


namespace j2cpp { namespace java { namespace io { class Serializable; } } }
namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class Cloneable; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace util { class Dictionary; } } }
namespace j2cpp { namespace java { namespace util { class Set; } } }
namespace j2cpp { namespace java { namespace util { class Map; } } }
namespace j2cpp { namespace java { namespace util { class Enumeration; } } }
namespace j2cpp { namespace java { namespace util { class Collection; } } }


#include <java/io/Serializable.hpp>
#include <java/lang/Cloneable.hpp>
#include <java/lang/Object.hpp>
#include <java/lang/String.hpp>
#include <java/util/Collection.hpp>
#include <java/util/Dictionary.hpp>
#include <java/util/Enumeration.hpp>
#include <java/util/Map.hpp>
#include <java/util/Set.hpp>


namespace j2cpp {

namespace java { namespace util {

	class Hashtable;
	class Hashtable
		: public object<Hashtable>
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
		J2CPP_DECLARE_METHOD(16)
		J2CPP_DECLARE_METHOD(17)
		J2CPP_DECLARE_METHOD(18)
		J2CPP_DECLARE_METHOD(19)
		J2CPP_DECLARE_METHOD(20)
		J2CPP_DECLARE_METHOD(21)
		J2CPP_DECLARE_METHOD(22)
		J2CPP_DECLARE_METHOD(23)

		explicit Hashtable(jobject jobj)
		: object<Hashtable>(jobj)
		{
		}

		operator local_ref<java::util::Dictionary>() const;
		operator local_ref<java::util::Map>() const;
		operator local_ref<java::lang::Cloneable>() const;
		operator local_ref<java::io::Serializable>() const;


		Hashtable();
		Hashtable(jint);
		Hashtable(jint, jfloat);
		Hashtable(local_ref< java::util::Map > const&);
		local_ref< java::lang::Object > clone();
		jboolean isEmpty();
		jint size();
		local_ref< java::lang::Object > get(local_ref< java::lang::Object >  const&);
		jboolean containsKey(local_ref< java::lang::Object >  const&);
		jboolean containsValue(local_ref< java::lang::Object >  const&);
		jboolean contains(local_ref< java::lang::Object >  const&);
		local_ref< java::lang::Object > put(local_ref< java::lang::Object >  const&, local_ref< java::lang::Object >  const&);
		void putAll(local_ref< java::util::Map >  const&);
		local_ref< java::lang::Object > remove(local_ref< java::lang::Object >  const&);
		void clear();
		local_ref< java::util::Set > keySet();
		local_ref< java::util::Collection > values();
		local_ref< java::util::Set > entrySet();
		local_ref< java::util::Enumeration > keys();
		local_ref< java::util::Enumeration > elements();
		jboolean equals(local_ref< java::lang::Object >  const&);
		jint hashCode();
		local_ref< java::lang::String > toString();
	}; //class Hashtable

} //namespace util
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_HASHTABLE_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_HASHTABLE_HPP_IMPL
#define J2CPP_JAVA_UTIL_HASHTABLE_HPP_IMPL

namespace j2cpp {



java::util::Hashtable::operator local_ref<java::util::Dictionary>() const
{
	return local_ref<java::util::Dictionary>(get_jobject());
}

java::util::Hashtable::operator local_ref<java::util::Map>() const
{
	return local_ref<java::util::Map>(get_jobject());
}

java::util::Hashtable::operator local_ref<java::lang::Cloneable>() const
{
	return local_ref<java::lang::Cloneable>(get_jobject());
}

java::util::Hashtable::operator local_ref<java::io::Serializable>() const
{
	return local_ref<java::io::Serializable>(get_jobject());
}


java::util::Hashtable::Hashtable()
: object<java::util::Hashtable>(
	call_new_object<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(0),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



java::util::Hashtable::Hashtable(jint a0)
: object<java::util::Hashtable>(
	call_new_object<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(1),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}



java::util::Hashtable::Hashtable(jint a0, jfloat a1)
: object<java::util::Hashtable>(
	call_new_object<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(2),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(2)
	>(a0, a1)
)
{
}



java::util::Hashtable::Hashtable(local_ref< java::util::Map > const &a0)
: object<java::util::Hashtable>(
	call_new_object<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(3),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(3)
	>(a0)
)
{
}


local_ref< java::lang::Object > java::util::Hashtable::clone()
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(4),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(4), 
		local_ref< java::lang::Object >
	>(get_jobject());
}

jboolean java::util::Hashtable::isEmpty()
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(5),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(5), 
		jboolean
	>(get_jobject());
}

jint java::util::Hashtable::size()
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(6),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(6), 
		jint
	>(get_jobject());
}

local_ref< java::lang::Object > java::util::Hashtable::get(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(7),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(7), 
		local_ref< java::lang::Object >
	>(get_jobject(), a0);
}

jboolean java::util::Hashtable::containsKey(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(8),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(8), 
		jboolean
	>(get_jobject(), a0);
}

jboolean java::util::Hashtable::containsValue(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(9),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(9), 
		jboolean
	>(get_jobject(), a0);
}

jboolean java::util::Hashtable::contains(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(10),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(10), 
		jboolean
	>(get_jobject(), a0);
}

local_ref< java::lang::Object > java::util::Hashtable::put(local_ref< java::lang::Object > const &a0, local_ref< java::lang::Object > const &a1)
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(11),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(11), 
		local_ref< java::lang::Object >
	>(get_jobject(), a0, a1);
}

void java::util::Hashtable::putAll(local_ref< java::util::Map > const &a0)
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(12),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(12), 
		void
	>(get_jobject(), a0);
}


local_ref< java::lang::Object > java::util::Hashtable::remove(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(14),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(14), 
		local_ref< java::lang::Object >
	>(get_jobject(), a0);
}

void java::util::Hashtable::clear()
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(15),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(15), 
		void
	>(get_jobject());
}

local_ref< java::util::Set > java::util::Hashtable::keySet()
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(16),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(16), 
		local_ref< java::util::Set >
	>(get_jobject());
}

local_ref< java::util::Collection > java::util::Hashtable::values()
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(17),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(17), 
		local_ref< java::util::Collection >
	>(get_jobject());
}

local_ref< java::util::Set > java::util::Hashtable::entrySet()
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(18),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(18), 
		local_ref< java::util::Set >
	>(get_jobject());
}

local_ref< java::util::Enumeration > java::util::Hashtable::keys()
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(19),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(19), 
		local_ref< java::util::Enumeration >
	>(get_jobject());
}

local_ref< java::util::Enumeration > java::util::Hashtable::elements()
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(20),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(20), 
		local_ref< java::util::Enumeration >
	>(get_jobject());
}

jboolean java::util::Hashtable::equals(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(21),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(21), 
		jboolean
	>(get_jobject(), a0);
}

jint java::util::Hashtable::hashCode()
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(22),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(22), 
		jint
	>(get_jobject());
}

local_ref< java::lang::String > java::util::Hashtable::toString()
{
	return call_method<
		java::util::Hashtable::J2CPP_CLASS_NAME,
		java::util::Hashtable::J2CPP_METHOD_NAME(23),
		java::util::Hashtable::J2CPP_METHOD_SIGNATURE(23), 
		local_ref< java::lang::String >
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(java::util::Hashtable,"java/util/Hashtable")
J2CPP_DEFINE_METHOD(java::util::Hashtable,0,"<init>","()V")
J2CPP_DEFINE_METHOD(java::util::Hashtable,1,"<init>","(I)V")
J2CPP_DEFINE_METHOD(java::util::Hashtable,2,"<init>","(IF)V")
J2CPP_DEFINE_METHOD(java::util::Hashtable,3,"<init>","(Ljava/util/Map;)V")
J2CPP_DEFINE_METHOD(java::util::Hashtable,4,"clone","()Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Hashtable,5,"isEmpty","()Z")
J2CPP_DEFINE_METHOD(java::util::Hashtable,6,"size","()I")
J2CPP_DEFINE_METHOD(java::util::Hashtable,7,"get","(Ljava/lang/Object;)Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Hashtable,8,"containsKey","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::Hashtable,9,"containsValue","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::Hashtable,10,"contains","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::Hashtable,11,"put","(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Hashtable,12,"putAll","(Ljava/util/Map;)V")
J2CPP_DEFINE_METHOD(java::util::Hashtable,13,"rehash","()V")
J2CPP_DEFINE_METHOD(java::util::Hashtable,14,"remove","(Ljava/lang/Object;)Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Hashtable,15,"clear","()V")
J2CPP_DEFINE_METHOD(java::util::Hashtable,16,"keySet","()Ljava/util/Set;")
J2CPP_DEFINE_METHOD(java::util::Hashtable,17,"values","()Ljava/util/Collection;")
J2CPP_DEFINE_METHOD(java::util::Hashtable,18,"entrySet","()Ljava/util/Set;")
J2CPP_DEFINE_METHOD(java::util::Hashtable,19,"keys","()Ljava/util/Enumeration;")
J2CPP_DEFINE_METHOD(java::util::Hashtable,20,"elements","()Ljava/util/Enumeration;")
J2CPP_DEFINE_METHOD(java::util::Hashtable,21,"equals","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::Hashtable,22,"hashCode","()I")
J2CPP_DEFINE_METHOD(java::util::Hashtable,23,"toString","()Ljava/lang/String;")

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_HASHTABLE_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
