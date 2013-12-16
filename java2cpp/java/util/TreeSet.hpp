/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.util.TreeSet
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_TREESET_HPP_DECL
#define J2CPP_JAVA_UTIL_TREESET_HPP_DECL


namespace j2cpp { namespace java { namespace io { class Serializable; } } }
namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class Cloneable; } } }
namespace j2cpp { namespace java { namespace util { class Comparator; } } }
namespace j2cpp { namespace java { namespace util { class SortedSet; } } }
namespace j2cpp { namespace java { namespace util { class Iterator; } } }
namespace j2cpp { namespace java { namespace util { class AbstractSet; } } }
namespace j2cpp { namespace java { namespace util { class Collection; } } }


#include <java/io/Serializable.hpp>
#include <java/lang/Cloneable.hpp>
#include <java/lang/Object.hpp>
#include <java/util/AbstractSet.hpp>
#include <java/util/Collection.hpp>
#include <java/util/Comparator.hpp>
#include <java/util/Iterator.hpp>
#include <java/util/SortedSet.hpp>


namespace j2cpp {

namespace java { namespace util {

	class TreeSet;
	class TreeSet
		: public object<TreeSet>
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

		explicit TreeSet(jobject jobj)
		: object<TreeSet>(jobj)
		{
		}

		operator local_ref<java::util::AbstractSet>() const;
		operator local_ref<java::util::SortedSet>() const;
		operator local_ref<java::lang::Cloneable>() const;
		operator local_ref<java::io::Serializable>() const;


		TreeSet();
		TreeSet(local_ref< java::util::Collection > const&);
		TreeSet(local_ref< java::util::Comparator > const&);
		TreeSet(local_ref< java::util::SortedSet > const&);
		jboolean add(local_ref< java::lang::Object >  const&);
		jboolean addAll(local_ref< java::util::Collection >  const&);
		void clear();
		local_ref< java::lang::Object > clone();
		local_ref< java::util::Comparator > comparator();
		jboolean contains(local_ref< java::lang::Object >  const&);
		local_ref< java::lang::Object > first();
		local_ref< java::util::SortedSet > headSet(local_ref< java::lang::Object >  const&);
		jboolean isEmpty();
		local_ref< java::util::Iterator > iterator();
		local_ref< java::lang::Object > last();
		jboolean remove(local_ref< java::lang::Object >  const&);
		jint size();
		local_ref< java::util::SortedSet > subSet(local_ref< java::lang::Object >  const&, local_ref< java::lang::Object >  const&);
		local_ref< java::util::SortedSet > tailSet(local_ref< java::lang::Object >  const&);
	}; //class TreeSet

} //namespace util
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_TREESET_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_TREESET_HPP_IMPL
#define J2CPP_JAVA_UTIL_TREESET_HPP_IMPL

namespace j2cpp {



java::util::TreeSet::operator local_ref<java::util::AbstractSet>() const
{
	return local_ref<java::util::AbstractSet>(get_jobject());
}

java::util::TreeSet::operator local_ref<java::util::SortedSet>() const
{
	return local_ref<java::util::SortedSet>(get_jobject());
}

java::util::TreeSet::operator local_ref<java::lang::Cloneable>() const
{
	return local_ref<java::lang::Cloneable>(get_jobject());
}

java::util::TreeSet::operator local_ref<java::io::Serializable>() const
{
	return local_ref<java::io::Serializable>(get_jobject());
}


java::util::TreeSet::TreeSet()
: object<java::util::TreeSet>(
	call_new_object<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(0),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



java::util::TreeSet::TreeSet(local_ref< java::util::Collection > const &a0)
: object<java::util::TreeSet>(
	call_new_object<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(1),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}



java::util::TreeSet::TreeSet(local_ref< java::util::Comparator > const &a0)
: object<java::util::TreeSet>(
	call_new_object<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(2),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(2)
	>(a0)
)
{
}



java::util::TreeSet::TreeSet(local_ref< java::util::SortedSet > const &a0)
: object<java::util::TreeSet>(
	call_new_object<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(3),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(3)
	>(a0)
)
{
}


jboolean java::util::TreeSet::add(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(4),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(4), 
		jboolean
	>(get_jobject(), a0);
}

jboolean java::util::TreeSet::addAll(local_ref< java::util::Collection > const &a0)
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(5),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(5), 
		jboolean
	>(get_jobject(), a0);
}

void java::util::TreeSet::clear()
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(6),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(6), 
		void
	>(get_jobject());
}

local_ref< java::lang::Object > java::util::TreeSet::clone()
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(7),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(7), 
		local_ref< java::lang::Object >
	>(get_jobject());
}

local_ref< java::util::Comparator > java::util::TreeSet::comparator()
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(8),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(8), 
		local_ref< java::util::Comparator >
	>(get_jobject());
}

jboolean java::util::TreeSet::contains(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(9),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(9), 
		jboolean
	>(get_jobject(), a0);
}

local_ref< java::lang::Object > java::util::TreeSet::first()
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(10),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(10), 
		local_ref< java::lang::Object >
	>(get_jobject());
}

local_ref< java::util::SortedSet > java::util::TreeSet::headSet(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(11),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(11), 
		local_ref< java::util::SortedSet >
	>(get_jobject(), a0);
}

jboolean java::util::TreeSet::isEmpty()
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(12),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(12), 
		jboolean
	>(get_jobject());
}

local_ref< java::util::Iterator > java::util::TreeSet::iterator()
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(13),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(13), 
		local_ref< java::util::Iterator >
	>(get_jobject());
}

local_ref< java::lang::Object > java::util::TreeSet::last()
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(14),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(14), 
		local_ref< java::lang::Object >
	>(get_jobject());
}

jboolean java::util::TreeSet::remove(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(15),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(15), 
		jboolean
	>(get_jobject(), a0);
}

jint java::util::TreeSet::size()
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(16),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(16), 
		jint
	>(get_jobject());
}

local_ref< java::util::SortedSet > java::util::TreeSet::subSet(local_ref< java::lang::Object > const &a0, local_ref< java::lang::Object > const &a1)
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(17),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(17), 
		local_ref< java::util::SortedSet >
	>(get_jobject(), a0, a1);
}

local_ref< java::util::SortedSet > java::util::TreeSet::tailSet(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::TreeSet::J2CPP_CLASS_NAME,
		java::util::TreeSet::J2CPP_METHOD_NAME(18),
		java::util::TreeSet::J2CPP_METHOD_SIGNATURE(18), 
		local_ref< java::util::SortedSet >
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(java::util::TreeSet,"java/util/TreeSet")
J2CPP_DEFINE_METHOD(java::util::TreeSet,0,"<init>","()V")
J2CPP_DEFINE_METHOD(java::util::TreeSet,1,"<init>","(Ljava/util/Collection;)V")
J2CPP_DEFINE_METHOD(java::util::TreeSet,2,"<init>","(Ljava/util/Comparator;)V")
J2CPP_DEFINE_METHOD(java::util::TreeSet,3,"<init>","(Ljava/util/SortedSet;)V")
J2CPP_DEFINE_METHOD(java::util::TreeSet,4,"add","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::TreeSet,5,"addAll","(Ljava/util/Collection;)Z")
J2CPP_DEFINE_METHOD(java::util::TreeSet,6,"clear","()V")
J2CPP_DEFINE_METHOD(java::util::TreeSet,7,"clone","()Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::TreeSet,8,"comparator","()Ljava/util/Comparator;")
J2CPP_DEFINE_METHOD(java::util::TreeSet,9,"contains","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::TreeSet,10,"first","()Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::TreeSet,11,"headSet","(Ljava/lang/Object;)Ljava/util/SortedSet;")
J2CPP_DEFINE_METHOD(java::util::TreeSet,12,"isEmpty","()Z")
J2CPP_DEFINE_METHOD(java::util::TreeSet,13,"iterator","()Ljava/util/Iterator;")
J2CPP_DEFINE_METHOD(java::util::TreeSet,14,"last","()Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::TreeSet,15,"remove","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::TreeSet,16,"size","()I")
J2CPP_DEFINE_METHOD(java::util::TreeSet,17,"subSet","(Ljava/lang/Object;Ljava/lang/Object;)Ljava/util/SortedSet;")
J2CPP_DEFINE_METHOD(java::util::TreeSet,18,"tailSet","(Ljava/lang/Object;)Ljava/util/SortedSet;")

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_TREESET_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
