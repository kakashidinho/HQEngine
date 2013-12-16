/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.util.List
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_LIST_HPP_DECL
#define J2CPP_JAVA_UTIL_LIST_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace util { class Iterator; } } }
namespace j2cpp { namespace java { namespace util { class ListIterator; } } }
namespace j2cpp { namespace java { namespace util { class Collection; } } }


#include <java/lang/Object.hpp>
#include <java/util/Collection.hpp>
#include <java/util/Iterator.hpp>
#include <java/util/ListIterator.hpp>


namespace j2cpp {

namespace java { namespace util {

	class List;
	class List
		: public object<List>
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
		J2CPP_DECLARE_METHOD(24)

		explicit List(jobject jobj)
		: object<List>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<java::util::Collection>() const;


		void add(jint, local_ref< java::lang::Object >  const&);
		jboolean add(local_ref< java::lang::Object >  const&);
		jboolean addAll(jint, local_ref< java::util::Collection >  const&);
		jboolean addAll(local_ref< java::util::Collection >  const&);
		void clear();
		jboolean contains(local_ref< java::lang::Object >  const&);
		jboolean containsAll(local_ref< java::util::Collection >  const&);
		jboolean equals(local_ref< java::lang::Object >  const&);
		local_ref< java::lang::Object > get(jint);
		jint hashCode();
		jint indexOf(local_ref< java::lang::Object >  const&);
		jboolean isEmpty();
		local_ref< java::util::Iterator > iterator();
		jint lastIndexOf(local_ref< java::lang::Object >  const&);
		local_ref< java::util::ListIterator > listIterator();
		local_ref< java::util::ListIterator > listIterator(jint);
		local_ref< java::lang::Object > remove(jint);
		jboolean remove(local_ref< java::lang::Object >  const&);
		jboolean removeAll(local_ref< java::util::Collection >  const&);
		jboolean retainAll(local_ref< java::util::Collection >  const&);
		local_ref< java::lang::Object > set(jint, local_ref< java::lang::Object >  const&);
		jint size();
		local_ref< java::util::List > subList(jint, jint);
		local_ref< array< local_ref< java::lang::Object >, 1> > toArray();
		local_ref< array< local_ref< java::lang::Object >, 1> > toArray(local_ref< array< local_ref< java::lang::Object >, 1> >  const&);
	}; //class List

} //namespace util
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_LIST_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_LIST_HPP_IMPL
#define J2CPP_JAVA_UTIL_LIST_HPP_IMPL

namespace j2cpp {



java::util::List::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

java::util::List::operator local_ref<java::util::Collection>() const
{
	return local_ref<java::util::Collection>(get_jobject());
}

void java::util::List::add(jint a0, local_ref< java::lang::Object > const &a1)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(0),
		java::util::List::J2CPP_METHOD_SIGNATURE(0), 
		void
	>(get_jobject(), a0, a1);
}

jboolean java::util::List::add(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(1),
		java::util::List::J2CPP_METHOD_SIGNATURE(1), 
		jboolean
	>(get_jobject(), a0);
}

jboolean java::util::List::addAll(jint a0, local_ref< java::util::Collection > const &a1)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(2),
		java::util::List::J2CPP_METHOD_SIGNATURE(2), 
		jboolean
	>(get_jobject(), a0, a1);
}

jboolean java::util::List::addAll(local_ref< java::util::Collection > const &a0)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(3),
		java::util::List::J2CPP_METHOD_SIGNATURE(3), 
		jboolean
	>(get_jobject(), a0);
}

void java::util::List::clear()
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(4),
		java::util::List::J2CPP_METHOD_SIGNATURE(4), 
		void
	>(get_jobject());
}

jboolean java::util::List::contains(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(5),
		java::util::List::J2CPP_METHOD_SIGNATURE(5), 
		jboolean
	>(get_jobject(), a0);
}

jboolean java::util::List::containsAll(local_ref< java::util::Collection > const &a0)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(6),
		java::util::List::J2CPP_METHOD_SIGNATURE(6), 
		jboolean
	>(get_jobject(), a0);
}

jboolean java::util::List::equals(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(7),
		java::util::List::J2CPP_METHOD_SIGNATURE(7), 
		jboolean
	>(get_jobject(), a0);
}

local_ref< java::lang::Object > java::util::List::get(jint a0)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(8),
		java::util::List::J2CPP_METHOD_SIGNATURE(8), 
		local_ref< java::lang::Object >
	>(get_jobject(), a0);
}

jint java::util::List::hashCode()
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(9),
		java::util::List::J2CPP_METHOD_SIGNATURE(9), 
		jint
	>(get_jobject());
}

jint java::util::List::indexOf(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(10),
		java::util::List::J2CPP_METHOD_SIGNATURE(10), 
		jint
	>(get_jobject(), a0);
}

jboolean java::util::List::isEmpty()
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(11),
		java::util::List::J2CPP_METHOD_SIGNATURE(11), 
		jboolean
	>(get_jobject());
}

local_ref< java::util::Iterator > java::util::List::iterator()
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(12),
		java::util::List::J2CPP_METHOD_SIGNATURE(12), 
		local_ref< java::util::Iterator >
	>(get_jobject());
}

jint java::util::List::lastIndexOf(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(13),
		java::util::List::J2CPP_METHOD_SIGNATURE(13), 
		jint
	>(get_jobject(), a0);
}

local_ref< java::util::ListIterator > java::util::List::listIterator()
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(14),
		java::util::List::J2CPP_METHOD_SIGNATURE(14), 
		local_ref< java::util::ListIterator >
	>(get_jobject());
}

local_ref< java::util::ListIterator > java::util::List::listIterator(jint a0)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(15),
		java::util::List::J2CPP_METHOD_SIGNATURE(15), 
		local_ref< java::util::ListIterator >
	>(get_jobject(), a0);
}

local_ref< java::lang::Object > java::util::List::remove(jint a0)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(16),
		java::util::List::J2CPP_METHOD_SIGNATURE(16), 
		local_ref< java::lang::Object >
	>(get_jobject(), a0);
}

jboolean java::util::List::remove(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(17),
		java::util::List::J2CPP_METHOD_SIGNATURE(17), 
		jboolean
	>(get_jobject(), a0);
}

jboolean java::util::List::removeAll(local_ref< java::util::Collection > const &a0)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(18),
		java::util::List::J2CPP_METHOD_SIGNATURE(18), 
		jboolean
	>(get_jobject(), a0);
}

jboolean java::util::List::retainAll(local_ref< java::util::Collection > const &a0)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(19),
		java::util::List::J2CPP_METHOD_SIGNATURE(19), 
		jboolean
	>(get_jobject(), a0);
}

local_ref< java::lang::Object > java::util::List::set(jint a0, local_ref< java::lang::Object > const &a1)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(20),
		java::util::List::J2CPP_METHOD_SIGNATURE(20), 
		local_ref< java::lang::Object >
	>(get_jobject(), a0, a1);
}

jint java::util::List::size()
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(21),
		java::util::List::J2CPP_METHOD_SIGNATURE(21), 
		jint
	>(get_jobject());
}

local_ref< java::util::List > java::util::List::subList(jint a0, jint a1)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(22),
		java::util::List::J2CPP_METHOD_SIGNATURE(22), 
		local_ref< java::util::List >
	>(get_jobject(), a0, a1);
}

local_ref< array< local_ref< java::lang::Object >, 1> > java::util::List::toArray()
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(23),
		java::util::List::J2CPP_METHOD_SIGNATURE(23), 
		local_ref< array< local_ref< java::lang::Object >, 1> >
	>(get_jobject());
}

local_ref< array< local_ref< java::lang::Object >, 1> > java::util::List::toArray(local_ref< array< local_ref< java::lang::Object >, 1> > const &a0)
{
	return call_method<
		java::util::List::J2CPP_CLASS_NAME,
		java::util::List::J2CPP_METHOD_NAME(24),
		java::util::List::J2CPP_METHOD_SIGNATURE(24), 
		local_ref< array< local_ref< java::lang::Object >, 1> >
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(java::util::List,"java/util/List")
J2CPP_DEFINE_METHOD(java::util::List,0,"add","(ILjava/lang/Object;)V")
J2CPP_DEFINE_METHOD(java::util::List,1,"add","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::List,2,"addAll","(ILjava/util/Collection;)Z")
J2CPP_DEFINE_METHOD(java::util::List,3,"addAll","(Ljava/util/Collection;)Z")
J2CPP_DEFINE_METHOD(java::util::List,4,"clear","()V")
J2CPP_DEFINE_METHOD(java::util::List,5,"contains","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::List,6,"containsAll","(Ljava/util/Collection;)Z")
J2CPP_DEFINE_METHOD(java::util::List,7,"equals","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::List,8,"get","(I)Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::List,9,"hashCode","()I")
J2CPP_DEFINE_METHOD(java::util::List,10,"indexOf","(Ljava/lang/Object;)I")
J2CPP_DEFINE_METHOD(java::util::List,11,"isEmpty","()Z")
J2CPP_DEFINE_METHOD(java::util::List,12,"iterator","()Ljava/util/Iterator;")
J2CPP_DEFINE_METHOD(java::util::List,13,"lastIndexOf","(Ljava/lang/Object;)I")
J2CPP_DEFINE_METHOD(java::util::List,14,"listIterator","()Ljava/util/ListIterator;")
J2CPP_DEFINE_METHOD(java::util::List,15,"listIterator","(I)Ljava/util/ListIterator;")
J2CPP_DEFINE_METHOD(java::util::List,16,"remove","(I)Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::List,17,"remove","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::List,18,"removeAll","(Ljava/util/Collection;)Z")
J2CPP_DEFINE_METHOD(java::util::List,19,"retainAll","(Ljava/util/Collection;)Z")
J2CPP_DEFINE_METHOD(java::util::List,20,"set","(ILjava/lang/Object;)Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::List,21,"size","()I")
J2CPP_DEFINE_METHOD(java::util::List,22,"subList","(II)Ljava/util/List;")
J2CPP_DEFINE_METHOD(java::util::List,23,"toArray","()[java.lang.Object")
J2CPP_DEFINE_METHOD(java::util::List,24,"toArray","([java.lang.Object)[java.lang.Object")

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_LIST_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
