/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.util.Vector
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_VECTOR_HPP_DECL
#define J2CPP_JAVA_UTIL_VECTOR_HPP_DECL


namespace j2cpp { namespace java { namespace io { class Serializable; } } }
namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class Cloneable; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace util { class AbstractList; } } }
namespace j2cpp { namespace java { namespace util { class Enumeration; } } }
namespace j2cpp { namespace java { namespace util { class RandomAccess; } } }
namespace j2cpp { namespace java { namespace util { class Collection; } } }
namespace j2cpp { namespace java { namespace util { class List; } } }


#include <java/io/Serializable.hpp>
#include <java/lang/Cloneable.hpp>
#include <java/lang/Object.hpp>
#include <java/lang/String.hpp>
#include <java/util/AbstractList.hpp>
#include <java/util/Collection.hpp>
#include <java/util/Enumeration.hpp>
#include <java/util/List.hpp>
#include <java/util/RandomAccess.hpp>


namespace j2cpp {

namespace java { namespace util {

	class Vector;
	class Vector
		: public object<Vector>
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
		J2CPP_DECLARE_METHOD(25)
		J2CPP_DECLARE_METHOD(26)
		J2CPP_DECLARE_METHOD(27)
		J2CPP_DECLARE_METHOD(28)
		J2CPP_DECLARE_METHOD(29)
		J2CPP_DECLARE_METHOD(30)
		J2CPP_DECLARE_METHOD(31)
		J2CPP_DECLARE_METHOD(32)
		J2CPP_DECLARE_METHOD(33)
		J2CPP_DECLARE_METHOD(34)
		J2CPP_DECLARE_METHOD(35)
		J2CPP_DECLARE_METHOD(36)
		J2CPP_DECLARE_METHOD(37)
		J2CPP_DECLARE_METHOD(38)
		J2CPP_DECLARE_METHOD(39)
		J2CPP_DECLARE_METHOD(40)
		J2CPP_DECLARE_METHOD(41)
		J2CPP_DECLARE_METHOD(42)
		J2CPP_DECLARE_METHOD(43)
		J2CPP_DECLARE_METHOD(44)
		J2CPP_DECLARE_METHOD(45)
		J2CPP_DECLARE_FIELD(0)
		J2CPP_DECLARE_FIELD(1)
		J2CPP_DECLARE_FIELD(2)

		explicit Vector(jobject jobj)
		: object<Vector>(jobj)
		{
		}

		operator local_ref<java::util::AbstractList>() const;
		operator local_ref<java::util::List>() const;
		operator local_ref<java::util::RandomAccess>() const;
		operator local_ref<java::lang::Cloneable>() const;
		operator local_ref<java::io::Serializable>() const;


		Vector();
		Vector(jint);
		Vector(jint, jint);
		Vector(local_ref< java::util::Collection > const&);
		void add(jint, local_ref< java::lang::Object >  const&);
		jboolean add(local_ref< java::lang::Object >  const&);
		jboolean addAll(jint, local_ref< java::util::Collection >  const&);
		jboolean addAll(local_ref< java::util::Collection >  const&);
		void addElement(local_ref< java::lang::Object >  const&);
		jint capacity();
		void clear();
		local_ref< java::lang::Object > clone();
		jboolean contains(local_ref< java::lang::Object >  const&);
		jboolean containsAll(local_ref< java::util::Collection >  const&);
		void copyInto(local_ref< array< local_ref< java::lang::Object >, 1> >  const&);
		local_ref< java::lang::Object > elementAt(jint);
		local_ref< java::util::Enumeration > elements();
		void ensureCapacity(jint);
		jboolean equals(local_ref< java::lang::Object >  const&);
		local_ref< java::lang::Object > firstElement();
		local_ref< java::lang::Object > get(jint);
		jint hashCode();
		jint indexOf(local_ref< java::lang::Object >  const&);
		jint indexOf(local_ref< java::lang::Object >  const&, jint);
		void insertElementAt(local_ref< java::lang::Object >  const&, jint);
		jboolean isEmpty();
		local_ref< java::lang::Object > lastElement();
		jint lastIndexOf(local_ref< java::lang::Object >  const&);
		jint lastIndexOf(local_ref< java::lang::Object >  const&, jint);
		local_ref< java::lang::Object > remove(jint);
		jboolean remove(local_ref< java::lang::Object >  const&);
		jboolean removeAll(local_ref< java::util::Collection >  const&);
		void removeAllElements();
		jboolean removeElement(local_ref< java::lang::Object >  const&);
		void removeElementAt(jint);
		jboolean retainAll(local_ref< java::util::Collection >  const&);
		local_ref< java::lang::Object > set(jint, local_ref< java::lang::Object >  const&);
		void setElementAt(local_ref< java::lang::Object >  const&, jint);
		void setSize(jint);
		jint size();
		local_ref< java::util::List > subList(jint, jint);
		local_ref< array< local_ref< java::lang::Object >, 1> > toArray();
		local_ref< array< local_ref< java::lang::Object >, 1> > toArray(local_ref< array< local_ref< java::lang::Object >, 1> >  const&);
		local_ref< java::lang::String > toString();
		void trimToSize();

	}; //class Vector

} //namespace util
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_VECTOR_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_UTIL_VECTOR_HPP_IMPL
#define J2CPP_JAVA_UTIL_VECTOR_HPP_IMPL

namespace j2cpp {



java::util::Vector::operator local_ref<java::util::AbstractList>() const
{
	return local_ref<java::util::AbstractList>(get_jobject());
}

java::util::Vector::operator local_ref<java::util::List>() const
{
	return local_ref<java::util::List>(get_jobject());
}

java::util::Vector::operator local_ref<java::util::RandomAccess>() const
{
	return local_ref<java::util::RandomAccess>(get_jobject());
}

java::util::Vector::operator local_ref<java::lang::Cloneable>() const
{
	return local_ref<java::lang::Cloneable>(get_jobject());
}

java::util::Vector::operator local_ref<java::io::Serializable>() const
{
	return local_ref<java::io::Serializable>(get_jobject());
}


java::util::Vector::Vector()
: object<java::util::Vector>(
	call_new_object<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(0),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



java::util::Vector::Vector(jint a0)
: object<java::util::Vector>(
	call_new_object<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(1),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}



java::util::Vector::Vector(jint a0, jint a1)
: object<java::util::Vector>(
	call_new_object<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(2),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(2)
	>(a0, a1)
)
{
}



java::util::Vector::Vector(local_ref< java::util::Collection > const &a0)
: object<java::util::Vector>(
	call_new_object<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(3),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(3)
	>(a0)
)
{
}


void java::util::Vector::add(jint a0, local_ref< java::lang::Object > const &a1)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(4),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(4), 
		void
	>(get_jobject(), a0, a1);
}

jboolean java::util::Vector::add(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(5),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(5), 
		jboolean
	>(get_jobject(), a0);
}

jboolean java::util::Vector::addAll(jint a0, local_ref< java::util::Collection > const &a1)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(6),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(6), 
		jboolean
	>(get_jobject(), a0, a1);
}

jboolean java::util::Vector::addAll(local_ref< java::util::Collection > const &a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(7),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(7), 
		jboolean
	>(get_jobject(), a0);
}

void java::util::Vector::addElement(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(8),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(8), 
		void
	>(get_jobject(), a0);
}

jint java::util::Vector::capacity()
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(9),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(9), 
		jint
	>(get_jobject());
}

void java::util::Vector::clear()
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(10),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(10), 
		void
	>(get_jobject());
}

local_ref< java::lang::Object > java::util::Vector::clone()
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(11),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(11), 
		local_ref< java::lang::Object >
	>(get_jobject());
}

jboolean java::util::Vector::contains(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(12),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(12), 
		jboolean
	>(get_jobject(), a0);
}

jboolean java::util::Vector::containsAll(local_ref< java::util::Collection > const &a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(13),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(13), 
		jboolean
	>(get_jobject(), a0);
}

void java::util::Vector::copyInto(local_ref< array< local_ref< java::lang::Object >, 1> > const &a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(14),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(14), 
		void
	>(get_jobject(), a0);
}

local_ref< java::lang::Object > java::util::Vector::elementAt(jint a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(15),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(15), 
		local_ref< java::lang::Object >
	>(get_jobject(), a0);
}

local_ref< java::util::Enumeration > java::util::Vector::elements()
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(16),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(16), 
		local_ref< java::util::Enumeration >
	>(get_jobject());
}

void java::util::Vector::ensureCapacity(jint a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(17),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(17), 
		void
	>(get_jobject(), a0);
}

jboolean java::util::Vector::equals(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(18),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(18), 
		jboolean
	>(get_jobject(), a0);
}

local_ref< java::lang::Object > java::util::Vector::firstElement()
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(19),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(19), 
		local_ref< java::lang::Object >
	>(get_jobject());
}

local_ref< java::lang::Object > java::util::Vector::get(jint a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(20),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(20), 
		local_ref< java::lang::Object >
	>(get_jobject(), a0);
}

jint java::util::Vector::hashCode()
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(21),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(21), 
		jint
	>(get_jobject());
}

jint java::util::Vector::indexOf(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(22),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(22), 
		jint
	>(get_jobject(), a0);
}

jint java::util::Vector::indexOf(local_ref< java::lang::Object > const &a0, jint a1)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(23),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(23), 
		jint
	>(get_jobject(), a0, a1);
}

void java::util::Vector::insertElementAt(local_ref< java::lang::Object > const &a0, jint a1)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(24),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(24), 
		void
	>(get_jobject(), a0, a1);
}

jboolean java::util::Vector::isEmpty()
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(25),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(25), 
		jboolean
	>(get_jobject());
}

local_ref< java::lang::Object > java::util::Vector::lastElement()
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(26),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(26), 
		local_ref< java::lang::Object >
	>(get_jobject());
}

jint java::util::Vector::lastIndexOf(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(27),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(27), 
		jint
	>(get_jobject(), a0);
}

jint java::util::Vector::lastIndexOf(local_ref< java::lang::Object > const &a0, jint a1)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(28),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(28), 
		jint
	>(get_jobject(), a0, a1);
}

local_ref< java::lang::Object > java::util::Vector::remove(jint a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(29),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(29), 
		local_ref< java::lang::Object >
	>(get_jobject(), a0);
}

jboolean java::util::Vector::remove(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(30),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(30), 
		jboolean
	>(get_jobject(), a0);
}

jboolean java::util::Vector::removeAll(local_ref< java::util::Collection > const &a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(31),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(31), 
		jboolean
	>(get_jobject(), a0);
}

void java::util::Vector::removeAllElements()
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(32),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(32), 
		void
	>(get_jobject());
}

jboolean java::util::Vector::removeElement(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(33),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(33), 
		jboolean
	>(get_jobject(), a0);
}

void java::util::Vector::removeElementAt(jint a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(34),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(34), 
		void
	>(get_jobject(), a0);
}


jboolean java::util::Vector::retainAll(local_ref< java::util::Collection > const &a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(36),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(36), 
		jboolean
	>(get_jobject(), a0);
}

local_ref< java::lang::Object > java::util::Vector::set(jint a0, local_ref< java::lang::Object > const &a1)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(37),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(37), 
		local_ref< java::lang::Object >
	>(get_jobject(), a0, a1);
}

void java::util::Vector::setElementAt(local_ref< java::lang::Object > const &a0, jint a1)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(38),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(38), 
		void
	>(get_jobject(), a0, a1);
}

void java::util::Vector::setSize(jint a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(39),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(39), 
		void
	>(get_jobject(), a0);
}

jint java::util::Vector::size()
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(40),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(40), 
		jint
	>(get_jobject());
}

local_ref< java::util::List > java::util::Vector::subList(jint a0, jint a1)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(41),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(41), 
		local_ref< java::util::List >
	>(get_jobject(), a0, a1);
}

local_ref< array< local_ref< java::lang::Object >, 1> > java::util::Vector::toArray()
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(42),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(42), 
		local_ref< array< local_ref< java::lang::Object >, 1> >
	>(get_jobject());
}

local_ref< array< local_ref< java::lang::Object >, 1> > java::util::Vector::toArray(local_ref< array< local_ref< java::lang::Object >, 1> > const &a0)
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(43),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(43), 
		local_ref< array< local_ref< java::lang::Object >, 1> >
	>(get_jobject(), a0);
}

local_ref< java::lang::String > java::util::Vector::toString()
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(44),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(44), 
		local_ref< java::lang::String >
	>(get_jobject());
}

void java::util::Vector::trimToSize()
{
	return call_method<
		java::util::Vector::J2CPP_CLASS_NAME,
		java::util::Vector::J2CPP_METHOD_NAME(45),
		java::util::Vector::J2CPP_METHOD_SIGNATURE(45), 
		void
	>(get_jobject());
}



J2CPP_DEFINE_CLASS(java::util::Vector,"java/util/Vector")
J2CPP_DEFINE_METHOD(java::util::Vector,0,"<init>","()V")
J2CPP_DEFINE_METHOD(java::util::Vector,1,"<init>","(I)V")
J2CPP_DEFINE_METHOD(java::util::Vector,2,"<init>","(II)V")
J2CPP_DEFINE_METHOD(java::util::Vector,3,"<init>","(Ljava/util/Collection;)V")
J2CPP_DEFINE_METHOD(java::util::Vector,4,"add","(ILjava/lang/Object;)V")
J2CPP_DEFINE_METHOD(java::util::Vector,5,"add","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::Vector,6,"addAll","(ILjava/util/Collection;)Z")
J2CPP_DEFINE_METHOD(java::util::Vector,7,"addAll","(Ljava/util/Collection;)Z")
J2CPP_DEFINE_METHOD(java::util::Vector,8,"addElement","(Ljava/lang/Object;)V")
J2CPP_DEFINE_METHOD(java::util::Vector,9,"capacity","()I")
J2CPP_DEFINE_METHOD(java::util::Vector,10,"clear","()V")
J2CPP_DEFINE_METHOD(java::util::Vector,11,"clone","()Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Vector,12,"contains","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::Vector,13,"containsAll","(Ljava/util/Collection;)Z")
J2CPP_DEFINE_METHOD(java::util::Vector,14,"copyInto","([java.lang.Object)V")
J2CPP_DEFINE_METHOD(java::util::Vector,15,"elementAt","(I)Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Vector,16,"elements","()Ljava/util/Enumeration;")
J2CPP_DEFINE_METHOD(java::util::Vector,17,"ensureCapacity","(I)V")
J2CPP_DEFINE_METHOD(java::util::Vector,18,"equals","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::Vector,19,"firstElement","()Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Vector,20,"get","(I)Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Vector,21,"hashCode","()I")
J2CPP_DEFINE_METHOD(java::util::Vector,22,"indexOf","(Ljava/lang/Object;)I")
J2CPP_DEFINE_METHOD(java::util::Vector,23,"indexOf","(Ljava/lang/Object;I)I")
J2CPP_DEFINE_METHOD(java::util::Vector,24,"insertElementAt","(Ljava/lang/Object;I)V")
J2CPP_DEFINE_METHOD(java::util::Vector,25,"isEmpty","()Z")
J2CPP_DEFINE_METHOD(java::util::Vector,26,"lastElement","()Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Vector,27,"lastIndexOf","(Ljava/lang/Object;)I")
J2CPP_DEFINE_METHOD(java::util::Vector,28,"lastIndexOf","(Ljava/lang/Object;I)I")
J2CPP_DEFINE_METHOD(java::util::Vector,29,"remove","(I)Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Vector,30,"remove","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::Vector,31,"removeAll","(Ljava/util/Collection;)Z")
J2CPP_DEFINE_METHOD(java::util::Vector,32,"removeAllElements","()V")
J2CPP_DEFINE_METHOD(java::util::Vector,33,"removeElement","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::util::Vector,34,"removeElementAt","(I)V")
J2CPP_DEFINE_METHOD(java::util::Vector,35,"removeRange","(II)V")
J2CPP_DEFINE_METHOD(java::util::Vector,36,"retainAll","(Ljava/util/Collection;)Z")
J2CPP_DEFINE_METHOD(java::util::Vector,37,"set","(ILjava/lang/Object;)Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::util::Vector,38,"setElementAt","(Ljava/lang/Object;I)V")
J2CPP_DEFINE_METHOD(java::util::Vector,39,"setSize","(I)V")
J2CPP_DEFINE_METHOD(java::util::Vector,40,"size","()I")
J2CPP_DEFINE_METHOD(java::util::Vector,41,"subList","(II)Ljava/util/List;")
J2CPP_DEFINE_METHOD(java::util::Vector,42,"toArray","()[java.lang.Object")
J2CPP_DEFINE_METHOD(java::util::Vector,43,"toArray","([java.lang.Object)[java.lang.Object")
J2CPP_DEFINE_METHOD(java::util::Vector,44,"toString","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(java::util::Vector,45,"trimToSize","()V")
J2CPP_DEFINE_FIELD(java::util::Vector,0,"elementCount","I")
J2CPP_DEFINE_FIELD(java::util::Vector,1,"elementData","[java.lang.Object")
J2CPP_DEFINE_FIELD(java::util::Vector,2,"capacityIncrement","I")

} //namespace j2cpp

#endif //J2CPP_JAVA_UTIL_VECTOR_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
