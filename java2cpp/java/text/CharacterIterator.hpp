/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.text.CharacterIterator
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_TEXT_CHARACTERITERATOR_HPP_DECL
#define J2CPP_JAVA_TEXT_CHARACTERITERATOR_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class Cloneable; } } }


#include <java/lang/Cloneable.hpp>
#include <java/lang/Object.hpp>


namespace j2cpp {

namespace java { namespace text {

	class CharacterIterator;
	class CharacterIterator
		: public object<CharacterIterator>
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
		J2CPP_DECLARE_FIELD(0)

		explicit CharacterIterator(jobject jobj)
		: object<CharacterIterator>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<java::lang::Cloneable>() const;


		local_ref< java::lang::Object > clone();
		jchar current();
		jchar first();
		jint getBeginIndex();
		jint getEndIndex();
		jint getIndex();
		jchar last();
		jchar next();
		jchar previous();
		jchar setIndex(jint);

		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(0), J2CPP_FIELD_SIGNATURE(0), jchar > DONE;
	}; //class CharacterIterator

} //namespace text
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_TEXT_CHARACTERITERATOR_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_TEXT_CHARACTERITERATOR_HPP_IMPL
#define J2CPP_JAVA_TEXT_CHARACTERITERATOR_HPP_IMPL

namespace j2cpp {



java::text::CharacterIterator::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

java::text::CharacterIterator::operator local_ref<java::lang::Cloneable>() const
{
	return local_ref<java::lang::Cloneable>(get_jobject());
}

local_ref< java::lang::Object > java::text::CharacterIterator::clone()
{
	return call_method<
		java::text::CharacterIterator::J2CPP_CLASS_NAME,
		java::text::CharacterIterator::J2CPP_METHOD_NAME(0),
		java::text::CharacterIterator::J2CPP_METHOD_SIGNATURE(0), 
		local_ref< java::lang::Object >
	>(get_jobject());
}

jchar java::text::CharacterIterator::current()
{
	return call_method<
		java::text::CharacterIterator::J2CPP_CLASS_NAME,
		java::text::CharacterIterator::J2CPP_METHOD_NAME(1),
		java::text::CharacterIterator::J2CPP_METHOD_SIGNATURE(1), 
		jchar
	>(get_jobject());
}

jchar java::text::CharacterIterator::first()
{
	return call_method<
		java::text::CharacterIterator::J2CPP_CLASS_NAME,
		java::text::CharacterIterator::J2CPP_METHOD_NAME(2),
		java::text::CharacterIterator::J2CPP_METHOD_SIGNATURE(2), 
		jchar
	>(get_jobject());
}

jint java::text::CharacterIterator::getBeginIndex()
{
	return call_method<
		java::text::CharacterIterator::J2CPP_CLASS_NAME,
		java::text::CharacterIterator::J2CPP_METHOD_NAME(3),
		java::text::CharacterIterator::J2CPP_METHOD_SIGNATURE(3), 
		jint
	>(get_jobject());
}

jint java::text::CharacterIterator::getEndIndex()
{
	return call_method<
		java::text::CharacterIterator::J2CPP_CLASS_NAME,
		java::text::CharacterIterator::J2CPP_METHOD_NAME(4),
		java::text::CharacterIterator::J2CPP_METHOD_SIGNATURE(4), 
		jint
	>(get_jobject());
}

jint java::text::CharacterIterator::getIndex()
{
	return call_method<
		java::text::CharacterIterator::J2CPP_CLASS_NAME,
		java::text::CharacterIterator::J2CPP_METHOD_NAME(5),
		java::text::CharacterIterator::J2CPP_METHOD_SIGNATURE(5), 
		jint
	>(get_jobject());
}

jchar java::text::CharacterIterator::last()
{
	return call_method<
		java::text::CharacterIterator::J2CPP_CLASS_NAME,
		java::text::CharacterIterator::J2CPP_METHOD_NAME(6),
		java::text::CharacterIterator::J2CPP_METHOD_SIGNATURE(6), 
		jchar
	>(get_jobject());
}

jchar java::text::CharacterIterator::next()
{
	return call_method<
		java::text::CharacterIterator::J2CPP_CLASS_NAME,
		java::text::CharacterIterator::J2CPP_METHOD_NAME(7),
		java::text::CharacterIterator::J2CPP_METHOD_SIGNATURE(7), 
		jchar
	>(get_jobject());
}

jchar java::text::CharacterIterator::previous()
{
	return call_method<
		java::text::CharacterIterator::J2CPP_CLASS_NAME,
		java::text::CharacterIterator::J2CPP_METHOD_NAME(8),
		java::text::CharacterIterator::J2CPP_METHOD_SIGNATURE(8), 
		jchar
	>(get_jobject());
}

jchar java::text::CharacterIterator::setIndex(jint a0)
{
	return call_method<
		java::text::CharacterIterator::J2CPP_CLASS_NAME,
		java::text::CharacterIterator::J2CPP_METHOD_NAME(9),
		java::text::CharacterIterator::J2CPP_METHOD_SIGNATURE(9), 
		jchar
	>(get_jobject(), a0);
}


static_field<
	java::text::CharacterIterator::J2CPP_CLASS_NAME,
	java::text::CharacterIterator::J2CPP_FIELD_NAME(0),
	java::text::CharacterIterator::J2CPP_FIELD_SIGNATURE(0),
	jchar
> java::text::CharacterIterator::DONE;


J2CPP_DEFINE_CLASS(java::text::CharacterIterator,"java/text/CharacterIterator")
J2CPP_DEFINE_METHOD(java::text::CharacterIterator,0,"clone","()Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::text::CharacterIterator,1,"current","()C")
J2CPP_DEFINE_METHOD(java::text::CharacterIterator,2,"first","()C")
J2CPP_DEFINE_METHOD(java::text::CharacterIterator,3,"getBeginIndex","()I")
J2CPP_DEFINE_METHOD(java::text::CharacterIterator,4,"getEndIndex","()I")
J2CPP_DEFINE_METHOD(java::text::CharacterIterator,5,"getIndex","()I")
J2CPP_DEFINE_METHOD(java::text::CharacterIterator,6,"last","()C")
J2CPP_DEFINE_METHOD(java::text::CharacterIterator,7,"next","()C")
J2CPP_DEFINE_METHOD(java::text::CharacterIterator,8,"previous","()C")
J2CPP_DEFINE_METHOD(java::text::CharacterIterator,9,"setIndex","(I)C")
J2CPP_DEFINE_FIELD(java::text::CharacterIterator,0,"DONE","C")

} //namespace j2cpp

#endif //J2CPP_JAVA_TEXT_CHARACTERITERATOR_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
