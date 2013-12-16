/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.text.StringCharacterIterator
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_TEXT_STRINGCHARACTERITERATOR_HPP_DECL
#define J2CPP_JAVA_TEXT_STRINGCHARACTERITERATOR_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace text { class CharacterIterator; } } }


#include <java/lang/Object.hpp>
#include <java/lang/String.hpp>
#include <java/text/CharacterIterator.hpp>


namespace j2cpp {

namespace java { namespace text {

	class StringCharacterIterator;
	class StringCharacterIterator
		: public object<StringCharacterIterator>
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

		explicit StringCharacterIterator(jobject jobj)
		: object<StringCharacterIterator>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<java::text::CharacterIterator>() const;


		StringCharacterIterator(local_ref< java::lang::String > const&);
		StringCharacterIterator(local_ref< java::lang::String > const&, jint);
		StringCharacterIterator(local_ref< java::lang::String > const&, jint, jint, jint);
		local_ref< java::lang::Object > clone();
		jchar current();
		jboolean equals(local_ref< java::lang::Object >  const&);
		jchar first();
		jint getBeginIndex();
		jint getEndIndex();
		jint getIndex();
		jint hashCode();
		jchar last();
		jchar next();
		jchar previous();
		jchar setIndex(jint);
		void setText(local_ref< java::lang::String >  const&);
	}; //class StringCharacterIterator

} //namespace text
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_TEXT_STRINGCHARACTERITERATOR_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_TEXT_STRINGCHARACTERITERATOR_HPP_IMPL
#define J2CPP_JAVA_TEXT_STRINGCHARACTERITERATOR_HPP_IMPL

namespace j2cpp {



java::text::StringCharacterIterator::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

java::text::StringCharacterIterator::operator local_ref<java::text::CharacterIterator>() const
{
	return local_ref<java::text::CharacterIterator>(get_jobject());
}


java::text::StringCharacterIterator::StringCharacterIterator(local_ref< java::lang::String > const &a0)
: object<java::text::StringCharacterIterator>(
	call_new_object<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(0),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}



java::text::StringCharacterIterator::StringCharacterIterator(local_ref< java::lang::String > const &a0, jint a1)
: object<java::text::StringCharacterIterator>(
	call_new_object<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(1),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1)
)
{
}



java::text::StringCharacterIterator::StringCharacterIterator(local_ref< java::lang::String > const &a0, jint a1, jint a2, jint a3)
: object<java::text::StringCharacterIterator>(
	call_new_object<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(2),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(2)
	>(a0, a1, a2, a3)
)
{
}


local_ref< java::lang::Object > java::text::StringCharacterIterator::clone()
{
	return call_method<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(3),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(3), 
		local_ref< java::lang::Object >
	>(get_jobject());
}

jchar java::text::StringCharacterIterator::current()
{
	return call_method<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(4),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(4), 
		jchar
	>(get_jobject());
}

jboolean java::text::StringCharacterIterator::equals(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(5),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(5), 
		jboolean
	>(get_jobject(), a0);
}

jchar java::text::StringCharacterIterator::first()
{
	return call_method<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(6),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(6), 
		jchar
	>(get_jobject());
}

jint java::text::StringCharacterIterator::getBeginIndex()
{
	return call_method<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(7),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(7), 
		jint
	>(get_jobject());
}

jint java::text::StringCharacterIterator::getEndIndex()
{
	return call_method<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(8),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(8), 
		jint
	>(get_jobject());
}

jint java::text::StringCharacterIterator::getIndex()
{
	return call_method<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(9),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(9), 
		jint
	>(get_jobject());
}

jint java::text::StringCharacterIterator::hashCode()
{
	return call_method<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(10),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(10), 
		jint
	>(get_jobject());
}

jchar java::text::StringCharacterIterator::last()
{
	return call_method<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(11),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(11), 
		jchar
	>(get_jobject());
}

jchar java::text::StringCharacterIterator::next()
{
	return call_method<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(12),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(12), 
		jchar
	>(get_jobject());
}

jchar java::text::StringCharacterIterator::previous()
{
	return call_method<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(13),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(13), 
		jchar
	>(get_jobject());
}

jchar java::text::StringCharacterIterator::setIndex(jint a0)
{
	return call_method<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(14),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(14), 
		jchar
	>(get_jobject(), a0);
}

void java::text::StringCharacterIterator::setText(local_ref< java::lang::String > const &a0)
{
	return call_method<
		java::text::StringCharacterIterator::J2CPP_CLASS_NAME,
		java::text::StringCharacterIterator::J2CPP_METHOD_NAME(15),
		java::text::StringCharacterIterator::J2CPP_METHOD_SIGNATURE(15), 
		void
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(java::text::StringCharacterIterator,"java/text/StringCharacterIterator")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,0,"<init>","(Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,1,"<init>","(Ljava/lang/String;I)V")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,2,"<init>","(Ljava/lang/String;III)V")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,3,"clone","()Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,4,"current","()C")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,5,"equals","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,6,"first","()C")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,7,"getBeginIndex","()I")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,8,"getEndIndex","()I")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,9,"getIndex","()I")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,10,"hashCode","()I")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,11,"last","()C")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,12,"next","()C")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,13,"previous","()C")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,14,"setIndex","(I)C")
J2CPP_DEFINE_METHOD(java::text::StringCharacterIterator,15,"setText","(Ljava/lang/String;)V")

} //namespace j2cpp

#endif //J2CPP_JAVA_TEXT_STRINGCHARACTERITERATOR_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
