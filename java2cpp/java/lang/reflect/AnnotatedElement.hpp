/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.lang.reflect.AnnotatedElement
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_LANG_REFLECT_ANNOTATEDELEMENT_HPP_DECL
#define J2CPP_JAVA_LANG_REFLECT_ANNOTATEDELEMENT_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class Class; } } }
namespace j2cpp { namespace java { namespace lang { namespace annotation { class Annotation; } } } }


#include <java/lang/Class.hpp>
#include <java/lang/Object.hpp>
#include <java/lang/annotation/Annotation.hpp>


namespace j2cpp {

namespace java { namespace lang { namespace reflect {

	class AnnotatedElement;
	class AnnotatedElement
		: public object<AnnotatedElement>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)

		explicit AnnotatedElement(jobject jobj)
		: object<AnnotatedElement>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;


		local_ref< java::lang::annotation::Annotation > getAnnotation(local_ref< java::lang::Class >  const&);
		local_ref< array< local_ref< java::lang::annotation::Annotation >, 1> > getAnnotations();
		local_ref< array< local_ref< java::lang::annotation::Annotation >, 1> > getDeclaredAnnotations();
		jboolean isAnnotationPresent(local_ref< java::lang::Class >  const&);
	}; //class AnnotatedElement

} //namespace reflect
} //namespace lang
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_LANG_REFLECT_ANNOTATEDELEMENT_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_LANG_REFLECT_ANNOTATEDELEMENT_HPP_IMPL
#define J2CPP_JAVA_LANG_REFLECT_ANNOTATEDELEMENT_HPP_IMPL

namespace j2cpp {



java::lang::reflect::AnnotatedElement::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

local_ref< java::lang::annotation::Annotation > java::lang::reflect::AnnotatedElement::getAnnotation(local_ref< java::lang::Class > const &a0)
{
	return call_method<
		java::lang::reflect::AnnotatedElement::J2CPP_CLASS_NAME,
		java::lang::reflect::AnnotatedElement::J2CPP_METHOD_NAME(0),
		java::lang::reflect::AnnotatedElement::J2CPP_METHOD_SIGNATURE(0), 
		local_ref< java::lang::annotation::Annotation >
	>(get_jobject(), a0);
}

local_ref< array< local_ref< java::lang::annotation::Annotation >, 1> > java::lang::reflect::AnnotatedElement::getAnnotations()
{
	return call_method<
		java::lang::reflect::AnnotatedElement::J2CPP_CLASS_NAME,
		java::lang::reflect::AnnotatedElement::J2CPP_METHOD_NAME(1),
		java::lang::reflect::AnnotatedElement::J2CPP_METHOD_SIGNATURE(1), 
		local_ref< array< local_ref< java::lang::annotation::Annotation >, 1> >
	>(get_jobject());
}

local_ref< array< local_ref< java::lang::annotation::Annotation >, 1> > java::lang::reflect::AnnotatedElement::getDeclaredAnnotations()
{
	return call_method<
		java::lang::reflect::AnnotatedElement::J2CPP_CLASS_NAME,
		java::lang::reflect::AnnotatedElement::J2CPP_METHOD_NAME(2),
		java::lang::reflect::AnnotatedElement::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< array< local_ref< java::lang::annotation::Annotation >, 1> >
	>(get_jobject());
}

jboolean java::lang::reflect::AnnotatedElement::isAnnotationPresent(local_ref< java::lang::Class > const &a0)
{
	return call_method<
		java::lang::reflect::AnnotatedElement::J2CPP_CLASS_NAME,
		java::lang::reflect::AnnotatedElement::J2CPP_METHOD_NAME(3),
		java::lang::reflect::AnnotatedElement::J2CPP_METHOD_SIGNATURE(3), 
		jboolean
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(java::lang::reflect::AnnotatedElement,"java/lang/reflect/AnnotatedElement")
J2CPP_DEFINE_METHOD(java::lang::reflect::AnnotatedElement,0,"getAnnotation","(Ljava/lang/Class;)Ljava/lang/annotation/Annotation;")
J2CPP_DEFINE_METHOD(java::lang::reflect::AnnotatedElement,1,"getAnnotations","()[java.lang.annotation.Annotation")
J2CPP_DEFINE_METHOD(java::lang::reflect::AnnotatedElement,2,"getDeclaredAnnotations","()[java.lang.annotation.Annotation")
J2CPP_DEFINE_METHOD(java::lang::reflect::AnnotatedElement,3,"isAnnotationPresent","(Ljava/lang/Class;)Z")

} //namespace j2cpp

#endif //J2CPP_JAVA_LANG_REFLECT_ANNOTATEDELEMENT_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
