/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.sax.RootElement
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_SAX_ROOTELEMENT_HPP_DECL
#define J2CPP_ANDROID_SAX_ROOTELEMENT_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace org { namespace xml { namespace sax { class ContentHandler; } } } }
namespace j2cpp { namespace android { namespace sax { class Element; } } }


#include <android/sax/Element.hpp>
#include <java/lang/String.hpp>
#include <org/xml/sax/ContentHandler.hpp>


namespace j2cpp {

namespace android { namespace sax {

	class RootElement;
	class RootElement
		: public object<RootElement>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)

		explicit RootElement(jobject jobj)
		: object<RootElement>(jobj)
		{
		}

		operator local_ref<android::sax::Element>() const;


		RootElement(local_ref< java::lang::String > const&, local_ref< java::lang::String > const&);
		RootElement(local_ref< java::lang::String > const&);
		local_ref< org::xml::sax::ContentHandler > getContentHandler();
	}; //class RootElement

} //namespace sax
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_SAX_ROOTELEMENT_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_SAX_ROOTELEMENT_HPP_IMPL
#define J2CPP_ANDROID_SAX_ROOTELEMENT_HPP_IMPL

namespace j2cpp {



android::sax::RootElement::operator local_ref<android::sax::Element>() const
{
	return local_ref<android::sax::Element>(get_jobject());
}


android::sax::RootElement::RootElement(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1)
: object<android::sax::RootElement>(
	call_new_object<
		android::sax::RootElement::J2CPP_CLASS_NAME,
		android::sax::RootElement::J2CPP_METHOD_NAME(0),
		android::sax::RootElement::J2CPP_METHOD_SIGNATURE(0)
	>(a0, a1)
)
{
}



android::sax::RootElement::RootElement(local_ref< java::lang::String > const &a0)
: object<android::sax::RootElement>(
	call_new_object<
		android::sax::RootElement::J2CPP_CLASS_NAME,
		android::sax::RootElement::J2CPP_METHOD_NAME(1),
		android::sax::RootElement::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}


local_ref< org::xml::sax::ContentHandler > android::sax::RootElement::getContentHandler()
{
	return call_method<
		android::sax::RootElement::J2CPP_CLASS_NAME,
		android::sax::RootElement::J2CPP_METHOD_NAME(2),
		android::sax::RootElement::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< org::xml::sax::ContentHandler >
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(android::sax::RootElement,"android/sax/RootElement")
J2CPP_DEFINE_METHOD(android::sax::RootElement,0,"<init>","(Ljava/lang/String;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(android::sax::RootElement,1,"<init>","(Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(android::sax::RootElement,2,"getContentHandler","()Lorg/xml/sax/ContentHandler;")

} //namespace j2cpp

#endif //J2CPP_ANDROID_SAX_ROOTELEMENT_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
