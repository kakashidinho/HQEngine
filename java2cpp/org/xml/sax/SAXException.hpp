/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: org.xml.sax.SAXException
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_XML_SAX_SAXEXCEPTION_HPP_DECL
#define J2CPP_ORG_XML_SAX_SAXEXCEPTION_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace lang { class Exception; } } }


#include <java/lang/Exception.hpp>
#include <java/lang/String.hpp>


namespace j2cpp {

namespace org { namespace xml { namespace sax {

	class SAXException;
	class SAXException
		: public object<SAXException>
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

		explicit SAXException(jobject jobj)
		: object<SAXException>(jobj)
		{
		}

		operator local_ref<java::lang::Exception>() const;


		SAXException();
		SAXException(local_ref< java::lang::String > const&);
		SAXException(local_ref< java::lang::Exception > const&);
		SAXException(local_ref< java::lang::String > const&, local_ref< java::lang::Exception > const&);
		local_ref< java::lang::String > getMessage();
		local_ref< java::lang::Exception > getException();
		local_ref< java::lang::String > toString();
	}; //class SAXException

} //namespace sax
} //namespace xml
} //namespace org

} //namespace j2cpp

#endif //J2CPP_ORG_XML_SAX_SAXEXCEPTION_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_XML_SAX_SAXEXCEPTION_HPP_IMPL
#define J2CPP_ORG_XML_SAX_SAXEXCEPTION_HPP_IMPL

namespace j2cpp {



org::xml::sax::SAXException::operator local_ref<java::lang::Exception>() const
{
	return local_ref<java::lang::Exception>(get_jobject());
}


org::xml::sax::SAXException::SAXException()
: object<org::xml::sax::SAXException>(
	call_new_object<
		org::xml::sax::SAXException::J2CPP_CLASS_NAME,
		org::xml::sax::SAXException::J2CPP_METHOD_NAME(0),
		org::xml::sax::SAXException::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



org::xml::sax::SAXException::SAXException(local_ref< java::lang::String > const &a0)
: object<org::xml::sax::SAXException>(
	call_new_object<
		org::xml::sax::SAXException::J2CPP_CLASS_NAME,
		org::xml::sax::SAXException::J2CPP_METHOD_NAME(1),
		org::xml::sax::SAXException::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}



org::xml::sax::SAXException::SAXException(local_ref< java::lang::Exception > const &a0)
: object<org::xml::sax::SAXException>(
	call_new_object<
		org::xml::sax::SAXException::J2CPP_CLASS_NAME,
		org::xml::sax::SAXException::J2CPP_METHOD_NAME(2),
		org::xml::sax::SAXException::J2CPP_METHOD_SIGNATURE(2)
	>(a0)
)
{
}



org::xml::sax::SAXException::SAXException(local_ref< java::lang::String > const &a0, local_ref< java::lang::Exception > const &a1)
: object<org::xml::sax::SAXException>(
	call_new_object<
		org::xml::sax::SAXException::J2CPP_CLASS_NAME,
		org::xml::sax::SAXException::J2CPP_METHOD_NAME(3),
		org::xml::sax::SAXException::J2CPP_METHOD_SIGNATURE(3)
	>(a0, a1)
)
{
}


local_ref< java::lang::String > org::xml::sax::SAXException::getMessage()
{
	return call_method<
		org::xml::sax::SAXException::J2CPP_CLASS_NAME,
		org::xml::sax::SAXException::J2CPP_METHOD_NAME(4),
		org::xml::sax::SAXException::J2CPP_METHOD_SIGNATURE(4), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::Exception > org::xml::sax::SAXException::getException()
{
	return call_method<
		org::xml::sax::SAXException::J2CPP_CLASS_NAME,
		org::xml::sax::SAXException::J2CPP_METHOD_NAME(5),
		org::xml::sax::SAXException::J2CPP_METHOD_SIGNATURE(5), 
		local_ref< java::lang::Exception >
	>(get_jobject());
}

local_ref< java::lang::String > org::xml::sax::SAXException::toString()
{
	return call_method<
		org::xml::sax::SAXException::J2CPP_CLASS_NAME,
		org::xml::sax::SAXException::J2CPP_METHOD_NAME(6),
		org::xml::sax::SAXException::J2CPP_METHOD_SIGNATURE(6), 
		local_ref< java::lang::String >
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(org::xml::sax::SAXException,"org/xml/sax/SAXException")
J2CPP_DEFINE_METHOD(org::xml::sax::SAXException,0,"<init>","()V")
J2CPP_DEFINE_METHOD(org::xml::sax::SAXException,1,"<init>","(Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::SAXException,2,"<init>","(Ljava/lang/Exception;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::SAXException,3,"<init>","(Ljava/lang/String;Ljava/lang/Exception;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::SAXException,4,"getMessage","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(org::xml::sax::SAXException,5,"getException","()Ljava/lang/Exception;")
J2CPP_DEFINE_METHOD(org::xml::sax::SAXException,6,"toString","()Ljava/lang/String;")

} //namespace j2cpp

#endif //J2CPP_ORG_XML_SAX_SAXEXCEPTION_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
