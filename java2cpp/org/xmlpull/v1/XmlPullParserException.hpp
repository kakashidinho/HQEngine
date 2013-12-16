/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: org.xmlpull.v1.XmlPullParserException
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_XMLPULL_V1_XMLPULLPARSEREXCEPTION_HPP_DECL
#define J2CPP_ORG_XMLPULL_V1_XMLPULLPARSEREXCEPTION_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace lang { class Throwable; } } }
namespace j2cpp { namespace java { namespace lang { class Exception; } } }
namespace j2cpp { namespace org { namespace xmlpull { namespace v1 { class XmlPullParser; } } } }


#include <java/lang/Exception.hpp>
#include <java/lang/String.hpp>
#include <java/lang/Throwable.hpp>
#include <org/xmlpull/v1/XmlPullParser.hpp>


namespace j2cpp {

namespace org { namespace xmlpull { namespace v1 {

	class XmlPullParserException;
	class XmlPullParserException
		: public object<XmlPullParserException>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)
		J2CPP_DECLARE_METHOD(4)
		J2CPP_DECLARE_METHOD(5)
		J2CPP_DECLARE_FIELD(0)
		J2CPP_DECLARE_FIELD(1)
		J2CPP_DECLARE_FIELD(2)

		explicit XmlPullParserException(jobject jobj)
		: object<XmlPullParserException>(jobj)
		{
		}

		operator local_ref<java::lang::Exception>() const;


		XmlPullParserException(local_ref< java::lang::String > const&);
		XmlPullParserException(local_ref< java::lang::String > const&, local_ref< org::xmlpull::v1::XmlPullParser > const&, local_ref< java::lang::Throwable > const&);
		local_ref< java::lang::Throwable > getDetail();
		jint getLineNumber();
		jint getColumnNumber();
		void printStackTrace();

	}; //class XmlPullParserException

} //namespace v1
} //namespace xmlpull
} //namespace org

} //namespace j2cpp

#endif //J2CPP_ORG_XMLPULL_V1_XMLPULLPARSEREXCEPTION_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_XMLPULL_V1_XMLPULLPARSEREXCEPTION_HPP_IMPL
#define J2CPP_ORG_XMLPULL_V1_XMLPULLPARSEREXCEPTION_HPP_IMPL

namespace j2cpp {



org::xmlpull::v1::XmlPullParserException::operator local_ref<java::lang::Exception>() const
{
	return local_ref<java::lang::Exception>(get_jobject());
}


org::xmlpull::v1::XmlPullParserException::XmlPullParserException(local_ref< java::lang::String > const &a0)
: object<org::xmlpull::v1::XmlPullParserException>(
	call_new_object<
		org::xmlpull::v1::XmlPullParserException::J2CPP_CLASS_NAME,
		org::xmlpull::v1::XmlPullParserException::J2CPP_METHOD_NAME(0),
		org::xmlpull::v1::XmlPullParserException::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}



org::xmlpull::v1::XmlPullParserException::XmlPullParserException(local_ref< java::lang::String > const &a0, local_ref< org::xmlpull::v1::XmlPullParser > const &a1, local_ref< java::lang::Throwable > const &a2)
: object<org::xmlpull::v1::XmlPullParserException>(
	call_new_object<
		org::xmlpull::v1::XmlPullParserException::J2CPP_CLASS_NAME,
		org::xmlpull::v1::XmlPullParserException::J2CPP_METHOD_NAME(1),
		org::xmlpull::v1::XmlPullParserException::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1, a2)
)
{
}


local_ref< java::lang::Throwable > org::xmlpull::v1::XmlPullParserException::getDetail()
{
	return call_method<
		org::xmlpull::v1::XmlPullParserException::J2CPP_CLASS_NAME,
		org::xmlpull::v1::XmlPullParserException::J2CPP_METHOD_NAME(2),
		org::xmlpull::v1::XmlPullParserException::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< java::lang::Throwable >
	>(get_jobject());
}

jint org::xmlpull::v1::XmlPullParserException::getLineNumber()
{
	return call_method<
		org::xmlpull::v1::XmlPullParserException::J2CPP_CLASS_NAME,
		org::xmlpull::v1::XmlPullParserException::J2CPP_METHOD_NAME(3),
		org::xmlpull::v1::XmlPullParserException::J2CPP_METHOD_SIGNATURE(3), 
		jint
	>(get_jobject());
}

jint org::xmlpull::v1::XmlPullParserException::getColumnNumber()
{
	return call_method<
		org::xmlpull::v1::XmlPullParserException::J2CPP_CLASS_NAME,
		org::xmlpull::v1::XmlPullParserException::J2CPP_METHOD_NAME(4),
		org::xmlpull::v1::XmlPullParserException::J2CPP_METHOD_SIGNATURE(4), 
		jint
	>(get_jobject());
}

void org::xmlpull::v1::XmlPullParserException::printStackTrace()
{
	return call_method<
		org::xmlpull::v1::XmlPullParserException::J2CPP_CLASS_NAME,
		org::xmlpull::v1::XmlPullParserException::J2CPP_METHOD_NAME(5),
		org::xmlpull::v1::XmlPullParserException::J2CPP_METHOD_SIGNATURE(5), 
		void
	>(get_jobject());
}



J2CPP_DEFINE_CLASS(org::xmlpull::v1::XmlPullParserException,"org/xmlpull/v1/XmlPullParserException")
J2CPP_DEFINE_METHOD(org::xmlpull::v1::XmlPullParserException,0,"<init>","(Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(org::xmlpull::v1::XmlPullParserException,1,"<init>","(Ljava/lang/String;Lorg/xmlpull/v1/XmlPullParser;Ljava/lang/Throwable;)V")
J2CPP_DEFINE_METHOD(org::xmlpull::v1::XmlPullParserException,2,"getDetail","()Ljava/lang/Throwable;")
J2CPP_DEFINE_METHOD(org::xmlpull::v1::XmlPullParserException,3,"getLineNumber","()I")
J2CPP_DEFINE_METHOD(org::xmlpull::v1::XmlPullParserException,4,"getColumnNumber","()I")
J2CPP_DEFINE_METHOD(org::xmlpull::v1::XmlPullParserException,5,"printStackTrace","()V")
J2CPP_DEFINE_FIELD(org::xmlpull::v1::XmlPullParserException,0,"detail","Ljava/lang/Throwable;")
J2CPP_DEFINE_FIELD(org::xmlpull::v1::XmlPullParserException,1,"row","I")
J2CPP_DEFINE_FIELD(org::xmlpull::v1::XmlPullParserException,2,"column","I")

} //namespace j2cpp

#endif //J2CPP_ORG_XMLPULL_V1_XMLPULLPARSEREXCEPTION_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
