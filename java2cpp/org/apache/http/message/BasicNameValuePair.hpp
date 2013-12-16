/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: org.apache.http.message.BasicNameValuePair
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_APACHE_HTTP_MESSAGE_BASICNAMEVALUEPAIR_HPP_DECL
#define J2CPP_ORG_APACHE_HTTP_MESSAGE_BASICNAMEVALUEPAIR_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class Cloneable; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace org { namespace apache { namespace http { class NameValuePair; } } } }


#include <java/lang/Cloneable.hpp>
#include <java/lang/Object.hpp>
#include <java/lang/String.hpp>
#include <org/apache/http/NameValuePair.hpp>


namespace j2cpp {

namespace org { namespace apache { namespace http { namespace message {

	class BasicNameValuePair;
	class BasicNameValuePair
		: public object<BasicNameValuePair>
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

		explicit BasicNameValuePair(jobject jobj)
		: object<BasicNameValuePair>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<org::apache::http::NameValuePair>() const;
		operator local_ref<java::lang::Cloneable>() const;


		BasicNameValuePair(local_ref< java::lang::String > const&, local_ref< java::lang::String > const&);
		local_ref< java::lang::String > getName();
		local_ref< java::lang::String > getValue();
		local_ref< java::lang::String > toString();
		jboolean equals(local_ref< java::lang::Object >  const&);
		jint hashCode();
		local_ref< java::lang::Object > clone();
	}; //class BasicNameValuePair

} //namespace message
} //namespace http
} //namespace apache
} //namespace org

} //namespace j2cpp

#endif //J2CPP_ORG_APACHE_HTTP_MESSAGE_BASICNAMEVALUEPAIR_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_APACHE_HTTP_MESSAGE_BASICNAMEVALUEPAIR_HPP_IMPL
#define J2CPP_ORG_APACHE_HTTP_MESSAGE_BASICNAMEVALUEPAIR_HPP_IMPL

namespace j2cpp {



org::apache::http::message::BasicNameValuePair::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

org::apache::http::message::BasicNameValuePair::operator local_ref<org::apache::http::NameValuePair>() const
{
	return local_ref<org::apache::http::NameValuePair>(get_jobject());
}

org::apache::http::message::BasicNameValuePair::operator local_ref<java::lang::Cloneable>() const
{
	return local_ref<java::lang::Cloneable>(get_jobject());
}


org::apache::http::message::BasicNameValuePair::BasicNameValuePair(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1)
: object<org::apache::http::message::BasicNameValuePair>(
	call_new_object<
		org::apache::http::message::BasicNameValuePair::J2CPP_CLASS_NAME,
		org::apache::http::message::BasicNameValuePair::J2CPP_METHOD_NAME(0),
		org::apache::http::message::BasicNameValuePair::J2CPP_METHOD_SIGNATURE(0)
	>(a0, a1)
)
{
}


local_ref< java::lang::String > org::apache::http::message::BasicNameValuePair::getName()
{
	return call_method<
		org::apache::http::message::BasicNameValuePair::J2CPP_CLASS_NAME,
		org::apache::http::message::BasicNameValuePair::J2CPP_METHOD_NAME(1),
		org::apache::http::message::BasicNameValuePair::J2CPP_METHOD_SIGNATURE(1), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > org::apache::http::message::BasicNameValuePair::getValue()
{
	return call_method<
		org::apache::http::message::BasicNameValuePair::J2CPP_CLASS_NAME,
		org::apache::http::message::BasicNameValuePair::J2CPP_METHOD_NAME(2),
		org::apache::http::message::BasicNameValuePair::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > org::apache::http::message::BasicNameValuePair::toString()
{
	return call_method<
		org::apache::http::message::BasicNameValuePair::J2CPP_CLASS_NAME,
		org::apache::http::message::BasicNameValuePair::J2CPP_METHOD_NAME(3),
		org::apache::http::message::BasicNameValuePair::J2CPP_METHOD_SIGNATURE(3), 
		local_ref< java::lang::String >
	>(get_jobject());
}

jboolean org::apache::http::message::BasicNameValuePair::equals(local_ref< java::lang::Object > const &a0)
{
	return call_method<
		org::apache::http::message::BasicNameValuePair::J2CPP_CLASS_NAME,
		org::apache::http::message::BasicNameValuePair::J2CPP_METHOD_NAME(4),
		org::apache::http::message::BasicNameValuePair::J2CPP_METHOD_SIGNATURE(4), 
		jboolean
	>(get_jobject(), a0);
}

jint org::apache::http::message::BasicNameValuePair::hashCode()
{
	return call_method<
		org::apache::http::message::BasicNameValuePair::J2CPP_CLASS_NAME,
		org::apache::http::message::BasicNameValuePair::J2CPP_METHOD_NAME(5),
		org::apache::http::message::BasicNameValuePair::J2CPP_METHOD_SIGNATURE(5), 
		jint
	>(get_jobject());
}

local_ref< java::lang::Object > org::apache::http::message::BasicNameValuePair::clone()
{
	return call_method<
		org::apache::http::message::BasicNameValuePair::J2CPP_CLASS_NAME,
		org::apache::http::message::BasicNameValuePair::J2CPP_METHOD_NAME(6),
		org::apache::http::message::BasicNameValuePair::J2CPP_METHOD_SIGNATURE(6), 
		local_ref< java::lang::Object >
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(org::apache::http::message::BasicNameValuePair,"org/apache/http/message/BasicNameValuePair")
J2CPP_DEFINE_METHOD(org::apache::http::message::BasicNameValuePair,0,"<init>","(Ljava/lang/String;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(org::apache::http::message::BasicNameValuePair,1,"getName","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(org::apache::http::message::BasicNameValuePair,2,"getValue","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(org::apache::http::message::BasicNameValuePair,3,"toString","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(org::apache::http::message::BasicNameValuePair,4,"equals","(Ljava/lang/Object;)Z")
J2CPP_DEFINE_METHOD(org::apache::http::message::BasicNameValuePair,5,"hashCode","()I")
J2CPP_DEFINE_METHOD(org::apache::http::message::BasicNameValuePair,6,"clone","()Ljava/lang/Object;")

} //namespace j2cpp

#endif //J2CPP_ORG_APACHE_HTTP_MESSAGE_BASICNAMEVALUEPAIR_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
