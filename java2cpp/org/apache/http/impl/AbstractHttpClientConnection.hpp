/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: org.apache.http.impl.AbstractHttpClientConnection
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_APACHE_HTTP_IMPL_ABSTRACTHTTPCLIENTCONNECTION_HPP_DECL
#define J2CPP_ORG_APACHE_HTTP_IMPL_ABSTRACTHTTPCLIENTCONNECTION_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace org { namespace apache { namespace http { class HttpConnectionMetrics; } } } }
namespace j2cpp { namespace org { namespace apache { namespace http { class HttpRequest; } } } }
namespace j2cpp { namespace org { namespace apache { namespace http { class HttpClientConnection; } } } }
namespace j2cpp { namespace org { namespace apache { namespace http { class HttpEntityEnclosingRequest; } } } }
namespace j2cpp { namespace org { namespace apache { namespace http { class HttpResponse; } } } }


#include <java/lang/Object.hpp>
#include <org/apache/http/HttpClientConnection.hpp>
#include <org/apache/http/HttpConnectionMetrics.hpp>
#include <org/apache/http/HttpEntityEnclosingRequest.hpp>
#include <org/apache/http/HttpRequest.hpp>
#include <org/apache/http/HttpResponse.hpp>


namespace j2cpp {

namespace org { namespace apache { namespace http { namespace impl {

	class AbstractHttpClientConnection;
	class AbstractHttpClientConnection
		: public object<AbstractHttpClientConnection>
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

		explicit AbstractHttpClientConnection(jobject jobj)
		: object<AbstractHttpClientConnection>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<org::apache::http::HttpClientConnection>() const;


		AbstractHttpClientConnection();
		jboolean isResponseAvailable(jint);
		void sendRequestHeader(local_ref< org::apache::http::HttpRequest >  const&);
		void sendRequestEntity(local_ref< org::apache::http::HttpEntityEnclosingRequest >  const&);
		void flush();
		local_ref< org::apache::http::HttpResponse > receiveResponseHeader();
		void receiveResponseEntity(local_ref< org::apache::http::HttpResponse >  const&);
		jboolean isStale();
		local_ref< org::apache::http::HttpConnectionMetrics > getMetrics();
	}; //class AbstractHttpClientConnection

} //namespace impl
} //namespace http
} //namespace apache
} //namespace org

} //namespace j2cpp

#endif //J2CPP_ORG_APACHE_HTTP_IMPL_ABSTRACTHTTPCLIENTCONNECTION_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_APACHE_HTTP_IMPL_ABSTRACTHTTPCLIENTCONNECTION_HPP_IMPL
#define J2CPP_ORG_APACHE_HTTP_IMPL_ABSTRACTHTTPCLIENTCONNECTION_HPP_IMPL

namespace j2cpp {



org::apache::http::impl::AbstractHttpClientConnection::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

org::apache::http::impl::AbstractHttpClientConnection::operator local_ref<org::apache::http::HttpClientConnection>() const
{
	return local_ref<org::apache::http::HttpClientConnection>(get_jobject());
}


org::apache::http::impl::AbstractHttpClientConnection::AbstractHttpClientConnection()
: object<org::apache::http::impl::AbstractHttpClientConnection>(
	call_new_object<
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_CLASS_NAME,
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_NAME(0),
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}









jboolean org::apache::http::impl::AbstractHttpClientConnection::isResponseAvailable(jint a0)
{
	return call_method<
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_CLASS_NAME,
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_NAME(8),
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_SIGNATURE(8), 
		jboolean
	>(get_jobject(), a0);
}

void org::apache::http::impl::AbstractHttpClientConnection::sendRequestHeader(local_ref< org::apache::http::HttpRequest > const &a0)
{
	return call_method<
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_CLASS_NAME,
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_NAME(9),
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_SIGNATURE(9), 
		void
	>(get_jobject(), a0);
}

void org::apache::http::impl::AbstractHttpClientConnection::sendRequestEntity(local_ref< org::apache::http::HttpEntityEnclosingRequest > const &a0)
{
	return call_method<
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_CLASS_NAME,
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_NAME(10),
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_SIGNATURE(10), 
		void
	>(get_jobject(), a0);
}


void org::apache::http::impl::AbstractHttpClientConnection::flush()
{
	return call_method<
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_CLASS_NAME,
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_NAME(12),
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_SIGNATURE(12), 
		void
	>(get_jobject());
}

local_ref< org::apache::http::HttpResponse > org::apache::http::impl::AbstractHttpClientConnection::receiveResponseHeader()
{
	return call_method<
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_CLASS_NAME,
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_NAME(13),
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_SIGNATURE(13), 
		local_ref< org::apache::http::HttpResponse >
	>(get_jobject());
}

void org::apache::http::impl::AbstractHttpClientConnection::receiveResponseEntity(local_ref< org::apache::http::HttpResponse > const &a0)
{
	return call_method<
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_CLASS_NAME,
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_NAME(14),
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_SIGNATURE(14), 
		void
	>(get_jobject(), a0);
}

jboolean org::apache::http::impl::AbstractHttpClientConnection::isStale()
{
	return call_method<
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_CLASS_NAME,
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_NAME(15),
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_SIGNATURE(15), 
		jboolean
	>(get_jobject());
}

local_ref< org::apache::http::HttpConnectionMetrics > org::apache::http::impl::AbstractHttpClientConnection::getMetrics()
{
	return call_method<
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_CLASS_NAME,
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_NAME(16),
		org::apache::http::impl::AbstractHttpClientConnection::J2CPP_METHOD_SIGNATURE(16), 
		local_ref< org::apache::http::HttpConnectionMetrics >
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(org::apache::http::impl::AbstractHttpClientConnection,"org/apache/http/impl/AbstractHttpClientConnection")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,0,"<init>","()V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,1,"assertOpen","()V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,2,"createEntityDeserializer","()Lorg/apache/http/impl/entity/EntityDeserializer;")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,3,"createEntitySerializer","()Lorg/apache/http/impl/entity/EntitySerializer;")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,4,"createHttpResponseFactory","()Lorg/apache/http/HttpResponseFactory;")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,5,"createResponseParser","(Lorg/apache/http/io/SessionInputBuffer;Lorg/apache/http/HttpResponseFactory;Lorg/apache/http/params/HttpParams;)Lorg/apache/http/io/HttpMessageParser;")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,6,"createRequestWriter","(Lorg/apache/http/io/SessionOutputBuffer;Lorg/apache/http/params/HttpParams;)Lorg/apache/http/io/HttpMessageWriter;")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,7,"init","(Lorg/apache/http/io/SessionInputBuffer;Lorg/apache/http/io/SessionOutputBuffer;Lorg/apache/http/params/HttpParams;)V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,8,"isResponseAvailable","(I)Z")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,9,"sendRequestHeader","(Lorg/apache/http/HttpRequest;)V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,10,"sendRequestEntity","(Lorg/apache/http/HttpEntityEnclosingRequest;)V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,11,"doFlush","()V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,12,"flush","()V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,13,"receiveResponseHeader","()Lorg/apache/http/HttpResponse;")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,14,"receiveResponseEntity","(Lorg/apache/http/HttpResponse;)V")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,15,"isStale","()Z")
J2CPP_DEFINE_METHOD(org::apache::http::impl::AbstractHttpClientConnection,16,"getMetrics","()Lorg/apache/http/HttpConnectionMetrics;")

} //namespace j2cpp

#endif //J2CPP_ORG_APACHE_HTTP_IMPL_ABSTRACTHTTPCLIENTCONNECTION_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
