/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: org.apache.http.entity.EntityTemplate
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_APACHE_HTTP_ENTITY_ENTITYTEMPLATE_HPP_DECL
#define J2CPP_ORG_APACHE_HTTP_ENTITY_ENTITYTEMPLATE_HPP_DECL


namespace j2cpp { namespace java { namespace io { class InputStream; } } }
namespace j2cpp { namespace java { namespace io { class OutputStream; } } }
namespace j2cpp { namespace org { namespace apache { namespace http { namespace entity { class AbstractHttpEntity; } } } } }
namespace j2cpp { namespace org { namespace apache { namespace http { namespace entity { class ContentProducer; } } } } }


#include <java/io/InputStream.hpp>
#include <java/io/OutputStream.hpp>
#include <org/apache/http/entity/AbstractHttpEntity.hpp>
#include <org/apache/http/entity/ContentProducer.hpp>


namespace j2cpp {

namespace org { namespace apache { namespace http { namespace entity {

	class EntityTemplate;
	class EntityTemplate
		: public object<EntityTemplate>
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

		explicit EntityTemplate(jobject jobj)
		: object<EntityTemplate>(jobj)
		{
		}

		operator local_ref<org::apache::http::entity::AbstractHttpEntity>() const;


		EntityTemplate(local_ref< org::apache::http::entity::ContentProducer > const&);
		jlong getContentLength();
		local_ref< java::io::InputStream > getContent();
		jboolean isRepeatable();
		void writeTo(local_ref< java::io::OutputStream >  const&);
		jboolean isStreaming();
		void consumeContent();
	}; //class EntityTemplate

} //namespace entity
} //namespace http
} //namespace apache
} //namespace org

} //namespace j2cpp

#endif //J2CPP_ORG_APACHE_HTTP_ENTITY_ENTITYTEMPLATE_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_APACHE_HTTP_ENTITY_ENTITYTEMPLATE_HPP_IMPL
#define J2CPP_ORG_APACHE_HTTP_ENTITY_ENTITYTEMPLATE_HPP_IMPL

namespace j2cpp {



org::apache::http::entity::EntityTemplate::operator local_ref<org::apache::http::entity::AbstractHttpEntity>() const
{
	return local_ref<org::apache::http::entity::AbstractHttpEntity>(get_jobject());
}


org::apache::http::entity::EntityTemplate::EntityTemplate(local_ref< org::apache::http::entity::ContentProducer > const &a0)
: object<org::apache::http::entity::EntityTemplate>(
	call_new_object<
		org::apache::http::entity::EntityTemplate::J2CPP_CLASS_NAME,
		org::apache::http::entity::EntityTemplate::J2CPP_METHOD_NAME(0),
		org::apache::http::entity::EntityTemplate::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}


jlong org::apache::http::entity::EntityTemplate::getContentLength()
{
	return call_method<
		org::apache::http::entity::EntityTemplate::J2CPP_CLASS_NAME,
		org::apache::http::entity::EntityTemplate::J2CPP_METHOD_NAME(1),
		org::apache::http::entity::EntityTemplate::J2CPP_METHOD_SIGNATURE(1), 
		jlong
	>(get_jobject());
}

local_ref< java::io::InputStream > org::apache::http::entity::EntityTemplate::getContent()
{
	return call_method<
		org::apache::http::entity::EntityTemplate::J2CPP_CLASS_NAME,
		org::apache::http::entity::EntityTemplate::J2CPP_METHOD_NAME(2),
		org::apache::http::entity::EntityTemplate::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< java::io::InputStream >
	>(get_jobject());
}

jboolean org::apache::http::entity::EntityTemplate::isRepeatable()
{
	return call_method<
		org::apache::http::entity::EntityTemplate::J2CPP_CLASS_NAME,
		org::apache::http::entity::EntityTemplate::J2CPP_METHOD_NAME(3),
		org::apache::http::entity::EntityTemplate::J2CPP_METHOD_SIGNATURE(3), 
		jboolean
	>(get_jobject());
}

void org::apache::http::entity::EntityTemplate::writeTo(local_ref< java::io::OutputStream > const &a0)
{
	return call_method<
		org::apache::http::entity::EntityTemplate::J2CPP_CLASS_NAME,
		org::apache::http::entity::EntityTemplate::J2CPP_METHOD_NAME(4),
		org::apache::http::entity::EntityTemplate::J2CPP_METHOD_SIGNATURE(4), 
		void
	>(get_jobject(), a0);
}

jboolean org::apache::http::entity::EntityTemplate::isStreaming()
{
	return call_method<
		org::apache::http::entity::EntityTemplate::J2CPP_CLASS_NAME,
		org::apache::http::entity::EntityTemplate::J2CPP_METHOD_NAME(5),
		org::apache::http::entity::EntityTemplate::J2CPP_METHOD_SIGNATURE(5), 
		jboolean
	>(get_jobject());
}

void org::apache::http::entity::EntityTemplate::consumeContent()
{
	return call_method<
		org::apache::http::entity::EntityTemplate::J2CPP_CLASS_NAME,
		org::apache::http::entity::EntityTemplate::J2CPP_METHOD_NAME(6),
		org::apache::http::entity::EntityTemplate::J2CPP_METHOD_SIGNATURE(6), 
		void
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(org::apache::http::entity::EntityTemplate,"org/apache/http/entity/EntityTemplate")
J2CPP_DEFINE_METHOD(org::apache::http::entity::EntityTemplate,0,"<init>","(Lorg/apache/http/entity/ContentProducer;)V")
J2CPP_DEFINE_METHOD(org::apache::http::entity::EntityTemplate,1,"getContentLength","()J")
J2CPP_DEFINE_METHOD(org::apache::http::entity::EntityTemplate,2,"getContent","()Ljava/io/InputStream;")
J2CPP_DEFINE_METHOD(org::apache::http::entity::EntityTemplate,3,"isRepeatable","()Z")
J2CPP_DEFINE_METHOD(org::apache::http::entity::EntityTemplate,4,"writeTo","(Ljava/io/OutputStream;)V")
J2CPP_DEFINE_METHOD(org::apache::http::entity::EntityTemplate,5,"isStreaming","()Z")
J2CPP_DEFINE_METHOD(org::apache::http::entity::EntityTemplate,6,"consumeContent","()V")

} //namespace j2cpp

#endif //J2CPP_ORG_APACHE_HTTP_ENTITY_ENTITYTEMPLATE_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
