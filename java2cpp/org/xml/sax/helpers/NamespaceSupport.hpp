/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: org.xml.sax.helpers.NamespaceSupport
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_XML_SAX_HELPERS_NAMESPACESUPPORT_HPP_DECL
#define J2CPP_ORG_XML_SAX_HELPERS_NAMESPACESUPPORT_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace util { class Enumeration; } } }


#include <java/lang/Object.hpp>
#include <java/lang/String.hpp>
#include <java/util/Enumeration.hpp>


namespace j2cpp {

namespace org { namespace xml { namespace sax { namespace helpers {

	class NamespaceSupport;
	class NamespaceSupport
		: public object<NamespaceSupport>
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
		J2CPP_DECLARE_FIELD(0)
		J2CPP_DECLARE_FIELD(1)

		explicit NamespaceSupport(jobject jobj)
		: object<NamespaceSupport>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;


		NamespaceSupport();
		void reset();
		void pushContext();
		void popContext();
		jboolean declarePrefix(local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&);
		local_ref< array< local_ref< java::lang::String >, 1> > processName(local_ref< java::lang::String >  const&, local_ref< array< local_ref< java::lang::String >, 1> >  const&, jboolean);
		local_ref< java::lang::String > getURI(local_ref< java::lang::String >  const&);
		local_ref< java::util::Enumeration > getPrefixes();
		local_ref< java::lang::String > getPrefix(local_ref< java::lang::String >  const&);
		local_ref< java::util::Enumeration > getPrefixes(local_ref< java::lang::String >  const&);
		local_ref< java::util::Enumeration > getDeclaredPrefixes();
		void setNamespaceDeclUris(jboolean);
		jboolean isNamespaceDeclUris();

		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(0), J2CPP_FIELD_SIGNATURE(0), local_ref< java::lang::String > > XMLNS;
		static static_field< J2CPP_CLASS_NAME, J2CPP_FIELD_NAME(1), J2CPP_FIELD_SIGNATURE(1), local_ref< java::lang::String > > NSDECL;
	}; //class NamespaceSupport

} //namespace helpers
} //namespace sax
} //namespace xml
} //namespace org

} //namespace j2cpp

#endif //J2CPP_ORG_XML_SAX_HELPERS_NAMESPACESUPPORT_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_XML_SAX_HELPERS_NAMESPACESUPPORT_HPP_IMPL
#define J2CPP_ORG_XML_SAX_HELPERS_NAMESPACESUPPORT_HPP_IMPL

namespace j2cpp {



org::xml::sax::helpers::NamespaceSupport::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}


org::xml::sax::helpers::NamespaceSupport::NamespaceSupport()
: object<org::xml::sax::helpers::NamespaceSupport>(
	call_new_object<
		org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_NAME(0),
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}


void org::xml::sax::helpers::NamespaceSupport::reset()
{
	return call_method<
		org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_NAME(1),
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_SIGNATURE(1), 
		void
	>(get_jobject());
}

void org::xml::sax::helpers::NamespaceSupport::pushContext()
{
	return call_method<
		org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_NAME(2),
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_SIGNATURE(2), 
		void
	>(get_jobject());
}

void org::xml::sax::helpers::NamespaceSupport::popContext()
{
	return call_method<
		org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_NAME(3),
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject());
}

jboolean org::xml::sax::helpers::NamespaceSupport::declarePrefix(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1)
{
	return call_method<
		org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_NAME(4),
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_SIGNATURE(4), 
		jboolean
	>(get_jobject(), a0, a1);
}

local_ref< array< local_ref< java::lang::String >, 1> > org::xml::sax::helpers::NamespaceSupport::processName(local_ref< java::lang::String > const &a0, local_ref< array< local_ref< java::lang::String >, 1> > const &a1, jboolean a2)
{
	return call_method<
		org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_NAME(5),
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_SIGNATURE(5), 
		local_ref< array< local_ref< java::lang::String >, 1> >
	>(get_jobject(), a0, a1, a2);
}

local_ref< java::lang::String > org::xml::sax::helpers::NamespaceSupport::getURI(local_ref< java::lang::String > const &a0)
{
	return call_method<
		org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_NAME(6),
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_SIGNATURE(6), 
		local_ref< java::lang::String >
	>(get_jobject(), a0);
}

local_ref< java::util::Enumeration > org::xml::sax::helpers::NamespaceSupport::getPrefixes()
{
	return call_method<
		org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_NAME(7),
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_SIGNATURE(7), 
		local_ref< java::util::Enumeration >
	>(get_jobject());
}

local_ref< java::lang::String > org::xml::sax::helpers::NamespaceSupport::getPrefix(local_ref< java::lang::String > const &a0)
{
	return call_method<
		org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_NAME(8),
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_SIGNATURE(8), 
		local_ref< java::lang::String >
	>(get_jobject(), a0);
}

local_ref< java::util::Enumeration > org::xml::sax::helpers::NamespaceSupport::getPrefixes(local_ref< java::lang::String > const &a0)
{
	return call_method<
		org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_NAME(9),
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_SIGNATURE(9), 
		local_ref< java::util::Enumeration >
	>(get_jobject(), a0);
}

local_ref< java::util::Enumeration > org::xml::sax::helpers::NamespaceSupport::getDeclaredPrefixes()
{
	return call_method<
		org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_NAME(10),
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_SIGNATURE(10), 
		local_ref< java::util::Enumeration >
	>(get_jobject());
}

void org::xml::sax::helpers::NamespaceSupport::setNamespaceDeclUris(jboolean a0)
{
	return call_method<
		org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_NAME(11),
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_SIGNATURE(11), 
		void
	>(get_jobject(), a0);
}

jboolean org::xml::sax::helpers::NamespaceSupport::isNamespaceDeclUris()
{
	return call_method<
		org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_NAME(12),
		org::xml::sax::helpers::NamespaceSupport::J2CPP_METHOD_SIGNATURE(12), 
		jboolean
	>(get_jobject());
}


static_field<
	org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
	org::xml::sax::helpers::NamespaceSupport::J2CPP_FIELD_NAME(0),
	org::xml::sax::helpers::NamespaceSupport::J2CPP_FIELD_SIGNATURE(0),
	local_ref< java::lang::String >
> org::xml::sax::helpers::NamespaceSupport::XMLNS;

static_field<
	org::xml::sax::helpers::NamespaceSupport::J2CPP_CLASS_NAME,
	org::xml::sax::helpers::NamespaceSupport::J2CPP_FIELD_NAME(1),
	org::xml::sax::helpers::NamespaceSupport::J2CPP_FIELD_SIGNATURE(1),
	local_ref< java::lang::String >
> org::xml::sax::helpers::NamespaceSupport::NSDECL;


J2CPP_DEFINE_CLASS(org::xml::sax::helpers::NamespaceSupport,"org/xml/sax/helpers/NamespaceSupport")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::NamespaceSupport,0,"<init>","()V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::NamespaceSupport,1,"reset","()V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::NamespaceSupport,2,"pushContext","()V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::NamespaceSupport,3,"popContext","()V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::NamespaceSupport,4,"declarePrefix","(Ljava/lang/String;Ljava/lang/String;)Z")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::NamespaceSupport,5,"processName","(Ljava/lang/String;[java.lang.StringZ)[java.lang.String")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::NamespaceSupport,6,"getURI","(Ljava/lang/String;)Ljava/lang/String;")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::NamespaceSupport,7,"getPrefixes","()Ljava/util/Enumeration;")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::NamespaceSupport,8,"getPrefix","(Ljava/lang/String;)Ljava/lang/String;")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::NamespaceSupport,9,"getPrefixes","(Ljava/lang/String;)Ljava/util/Enumeration;")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::NamespaceSupport,10,"getDeclaredPrefixes","()Ljava/util/Enumeration;")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::NamespaceSupport,11,"setNamespaceDeclUris","(Z)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::NamespaceSupport,12,"isNamespaceDeclUris","()Z")
J2CPP_DEFINE_FIELD(org::xml::sax::helpers::NamespaceSupport,0,"XMLNS","Ljava/lang/String;")
J2CPP_DEFINE_FIELD(org::xml::sax::helpers::NamespaceSupport,1,"NSDECL","Ljava/lang/String;")

} //namespace j2cpp

#endif //J2CPP_ORG_XML_SAX_HELPERS_NAMESPACESUPPORT_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
