/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: org.xml.sax.helpers.XMLFilterImpl
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_XML_SAX_HELPERS_XMLFILTERIMPL_HPP_DECL
#define J2CPP_ORG_XML_SAX_HELPERS_XMLFILTERIMPL_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace org { namespace xml { namespace sax { class ContentHandler; } } } }
namespace j2cpp { namespace org { namespace xml { namespace sax { class EntityResolver; } } } }
namespace j2cpp { namespace org { namespace xml { namespace sax { class ErrorHandler; } } } }
namespace j2cpp { namespace org { namespace xml { namespace sax { class InputSource; } } } }
namespace j2cpp { namespace org { namespace xml { namespace sax { class SAXParseException; } } } }
namespace j2cpp { namespace org { namespace xml { namespace sax { class DTDHandler; } } } }
namespace j2cpp { namespace org { namespace xml { namespace sax { class Attributes; } } } }
namespace j2cpp { namespace org { namespace xml { namespace sax { class XMLFilter; } } } }
namespace j2cpp { namespace org { namespace xml { namespace sax { class Locator; } } } }
namespace j2cpp { namespace org { namespace xml { namespace sax { class XMLReader; } } } }


#include <java/lang/Object.hpp>
#include <java/lang/String.hpp>
#include <org/xml/sax/Attributes.hpp>
#include <org/xml/sax/ContentHandler.hpp>
#include <org/xml/sax/DTDHandler.hpp>
#include <org/xml/sax/EntityResolver.hpp>
#include <org/xml/sax/ErrorHandler.hpp>
#include <org/xml/sax/InputSource.hpp>
#include <org/xml/sax/Locator.hpp>
#include <org/xml/sax/SAXParseException.hpp>
#include <org/xml/sax/XMLFilter.hpp>
#include <org/xml/sax/XMLReader.hpp>


namespace j2cpp {

namespace org { namespace xml { namespace sax { namespace helpers {

	class XMLFilterImpl;
	class XMLFilterImpl
		: public object<XMLFilterImpl>
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
		J2CPP_DECLARE_METHOD(17)
		J2CPP_DECLARE_METHOD(18)
		J2CPP_DECLARE_METHOD(19)
		J2CPP_DECLARE_METHOD(20)
		J2CPP_DECLARE_METHOD(21)
		J2CPP_DECLARE_METHOD(22)
		J2CPP_DECLARE_METHOD(23)
		J2CPP_DECLARE_METHOD(24)
		J2CPP_DECLARE_METHOD(25)
		J2CPP_DECLARE_METHOD(26)
		J2CPP_DECLARE_METHOD(27)
		J2CPP_DECLARE_METHOD(28)
		J2CPP_DECLARE_METHOD(29)
		J2CPP_DECLARE_METHOD(30)
		J2CPP_DECLARE_METHOD(31)
		J2CPP_DECLARE_METHOD(32)
		J2CPP_DECLARE_METHOD(33)
		J2CPP_DECLARE_METHOD(34)

		explicit XMLFilterImpl(jobject jobj)
		: object<XMLFilterImpl>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<org::xml::sax::XMLFilter>() const;
		operator local_ref<org::xml::sax::EntityResolver>() const;
		operator local_ref<org::xml::sax::DTDHandler>() const;
		operator local_ref<org::xml::sax::ContentHandler>() const;
		operator local_ref<org::xml::sax::ErrorHandler>() const;


		XMLFilterImpl();
		XMLFilterImpl(local_ref< org::xml::sax::XMLReader > const&);
		void setParent(local_ref< org::xml::sax::XMLReader >  const&);
		local_ref< org::xml::sax::XMLReader > getParent();
		void setFeature(local_ref< java::lang::String >  const&, jboolean);
		jboolean getFeature(local_ref< java::lang::String >  const&);
		void setProperty(local_ref< java::lang::String >  const&, local_ref< java::lang::Object >  const&);
		local_ref< java::lang::Object > getProperty(local_ref< java::lang::String >  const&);
		void setEntityResolver(local_ref< org::xml::sax::EntityResolver >  const&);
		local_ref< org::xml::sax::EntityResolver > getEntityResolver();
		void setDTDHandler(local_ref< org::xml::sax::DTDHandler >  const&);
		local_ref< org::xml::sax::DTDHandler > getDTDHandler();
		void setContentHandler(local_ref< org::xml::sax::ContentHandler >  const&);
		local_ref< org::xml::sax::ContentHandler > getContentHandler();
		void setErrorHandler(local_ref< org::xml::sax::ErrorHandler >  const&);
		local_ref< org::xml::sax::ErrorHandler > getErrorHandler();
		void parse(local_ref< org::xml::sax::InputSource >  const&);
		void parse(local_ref< java::lang::String >  const&);
		local_ref< org::xml::sax::InputSource > resolveEntity(local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&);
		void notationDecl(local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&);
		void unparsedEntityDecl(local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&);
		void setDocumentLocator(local_ref< org::xml::sax::Locator >  const&);
		void startDocument();
		void endDocument();
		void startPrefixMapping(local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&);
		void endPrefixMapping(local_ref< java::lang::String >  const&);
		void startElement(local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&, local_ref< org::xml::sax::Attributes >  const&);
		void endElement(local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&);
		void characters(local_ref< array<jchar,1> >  const&, jint, jint);
		void ignorableWhitespace(local_ref< array<jchar,1> >  const&, jint, jint);
		void processingInstruction(local_ref< java::lang::String >  const&, local_ref< java::lang::String >  const&);
		void skippedEntity(local_ref< java::lang::String >  const&);
		void warning(local_ref< org::xml::sax::SAXParseException >  const&);
		void error(local_ref< org::xml::sax::SAXParseException >  const&);
		void fatalError(local_ref< org::xml::sax::SAXParseException >  const&);
	}; //class XMLFilterImpl

} //namespace helpers
} //namespace sax
} //namespace xml
} //namespace org

} //namespace j2cpp

#endif //J2CPP_ORG_XML_SAX_HELPERS_XMLFILTERIMPL_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ORG_XML_SAX_HELPERS_XMLFILTERIMPL_HPP_IMPL
#define J2CPP_ORG_XML_SAX_HELPERS_XMLFILTERIMPL_HPP_IMPL

namespace j2cpp {



org::xml::sax::helpers::XMLFilterImpl::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

org::xml::sax::helpers::XMLFilterImpl::operator local_ref<org::xml::sax::XMLFilter>() const
{
	return local_ref<org::xml::sax::XMLFilter>(get_jobject());
}

org::xml::sax::helpers::XMLFilterImpl::operator local_ref<org::xml::sax::EntityResolver>() const
{
	return local_ref<org::xml::sax::EntityResolver>(get_jobject());
}

org::xml::sax::helpers::XMLFilterImpl::operator local_ref<org::xml::sax::DTDHandler>() const
{
	return local_ref<org::xml::sax::DTDHandler>(get_jobject());
}

org::xml::sax::helpers::XMLFilterImpl::operator local_ref<org::xml::sax::ContentHandler>() const
{
	return local_ref<org::xml::sax::ContentHandler>(get_jobject());
}

org::xml::sax::helpers::XMLFilterImpl::operator local_ref<org::xml::sax::ErrorHandler>() const
{
	return local_ref<org::xml::sax::ErrorHandler>(get_jobject());
}


org::xml::sax::helpers::XMLFilterImpl::XMLFilterImpl()
: object<org::xml::sax::helpers::XMLFilterImpl>(
	call_new_object<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(0),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



org::xml::sax::helpers::XMLFilterImpl::XMLFilterImpl(local_ref< org::xml::sax::XMLReader > const &a0)
: object<org::xml::sax::helpers::XMLFilterImpl>(
	call_new_object<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(1),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}


void org::xml::sax::helpers::XMLFilterImpl::setParent(local_ref< org::xml::sax::XMLReader > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(2),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(2), 
		void
	>(get_jobject(), a0);
}

local_ref< org::xml::sax::XMLReader > org::xml::sax::helpers::XMLFilterImpl::getParent()
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(3),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(3), 
		local_ref< org::xml::sax::XMLReader >
	>(get_jobject());
}

void org::xml::sax::helpers::XMLFilterImpl::setFeature(local_ref< java::lang::String > const &a0, jboolean a1)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(4),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(4), 
		void
	>(get_jobject(), a0, a1);
}

jboolean org::xml::sax::helpers::XMLFilterImpl::getFeature(local_ref< java::lang::String > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(5),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(5), 
		jboolean
	>(get_jobject(), a0);
}

void org::xml::sax::helpers::XMLFilterImpl::setProperty(local_ref< java::lang::String > const &a0, local_ref< java::lang::Object > const &a1)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(6),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(6), 
		void
	>(get_jobject(), a0, a1);
}

local_ref< java::lang::Object > org::xml::sax::helpers::XMLFilterImpl::getProperty(local_ref< java::lang::String > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(7),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(7), 
		local_ref< java::lang::Object >
	>(get_jobject(), a0);
}

void org::xml::sax::helpers::XMLFilterImpl::setEntityResolver(local_ref< org::xml::sax::EntityResolver > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(8),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(8), 
		void
	>(get_jobject(), a0);
}

local_ref< org::xml::sax::EntityResolver > org::xml::sax::helpers::XMLFilterImpl::getEntityResolver()
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(9),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(9), 
		local_ref< org::xml::sax::EntityResolver >
	>(get_jobject());
}

void org::xml::sax::helpers::XMLFilterImpl::setDTDHandler(local_ref< org::xml::sax::DTDHandler > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(10),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(10), 
		void
	>(get_jobject(), a0);
}

local_ref< org::xml::sax::DTDHandler > org::xml::sax::helpers::XMLFilterImpl::getDTDHandler()
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(11),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(11), 
		local_ref< org::xml::sax::DTDHandler >
	>(get_jobject());
}

void org::xml::sax::helpers::XMLFilterImpl::setContentHandler(local_ref< org::xml::sax::ContentHandler > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(12),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(12), 
		void
	>(get_jobject(), a0);
}

local_ref< org::xml::sax::ContentHandler > org::xml::sax::helpers::XMLFilterImpl::getContentHandler()
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(13),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(13), 
		local_ref< org::xml::sax::ContentHandler >
	>(get_jobject());
}

void org::xml::sax::helpers::XMLFilterImpl::setErrorHandler(local_ref< org::xml::sax::ErrorHandler > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(14),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(14), 
		void
	>(get_jobject(), a0);
}

local_ref< org::xml::sax::ErrorHandler > org::xml::sax::helpers::XMLFilterImpl::getErrorHandler()
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(15),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(15), 
		local_ref< org::xml::sax::ErrorHandler >
	>(get_jobject());
}

void org::xml::sax::helpers::XMLFilterImpl::parse(local_ref< org::xml::sax::InputSource > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(16),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(16), 
		void
	>(get_jobject(), a0);
}

void org::xml::sax::helpers::XMLFilterImpl::parse(local_ref< java::lang::String > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(17),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(17), 
		void
	>(get_jobject(), a0);
}

local_ref< org::xml::sax::InputSource > org::xml::sax::helpers::XMLFilterImpl::resolveEntity(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(18),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(18), 
		local_ref< org::xml::sax::InputSource >
	>(get_jobject(), a0, a1);
}

void org::xml::sax::helpers::XMLFilterImpl::notationDecl(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1, local_ref< java::lang::String > const &a2)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(19),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(19), 
		void
	>(get_jobject(), a0, a1, a2);
}

void org::xml::sax::helpers::XMLFilterImpl::unparsedEntityDecl(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1, local_ref< java::lang::String > const &a2, local_ref< java::lang::String > const &a3)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(20),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(20), 
		void
	>(get_jobject(), a0, a1, a2, a3);
}

void org::xml::sax::helpers::XMLFilterImpl::setDocumentLocator(local_ref< org::xml::sax::Locator > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(21),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(21), 
		void
	>(get_jobject(), a0);
}

void org::xml::sax::helpers::XMLFilterImpl::startDocument()
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(22),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(22), 
		void
	>(get_jobject());
}

void org::xml::sax::helpers::XMLFilterImpl::endDocument()
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(23),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(23), 
		void
	>(get_jobject());
}

void org::xml::sax::helpers::XMLFilterImpl::startPrefixMapping(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(24),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(24), 
		void
	>(get_jobject(), a0, a1);
}

void org::xml::sax::helpers::XMLFilterImpl::endPrefixMapping(local_ref< java::lang::String > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(25),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(25), 
		void
	>(get_jobject(), a0);
}

void org::xml::sax::helpers::XMLFilterImpl::startElement(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1, local_ref< java::lang::String > const &a2, local_ref< org::xml::sax::Attributes > const &a3)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(26),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(26), 
		void
	>(get_jobject(), a0, a1, a2, a3);
}

void org::xml::sax::helpers::XMLFilterImpl::endElement(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1, local_ref< java::lang::String > const &a2)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(27),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(27), 
		void
	>(get_jobject(), a0, a1, a2);
}

void org::xml::sax::helpers::XMLFilterImpl::characters(local_ref< array<jchar,1> > const &a0, jint a1, jint a2)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(28),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(28), 
		void
	>(get_jobject(), a0, a1, a2);
}

void org::xml::sax::helpers::XMLFilterImpl::ignorableWhitespace(local_ref< array<jchar,1> > const &a0, jint a1, jint a2)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(29),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(29), 
		void
	>(get_jobject(), a0, a1, a2);
}

void org::xml::sax::helpers::XMLFilterImpl::processingInstruction(local_ref< java::lang::String > const &a0, local_ref< java::lang::String > const &a1)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(30),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(30), 
		void
	>(get_jobject(), a0, a1);
}

void org::xml::sax::helpers::XMLFilterImpl::skippedEntity(local_ref< java::lang::String > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(31),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(31), 
		void
	>(get_jobject(), a0);
}

void org::xml::sax::helpers::XMLFilterImpl::warning(local_ref< org::xml::sax::SAXParseException > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(32),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(32), 
		void
	>(get_jobject(), a0);
}

void org::xml::sax::helpers::XMLFilterImpl::error(local_ref< org::xml::sax::SAXParseException > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(33),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(33), 
		void
	>(get_jobject(), a0);
}

void org::xml::sax::helpers::XMLFilterImpl::fatalError(local_ref< org::xml::sax::SAXParseException > const &a0)
{
	return call_method<
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_CLASS_NAME,
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_NAME(34),
		org::xml::sax::helpers::XMLFilterImpl::J2CPP_METHOD_SIGNATURE(34), 
		void
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(org::xml::sax::helpers::XMLFilterImpl,"org/xml/sax/helpers/XMLFilterImpl")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,0,"<init>","()V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,1,"<init>","(Lorg/xml/sax/XMLReader;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,2,"setParent","(Lorg/xml/sax/XMLReader;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,3,"getParent","()Lorg/xml/sax/XMLReader;")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,4,"setFeature","(Ljava/lang/String;Z)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,5,"getFeature","(Ljava/lang/String;)Z")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,6,"setProperty","(Ljava/lang/String;Ljava/lang/Object;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,7,"getProperty","(Ljava/lang/String;)Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,8,"setEntityResolver","(Lorg/xml/sax/EntityResolver;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,9,"getEntityResolver","()Lorg/xml/sax/EntityResolver;")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,10,"setDTDHandler","(Lorg/xml/sax/DTDHandler;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,11,"getDTDHandler","()Lorg/xml/sax/DTDHandler;")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,12,"setContentHandler","(Lorg/xml/sax/ContentHandler;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,13,"getContentHandler","()Lorg/xml/sax/ContentHandler;")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,14,"setErrorHandler","(Lorg/xml/sax/ErrorHandler;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,15,"getErrorHandler","()Lorg/xml/sax/ErrorHandler;")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,16,"parse","(Lorg/xml/sax/InputSource;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,17,"parse","(Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,18,"resolveEntity","(Ljava/lang/String;Ljava/lang/String;)Lorg/xml/sax/InputSource;")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,19,"notationDecl","(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,20,"unparsedEntityDecl","(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,21,"setDocumentLocator","(Lorg/xml/sax/Locator;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,22,"startDocument","()V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,23,"endDocument","()V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,24,"startPrefixMapping","(Ljava/lang/String;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,25,"endPrefixMapping","(Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,26,"startElement","(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Lorg/xml/sax/Attributes;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,27,"endElement","(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,28,"characters","([CII)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,29,"ignorableWhitespace","([CII)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,30,"processingInstruction","(Ljava/lang/String;Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,31,"skippedEntity","(Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,32,"warning","(Lorg/xml/sax/SAXParseException;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,33,"error","(Lorg/xml/sax/SAXParseException;)V")
J2CPP_DEFINE_METHOD(org::xml::sax::helpers::XMLFilterImpl,34,"fatalError","(Lorg/xml/sax/SAXParseException;)V")

} //namespace j2cpp

#endif //J2CPP_ORG_XML_SAX_HELPERS_XMLFILTERIMPL_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
