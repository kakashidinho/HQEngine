/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: javax.security.cert.X509Certificate
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVAX_SECURITY_CERT_X509CERTIFICATE_HPP_DECL
#define J2CPP_JAVAX_SECURITY_CERT_X509CERTIFICATE_HPP_DECL


namespace j2cpp { namespace javax { namespace security { namespace cert { class Certificate; } } } }
namespace j2cpp { namespace java { namespace io { class InputStream; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace math { class BigInteger; } } }
namespace j2cpp { namespace java { namespace security { class Principal; } } }
namespace j2cpp { namespace java { namespace util { class Date; } } }


#include <java/io/InputStream.hpp>
#include <java/lang/String.hpp>
#include <java/math/BigInteger.hpp>
#include <java/security/Principal.hpp>
#include <java/util/Date.hpp>
#include <javax/security/cert/Certificate.hpp>


namespace j2cpp {

namespace javax { namespace security { namespace cert {

	class X509Certificate;
	class X509Certificate
		: public object<X509Certificate>
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

		explicit X509Certificate(jobject jobj)
		: object<X509Certificate>(jobj)
		{
		}

		operator local_ref<javax::security::cert::Certificate>() const;


		X509Certificate();
		static local_ref< javax::security::cert::X509Certificate > getInstance(local_ref< java::io::InputStream >  const&);
		static local_ref< javax::security::cert::X509Certificate > getInstance(local_ref< array<jbyte,1> >  const&);
		void checkValidity();
		void checkValidity(local_ref< java::util::Date >  const&);
		jint getVersion();
		local_ref< java::math::BigInteger > getSerialNumber();
		local_ref< java::security::Principal > getIssuerDN();
		local_ref< java::security::Principal > getSubjectDN();
		local_ref< java::util::Date > getNotBefore();
		local_ref< java::util::Date > getNotAfter();
		local_ref< java::lang::String > getSigAlgName();
		local_ref< java::lang::String > getSigAlgOID();
		local_ref< array<jbyte,1> > getSigAlgParams();
	}; //class X509Certificate

} //namespace cert
} //namespace security
} //namespace javax

} //namespace j2cpp

#endif //J2CPP_JAVAX_SECURITY_CERT_X509CERTIFICATE_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVAX_SECURITY_CERT_X509CERTIFICATE_HPP_IMPL
#define J2CPP_JAVAX_SECURITY_CERT_X509CERTIFICATE_HPP_IMPL

namespace j2cpp {



javax::security::cert::X509Certificate::operator local_ref<javax::security::cert::Certificate>() const
{
	return local_ref<javax::security::cert::Certificate>(get_jobject());
}


javax::security::cert::X509Certificate::X509Certificate()
: object<javax::security::cert::X509Certificate>(
	call_new_object<
		javax::security::cert::X509Certificate::J2CPP_CLASS_NAME,
		javax::security::cert::X509Certificate::J2CPP_METHOD_NAME(0),
		javax::security::cert::X509Certificate::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}


local_ref< javax::security::cert::X509Certificate > javax::security::cert::X509Certificate::getInstance(local_ref< java::io::InputStream > const &a0)
{
	return call_static_method<
		javax::security::cert::X509Certificate::J2CPP_CLASS_NAME,
		javax::security::cert::X509Certificate::J2CPP_METHOD_NAME(1),
		javax::security::cert::X509Certificate::J2CPP_METHOD_SIGNATURE(1), 
		local_ref< javax::security::cert::X509Certificate >
	>(a0);
}

local_ref< javax::security::cert::X509Certificate > javax::security::cert::X509Certificate::getInstance(local_ref< array<jbyte,1> > const &a0)
{
	return call_static_method<
		javax::security::cert::X509Certificate::J2CPP_CLASS_NAME,
		javax::security::cert::X509Certificate::J2CPP_METHOD_NAME(2),
		javax::security::cert::X509Certificate::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< javax::security::cert::X509Certificate >
	>(a0);
}

void javax::security::cert::X509Certificate::checkValidity()
{
	return call_method<
		javax::security::cert::X509Certificate::J2CPP_CLASS_NAME,
		javax::security::cert::X509Certificate::J2CPP_METHOD_NAME(3),
		javax::security::cert::X509Certificate::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject());
}

void javax::security::cert::X509Certificate::checkValidity(local_ref< java::util::Date > const &a0)
{
	return call_method<
		javax::security::cert::X509Certificate::J2CPP_CLASS_NAME,
		javax::security::cert::X509Certificate::J2CPP_METHOD_NAME(4),
		javax::security::cert::X509Certificate::J2CPP_METHOD_SIGNATURE(4), 
		void
	>(get_jobject(), a0);
}

jint javax::security::cert::X509Certificate::getVersion()
{
	return call_method<
		javax::security::cert::X509Certificate::J2CPP_CLASS_NAME,
		javax::security::cert::X509Certificate::J2CPP_METHOD_NAME(5),
		javax::security::cert::X509Certificate::J2CPP_METHOD_SIGNATURE(5), 
		jint
	>(get_jobject());
}

local_ref< java::math::BigInteger > javax::security::cert::X509Certificate::getSerialNumber()
{
	return call_method<
		javax::security::cert::X509Certificate::J2CPP_CLASS_NAME,
		javax::security::cert::X509Certificate::J2CPP_METHOD_NAME(6),
		javax::security::cert::X509Certificate::J2CPP_METHOD_SIGNATURE(6), 
		local_ref< java::math::BigInteger >
	>(get_jobject());
}

local_ref< java::security::Principal > javax::security::cert::X509Certificate::getIssuerDN()
{
	return call_method<
		javax::security::cert::X509Certificate::J2CPP_CLASS_NAME,
		javax::security::cert::X509Certificate::J2CPP_METHOD_NAME(7),
		javax::security::cert::X509Certificate::J2CPP_METHOD_SIGNATURE(7), 
		local_ref< java::security::Principal >
	>(get_jobject());
}

local_ref< java::security::Principal > javax::security::cert::X509Certificate::getSubjectDN()
{
	return call_method<
		javax::security::cert::X509Certificate::J2CPP_CLASS_NAME,
		javax::security::cert::X509Certificate::J2CPP_METHOD_NAME(8),
		javax::security::cert::X509Certificate::J2CPP_METHOD_SIGNATURE(8), 
		local_ref< java::security::Principal >
	>(get_jobject());
}

local_ref< java::util::Date > javax::security::cert::X509Certificate::getNotBefore()
{
	return call_method<
		javax::security::cert::X509Certificate::J2CPP_CLASS_NAME,
		javax::security::cert::X509Certificate::J2CPP_METHOD_NAME(9),
		javax::security::cert::X509Certificate::J2CPP_METHOD_SIGNATURE(9), 
		local_ref< java::util::Date >
	>(get_jobject());
}

local_ref< java::util::Date > javax::security::cert::X509Certificate::getNotAfter()
{
	return call_method<
		javax::security::cert::X509Certificate::J2CPP_CLASS_NAME,
		javax::security::cert::X509Certificate::J2CPP_METHOD_NAME(10),
		javax::security::cert::X509Certificate::J2CPP_METHOD_SIGNATURE(10), 
		local_ref< java::util::Date >
	>(get_jobject());
}

local_ref< java::lang::String > javax::security::cert::X509Certificate::getSigAlgName()
{
	return call_method<
		javax::security::cert::X509Certificate::J2CPP_CLASS_NAME,
		javax::security::cert::X509Certificate::J2CPP_METHOD_NAME(11),
		javax::security::cert::X509Certificate::J2CPP_METHOD_SIGNATURE(11), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< java::lang::String > javax::security::cert::X509Certificate::getSigAlgOID()
{
	return call_method<
		javax::security::cert::X509Certificate::J2CPP_CLASS_NAME,
		javax::security::cert::X509Certificate::J2CPP_METHOD_NAME(12),
		javax::security::cert::X509Certificate::J2CPP_METHOD_SIGNATURE(12), 
		local_ref< java::lang::String >
	>(get_jobject());
}

local_ref< array<jbyte,1> > javax::security::cert::X509Certificate::getSigAlgParams()
{
	return call_method<
		javax::security::cert::X509Certificate::J2CPP_CLASS_NAME,
		javax::security::cert::X509Certificate::J2CPP_METHOD_NAME(13),
		javax::security::cert::X509Certificate::J2CPP_METHOD_SIGNATURE(13), 
		local_ref< array<jbyte,1> >
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(javax::security::cert::X509Certificate,"javax/security/cert/X509Certificate")
J2CPP_DEFINE_METHOD(javax::security::cert::X509Certificate,0,"<init>","()V")
J2CPP_DEFINE_METHOD(javax::security::cert::X509Certificate,1,"getInstance","(Ljava/io/InputStream;)Ljavax/security/cert/X509Certificate;")
J2CPP_DEFINE_METHOD(javax::security::cert::X509Certificate,2,"getInstance","([B)Ljavax/security/cert/X509Certificate;")
J2CPP_DEFINE_METHOD(javax::security::cert::X509Certificate,3,"checkValidity","()V")
J2CPP_DEFINE_METHOD(javax::security::cert::X509Certificate,4,"checkValidity","(Ljava/util/Date;)V")
J2CPP_DEFINE_METHOD(javax::security::cert::X509Certificate,5,"getVersion","()I")
J2CPP_DEFINE_METHOD(javax::security::cert::X509Certificate,6,"getSerialNumber","()Ljava/math/BigInteger;")
J2CPP_DEFINE_METHOD(javax::security::cert::X509Certificate,7,"getIssuerDN","()Ljava/security/Principal;")
J2CPP_DEFINE_METHOD(javax::security::cert::X509Certificate,8,"getSubjectDN","()Ljava/security/Principal;")
J2CPP_DEFINE_METHOD(javax::security::cert::X509Certificate,9,"getNotBefore","()Ljava/util/Date;")
J2CPP_DEFINE_METHOD(javax::security::cert::X509Certificate,10,"getNotAfter","()Ljava/util/Date;")
J2CPP_DEFINE_METHOD(javax::security::cert::X509Certificate,11,"getSigAlgName","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(javax::security::cert::X509Certificate,12,"getSigAlgOID","()Ljava/lang/String;")
J2CPP_DEFINE_METHOD(javax::security::cert::X509Certificate,13,"getSigAlgParams","()[B")

} //namespace j2cpp

#endif //J2CPP_JAVAX_SECURITY_CERT_X509CERTIFICATE_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
