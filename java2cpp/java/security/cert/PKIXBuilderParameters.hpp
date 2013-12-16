/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.security.cert.PKIXBuilderParameters
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_SECURITY_CERT_PKIXBUILDERPARAMETERS_HPP_DECL
#define J2CPP_JAVA_SECURITY_CERT_PKIXBUILDERPARAMETERS_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace security { class KeyStore; } } }
namespace j2cpp { namespace java { namespace security { namespace cert { class CertSelector; } } } }
namespace j2cpp { namespace java { namespace security { namespace cert { class PKIXParameters; } } } }
namespace j2cpp { namespace java { namespace util { class Set; } } }


#include <java/lang/String.hpp>
#include <java/security/KeyStore.hpp>
#include <java/security/cert/CertSelector.hpp>
#include <java/security/cert/PKIXParameters.hpp>
#include <java/util/Set.hpp>


namespace j2cpp {

namespace java { namespace security { namespace cert {

	class PKIXBuilderParameters;
	class PKIXBuilderParameters
		: public object<PKIXBuilderParameters>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)
		J2CPP_DECLARE_METHOD(4)

		explicit PKIXBuilderParameters(jobject jobj)
		: object<PKIXBuilderParameters>(jobj)
		{
		}

		operator local_ref<java::security::cert::PKIXParameters>() const;


		PKIXBuilderParameters(local_ref< java::util::Set > const&, local_ref< java::security::cert::CertSelector > const&);
		PKIXBuilderParameters(local_ref< java::security::KeyStore > const&, local_ref< java::security::cert::CertSelector > const&);
		jint getMaxPathLength();
		void setMaxPathLength(jint);
		local_ref< java::lang::String > toString();
	}; //class PKIXBuilderParameters

} //namespace cert
} //namespace security
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_SECURITY_CERT_PKIXBUILDERPARAMETERS_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_SECURITY_CERT_PKIXBUILDERPARAMETERS_HPP_IMPL
#define J2CPP_JAVA_SECURITY_CERT_PKIXBUILDERPARAMETERS_HPP_IMPL

namespace j2cpp {



java::security::cert::PKIXBuilderParameters::operator local_ref<java::security::cert::PKIXParameters>() const
{
	return local_ref<java::security::cert::PKIXParameters>(get_jobject());
}


java::security::cert::PKIXBuilderParameters::PKIXBuilderParameters(local_ref< java::util::Set > const &a0, local_ref< java::security::cert::CertSelector > const &a1)
: object<java::security::cert::PKIXBuilderParameters>(
	call_new_object<
		java::security::cert::PKIXBuilderParameters::J2CPP_CLASS_NAME,
		java::security::cert::PKIXBuilderParameters::J2CPP_METHOD_NAME(0),
		java::security::cert::PKIXBuilderParameters::J2CPP_METHOD_SIGNATURE(0)
	>(a0, a1)
)
{
}



java::security::cert::PKIXBuilderParameters::PKIXBuilderParameters(local_ref< java::security::KeyStore > const &a0, local_ref< java::security::cert::CertSelector > const &a1)
: object<java::security::cert::PKIXBuilderParameters>(
	call_new_object<
		java::security::cert::PKIXBuilderParameters::J2CPP_CLASS_NAME,
		java::security::cert::PKIXBuilderParameters::J2CPP_METHOD_NAME(1),
		java::security::cert::PKIXBuilderParameters::J2CPP_METHOD_SIGNATURE(1)
	>(a0, a1)
)
{
}


jint java::security::cert::PKIXBuilderParameters::getMaxPathLength()
{
	return call_method<
		java::security::cert::PKIXBuilderParameters::J2CPP_CLASS_NAME,
		java::security::cert::PKIXBuilderParameters::J2CPP_METHOD_NAME(2),
		java::security::cert::PKIXBuilderParameters::J2CPP_METHOD_SIGNATURE(2), 
		jint
	>(get_jobject());
}

void java::security::cert::PKIXBuilderParameters::setMaxPathLength(jint a0)
{
	return call_method<
		java::security::cert::PKIXBuilderParameters::J2CPP_CLASS_NAME,
		java::security::cert::PKIXBuilderParameters::J2CPP_METHOD_NAME(3),
		java::security::cert::PKIXBuilderParameters::J2CPP_METHOD_SIGNATURE(3), 
		void
	>(get_jobject(), a0);
}

local_ref< java::lang::String > java::security::cert::PKIXBuilderParameters::toString()
{
	return call_method<
		java::security::cert::PKIXBuilderParameters::J2CPP_CLASS_NAME,
		java::security::cert::PKIXBuilderParameters::J2CPP_METHOD_NAME(4),
		java::security::cert::PKIXBuilderParameters::J2CPP_METHOD_SIGNATURE(4), 
		local_ref< java::lang::String >
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(java::security::cert::PKIXBuilderParameters,"java/security/cert/PKIXBuilderParameters")
J2CPP_DEFINE_METHOD(java::security::cert::PKIXBuilderParameters,0,"<init>","(Ljava/util/Set;Ljava/security/cert/CertSelector;)V")
J2CPP_DEFINE_METHOD(java::security::cert::PKIXBuilderParameters,1,"<init>","(Ljava/security/KeyStore;Ljava/security/cert/CertSelector;)V")
J2CPP_DEFINE_METHOD(java::security::cert::PKIXBuilderParameters,2,"getMaxPathLength","()I")
J2CPP_DEFINE_METHOD(java::security::cert::PKIXBuilderParameters,3,"setMaxPathLength","(I)V")
J2CPP_DEFINE_METHOD(java::security::cert::PKIXBuilderParameters,4,"toString","()Ljava/lang/String;")

} //namespace j2cpp

#endif //J2CPP_JAVA_SECURITY_CERT_PKIXBUILDERPARAMETERS_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
