/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: java.security.cert.PKIXCertPathValidatorResult
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_SECURITY_CERT_PKIXCERTPATHVALIDATORRESULT_HPP_DECL
#define J2CPP_JAVA_SECURITY_CERT_PKIXCERTPATHVALIDATORRESULT_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace security { class PublicKey; } } }
namespace j2cpp { namespace java { namespace security { namespace cert { class CertPathValidatorResult; } } } }
namespace j2cpp { namespace java { namespace security { namespace cert { class PolicyNode; } } } }
namespace j2cpp { namespace java { namespace security { namespace cert { class TrustAnchor; } } } }


#include <java/lang/Object.hpp>
#include <java/lang/String.hpp>
#include <java/security/PublicKey.hpp>
#include <java/security/cert/CertPathValidatorResult.hpp>
#include <java/security/cert/PolicyNode.hpp>
#include <java/security/cert/TrustAnchor.hpp>


namespace j2cpp {

namespace java { namespace security { namespace cert {

	class PKIXCertPathValidatorResult;
	class PKIXCertPathValidatorResult
		: public object<PKIXCertPathValidatorResult>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)
		J2CPP_DECLARE_METHOD(2)
		J2CPP_DECLARE_METHOD(3)
		J2CPP_DECLARE_METHOD(4)
		J2CPP_DECLARE_METHOD(5)

		explicit PKIXCertPathValidatorResult(jobject jobj)
		: object<PKIXCertPathValidatorResult>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;
		operator local_ref<java::security::cert::CertPathValidatorResult>() const;


		PKIXCertPathValidatorResult(local_ref< java::security::cert::TrustAnchor > const&, local_ref< java::security::cert::PolicyNode > const&, local_ref< java::security::PublicKey > const&);
		local_ref< java::security::cert::PolicyNode > getPolicyTree();
		local_ref< java::security::PublicKey > getPublicKey();
		local_ref< java::security::cert::TrustAnchor > getTrustAnchor();
		local_ref< java::lang::Object > clone();
		local_ref< java::lang::String > toString();
	}; //class PKIXCertPathValidatorResult

} //namespace cert
} //namespace security
} //namespace java

} //namespace j2cpp

#endif //J2CPP_JAVA_SECURITY_CERT_PKIXCERTPATHVALIDATORRESULT_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVA_SECURITY_CERT_PKIXCERTPATHVALIDATORRESULT_HPP_IMPL
#define J2CPP_JAVA_SECURITY_CERT_PKIXCERTPATHVALIDATORRESULT_HPP_IMPL

namespace j2cpp {



java::security::cert::PKIXCertPathValidatorResult::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

java::security::cert::PKIXCertPathValidatorResult::operator local_ref<java::security::cert::CertPathValidatorResult>() const
{
	return local_ref<java::security::cert::CertPathValidatorResult>(get_jobject());
}


java::security::cert::PKIXCertPathValidatorResult::PKIXCertPathValidatorResult(local_ref< java::security::cert::TrustAnchor > const &a0, local_ref< java::security::cert::PolicyNode > const &a1, local_ref< java::security::PublicKey > const &a2)
: object<java::security::cert::PKIXCertPathValidatorResult>(
	call_new_object<
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_CLASS_NAME,
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_METHOD_NAME(0),
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_METHOD_SIGNATURE(0)
	>(a0, a1, a2)
)
{
}


local_ref< java::security::cert::PolicyNode > java::security::cert::PKIXCertPathValidatorResult::getPolicyTree()
{
	return call_method<
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_CLASS_NAME,
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_METHOD_NAME(1),
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_METHOD_SIGNATURE(1), 
		local_ref< java::security::cert::PolicyNode >
	>(get_jobject());
}

local_ref< java::security::PublicKey > java::security::cert::PKIXCertPathValidatorResult::getPublicKey()
{
	return call_method<
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_CLASS_NAME,
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_METHOD_NAME(2),
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_METHOD_SIGNATURE(2), 
		local_ref< java::security::PublicKey >
	>(get_jobject());
}

local_ref< java::security::cert::TrustAnchor > java::security::cert::PKIXCertPathValidatorResult::getTrustAnchor()
{
	return call_method<
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_CLASS_NAME,
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_METHOD_NAME(3),
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_METHOD_SIGNATURE(3), 
		local_ref< java::security::cert::TrustAnchor >
	>(get_jobject());
}

local_ref< java::lang::Object > java::security::cert::PKIXCertPathValidatorResult::clone()
{
	return call_method<
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_CLASS_NAME,
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_METHOD_NAME(4),
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_METHOD_SIGNATURE(4), 
		local_ref< java::lang::Object >
	>(get_jobject());
}

local_ref< java::lang::String > java::security::cert::PKIXCertPathValidatorResult::toString()
{
	return call_method<
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_CLASS_NAME,
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_METHOD_NAME(5),
		java::security::cert::PKIXCertPathValidatorResult::J2CPP_METHOD_SIGNATURE(5), 
		local_ref< java::lang::String >
	>(get_jobject());
}


J2CPP_DEFINE_CLASS(java::security::cert::PKIXCertPathValidatorResult,"java/security/cert/PKIXCertPathValidatorResult")
J2CPP_DEFINE_METHOD(java::security::cert::PKIXCertPathValidatorResult,0,"<init>","(Ljava/security/cert/TrustAnchor;Ljava/security/cert/PolicyNode;Ljava/security/PublicKey;)V")
J2CPP_DEFINE_METHOD(java::security::cert::PKIXCertPathValidatorResult,1,"getPolicyTree","()Ljava/security/cert/PolicyNode;")
J2CPP_DEFINE_METHOD(java::security::cert::PKIXCertPathValidatorResult,2,"getPublicKey","()Ljava/security/PublicKey;")
J2CPP_DEFINE_METHOD(java::security::cert::PKIXCertPathValidatorResult,3,"getTrustAnchor","()Ljava/security/cert/TrustAnchor;")
J2CPP_DEFINE_METHOD(java::security::cert::PKIXCertPathValidatorResult,4,"clone","()Ljava/lang/Object;")
J2CPP_DEFINE_METHOD(java::security::cert::PKIXCertPathValidatorResult,5,"toString","()Ljava/lang/String;")

} //namespace j2cpp

#endif //J2CPP_JAVA_SECURITY_CERT_PKIXCERTPATHVALIDATORRESULT_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
