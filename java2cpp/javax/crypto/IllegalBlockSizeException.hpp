/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: javax.crypto.IllegalBlockSizeException
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVAX_CRYPTO_ILLEGALBLOCKSIZEEXCEPTION_HPP_DECL
#define J2CPP_JAVAX_CRYPTO_ILLEGALBLOCKSIZEEXCEPTION_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class String; } } }
namespace j2cpp { namespace java { namespace security { class GeneralSecurityException; } } }


#include <java/lang/String.hpp>
#include <java/security/GeneralSecurityException.hpp>


namespace j2cpp {

namespace javax { namespace crypto {

	class IllegalBlockSizeException;
	class IllegalBlockSizeException
		: public object<IllegalBlockSizeException>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)

		explicit IllegalBlockSizeException(jobject jobj)
		: object<IllegalBlockSizeException>(jobj)
		{
		}

		operator local_ref<java::security::GeneralSecurityException>() const;


		IllegalBlockSizeException(local_ref< java::lang::String > const&);
		IllegalBlockSizeException();
	}; //class IllegalBlockSizeException

} //namespace crypto
} //namespace javax

} //namespace j2cpp

#endif //J2CPP_JAVAX_CRYPTO_ILLEGALBLOCKSIZEEXCEPTION_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_JAVAX_CRYPTO_ILLEGALBLOCKSIZEEXCEPTION_HPP_IMPL
#define J2CPP_JAVAX_CRYPTO_ILLEGALBLOCKSIZEEXCEPTION_HPP_IMPL

namespace j2cpp {



javax::crypto::IllegalBlockSizeException::operator local_ref<java::security::GeneralSecurityException>() const
{
	return local_ref<java::security::GeneralSecurityException>(get_jobject());
}


javax::crypto::IllegalBlockSizeException::IllegalBlockSizeException(local_ref< java::lang::String > const &a0)
: object<javax::crypto::IllegalBlockSizeException>(
	call_new_object<
		javax::crypto::IllegalBlockSizeException::J2CPP_CLASS_NAME,
		javax::crypto::IllegalBlockSizeException::J2CPP_METHOD_NAME(0),
		javax::crypto::IllegalBlockSizeException::J2CPP_METHOD_SIGNATURE(0)
	>(a0)
)
{
}



javax::crypto::IllegalBlockSizeException::IllegalBlockSizeException()
: object<javax::crypto::IllegalBlockSizeException>(
	call_new_object<
		javax::crypto::IllegalBlockSizeException::J2CPP_CLASS_NAME,
		javax::crypto::IllegalBlockSizeException::J2CPP_METHOD_NAME(1),
		javax::crypto::IllegalBlockSizeException::J2CPP_METHOD_SIGNATURE(1)
	>()
)
{
}



J2CPP_DEFINE_CLASS(javax::crypto::IllegalBlockSizeException,"javax/crypto/IllegalBlockSizeException")
J2CPP_DEFINE_METHOD(javax::crypto::IllegalBlockSizeException,0,"<init>","(Ljava/lang/String;)V")
J2CPP_DEFINE_METHOD(javax::crypto::IllegalBlockSizeException,1,"<init>","()V")

} //namespace j2cpp

#endif //J2CPP_JAVAX_CRYPTO_ILLEGALBLOCKSIZEEXCEPTION_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
