/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: android.accounts.AccountManagerCallback
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_ACCOUNTS_ACCOUNTMANAGERCALLBACK_HPP_DECL
#define J2CPP_ANDROID_ACCOUNTS_ACCOUNTMANAGERCALLBACK_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class Object; } } }
namespace j2cpp { namespace android { namespace accounts { class AccountManagerFuture; } } }


#include <android/accounts/AccountManagerFuture.hpp>
#include <java/lang/Object.hpp>


namespace j2cpp {

namespace android { namespace accounts {

	class AccountManagerCallback;
	class AccountManagerCallback
		: public object<AccountManagerCallback>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)

		explicit AccountManagerCallback(jobject jobj)
		: object<AccountManagerCallback>(jobj)
		{
		}

		operator local_ref<java::lang::Object>() const;


		void run(local_ref< android::accounts::AccountManagerFuture >  const&);
	}; //class AccountManagerCallback

} //namespace accounts
} //namespace android

} //namespace j2cpp

#endif //J2CPP_ANDROID_ACCOUNTS_ACCOUNTMANAGERCALLBACK_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_ANDROID_ACCOUNTS_ACCOUNTMANAGERCALLBACK_HPP_IMPL
#define J2CPP_ANDROID_ACCOUNTS_ACCOUNTMANAGERCALLBACK_HPP_IMPL

namespace j2cpp {



android::accounts::AccountManagerCallback::operator local_ref<java::lang::Object>() const
{
	return local_ref<java::lang::Object>(get_jobject());
}

void android::accounts::AccountManagerCallback::run(local_ref< android::accounts::AccountManagerFuture > const &a0)
{
	return call_method<
		android::accounts::AccountManagerCallback::J2CPP_CLASS_NAME,
		android::accounts::AccountManagerCallback::J2CPP_METHOD_NAME(0),
		android::accounts::AccountManagerCallback::J2CPP_METHOD_SIGNATURE(0), 
		void
	>(get_jobject(), a0);
}


J2CPP_DEFINE_CLASS(android::accounts::AccountManagerCallback,"android/accounts/AccountManagerCallback")
J2CPP_DEFINE_METHOD(android::accounts::AccountManagerCallback,0,"run","(Landroid/accounts/AccountManagerFuture;)V")

} //namespace j2cpp

#endif //J2CPP_ANDROID_ACCOUNTS_ACCOUNTMANAGERCALLBACK_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION
