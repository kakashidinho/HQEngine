/*================================================================================
  code generated by: java2cpp
  author: Zoran Angelov, mailto://baldzar@gmail.com
  class: dalvik.system.PotentialDeadlockError
================================================================================*/


#ifndef J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_DALVIK_SYSTEM_POTENTIALDEADLOCKERROR_HPP_DECL
#define J2CPP_DALVIK_SYSTEM_POTENTIALDEADLOCKERROR_HPP_DECL


namespace j2cpp { namespace java { namespace lang { class VirtualMachineError; } } }
namespace j2cpp { namespace java { namespace lang { class String; } } }


#include <java/lang/String.hpp>
#include <java/lang/VirtualMachineError.hpp>


namespace j2cpp {

namespace dalvik { namespace system {

	class PotentialDeadlockError;
	class PotentialDeadlockError
		: public object<PotentialDeadlockError>
	{
	public:

		J2CPP_DECLARE_CLASS

		J2CPP_DECLARE_METHOD(0)
		J2CPP_DECLARE_METHOD(1)

		explicit PotentialDeadlockError(jobject jobj)
		: object<PotentialDeadlockError>(jobj)
		{
		}

		operator local_ref<java::lang::VirtualMachineError>() const;


		PotentialDeadlockError();
		PotentialDeadlockError(local_ref< java::lang::String > const&);
	}; //class PotentialDeadlockError

} //namespace system
} //namespace dalvik

} //namespace j2cpp

#endif //J2CPP_DALVIK_SYSTEM_POTENTIALDEADLOCKERROR_HPP_DECL

#else //J2CPP_INCLUDE_IMPLEMENTATION

#ifndef J2CPP_DALVIK_SYSTEM_POTENTIALDEADLOCKERROR_HPP_IMPL
#define J2CPP_DALVIK_SYSTEM_POTENTIALDEADLOCKERROR_HPP_IMPL

namespace j2cpp {



dalvik::system::PotentialDeadlockError::operator local_ref<java::lang::VirtualMachineError>() const
{
	return local_ref<java::lang::VirtualMachineError>(get_jobject());
}


dalvik::system::PotentialDeadlockError::PotentialDeadlockError()
: object<dalvik::system::PotentialDeadlockError>(
	call_new_object<
		dalvik::system::PotentialDeadlockError::J2CPP_CLASS_NAME,
		dalvik::system::PotentialDeadlockError::J2CPP_METHOD_NAME(0),
		dalvik::system::PotentialDeadlockError::J2CPP_METHOD_SIGNATURE(0)
	>()
)
{
}



dalvik::system::PotentialDeadlockError::PotentialDeadlockError(local_ref< java::lang::String > const &a0)
: object<dalvik::system::PotentialDeadlockError>(
	call_new_object<
		dalvik::system::PotentialDeadlockError::J2CPP_CLASS_NAME,
		dalvik::system::PotentialDeadlockError::J2CPP_METHOD_NAME(1),
		dalvik::system::PotentialDeadlockError::J2CPP_METHOD_SIGNATURE(1)
	>(a0)
)
{
}



J2CPP_DEFINE_CLASS(dalvik::system::PotentialDeadlockError,"dalvik/system/PotentialDeadlockError")
J2CPP_DEFINE_METHOD(dalvik::system::PotentialDeadlockError,0,"<init>","()V")
J2CPP_DEFINE_METHOD(dalvik::system::PotentialDeadlockError,1,"<init>","(Ljava/lang/String;)V")

} //namespace j2cpp

#endif //J2CPP_DALVIK_SYSTEM_POTENTIALDEADLOCKERROR_HPP_IMPL

#endif //J2CPP_INCLUDE_IMPLEMENTATION