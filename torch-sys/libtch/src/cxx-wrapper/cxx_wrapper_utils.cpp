#include "cxx_wrapper_utils.h"

IntrusivePtrContainer new_target(){
    printf("create a target ref\n");
    return {c10::make_intrusive<MyIntrusiveTarget>()};
}


IntrusivePtrContainer
clone_target(IntrusivePtrContainer &src){
    printf("clone a target ref\n");
    return {src};
}