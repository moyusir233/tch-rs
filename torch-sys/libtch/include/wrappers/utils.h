#ifndef TCH_CXX_WRAPPER_UTILS_H
#define TCH_CXX_WRAPPER_UTILS_H

#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <cstdint>
#include <vector>

/*
 *  用于创建包装torch常用的引用计数指针的类的宏
 * */
#define CREATE_CONTAINER_CLASS(CLASS_NAME_, WRAP_TYPE_)                        \
  class CLASS_NAME_ {                                                          \
  protected:                                                                   \
    CLASS_NAME_() = delete;                                                    \
    explicit CLASS_NAME_(c10::intrusive_ptr<WRAP_TYPE_> &&value) noexcept      \
        : inner(std::forward<c10::intrusive_ptr<WRAP_TYPE_>>(value)) {}        \
                                                                               \
  public:                                                                      \
    c10::intrusive_ptr<WRAP_TYPE_> inner;                                      \
  };

// 用于包装指针数组的
#define CREATE_PTR_ARRAY(TYPE_NAME_, T)                                        \
  struct TYPE_NAME_ {                                                          \
    T *ptr;                                                                    \
    std::uint64_t size;                                                        \
  };

#endif // TCH_CXX_WRAPPER_UTILS_H
