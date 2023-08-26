// 从cxx v1.0.106源码处复制与修改而来,autocxx生成的文件依赖于cxx.h与cxx.cpp,而cxx的build.rs里所编译的cxx静态库链接后,会造成下列的函数的undefine reference(可能是版本的问题),因此这里手动进行补充
#include "cxx.h"
namespace rust {
Str::operator std::string() const {
  return std::string(this->data(), this->size());
}
} // namespace rust