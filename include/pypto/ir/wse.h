/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_IR_WSE_H_
#define PYPTO_IR_WSE_H_
#include <charconv>
#include <string>
#include <vector>

#include "pypto/core/logging.h"
namespace pypto::ir {
inline std::vector<int> ParseWseMembers(const std::string& text) {
  std::vector<int> ids;
  const char* cursor = text.data();
  const char* end = cursor + text.size();
  while (cursor < end) {
    int id;
    auto result = std::from_chars(cursor, end, id);
    CHECK(result.ec == std::errc() && id >= 0 &&
          (result.ptr == end || (*result.ptr == ',' && result.ptr + 1 < end)))
        << "WSE members must be comma-separated nonnegative int32 logical IDs";
    ids.push_back(id);
    cursor = result.ptr == end ? end : result.ptr + 1;
  }
  return ids;
}
}  // namespace pypto::ir
#endif  // PYPTO_IR_WSE_H_
