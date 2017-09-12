add_llvm_library(LLVMCanonicalization
  Canonicalization.cpp
  CanonicalizationVisitors.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Canonicalization

  DEPENDS
  intrinsics_gen
  )
