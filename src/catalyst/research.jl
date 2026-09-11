# Entry points implemented by optional extensions (research track, PLAN.md §3.5).

const _NEEDS_GPUCOMPILER = "this needs the GPUCompiler package: add GPUCompiler to your project and `using GPUCompiler`"
"""
    compile_native(qn::QNode, args...; dir=mktempdir(), name) -> NativeLibrary

Compile the program traced for `args` to a standalone shared library with Julia's compiler
(GPUCompiler, no MLIR): a QIR-style native program whose only external symbols are the Catalyst
runtime's C API. Call the result like the QNode. Needs `using GPUCompiler`.
"""
compile_native(args...; kwargs...) = error(_NEEDS_GPUCOMPILER)
"""`native_llvm_ir(qn, args...)`: LLVM IR of the generated runtime program (needs `using GPUCompiler`)."""
native_llvm_ir(args...; kwargs...) = error(_NEEDS_GPUCOMPILER)

const _NEEDS_REACTANT = "this needs the Reactant package (MLIR bindings): add Reactant to your project and `using Reactant`"
"""
    mlir_pass(text::AbstractString, passes...) -> String

Run Julia-implemented passes on Catalyst MLIR text through Reactant's MLIR bindings (the
Catalyst ops are handled as unregistered operations). Available: `:cancel_inverses`. Use as
`CatalystDevice(mlir_transform = src -> mlir_pass(src, :cancel_inverses))`. Needs `using Reactant`.
"""
mlir_pass(args...; kwargs...) = error(_NEEDS_REACTANT)
