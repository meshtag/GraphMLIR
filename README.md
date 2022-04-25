# GraphMLIR
An IR based solution for optimising graph algorithms.

```
git clone https://github.com/meshtag/GraphMLIR.git
cd GraphMLIR
git submodule update --init

cd llvm && mkdir build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
ninja
ninja check-mlir

cd ../../ && mkdir build && cd build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DGraphMLIR_EXAMPLES=ON
ninja bfsExample
cd bin && ./bfsExample
```
