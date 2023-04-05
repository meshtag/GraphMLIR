# GraphMLIR

An IR based solution for optimising graph algorithms.

## Clone repository and its dependencies

```
git clone https://github.com/meshtag/GraphMLIR.git
cd GraphMLIR
git submodule update --init
```

## Build LLVM

```
cd llvm && mkdir build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host;X86" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
ninja
ninja check-mlir
```

## Build project

```
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

## Benchmark project

```
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DGraphMLIR_BENCHMARK=ON \
    -DLEMON_DIR=/home/tushar/Downloads/lemon
ninja graph-processing-benchmark
cd bin && ./graph-processing-benchmark
```

_Note_ : Rename the `lemon.1.x.x` folder to `lemon`. For benchmarking install `BOOST` library in system.

## Instructions for generating docs

```
Use doxywizard for generating docs automatically from relevant source directories.
```

#### After this go to docs and open index.html in the html subdirectory with your prefered browser.
