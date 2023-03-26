[Site](https://uncomfy.github.io/braiderv1)

https://emscripten.org/
```bash
emcc -O3 -s WASM=1 -s EXPORTED_RUNTIME_METHODS="['cwrap']" -s EXPORTED_FUNCTIONS="[ \
    '_malloc', \
    '_free', \
    '_getNailIndices', \
    '_doGreedyStep', \
    '_doGreedy', \
    '_createBraider', \
    '_deleteBraider' \
]" main.cpp
```
