[Site](https://uncomfy.github.io)

https://emscripten.org/
```bash
emcc -O2 -s WASM=1 -s EXPORTED_RUNTIME_METHODS="['cwrap']" -s EXPORTED_FUNCTIONS="['_malloc', '_free']" main.cpp
```
