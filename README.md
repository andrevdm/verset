# Verset

[![Haskell-CI](https://github.com/andrevdm/verset/actions/workflows/haskell-ci.yml/badge.svg?branch=master)](https://github.com/andrevdm/verset/actions/workflows/haskell-ci.yml)


Small Haskell alternative prelude based on [Protolude](https://hackage.haskell.org/package/protolude) and [Intro](https://hackage.haskell.org/package/intro)

*At least it is not a monad tutorial*

Goals
 - Very small
 - Minimal dependencies
 - Avoid `String`
 - Removes partial functions where possible
 - It be easy to switch from `Verset` to other preludes



Notes
 - `catch`, `finally` etc are not exported. This makes it easier to use e.g. `Control.Exception.Safe` for safer exception without having to hide all the defaults
   - I would have liked to depend on `Control.Exception.Safe` and expose these function, but then changing preludes drastically changes how exception handling works for your app. 
 - No transformers are exposed but `lift` is
 - Simple `IsString` helpers (from Protolude)
 - `Map`, `Set` types exposed from containers, but no other functions
 - `identity` rather than `id`
 - `String` is not exported use `[Char]`. This is to discourage its use.


 Use Verset if you want a small prelude and are not opposed to a few extra imports in your code base
