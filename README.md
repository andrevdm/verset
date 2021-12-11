# Verset

Small Haskell alternative prelude based on [Protolude](https://hackage.haskell.org/package/protolude) and [Intro](https://hackage.haskell.org/package/intro)

*At least it is not a monad tutorial*

Goals
 - Very small
 - Minimal dependencies
 - Avoid `String`
 - Removes partial functions where possible



Notes
 - `catch`, `finally` etc are not exported. This makes it easier to use e.g. `Control.Exception.Safe` for safer exception without having to hide all the defaults
 - No transformers are exposed but `lift` is
 - Simple `IsString` helpers (from Protolude)
 - `Map`, `Set` types exposed from containers, but no other functions


 Use Verset if you want a small prelude and are not opposed to a few extra imports in your code base
