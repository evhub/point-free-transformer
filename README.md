# point-free-transformer

A [point-free](https://en.wikipedia.org/wiki/Tacit_programming) implementation of a one-layer transformer in [Coconut](https://coconut-lang.org/).

![](https://i.imgur.com/QSmFE2m.png)

![](https://i.imgur.com/9WpR3Rf.png)

![](https://i.imgur.com/RB8iFGw.png)

See the full highlighted source [here](https://refined-github-html-preview.kidonng.workers.dev/evhub/point-free-transformer/raw/main/point_free_transformer.html).

Some help for those that don't read Coconut:
- `..>` is forward function composition (so `f ..> g` is `g . f` in Haskell)
- `f$(<args>)` is partial application (so `f$(<args>)` is `functools.partial(f, <args>)` in Python)
- `lift` “lifts” a function so that all of its arguments become unary functions (for a binary function, `lift` is the S' combinator, or `liftA2` in Haskell—e.g. `lift(f)(g, h)` is equivalent to `lambda x: f(g(x), h(x))` in Python)
- `(<op>)` is the operator function for `<op>` (so `(+)` is `operator.add` in Python)
- `(. <op> <arg>)` is equivalent to `<op>$(?, <arg>)` (`?` is a placeholder that means that argument isn't partially applied)
- `.method(<args>)` is equivalent to `lambda x: x.method(<args>)` in Python

Also, `einsum` here is [`better_einsum.einsum`](https://github.com/evhub/better_einsum).
