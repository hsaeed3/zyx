# `ZYX` - Examples

The `examples` directory contains a collection of notebooks that use the `marimo` library to provide interactive examples of the `zyx` library. `marimo` defines itself as a "reactive Python notebook", which in my opinion far surpasses the traditional `.ipynb` Jupyter
Notebooks.

To try out any of the examples for yourself, simply install `marimo` using either:

```bash
pip install marimo
```

or through the `zyx[examples]` extra:

```bash
pip install 'zyx[examples]'
```

Once you've installed the `marimo` library, you can run any of the examples by running the `marimo run` command, for example:

```bash
marimo run examples/make.py
```