"""zyx.beta

Temporary namespace for the beta API of the ``zyx`` package. All resources within
this module are meant to be used standalone from the rest of the package, and will
replace the entirety of the library on the ``v1.1.0`` release.

---

The beta API provides a complete rewrite of the core architecture of the package,
opting to normalize on ``pydantic_ai``'s capabilities and hooks rather than
the pre-existing graph-based approach, this allows a much more modular approach to the
components created within the library.
"""
