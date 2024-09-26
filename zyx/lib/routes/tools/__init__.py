__all__ = [
    "execute_code",
    "generate_audio",
    "generate_image",
    "web_search",
    "write_file",
]


from .._loader import loader


class execute_code(loader):
    pass


execute_code.init("zyx.lib.completions.tools.execute_code", "execute_code")


class generate_audio(loader):
    pass


generate_audio.init("zyx.lib.completions.tools.generate_audio", "generate_audio")


class generate_image(loader):
    pass


generate_image.init("zyx.lib.completions.tools.generate_image", "generate_image")


class web_search(loader):
    pass


web_search.init("zyx.lib.completions.tools.web_search", "web_search")


class write_file(loader):
    pass


write_file.init("zyx.lib.completions.tools.write_file", "write_file")
