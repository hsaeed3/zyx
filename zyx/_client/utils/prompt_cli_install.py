def prompt_cli_install(library: str) -> bool:
    """Prompts the user to install a given library"""

    import subprocess

    response = input(f"Would you like to install the {library} library? (y/n)")

    if response == "y":
        try:
            subprocess.run(["pip", "install", library])
            return True
        except Exception as e:
            raise e
    else:
        return False
