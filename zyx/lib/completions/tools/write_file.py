def write_file(content: str, filename: str) -> bool:
    """A tool that writes a file to the current working directory

    Parameters:
        - content: The content to write to the file
        - filename: The name of the file to write to (Just the filename & extension, no path)
    """
    import os

    # Get the current working directory
    try:
        cwd = os.getcwd()

        zyx_tools_dir = os.path.join(cwd, "zyx_tools")
        if not os.path.exists(zyx_tools_dir):
            os.makedirs(zyx_tools_dir)

        # Create the full path
        full_path = os.path.join(zyx_tools_dir, filename)

        # Write the file
        with open(full_path, "w") as f:
            f.write(content)
    except Exception as e:
        return False

    return True


if __name__ == "__main__":
    print(write_file("Hello, world!", "test.txt"))
