from typing import Any


def execute_code(imports: str, code: str) -> Any:
    """A tool that executes code. Ensure input code is separated by newlines, if needed.

    Parameters:
        - imports: Import statements as a string
        - code: The code to execute (Must be Python Code)

    Returns:
        The result of the code execution
    """
    import tempfile
    import subprocess
    import sys
    import os
    import importlib.util

    # Combine imports and code
    full_code = imports + "\n" + code + "\n\nresult = None\n" + code

    # Create a temporary file to hold the code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(full_code)

    try:
        # Execute the code
        env = os.environ.copy()
        env["PYTHONPATH"] = ":".join(sys.path)
        result = subprocess.run(
            [sys.executable, temp_file_path], capture_output=True, text=True, env=env
        )

        if result.returncode != 0:
            raise RuntimeError(f"Error executing code: {result.stderr}")

        # Import the generated module
        spec = importlib.util.spec_from_file_location(
            "generated_module", temp_file_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Return the result
        return module.result

    except Exception as e:
        return f"Error: {str(e)}"

    finally:
        # Remove the temporary file
        os.remove(temp_file_path)


if __name__ == "__main__":
    print(execute_code("import os", "print(os.getcwd())\nresult = os.getcwd()"))
