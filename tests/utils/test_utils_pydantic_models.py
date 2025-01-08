"""
zyx.utils.pydantic_models tests
"""

from typing import Any
import logging
import pytest
from rich import print
from pydantic import BaseModel, Field
from zyx.utils import pydantic_models as pydantic_models_helpers

module_tag = "[bold italic sky_blue3]zyx.utils.pydantic_models[/bold italic sky_blue3]"
logger = logging.getLogger("zyx")


# ===================================================================
# Field Mapping Creation
# ===================================================================


def test_utils_pydantic_models_parse_string_to_field_mapping() -> None:
    """Test parsing strings to Pydantic field mappings."""

    # Test basic type mapping
    mapping = pydantic_models_helpers.parse_string_to_pydantic_field_mapping("name: str")
    print(f"{module_tag} - [bold green]Basic Type Mapping[/bold green]")
    print(mapping)
    assert "name" in mapping
    assert mapping["name"][0] == str

    # Test with no type specified
    mapping = pydantic_models_helpers.parse_string_to_pydantic_field_mapping("name")
    print(f"{module_tag} - [bold green]No Type Mapping[/bold green]")
    print(mapping)
    assert "name" in mapping
    assert mapping["name"][0] == str

    # Test with capitalized model name
    mapping = pydantic_models_helpers.parse_string_to_pydantic_field_mapping("Person")
    print(f"{module_tag} - [bold green]Model Name Mapping[/bold green]")
    print(mapping)
    assert "value" in mapping
    assert mapping["value"][0] == str

    # Test different type aliases
    type_tests = {
        "field: string": str,
        "field: int": int,
        "field: integer": int,
        "field: float": float,
        "field: number": float,
        "field: bool": bool,
        "field: boolean": bool,
        "field: list": list,
        "field: array": list,
        "field: dict": dict,
        "field: object": dict,
    }

    for test_str, expected_type in type_tests.items():
        mapping = pydantic_models_helpers.parse_string_to_pydantic_field_mapping(test_str)
        print(f"{module_tag} - [bold green]Type Alias: {test_str}[/bold green]")
        print(mapping)
        assert "field" in mapping
        assert mapping["field"][0] == expected_type


def test_utils_pydantic_models_parse_type_to_field_mapping() -> None:
    """Test parsing Python types to Pydantic field mappings."""

    # Test basic type mapping
    mapping = pydantic_models_helpers.parse_type_to_pydantic_field_mapping(str)
    print(f"{module_tag} - [bold green]Basic Type Mapping[/bold green]")
    print(mapping)
    assert "string" in mapping
    assert mapping["string"][0] == str

    # Test with index
    mapping = pydantic_models_helpers.parse_type_to_pydantic_field_mapping(int, index=1)
    print(f"{module_tag} - [bold green]Indexed Type Mapping[/bold green]")
    print(mapping)
    assert "integer_1" in mapping
    assert mapping["integer_1"][0] == int

    # Test all type mappings
    type_tests = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
        tuple: "array",
        set: "array",
    }

    for test_type, expected_name in type_tests.items():
        mapping = pydantic_models_helpers.parse_type_to_pydantic_field_mapping(test_type)
        print(f"{module_tag} - [bold green]Type Mapping: {test_type}[/bold green]")
        print(mapping)
        assert expected_name in mapping
        assert mapping[expected_name][0] == test_type


# ===================================================================
# Model Creation
# ===================================================================


def test_utils_pydantic_models_convert_to_model_cls() -> None:
    """Test converting various inputs to Pydantic model classes."""

    # Test with string
    model = pydantic_models_helpers.convert_to_pydantic_model_cls("name: str")
    print(f"{module_tag} - [bold green]String Model[/bold green]")
    print(model)
    assert isinstance(model, type)  # Check that we get a model class
    field_dict = model.model_fields
    assert "name" in field_dict

    # Test with type
    model = pydantic_models_helpers.convert_to_pydantic_model_cls(str)
    print(f"{module_tag} - [bold green]Type Model[/bold green]")
    print(model)
    assert isinstance(model, type)  # Check that we get a model class
    field_dict = model.model_fields
    assert "string" in field_dict

    # Test with sequence
    model = pydantic_models_helpers.convert_to_pydantic_model_cls(["name: str", "age: int"])
    print(f"{module_tag} - [bold green]Sequence Model[/bold green]")
    print(model)
    assert isinstance(model, type)  # Check that we get a model class
    field_dict = model.model_fields
    assert "name" in field_dict
    assert "age" in field_dict

    # Test with model name in sequence
    model = pydantic_models_helpers.convert_to_pydantic_model_cls(["Person", "name: str"])
    print(f"{module_tag} - [bold green]Named Model[/bold green]")
    print(model)
    assert model.__name__ == "Person"
    assert isinstance(model, type)  # Check that we get a model class
    field_dict = model.model_fields
    assert "name" in field_dict

    # Test with existing model
    class TestModel(BaseModel):
        name: str
        age: int

    model = pydantic_models_helpers.convert_to_pydantic_model_cls(TestModel)
    print(f"{module_tag} - [bold green]Existing Model[/bold green]")
    print(model)
    model_fields = model.model_fields
    test_fields = TestModel.model_fields

    # Test with dict
    model = pydantic_models_helpers.convert_to_pydantic_model_cls({"name": (str, Field())})
    print(f"{module_tag} - [bold green]Dict Model[/bold green]")
    print(model)
    assert isinstance(model, type)  # Check that we get a model class
    field_dict = model.model_fields
    assert "name" in field_dict


def test_utils_pydantic_models_convert_function_to_model_cls() -> None:
    """Test converting Python functions to Pydantic model classes."""

    def sample_function(name: str, age: int = 30, active: bool = True):
        """
        Sample function for testing.

        Args:
            name: The name of the person.
            age: The age of the person.
            active: Whether the person is active.
        """
        pass

    # Convert the function to a Pydantic model class
    model = pydantic_models_helpers.convert_python_function_to_pydantic_model_cls(sample_function)
    print(f"{module_tag} - [bold green]Function Model[/bold green]")
    print(model)
    assert isinstance(model, type)  # Check that we get a model class
    field_dict = model.model_fields
    assert "name" in field_dict
    assert "age" in field_dict
    assert "active" in field_dict

    # Check field types and defaults
    assert field_dict["name"].annotation == str
    assert field_dict["age"].annotation == int
    assert field_dict["age"].default == 30
    assert field_dict["active"].annotation == bool
    assert field_dict["active"].default == True


if __name__ == "__main__":
    test_utils_pydantic_models_parse_string_to_field_mapping()
    test_utils_pydantic_models_parse_type_to_field_mapping()
    test_utils_pydantic_models_convert_to_model_cls()
    test_utils_pydantic_models_convert_function_to_model_cls()
