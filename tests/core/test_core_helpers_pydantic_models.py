"""
Pydantic model helper tests
"""

import pytest
from rich import print
from pydantic import BaseModel, Field
from zyx.core.helpers import pydantic_models as pydantic_models_helpers

module_tag = "[bold italic sky_blue3]zyx.helpers.pydantic_models[/bold italic sky_blue3]"


# ===================================================================
# Field Mapping Creation
# ===================================================================


def test_core_helpers_pydantic_models_parse_string_to_pydantic_field_mapping():
    # Test basic type mapping
    mapping = pydantic_models_helpers.parse_string_to_pydantic_field_mapping("name: str")
    print(mapping)
    assert "name" in mapping
    assert mapping["name"][0] == str

    # Test with no type specified
    mapping = pydantic_models_helpers.parse_string_to_pydantic_field_mapping("name")
    print(mapping)
    assert "name" in mapping
    assert mapping["name"][0] == str

    # Test with capitalized model name
    mapping = pydantic_models_helpers.parse_string_to_pydantic_field_mapping("Person")
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
        print(mapping)
        assert "field" in mapping
        assert mapping["field"][0] == expected_type


def test_core_helpers_pydantic_models_parse_type_to_pydantic_field_mapping():
    # Test basic type mapping
    mapping = pydantic_models_helpers.parse_type_to_pydantic_field_mapping(str)
    print(mapping)
    assert "string" in mapping
    assert mapping["string"][0] == str

    # Test with index
    mapping = pydantic_models_helpers.parse_type_to_pydantic_field_mapping(int, index=1)
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
        print(mapping)
        assert expected_name in mapping
        assert mapping[expected_name][0] == test_type


# ===================================================================
# Model Creation
# ===================================================================


def test_core_helpers_pydantic_models_convert_to_pydantic_model_cls():
    # Test with string
    model = pydantic_models_helpers.convert_to_pydantic_model_cls("name: str")
    print(model)
    assert isinstance(model, type)  # Check that we get a model class
    field_dict = model.model_fields
    assert "name" in field_dict

    # Test with type
    model = pydantic_models_helpers.convert_to_pydantic_model_cls(str)
    print(model)
    assert isinstance(model, type)  # Check that we get a model class
    field_dict = model.model_fields
    assert "string" in field_dict

    # Test with sequence
    model = pydantic_models_helpers.convert_to_pydantic_model_cls(["name: str", "age: int"])
    print(model)
    assert isinstance(model, type)  # Check that we get a model class
    field_dict = model.model_fields
    assert "name" in field_dict
    assert "age" in field_dict

    # Test with model name in sequence
    model = pydantic_models_helpers.convert_to_pydantic_model_cls(["Person", "name: str"])
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
    print(model)
    model_fields = model.model_fields
    test_fields = TestModel.model_fields

    # Test with dict
    model = pydantic_models_helpers.convert_to_pydantic_model_cls({"name": (str, Field())})
    print(model)
    assert isinstance(model, type)  # Check that we get a model class
    field_dict = model.model_fields
    assert "name" in field_dict


def test_core_helpers_pydantic_models_convert_python_function_to_pydantic_model_cls():
    # Define a sample function
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
    test_core_helpers_pydantic_models_parse_string_to_pydantic_field_mapping()
    test_core_helpers_pydantic_models_parse_type_to_pydantic_field_mapping()
    test_core_helpers_pydantic_models_convert_to_pydantic_model_cls()
    test_core_helpers_pydantic_models_convert_python_function_to_pydantic_model_cls()
