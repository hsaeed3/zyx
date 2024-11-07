# zyx.utils
# utility functions

import json
from pydantic import BaseModel, ValidationError, create_model
from .lib.exception import ZYXException
from typing import Optional


def parse_json_string_to_pydantic_model(json_string: str, response_format: Optional[BaseModel] = None) -> BaseModel:
    """
    Parses a JSON string to a pydantic model
    """

    try:
        # parse the json string
        json_content = json.loads(json_string)

        # dynamically create a pydantic model from the json content
        dynamic_model = create_model('ResponseModel', **{k: (type(v), ...) for k, v in json_content.items()})
        dynamic_model = dynamic_model(**json_content)

        # if response format is provided, convert the dynamic model to the response format
        if response_format:
            dynamic_model = create_model(response_format.__name__, **{k: (type(v), ...) for k, v in json_content.items()})
            dynamic_model = dynamic_model(**json_content)

        return dynamic_model
        
    except json.JSONDecodeError as e:
        raise ZYXException(f"Invalid JSON string: {e}")
    except ValidationError as e:
        raise ZYXException(f"Validation error: {e}")
    except Exception as e:
        raise ZYXException(f"Failed to parse JSON string to pydantic model: {e}")

if __name__ == "__main__":
    
    json_string = '{"name": "John", "age": 30}'

    object = parse_json_string_to_pydantic_model(json_string)

    print(object.name)
    print(object.age)

