Structured Outputs
Introduction
Structured Outputs is a feature that guarantees the model will always generate responses that adhere to your supplied JSON Schema, so you don't need to worry about the model omitting a required key, or hallucinating an invalid enum value.

Some benefits of Structed Outputs include:

Reliable type-safety: No need to validate or retry incorrectly formatted responses
Explicit refusals: Safety-based model refusals are now programmatically detectable
Simpler prompting: No need for strongly worded prompts to achieve consistent formatting
Getting a structured response
python

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    response_format=CalendarEvent,
)

event = completion.choices[0].message.parsed
Examples
Chain of thought
You can ask the model to output an answer in a structured, step-by-step way, to guide the user through the solution.

Structured Outputs for chain-of-thought math tutoring
python

python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class Step(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "You are a helpful math tutor. Guide the user through the solution step by step."},
        {"role": "user", "content": "how can I solve 8x + 7 = -23"}
    ],
    response_format=MathReasoning,
)

math_reasoning = completion.choices[0].message.parsed
Example response
{
  "steps": [
    {
      "explanation": "Start with the equation 8x + 7 = -23.",
      "output": "8x + 7 = -23"
    },
    {
      "explanation": "Subtract 7 from both sides to isolate the term with the variable.",
      "output": "8x = -23 - 7"
    },
    {
      "explanation": "Simplify the right side of the equation.",
      "output": "8x = -30"
    },
    {
      "explanation": "Divide both sides by 8 to solve for x.",
      "output": "x = -30 / 8"
    },
    {
      "explanation": "Simplify the fraction.",
      "output": "x = -15 / 4"
    }
  ],
  "final_answer": "x = -15 / 4"
}
```

Structured Outputs vs JSON mode
Structured Outputs is the evolution of JSON mode. While both ensure valid JSON is produced, only Structured Outputs ensure schema adherance. Both Structured Outputs and JSON mode are supported in the Chat Completions API, Assistants API, Fine-tuning API and Batch API.

We recommend always using Structured Outputs instead of JSON mode when possible. However, Structured Outputs is only supported with the gpt-4o-mini, gpt-4o-mini-2024-07-18, and gpt-4o-2024-08-06 model snapshots and later.

Structured Outputs	JSON Mode
Outputs Valid JSON	Yes	Yes
Adheres to Schema	Yes 1	No
Compatible Models	gpt-4o-mini, gpt-4o-2024-08-06, and later	All modern models
Enabling	response_format: { type: "json_schema", json_schema: {"strict": true, "schema": ...} }	response_format: { type: "json_object" }
Used in Function Calling	Only if strict: true in function definition 2	Always
Supported in Parallel Function Calling	No	Yes
Learn how to leverage function calling with Structured Outputs.

When to use Structured Outputs via function calling vs via response_format

Structured Outputs is available in two forms in the OpenAI API:

When using Function Calling
When using a json_schema response format
Function calling is useful when you are building an application that bridges the models and functionality of your application.

For example, you can give the model access to functions that query a database in order to build an AI assistant that can help users with their orders, or functions that can interact with the UI.

Conversely, Structured Outputs via response_format are more suitable when you want to indicate a structured schema for use when the model responds to the user, rather than when the model calls a tool.

For example, if you are building a math tutoring application, you might want the assistant to respond to your user using a specific JSON Schema so that you can generate a UI that displays different parts of the model's output in distinct ways.

Put simply:

If you are connecting the model to tools, functions, data, etc. in your system, then you should use function calling
If you want to structure the model's output when it responds to the user, then you should use a structured response_format
The remainder of this guide will focus on non-function calling use cases in the Chat Completions API. To learn more about how to use Structured Outputs with function calling, check out the Function Calling guide.

How to use Structured Outputs with response_format

You can use Structured Outputs with the new SDK helper to parse the model's output into your desired format, or you can specify the JSON schema directly.

Step 1: Define your object
Step 2: Supply your object in the API call
Step 3: Handle edge cases
Step 4: Use the generated structured data in a type-safe way
Refusals with Structured Outputs

When using Structured Outputs with user-generated input, OpenAI models may occasionally refuse to fulfill the request for safety reasons.

Since a refusal does not necessarily follow the schema you have supplied in response_format, the API has a new field refusal to indicate when the model refused.

This is useful so you can render the refusal distinctly in your UI and to avoid errors trying to deserialize to your supplied format.

When using Structured Outputs with response_format, you may see an output like the following in case of a refusal:

json

```json
{
  "id": "chatcmpl-9nYAG9LPNonX8DAyrkwYfemr3C8HC",
  "object": "chat.completion",
  "created": 1721596428,
  "model": "gpt-4o-2024-08-06",
  "choices": [
    {
	  "index": 0,
	  "message": {
            "role": "assistant",
            "refusal": "I'm sorry, I cannot assist with that request."
	  },
	  "logprobs": null,
	  "finish_reason": "stop"
	}
  ],
  "usage": {
      "prompt_tokens": 81,
      "completion_tokens": 11,
      "total_tokens": 92
  },
  "system_fingerprint": "fp_3407719c7f"
}
```
Tips and best practices

As you use Structured Outputs, you can keep these best practices in mind to ensure your application is robust.

Mistakes handling
Structured Outputs doesn't prevent all mistakes.

If you see mistakes, try adjusting instructions or providing examples in the system instructions or splitting tasks into simpler subtasks.

Avoid JSON schema divergence
Prevent your JSON Schema and corresponding types in your programming language from diverging.

For example, by adding some CI rule that flags when either is edited or by adding a CI step that auto-generates the JSON Schema from your type definitions (or vice-versa).

Supported schemas
Structured Outputs supports a subset of the JSON Schema language.

Supported types
The following types are supported for Structured Outputs:

String
Number
Boolean
Object
Array
Enum
anyOf
All fields must be required
To use Structured Outputs, all fields or function parameters must be specified as required.

json

```json
{
    "name": "get_weather",
    "description": "Fetches the weather in the given location",
    "strict": true,
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location to get the weather for"
            },
            "unit": {
                "type": "string",
                "description": "The unit to return the temperature in",
                "enum": ["F", "C"]
            }
        },
        "additionalProperties": false,
        "required": ["location", "unit"]
    }
}
```
Although all fields must be required (and the model will return a value for each parameter), it is possible to emulate an optional parameter by using a union type with null.

json

```json
{
    "name": "get_weather",
    "description": "Fetches the weather in the given location",
    "strict": true,
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location to get the weather for"
            },
            "unit": {
                "type": ["string", "null"],
                "description": "The unit to return the temperature in",
                "enum": ["F", "C"]
            }
        },
        "additionalProperties": false,
        "required": [
            "location", "unit"
        ]
    }
}
```

Objects have limitations on nesting depth and size
A schema may have up to 100 object properties total, with up to 5 levels of nesting.

additionalProperties: false must always be set in objects
additionalProperties controls whether it is allowable for an object to contain additional keys / values that were not defined in the JSON Schema.

Structured Outputs only supports generating specified keys / values, so we require developers to set additionalProperties: false to opt into Structured Outputs.

json

```json
{
    "name": "get_weather",
    "description": "Fetches the weather in the given location",
    "strict": true,
    "schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location to get the weather for"
            },
            "unit": {
                "type": "string",
                "description": "The unit to return the temperature in",
                "enum": ["F", "C"]
            }
        },
        "additionalProperties": false,
        "required": [
            "location", "unit"
        ]
    }
}
```

Key ordering
When using Structured Outputs, outputs will be produced in the same order as the ordering of keys in the schema.

Some type-specific keywords are not yet supported
Notable keywords not supported include:

For strings: minLength, maxLength, pattern, format
For numbers: minimum, maximum, multipleOf
For objects: patternProperties, unevaluatedProperties, propertyNames, minProperties, maxProperties
For arrays: unevaluatedItems, contains, minContains, maxContains, minItems, maxItems, uniqueItems
If you turn on Structured Outputs by supplying strict: true and call the API with an unsupported JSON Schema, you will receive an error.

For anyOf, the nested schemas must each be a valid JSON Schema per this subset
Here's an example supported anyOf schema:

json

```json
{
    "name": "get_weather",
    "description": "Fetches the weather in the given location",
    "strict": true,
    "schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location to get the weather for"
            },
            "unit": {
                "type": "string",
                "description": "The unit to return the temperature in",
                "enum": ["F", "C"]
            }
        },
        "additionalProperties": false,
        "required": [
            "location", "unit"
        ]
    }
}
```

oneOf and allOf are not supported but oneOf can be emulated via anyOf
Sometimes it's helpful to have a schema where the model can decide to return one of multiple possible objects.

For example, if you wanted to have a schema that can return either { result: { status: "success" } } or { result: { status: "failure", failure_reason: "..."} } you could do so with a schema such as this.

json

```json
{
  "type": "object",
  "properties": {
    "result": {
      "type": "object",
      "anyOf": [
        {
          "properties": {
            "status": {
              "enum": ["success"]
            }
          },
          "required": ["status"]
        },
        {
          "properties": {
            "status": {
              "enum": ["failure"]
            },
            "failure_reason": {
              "type": "string"
            }
          },
          "required": ["status", "failure_reason"]
        }
      ]
    }
  },
  "required": ["result"],
  "additionalProperties": false
}
```

Note that since we are using anyOf, it is possible for the model to include keys from a combination of the options. We therefore recommend your schema uses some “discriminating key” (i.e. a key that each option includes set to a distinct value such as status in the above example) so your application can easily determine which of the options the model picked (and if desired, to strip).

Definitions are supported
You can use definitions to define subschemas which are referenced throughout your schema. The following is a simple example.

json

json
{
	"type": "object",
	"properties": {
		"steps": {
			"type": "array",
			"items": {
				"$ref": "#/$defs/step"
			}
		},
		"final_answer": {
			"type": "string"
		}
	},
	"$defs": {
		"step": {
			"type": "object",
			"properties": {
				"explanation": {
					"type": "string"
				},
				"output": {
					"type": "string"
				}
			},
			"required": [
				"explanation",
				"output"
			],
			"additionalProperties": false
		}
	},
	"required": [
		"steps",
		"final_answer"
	],
	"additionalProperties": false
}
Recursive schemas are supported
Sample recursive schema using # to indicate root recursion.

json

```json
{
        "name": "ui",
        "description": "Dynamically generated UI",
        "strict": true,
        "schema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "The type of the UI component",
                    "enum": ["div", "button", "header", "section", "field", "form"]
                },
                "label": {
                    "type": "string",
                    "description": "The label of the UI component, used for buttons or form fields"
                },
                "children": {
                    "type": "array",
                    "description": "Nested UI components",
                    "items": {
                        "$ref": "#"
                    }
                },
                "attributes": {
                    "type": "array",
                    "description": "Arbitrary attributes for the UI component, suitable for any element",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the attribute, for example onClick or className"
                            },
                            "value": {
                                "type": "string",
                                "description": "The value of the attribute"
                            }
                        },
                      "additionalProperties": false,
                      "required": ["name", "value"]
                    }
                }
            },
            "required": ["type", "label", "children", "attributes"],
            "additionalProperties": false
        }
    }
Sample recursive schema using explicit recursion:
```

json

```json
{
	"type": "object",
	"properties": {
		"linked_list": {
			"$ref": "#/$defs/linked_list_node"
		}
	},
	"$defs": {
		"linked_list_node": {
			"type": "object",
			"properties": {
				"value": {
					"type": "number"
				},
				"next": {
					"anyOf": [
						{
							"$ref": "#/$defs/linked_list_node"
						},
						{
							"type": "null"
						}
					]
				}
			},
			"additionalProperties": false,
			"required": [
				"next",
				"value"
			]
		}
	},
	"additionalProperties": false,
	"required": [
		"linked_list"
	]
}
```

JSON mode
JSON mode is a more basic version of the Structured Outputs feature. While JSON mode ensures that model output is valid JSON, Structured Outputs further guarantees that the model's output matches the schema you specify. Learn more about Structured Outputs.

We recommend you use Structured Outputs if it is supported for your use case.

When JSON mode is turned on, the model's output is ensured to be valid JSON, except for in some edge cases that you should detect and handle appropriately.

To turn on JSON mode with the Chat Completions or Assistants API you can set the response_format to { "type": "json_object" }. If you are using function calling, JSON mode is always turned on.

Important notes:

When using JSON mode, you must always instruct the model to produce JSON via some message in the conversation, for example via your system message. If you don't include an explicit instruction to generate JSON, the model may generate an unending stream of whitespace and the request may run continually until it reaches the token limit. To help ensure you don't forget, the API will throw an error if the string "JSON" does not appear somewhere in the context.
JSON mode will not guarantee the output matches any specific schema, only that it is valid and parses without errors. You should use Structured Outputs to ensure it matches your schema, or if that is not possible, you should use a validation library and potentially retries to ensure that the output matches your desired schema.
Your application must detect and handle the edge cases that can result in the model output not being a complete JSON object (see below)
Handling edge cases
python

```python
we_did_not_specify_stop_tokens = True

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": "Who won the world series in 2020? Please respond in the format {winner: ...}"}
        ],
        response_format={"type": "json_object"}
    )

    # Check if the conversation was too long for the context window, resulting in incomplete JSON 
    if response.choices[0].message.finish_reason == "length":
        # your code should handle this error case
        pass

    # Check if the OpenAI safety system refused the request and generated a refusal instead
    if response.choices[0].message[0].get("refusal"):
        # your code should handle this error case
        # In this case, the .content field will contain the explanation (if any) that the model generated for why it is refusing
        print(response.choices[0].message[0]["refusal"])

    # Check if the model's output included restricted content, so the generation of JSON was halted and may be partial
    if response.choices[0].message.finish_reason == "content_filter":
        # your code should handle this error case
        pass

    if response.choices[0].message.finish_reason == "stop":
        # In this case the model has either successfully finished generating the JSON object according to your schema, or the model generated one of the tokens you provided as a "stop token"

        if we_did_not_specify_stop_tokens:
            # If you didn't specify any stop tokens, then the generation is complete and the content key will contain the serialized JSON object
            # This is guaranteed to parse successfully and should now contain  "{"winner": "Los Angeles Dodgers"}"
            print(response.choices[0].message.content)
        else:
            # Check if the response.choices[0].message.content ends with one of your stop tokens and handle appropriately
            pass
except Exception as e:
    # Your code should handle errors here, for example a network error calling the API
    print(e)
```