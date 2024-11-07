# zyx.batch
# handlers for batch completions
from __future__ import annotations

from ...zyx import utils

from .types.completions.completion_arguments import CompletionArguments


BaseClient = 'BaseClient'


# batch completion handler
# currently instructor does not patch litellm's batch completions
# this method uses the response_format method from litellm to handle structured outputs
# loss in accuracy, as it does not use any of the instructor generation wrappers
def run_structured_batch_completion(
        base_client: BaseClient,
        args: CompletionArguments
):
    
    # set response format if not provided
    if args.response_model and not args.response_format:
        args.response_format = args.response_model
    
    # run litellm batch completion
    responses = base_client.litellm_batch_completion(
        messages = args.messages,
        model = args.model,
        response_format = args.response_format
    )

    # init list of returned models
    returned_models = []
    
    # parse/iterate through responses
    for response in responses:
        try:
            response_content = response.choices[0].message.content
        except:
            response_content = response['choices'][0]['message']['content']

        # convert the json string to a pydantic model
        response_model = utils.parse_json_string_to_pydantic_model(response_content, args.response_format)

        # append the response model to the list
        returned_models.append(response_model)

    return returned_models