# Pydantic Outputs

### Working with Pydantic Models

The `completion()` function utilizes the [Instructor](https://github.com/jxnl/instructor)
library to provide a quick option to generate structured outputs using Pydantic.


```python
import zyx
from pydantic import BaseModel

class Character(BaseModel):
    superhero : str
    secret_identity : str

zyx.completion(
    "Who is spiderman?",
    model = "ollama/llama3.2",
    response_model = Character
)
```

---

### Using the BaseModel Subclass

`zyx` has a subclass of Pydantic's `BaseModel` that has an additional `generate()` method,
specifically built for generating quicker structured outputs with LLMs.

```python hl_lines="9"
from zyx import BaseModel # Subclass of the Pydantic BaseModel

class Character(BaseModel):
    superhero : str
    secret_identity : str

Character.generate("Who is batman?")

# Character(superhero="Batman", secret_identity="Bruce Wayne")
```

---

## Generating Batch Synthetic Data

*You can also generate batch synthetic data using the `generate` method.* Also,
instructions are not required when using the generate method. The
LLM will be automatically prompted to create synthetic **& diverse** data.

```python hl_lines="9 10 11 12 13 14 15 16 17 18 19 20"
class SyntheticData(BaseModel):
    food : str
    chemical_compounds : list[str]

SyntheticData.generate(
    n=10,
)

# SyntheticData(
#     food='Apple',
#     chemical_compounds=['Quercetin', 'Cyanidin', 'Chlorogenic acid']
# ),
# SyntheticData(
#     food='Banana',
#     chemical_compounds=['Dopamine', 'Serotonin', 'Catecholamines']
# ),
# SyntheticData(
#     food='Carrot',
#     chemical_compounds=['Beta-carotene', 'Lutein', 'Zeaxanthin']
# ), ....
```

## Chain of Thought with Pydantic Models

```python hl_lines="14"
from zyx import BaseModel

class LearningPlanWeek(BaseModel):
    tasks : list[str]

class LearningPlan(BaseModel):
    goal : str
    week_1 : LearningPlanWeek
    week_2 : LearningPlanWeek
    week_3 : LearningPlanWeek

LearningPlan.generate(
    "Generate a cohesive 3 week plan for learning python",
    process = "sequential" # This generates each field sequentially, emulating chain of thought
)


# [
#     LearningPlan(
#         goal='Achieve a 20% increase in sales by the end of Q4.',
#         week_1=LearningPlanWeek(
#             tasks=[
#                 'Conduct market research to identify potential customer segments.',
#                 'Develop a targeted marketing campaign to reach new customers.',
#                 'Train the sales team on new sales techniques and product knowledge.',
#                 'Set weekly sales targets and monitor progress.',
#                 'Review and analyze sales data to adjust strategies as needed.'
#             ]
#         ),
#         week_2=LearningPlanWeek(
#             tasks=[
#                 'Implement the marketing campaign and track its effectiveness.',
#                 'Host a webinar to showcase products to potential customers.',
#                 'Follow up with leads generated from the marketing efforts.',
#                 'Conduct a sales team meeting to discuss progress and challenges.',
#                 'Gather feedback from the sales team on customer interactions and adjust
# strategies accordingly.'
#             ]
#         ),
#         week_3=LearningPlanWeek(
#             tasks=[
#                 'Analyze the results of the marketing campaign and sales performance.',
#                 'Identify areas for improvement based on customer feedback and sales
# data.',
#                 'Refine the sales strategy to better target high-potential customer
# segments.',
#                 'Conduct additional training sessions for the sales team based on
# identified gaps.',
#                 'Prepare a report summarizing progress towards the sales goal and next
# steps.'
#             ]
#         )
#     )
# ]
```
