"""
🧱 zyx.core.objects

Objects are pydantic models that are set in the state,
and can be used to store data in a structured way.

The point of objects is that they can be directly 
augumented by the agent, and can be used to store / regenerate / read data
in a structured way.

This can be used for something as simple as:
user_name = "steve"

# or
front_door_light_status = "off"
backyard_light_status = "off"
kitchen_light_status = "off"
garage_light_status = "off"
upstairs_light_status = "off"
upstairs_bathroom_light_status = "off"
upstairs_bedroom_light_status = "off"

Burr's state already does most of the work, this is just to help
with zyx to create prompts with proper context when regenerating
these values.
"""