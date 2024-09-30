# Notebook LM (Creating Podcasts from Documents)

`zyx` provides an experimental module, built for simulating conversations between 2 or more agents. 

---

# **[Generated Audio](./podcast.mp3)**

---

### Quick Example - Podcast about the Large Language Monkeys Paper

```python
from zyx import agents
import zyx

# Lets retrieve our document
# Large Language Monkeys: Scaling Inference Compute With Repeated Sampling
document = zyx.read("https://arxiv.org/pdf/2407.21787", output = str)

# Create the characters
john = agents.Character(
    name = "John",
    personality = "The main speaker of the podcast, very genuine and knowledgable.",
    knowledge = document,
    voice = "alloy" # Supports OpenAI TTS voices
)

jane = agents.Character(
    name = "Jane",
    personality = "The second speaker of the podcast, not very knowledgable, but very good at asking questions."
)

# Now lets create our conversation
agents.conversation(
    "Generate a very intuitive and easy to follow podcast converastion about the Large Language Monkeys paper.",
    characters = [john, jane],
    generate_audio = True,  # Generates audio for the conversation 
    audio_output_file = "podcast.mp3"
)
```

---

## Breaking it Down

To start creating Notebook LM style podcast, we need to first retrieve the document we will be using as context. Lets utilize the zyx.read() function
to retrieve the paper from arXiv.

```python
from zyx import read

document = read("https://arxiv.org/pdf/2407.21787")
```

---

### Defining Characters

To create our podcast now, first we need to create our characters. We will be creating two characters, John and Jane. John will be the main speaker of the podcast, and Jane will be the second speaker.

```python
from zyx import Character

john = Character(
    name = "John",
    personality = "The main speaker of the podcast, very genuine and knowledgable.",
    knowledge = document
)

jane = Character(
    name = "Jane",
    personality = "The second speaker of the podcast, not very knowledgable, but very good at asking questions."
)
```

---

### Generating The Conversation

Now we can create our conversation. We will be passing in the topic we want to discuss, and the characters we want to have in the conversation. The conversation function supports more than 2 characters, and can even support group conversations (*adhering to the limitations of the LLM you are using*). For this example we will be using the `generate_audio` parameter to generate audio for the conversation.

```python
from zyx import conversation

conversation(
    "Generate a very intuitive and easy to follow podcast converastion about the Large Language Monkeys paper.",
    characters = [john, jane],
    generate_audio = True,  # Generates audio for the conversation 
    audio_output_file = "podcast.mp3",
    max_turns = 10 # Set this to any number you want
)
```

**More examples will be added soon**

---

## API Reference

::: zyx.resources.completions.agents.conversation.conversation


