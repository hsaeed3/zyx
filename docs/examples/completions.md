# Generating Completions

The primary module of zyx, is the universal <code>.completion()</code> function. This module is an extensive wrapper around the <code>litellm .completion()</code> function, as well as the Instructor library. </br>

The <code>.completion()</code> function is capable of

- **Generations with any LiteLLM compatible model**
    - Ollama, OpenAI, Anthropic, Groq, Mistral, and more!
- **Direct Instructor Pydantc structured outputs**
- **Tool calling & execution support. (Get a tool interpretation with one function)**
    - zyx provides a few prebuilt tools out of the box
    - Can take in a list of **Python functions**, **OpenAI dictionaries**, or **Pydantic models** as tools!
    - Automatic tool execution if a tool is called through the <code>run_tools</code> parameter
- **Streaming**
- **New** Vision support 
    - Pass in a list of urls
    - Currently uses multi shot prompting if a response model or tools were also passed.
-  **New** Prompt optimization 
    - Creates or optimizes a task tuned system prompt using either the *COSTAR* or *TIDD-EC* frameworks automatically.

## Standard Completion

```python
# Simplest Way to Generate
# Defaults to "gpt-4o-mini" if no model is provided
from zyx import completion

response = completion("Hi, how are you?")

# Returns a standard OpenAI style response object
print(response.choices[0].message.content)
```

```python
# Obviously takes in a normal list of messages as well
response = completion([
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hi, how are you?"}
])
```

```bash
# OUTPUT
Hello! I'm just a program, but I'm here and ready to help you. How can I assist you today?
```

## Instructor Output

```python
import zyx
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# We can pass in a system prompt to change the behavior of the assistant
response = zyx.completion(
    "Create a mock person",
    response_model = Person,
    mode = "md_json" # Change instructor parsing mode with this (default is md_json)
)

# Lets print the full response object
print(response)

print(f"Person Name: '{response.name}', Person Age: '{response.age}'")
```

```bash
# OUTPUT
Person(name='John Doe', age=30)
Person Name: 'John Doe', Person Age: '30'
```

## Tool Calling & Execution

```python
# Lets return a tool call
import zyx

# Lets use the prebuilt web search tool!
response = zyx.completion(
    "Who won the 2024 Euro Cup Final?",
    tools = [zyx.tools.web_search],
    run_tools = True # Set to true to execute tool calls
)

print(response.choices[0].message.content)
```

```bash
# OUTPUT
Spain won the 2024 Euro Cup Final, defeating England 2-1. The decisive goal was scored by substitute Mikel 
Oyarzabal in the 86th minute. This victory marked Spain's fourth European championship title. You can find more 
details about the match (https://en.wikipedia.org/wiki/UEFA_Euro_2024_Final).
```

## Prompt Optimization

```python
# No Optimization
from zyx import completion

response = completion(
    "Write a story about a man who travels through time to save his family from a deadly disease."
)

print(response.choices[0].message.content)
```

<details closed>
<summary>Output</summary>

```bash
# OUTPUT
In the small town of Eldridge, nestled between rolling hills and shimmering rivers, lived Ethan Forrester, a mild-mannered librarian with an 
extraordinary obsession: time travel. By day, he cataloged dusty old tomes and helped patrons uncover forgotten stories; by night, he scoured 
the internet and ancient texts for clues that could unlock the secrets of time.

Then came the news that shattered his world. His beloved wife, Clara, and their two children, Lucy and Sam, had contracted a mysterious and 
deadly disease that was sweeping through the town like wildfire. Despite the doctors’ best efforts and the hospital’s desperate measures, the 
prognosis was grim. Ethan learned that if he did not act fast, he would lose the most precious people in his life.

One stormy evening, as thunder rumbled ominously outside, Ethan discovered a brittle manuscript hidden behind the row of encyclopedias. It 
spoke of a time-traveling device, built by a forgotten inventor named Archimedes Blake—an intricate metal orb that pulsed with a strange 
energy. The instructions were hazy, but Ethan understood one thing: he had to find it.

He scoured every corner of the library for weeks, until, finally, he found an ancient map hidden in the spine of a long-neglected book. The 
map contained clues leading to Archimedes’s laboratory, which lay abandoned in a remote part of the nearby woods. Driven by desperation, Ethan
packed supplies and set off into the forest that night.

The air was thick with mist as he crossed over fallen branches and brambles. Guided by the map’s scribbled notes and an adrenaline-fueled 
determination, he reached a tumbledown cottage shrouded in vines and shadows. Inside lay the remnants of Archimedes’s past—a jumble of 
beakers, gears, and sketches. In the center, on a makeshift table, rested the metal orb, its surface gleaming like a star shrouded in dust.

After hours of tinkering and curse-laden frustration, Ethan managed to grasp the orb correctly, recalling the few instructions he had 
deciphered. His heart raced as he envisioned where he needed to go: the moment before the disease had struck the town; the point where he 
could warn his family and seek treatment before it ensnared them.

With a deep breath, Ethan activated the orb. Blinding light enveloped him, and he felt a strange sensation, as if he were being pulled apart 
and stitched back together again. Moments later, he collapsed onto the ground—different leaves under him, a new sense of calm in the air. It 
was late summer, two months earlier. The sun bathed the town of Eldridge in a warm glow, and he could hear the distant laughter of children 
playing in the park.

Ethan's heart swelled with hope. He set out for home, each step quickening in urgency. He burst through the front door, breathless and 
frazzled. Clara looked up from the kitchen, her smile melting into confusion.

“Ethan? You’re home early!” she exclaimed, wiping her hands on a dish towel.

“Clara, we need to talk! The kids—Lucy and Sam—they need to be checked by a doctor. There’s something terrible coming.”

Clara raised an eyebrow, bemused by Ethan’s intensity, but he could see the flicker of concern in her eyes. He explained everything—the 
disease, the timeline, the frantic search for a cure—all while Lucy and Sam played, oblivious to the urgency.

The following days were a whirlwind. Ethan persuaded Clara to take the children for a full checkup and lab tests. As they waited, Ethan shared
with Clara snippets from the future—their family struggles, the joys, and sorrows. He watched as realization dawned on her face about the 
fragility of life. 

A week later, the tests came back. Bit by bit, the pieces started to fall into place. With early intervention, the family received treatment 
from a specialist flown in from the city. The miraculous news hit them like a fresh spring breeze—the strain had been captured, and a vaccine 
was developed in record time. The disease had not yet had a chance to strike them down. 

As Ethan watched them heal and grow stronger, he felt a pang of bittersweet longing. He had a profound understanding of time, of its 
relentless march and its moments of beauty. One evening, as the sun dipped below the horizon, he returned to the woods where the orb lay 
hidden. 

Clutching it tightly, he whispered a silent thanks for the second chance before burying it deep beneath the earth. He had come for a purpose, 
and now, with a family he cherished more than anything, he was ready to live in the present.

Years later, as he read to Lucy and Sam in the sprawling garden of their home, Ethan smiled fondly at Clara. Life was fragile and precious, 
but, as he had learned, so was hope. He vowed never to forget the lessons of time and love, believing deeply that every moment shared was the 
greatest adventure of all.
```

</details>

```python
from zyx import completion

response = completion(
    "Write a story about a man who travels through time to save his family from a deadly disease.",
    optimize = "costar" # Optimize with COSTAR
)

print(response.choices[0].message.content)
```

<details closed>
<summary>Output</summary>

```bash
# OUTPUT
# Chapter 1: A World in Shadows

In the year 2142, the world had crumbled under the weight of a relentless epidemic. A once-vibrant society now flickered like a dying ember, 
choked by despair and grief. Ethan was no stranger to this darkness; it haunted his every waking moment. His family—his wife Megan and their 
ten-year-old son Lucas—had been caught in the relentless grip of a disease that science had yet to understand, let alone cure. 

Standing in the crumbling remnants of their home, Ethan clutched a faded photograph of the three of them—smiling amidst the towering oaks of 
their backyard during a summer picnic, the sun breathing life into their laughter. Those moments now felt like whispers from a different life,
swallowed by an abyss of hopelessness. Each day, he watched as the light dimmed in Megan’s eyes and as Lucas succumbed to fits of coughing 
that rattled his small frame. 

Desperation clawed at Ethan’s chest, gnawing away at his resolve. But buried deep within him lay a flicker of an idea—rumors of time travel, 
of bending the very fabric of reality to find a remedy in the past. He had to try.

# Chapter 2: The Chronosphere

Ethan's journey began in the bowels of an ancient research facility, long abandoned yet whispered about among the few who still dared to 
dream. There, he found tales of a machine, the Chronosphere—a device that could pierce the veil of time.

After days of rummaging through tattered blueprints and decoding fragmented notes left behind by hopeless scientists, Ethan unearthed a 
working prototype, a hissing contraption of gears and glowing screens that threatened to expel him into the unknown. Steeling himself, he 
whispered his intentions, “I need to save my family.” 

With a flick of a switch, the room around him dimmed. An orchestra of energy hummed, thrumming with the promise of travel as spirals of light 
enveloped him. One moment he was standing in the desolation of 2142, and the next, he was hurtling through time, the colors around him 
swirling like oil on water. 

# Chapter 3: The Success of the Past

Ethan landed in the year 1962, the air thick with the sweetness of blooming flowers and the hum of life—a stark contrast to the sterile 
silence of his home. The sun blazed in a way it hadn’t done in a long time. His heart raced with both exhilaration and anxiety, but he was 
focused—his mission was to find that elusive cure.

He found solace in the bustling energy of the city, absorbing the sights and sounds that felt foreign yet familiar. Guided by flickering 
fragments of his research, he sought out a brilliant scientist, Dr. Elizabeth Hale—a name that stood out in the annals of hope for a disease 
curiously similar to the one afflicting his family.

As he approached the tall brick laboratory, he felt the weight of his desperation intensify. Would she believe him? Could she help?

# Chapter 4: Connections to History

Ethan's heart pounded as he introduced himself, weaving a tale flirted with half-truths while holding the fragile essence of faith. Dr. Hale 
was skeptical initially, her brow raised in doubt. Yet, as Ethan shared his knowledge of the disease—its symptoms, its evolution—her 
expression softened. The allure of a mysterious visitor from the future intrigued her.

Days turned into weeks as they worked side by side. The bond grew, not just a collaboration of science, but a friendship forged in shared 
purpose. They found laughter in long nights spent pouring over research, and for the first time, Ethan began to feel a flicker of hope.

But with each day, the cruel clock reminded Ethan of the ticking minutes that threatened his family’s existence. He was torn between two 
worlds—one where everything was a struggle and another where he could rewrite the script of fate.

# Chapter 5: The Breaking Point

As the breakthroughs arose, so did challenges. There were moments where he would catch himself staring longingly at photographs of Megan and 
Lucas—haunting visions that circled his mind like vultures. The tension mounted, and with it came ethical dilemmas. Each decision he made had 
the potential to alter the course of history. Could he risk changing significant events, or worse, losing his family altogether if he 
faltered?

The weight of his choices bore down heavily, but he couldn’t afford hesitation. Finally, after a whirlwind of sleepless nights and 
experiments, they developed a serum—a concoction that shimmered with promise.

# Chapter 6: A Heartbreaking Choice

Ethan stood at the threshold of history as he prepared to return. Dr. Hale expressed her unease. “You have to understand, if you succeed in 
making this discovery, it could alter everything. Are you willing to let the past unfold as it is?” 

Ethan’s heart ached. “I have to try. My family—my son needs me.” The words hung in the air like a promise, heavy and unwavering.

With the serum secured, Ethan activated the Chronosphere once again, this time with the weight of a lifetime on his shoulders. He surrounded 
himself with thoughts of Lucas and Megan, channeling every ounce of his love into the device, feeling the pull of time drawing him back to his
crumbled present.

# Chapter 7: The Race Against Time

The whirlwind of colors enveloped him once more, the familiar sensation of being flung through space. He arrived home to a desolate silence, a
stillness that mirrored the void in his heart. 

Rushing through the door, he found Megan sprawled on the couch, feverish and pale—the epitome of fragility. Lucas lay sleeping, his small 
chest rising and falling, the sound unsteady.

“Please,” Ethan whispered, his hands trembling as he prepared the serum. Would it be too late? He administered it without hesitation, pouring 
every ounce of his fear and love into the moment. 

As the seconds ticked agonizingly by, he held his breath. Slowly, Megan's eyes fluttered open, confusion etching her features. “Ethan?” 

# Chapter 8: A New Dawn

The wakeful hours turned into a miracle; he watched as the color gradually returned to Megan’s cheeks and, in time, Lucas awoke with his 
once-thin voice growing stronger. Joy overrode the remnants of despair—their reunited laughter echoed through the hall, worming its way into 
every corner of their home.

As Ethan held them close, he knew things would never be the same. He had altered history, yet in this newfound brightness, he felt intense 
gratitude—an understanding of love that transcended every obstacle.

But lingering in the back of his mind was an echo of Dr. Hale’s caution. There was no telling what ripples his actions would cause in the 
future. Still, in that moment, as Lucas and Megan filled their home with light, Ethan realized his sacrifice was not for nothing. He had 
traveled through time not just to save his family, but to awaken the unyielding strength of love.

And in the chaotic dance of time and fate, love had authored a new chapter worth embracing.
```

</details>

## API Reference

::: zyx._client.completion.completion