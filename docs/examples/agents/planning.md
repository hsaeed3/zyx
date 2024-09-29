# Planning

A big piece in chain of thought reasoning is planning. The `zyx` library comes with a `plan()` function, allowing helpful utility
for planning for agentic tasks.

---

## Tree of Thought Planning

[**Paper**](https://arxiv.org/abs/2305.10601)

```python
from zyx import plan
from pydantic import BaseModel
from typing import List

goal = "Create a marketing strategy for a new smartphone"
tot_plan = plan(
    goal,
    steps=5,
    verbose = True
)

print(tot_plan)
```

<details>
  <summary>Output</summary>
  ```bash
  Plan(
      tasks=[
          Task(description='### Actionable Tasks for Integrated Multi-Channel Marketing Campaign', details=None),
          Task(description='1. **Develop a Comprehensive Marketing Plan:**', details=None),
          Task(
              description='- **Define Objectives:** Establish clear goals for the campaign, such as increasing brand awareness, driving traffic
  to the website, or achieving a specific sales target.',
              details=None
          ),
          Task(
              description='- **Identify Target Audience:** Conduct research to define the target audience segments, including demographics,
  interests, and behaviors.',
              details=None
          ),
          Task(
              description='- **Budget Allocation:** Determine the budget for each channel (social media, email, PPC, etc.) and allocate
  resources accordingly.',
              details=None
          ),
          Task(
              description="- **Key Messages:** Craft core messages that resonate with the target audience and align with the brand's voice.",
              details=None
          ),
          Task(
              description='- **Timeline:** Create a timeline for the campaign, including key milestones and deadlines for each phase of the
  marketing activities.',
              details=None
          ),
          Task(description='2. **Create Engaging Content:**', details=None),
          Task(
              description='- **Content Calendar:** Develop a content calendar that outlines what content will be published on which channels and
  when.',
              details=None
          ),
          Task(description='- **Tailored Content Creation:** Produce high-quality content tailored for each platform, such as:', details=None),
          Task(
              description='- **Social Media Posts:** Eye-catching graphics and engaging captions for platforms like Instagram, Facebook, and
  Twitter.',
              details=None
          ),
          Task(
              description="- **Email Newsletters:** Informative and visually appealing emails that highlight the smartphone's features and
  promotions.",
              details=None
          ),
          Task(
              description="- **Blog Articles:** In-depth articles that provide insights into the smartphone's technology, benefits, and user
  experiences.",
              details=None
          ),
          Task(
              description="- **Video Ads:** Create short, engaging videos showcasing the smartphone's features and user testimonials.",
              details=None
          ),
          Task(description='3. **Leverage Influencer Partnerships:**', details=None),
          Task(
              description="- **Identify Influencers:** Research and compile a list of influencers who align with the brand's values and have a
  following that matches the target audience.",
              details=None
          ),
          Task(
              description='- **Outreach Strategy:** Develop a strategy for reaching out to influencers, including personalized messages and
  collaboration proposals.',
              details=None
          ),
          Task(
              description='- **Content Collaboration:** Work with influencers to create authentic content that showcases the smartphone, such as
  unboxing videos, reviews, or lifestyle posts.',
              details=None
          ),
          Task(
              description='- **Track Engagement:** Monitor the performance of influencer content to assess reach, engagement, and conversion
  rates.',
              details=None
          ),
          Task(description='4. **Implement Tracking and Analytics:**', details=None),
          Task(
              description='- **Set Up Tracking Tools:** Utilize tools like Google Analytics, social media insights, and email marketing
  analytics to track campaign performance.',
              details=None
          ),
          Task(
              description='- **Define KPIs:** Establish key performance indicators (KPIs) to measure success, such as website traffic,
  conversion rates, and social media engagement.',
              details=None
          ),
          Task(
              description='- **Real-Time Monitoring:** Implement real-time monitoring to assess the effectiveness of each channel and make
  adjustments as needed.',
              details=None
          ),
          Task(
              description='- **Post-Campaign Analysis:** After the campaign, conduct a thorough analysis of the data to evaluate what worked
  well and what can be improved for future campaigns.',
              details=None
          ),
          Task(description='5. **Launch and Promote the Campaign:**', details=None),
          Task(
              description='- **Coordinated Launch:** Ensure that all channels are prepared for the launch, with content scheduled and ready to
  go live simultaneously.',
              details=None
          ),
          Task(
              description='- **Engagement Strategies:** Implement strategies to engage the audience during the launch, such as live Q&A
  sessions, giveaways, or contests.',
              details=None
          ),
          Task(
              description="- **Consistent Messaging:** Maintain consistent messaging across all channels to reinforce the campaign's key
  messages and brand identity.",
              details=None
          ),
          Task(
              description='- **Follow-Up Promotions:** Plan follow-up promotions or content to sustain engagement and interest after the initial
  launch.',
              details=None
          )
      ]
  )
  ```
</details>

---

## Planning with a Custom BaseModel Input

```python
import zyx
from pydantic import BaseModel
from typing import List

class ResearchTask(BaseModel):
    topic: str
    resources: List[str]
    estimated_time: int

plan_model_tot = zyx.plan(
    ResearchTask,
    steps=4,
    verbose=True
)

print(plan_model_tot)
```

<details>
  <summary>Output</summary>
  ```bash
  ResearchTaskPlan(
      tasks=[
          ResearchTask(
              topic='Understanding Machine Learning',
              resources=[
                  "Textbook: 'Pattern Recognition and Machine Learning'",
                  "Online Course: 'Machine Learning by Stanford'",
                  "Research Paper: 'A survey of machine learning'"
              ],
              estimated_time=12
          ),
          ResearchTask(
              topic='Exploring Neural Networks',
              resources=[
                  "Textbook: 'Deep Learning' by Ian Goodfellow",
                  'Video Lecture Series on YouTube',
                  "Research Paper: 'Neural Networks for Machine Learning'"
              ],
              estimated_time=15
          ),
          ResearchTask(
              topic='Data Preprocessing Techniques',
              resources=[
                  "Online Article: 'A Comprehensive Guide to Data Preprocessing'",
                  'Video Tutorial on Data Cleaning',
                  "Research Paper: 'Data Preprocessing for Machine Learning'"
              ],
              estimated_time=8
          ),
          ResearchTask(
              topic='Evaluation Metrics in Machine Learning',
              resources=[
                  "Online Course: 'Evaluation Metrics for ML Models'",
                  "Textbook: 'The Elements of Statistical Learning'",
                  "Research Paper: 'Statistical Methods for Evaluating Learning Algorithms'"
              ],
              estimated_time=10
          )
      ]
  )
  ```
</details>

---

## API Reference

::: zyx.resources.completions.agents.plan.plan
