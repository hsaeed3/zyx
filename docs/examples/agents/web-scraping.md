# Web Scraping

`zyx` comes with a quick way to scrape the web for information using agentic reasoning through LLMs with the `scrape()` function.

> Node-based scraping coming soon for more complex scraping tasks.

---

### Simple Scraping

```python
from zyx import scrape

result = scrape(
    "The latest & hottest AI hardware",
    model = "openai/gpt-4o"
    workers = 5,
    max_results = 3
)

print(result)
```

<details>
  <summary>Output</summary>
  ```bash
  ...
  'summary': "The AI hardware market has seen rapid advancements and fierce competition, with several key players releasing
  innovative products to meet the growing demand for AI capabilities. Here are the most notable companies and their contributions to AI hardware
  as of 2024:\n\n1. **Nvidia**: A leader in the AI hardware space, Nvidia's chips like the A100 and H100 are critical for data centers. The
  recent introduction of the H200 and B200 chips, along with the Grace Hopper superchip, emphasizes Nvidia's focus on performance and
  scalability in AI applications.\n\n2. **AMD**: AMD continues to compete with Nvidia, having launched its MI300 series of AI chips, which rival
  Nvidia's offerings in terms of memory capacity and bandwidth. The new Zen 5 CPU microarchitecture enhances AMD's capabilities in AI
  workloads.\n\n3. **Intel**: Intel has introduced its Xeon 6 processors and the Gaudi 3 AI accelerator, which aims to improve processing
  efficiency. Intel's longstanding presence in the CPU market is now complemented by its focus on AI-specific hardware.\n\n4. **Alphabet
  (Google)**: With its Cloud TPU v5p and the recently announced Trillium TPU, Alphabet is committed to developing powerful AI chips tailored for
  large-scale machine learning tasks.\n\n5. **Amazon Web Services (AWS)**: AWS has shifted towards chip production with its Trainium and
  Inferentia chips, designed for training and deploying machine learning models, respectively. Their latest instance types offer significant
  improvements in memory and processing power.\n\n6. **Cerebras Systems**: Known for its wafer-scale engine, the WSE-3, Cerebras has achieved
  remarkable performance with its massive core count and memory bandwidth, making it a strong contender in the AI hardware market.\n\n7.
  **IBM**: IBM's AI Unit and the upcoming NorthPole chip focus on energy efficiency and performance improvements, aiming to compete with
  existing AI processors.\n\n8. **Qualcomm**: Although newer to the AI hardware scene, Qualcomm's Cloud AI 100 chip has shown competitive
  performance against Nvidia, particularly in data center applications.\n\n9. **Tenstorrent**: Founded by a former AMD architect, Tenstorrent
  focuses on scalable AI hardware solutions, including its Wormhole processors.\n\n10. **Emerging Startups**: Companies like Groq, SambaNova
  Systems, and Mythic are also making strides in the AI hardware space, offering specialized solutions for AI workloads.\n\nIn summary, the
  competitive landscape for AI hardware is characterized by rapid innovation, with established tech giants and emerging startups alike vying to
  create the most powerful and efficient AI chips. This ongoing evolution is driven by the increasing demands of AI applications, particularly
  in data centers and for large-scale machine learning models.",
    'evaluation': {
        'is_successful': True,
        'explanation': 'The summary effectively captures the current landscape of AI hardware as of 2024, highlighting key players and
  their contributions. It provides relevant details about the advancements made by major companies like Nvidia, AMD, Intel, and others, which
  directly relates to the query about the latest and hottest AI hardware. The structure is clear, listing companies and their innovations,
  making it easy for readers to understand the competitive dynamics in the AI hardware market. Overall, the summary is comprehensive, relevant,
  and well-organized, making it a successful response to the query.',
        'content': None
    }
  }
  },
  messages=[]
  )
  ```
</details>

---

## API Reference

::: zyx.resources.completions.agents.scrape.scrape
