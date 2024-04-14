# Awesome-LLM-Agent
[![PR Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/zjwu0522/Awesome-LLM-Agent/pulls)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Welcome to our comprehensive collection on LLM-based agents, with an emphasis on **reasoning, memory, action, and related applications**. Dive into a diverse array of academic papers, benchmarks, and open-source projects that explore the depths of LLM capabilities. This repo is actively maintained and frequently updated 🧑‍💻. Stay tuned for the latest advancements in the field 🚀!

## Table of Contents

- [Papers](#papers)
  - [Survey](#survey-)
  - [Benchmark](#benchmark-)
  - [Reasoning and Prompt Engineering](#reasoning-and-prompt-engineering-)
  - [Memory and Retrieval Augmented Generation](#memory-and-retrieval-augmented-generation-️)
  - [Action and Tool Using](#action-and-tool-using-️)
  - [Agent Fine-Tuning](#agent-fine-tuning-)
  - [LLM Fine-Tuning](#llm-fine-tuning-)
  - [Applications](#applications-)
    - [Web Agents](#web-agents)
    - [Recommender Agents](#recommender-agents)
    - [Paper Review Agents](#paper-review-agents)
    - [Trading Agents](#trading-agents)
    - [Others](#others)
- [Open-Source Projects](#open-source-projects)
  - [LLM Platform](#llm-platform)
  - [Multi-Agent Framework](#multi-agent-framework)
  - [Vector Database](#vector-database)

## Papers

> 🔥 for papers with >100 citations or repositories with >500 stars.
>
> 🚀 for papers with >300 citations or repositories with >1500 stars.

### Survey 🔍

- 🔥 *(arXiv 2023.08)* A Survey on Large Language Model based Autonomous Agents [[Paper](https://arxiv.org/abs/2308.11432)] [[GitHub](https://github.com/Paitesanshi/LLM-Agent-Survey)]
- 🔥 *(arXiv 2023.09)* The Rise and Potential of Large Language Model Based Agents: A Survey [[Paper](https://arxiv.org/abs/2309.07864)] [[GitHub](https://github.com/WooooDyy/LLM-Agent-Paper-List)]
- *(arXiv 2023.10)* AI Alignment: A Comprehensive Survey [[Paper](https://arxiv.org/abs/2310.19852)]
- 🔥 (arXiv 2023.12) Retrieval-Augmented Generation for Large Language Models: A Survey [[Paper](https://arxiv.org/abs/2312.10997)] [[GitHub](https://github.com/Tongji-KGLLM/RAG-Survey)]
- *(arXiv 2024.01)* Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security [[Paper](https://arxiv.org/abs/2401.05459)] [[GitHub](https://github.com/MobileLLM/Personal_LLM_Agents_Survey)]
- *(arXiv 2024.01)* Large Language Model based Multi-Agents: A Survey of Progress and Challenges [[Paper](https://arxiv.org/abs/2402.01680)] [[GitHub](https://github.com/taichengguo/LLM_MultiAgents_Survey_Papers)]
- 🔥 *(TMLR'2024)* Cognitive Architectures for Language Agents [[Paper]()] [[GitHub](https://github.com/ysymyth/awesome-language-agents)]
- *(arXiv 2024.01)* Agent AI: Surveying the Horizons of Multimodal Interaction [[Paper](https://arxiv.org/abs/2401.03568)]

### Benchmark 📈

- 🔥 *(NeurIPS'2022)* WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents [[Paper](https://arxiv.org/abs/2207.01206)] [[GitHub](https://github.com/princeton-nlp/WebShop)] [[Website](https://webshop-pnlp.github.io/)]
- *(EACL'2023)* MTEB: Massive Text Embedding Benchmark [[Paper](https://arxiv.org/abs/2210.07316)] [[GitHub](https://github.com/embeddings-benchmark/mteb)] [[Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)]
- *(EMNLP'2023)* API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs [[Paper](https://arxiv.org/abs/2304.08244)] [[GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)]
- 🔥 *(NeurIPS'2023)* PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change [[Paper](https://arxiv.org/abs/2206.10498)] [[GitHub](https://github.com/karthikv792/LLMs-Planning)]
- *(NeurIPS'2023)* ToolQA: A Dataset for LLM Question Answering with External Tools [[Paper](https://arxiv.org/abs/2306.13304)] [[GitHub](https://github.com/night-chen/ToolQA)]
- *(arXiv 2023.09)* Benchmarking Large Language Models in Retrieval-Augmented Generation [[Paper](https://arxiv.org/abs/2309.01431)] [[GitHub](https://github.com/chen700564/RGB)]
- 🔥 *(ICLR'2024)* WebArena: A Realistic Web Environment for Building Autonomous Agents [[Paper](https://arxiv.org/abs/2307.13854)] [[GitHub](https://github.com/web-arena-x/webarena)] [[Website](https://webarena.dev/)]
- 🚀 *(ICLR'2024)* AgentBench: Evaluating LLMs as Agents [[Paper](https://arxiv.org/abs/2308.03688)] [[Github](https://github.com/THUDM/AgentBench)] [[Website](https://llmbench.ai/agent)]
- 🔥 *(ICLR'2024)* SWE-bench: Can Language Models Resolve Real-World Github Issues? [[Paper](https://arxiv.org/abs/2310.06770)] [[GitHub](https://github.com/princeton-nlp/SWE-bench)] [[Website](https://www.swebench.com/)]
- *(arXiv 2023.10)* Benchmarking Large Language Models As AI Research Agents [[Paper](https://arxiv.org/abs/2310.03302)] [[Github](https://github.com/snap-stanford/MLAgentBench)]
- *(arXiv 2023.12)* T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step [[Paper](https://arxiv.org/abs/2312.14033)] [[GitHub](https://github.com/open-compass/T-Eval?tab=readme-ov-file)] [[Website](https://open-compass.github.io/T-Eval/)]
- *(arXiv 2024.01)* VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks [[Paper](https://arxiv.org/abs/2401.13649)] [[GitHub](https://github.com/web-arena-x/visualwebarena)] [[Website](https://jykoh.com/vwa)]
- *(arXiv 2024.04)* AutoWebGLM: Bootstrap And Reinforce A Large Language Model-based Web Navigating Agent [[Paper](https://arxiv.org/abs/2404.03648)] [[GitHub](https://github.com/THUDM/AutoWebGLM)]

### Reasoning and Prompt Engineering 💡

- 🚀 *(NeurIPS'2022)* Chain-of-Thought Prompting Elicits Reasoning in Large Language Models [[Paper](https://arxiv.org/abs/2201.11903)]
- 🚀 *(ICLR'2023)* ReAct: Synergizing Reasoning and Acting in Language Models [[Paper](https://arxiv.org/abs/2210.03629)] [[GitHub](https://github.com/ysymyth/ReAct)] [[Website](https://react-lm.github.io/)]
- 🔥 *(arXiv 2023.05)* ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models [[Paper](https://arxiv.org/abs/2305.18323)] [[GitHub](https://github.com/billxbf/ReWOO)]
- 🔥 *(EMNLP'2023)* Reasoning with Language Model is Planning with World Model [[Paper](https://arxiv.org/abs/2305.14992)] [[GitHub](https://github.com/Ber666/llm-reasoners)]
- 🚀 *(NeurIPS'2023)* Tree of Thoughts: Deliberate Problem Solving with Large Language Models [[Paper](https://arxiv.org/abs/2305.10601)] [[GitHub](https://github.com/princeton-nlp/tree-of-thought-llm)]
- 🚀 *(NeurIPS'2023)* Reflexion: Language Agents with Verbal Reinforcement Learning [[Paper](https://arxiv.org/abs/2303.11366)] [[GitHub](https://github.com/noahshinn/reflexion)]
- 🔥 *(NeurIPS'2023)* Self-Refine: Iterative Refinement with Self-Feedback [[Paper](https://arxiv.org/abs/2303.17651)] [[GitHub](https://github.com/madaan/self-refine)]
- 🚀 *(arXiv 2023.08)* Graph of Thoughts: Solving Elaborate Problems with Large Language Models [[Paper]()] [[GitHub](https://github.com/spcl/graph-of-thoughts)]
- *(ICLR'2024)* Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph [[Paper](https://arxiv.org/abs/2307.07697)] [[GitHub](https://github.com/IDEA-FinAI/ToG)]
- *(ICLR'2024)* Thought Propagation: An Analogical Approach to Complex Reasoning with Large Language Models [[Paper](https://arxiv.org/abs/2310.03965)] [[GitHub](https://github.com/Samyu0304/thought-propagation)]
- *(arXiv 2024.01)* Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts [[Paper](https://arxiv.org/abs/2401.14295)]
- *(arXiv 2024.01)* Self-Rewarding Language Models [[Paper](https://arxiv.org/abs/2401.10020)]

### Memory and Retrieval Augmented Generation ⚙️

- 🚀 *(PMLR'2022)* Improving language models by retrieving from trillions of tokens [[Paper](https://arxiv.org/abs/2112.04426)] [[GitHub](https://github.com/lucidrains/RETRO-pytorch)]
- *(arXiv 2023.01)* REPLUG: Retrieval-Augmented Black-Box Language Models [[Paper](https://arxiv.org/abs/2301.12652)]
- 🔥 *(EMNLP'2023)* Active Retrieval Augmented Generation [[Paper](https://arxiv.org/abs/2305.06983)] [[GitHub](https://github.com/jzbjyb/FLARE)]
- *(EMNLP'2023 findings)* Self-Knowledge Guided Retrieval Augmentation for Large Language Models [[Paper](https://arxiv.org/abs/2310.05002)]
- 🚀 *(ICLR'2024)* DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines [[Paper](https://arxiv.org/abs/2310.03714)] [[GitHub](https://github.com/stanfordnlp/dspy)]
- *(ICLR'2024)* Retrieval meets Long Context Large Language Models [[Paper](https://arxiv.org/abs/2310.03025)]
- 🔥 *(ICLR'2024)* Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection [[Paper](https://arxiv.org/abs/2310.11511)] [[GitHub](https://github.com/AkariAsai/self-rag)] [[Website](https://selfrag.github.io/)]
- *(NAACL'2024)* REST: Retrieval-Based Speculative Decoding [[Paper](https://arxiv.org/abs/2311.08252)] [[GitHub](https://github.com/FasterDecoding/REST)]
- *(arXiv 2023.11)* Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models [[Paper](https://arxiv.org/abs/2311.09210)]
- *(arXiv 2023.03)* RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-Horizon Generation [[Paper](https://arxiv.org/abs/2403.05313)] [[GitHub](https://github.com/CraftJarvis/RAT)] [[Website](https://craftjarvis-jarvis.github.io/RAT)] [[Demo](https://huggingface.co/spaces/jeasinema/RAT)]

### Action and Tool Using 🛠️

- 🔥 *(CVPR'2023)* Visual Programming: Compositional visual reasoning without training [[Paper](https://arxiv.org/abs/2211.11559)] [[GitHub](https://github.com/allenai/visprog)]
- 🚀 *(NeurIPS'2023)* Toolformer: Language Models Can Teach Themselves to Use Tools [[Paper](https://arxiv.org/abs/2302.04761)] [[GitHub](https://github.com/conceptofmind/toolformer)]
- *(arXiv 2023.05)* ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings [[Paper](https://arxiv.org/abs/2305.11554)] [[GitHub](https://github.com/Ber666/ToolkenGPT)]

- *(arXiv 2023.06)* ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases [[Paper](https://arxiv.org/abs/2306.05301)] [[GitHub](https://github.com/tangqiaoyu/ToolAlpaca)]
- 🚀 *(ICLR'2024)* ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs [[Paper](https://arxiv.org/abs/2307.16789)] [[GitHub](https://github.com/OpenBMB/ToolBench)]
- 🚀 *(TMLR'2024)* Voyager: An Open-Ended Embodied Agent with Large Language Models [[Paper](https://arxiv.org/abs/2305.16291)] [[GitHub](https://github.com/MineDojo/Voyager)]

### Agent Fine-Tuning 🤖

- 🔥 *(arXiv 2023.10)* AgentTuning: Enabling Generalized Agent Abilities for LLMs [[Paper](https://arxiv.org/pdf/2310.12823.pdf)] [[GitHub](https://github.com/THUDM/AgentTuning)] [[Website](https://thudm.github.io/AgentTuning/)]

- *(arXiv 2023.10)* FireAct: Toward Language Agent Fine-tuning [[Paper](https://arxiv.org/abs/2310.05915)] [[GitHub](https://github.com/anchen1011/FireAct)] [[Website](https://fireact-agent.github.io/)]
- *(arXiv 2024.02)* AUTOACT: Automatic Agent Learning from Scratch via Self-Planning [[Paper](https://arxiv.org/abs/2401.05268)] [[GitHub](https://github.com/zjunlp/AutoAct)] [[Website](https://www.zjukg.org/project/AutoAct/)]

- *(arXiv 2024.03)* Agent Lumos: Unified and Modular Training for Open-Source Language Agents [[Paper](https://arxiv.org/abs/2311.05657)] [[GitHub](https://github.com/allenai/lumos)] [[Website](https://allenai.github.io/lumos/)]

- *(arXiv 2024.03)* Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for Large Language Models [[Paper](https://arxiv.org/abs/2403.12881v1)] [[GitHub](https://github.com/InternLM/Agent-FLAN)] [[Website](https://internlm.github.io/Agent-FLAN/)]

### LLM Fine-Tuning 🧠

- 🚀 *(NeurIPS'2022)* Training language models to follow instructions with human feedback [[Paper](https://arxiv.org/abs/2203.02155)] [[GitHub](https://github.com/openai/following-instructions-human-feedback)]
- 🚀 *(NeurIPS'2023)* Direct Preference Optimization: Your Language Model is Secretly a Reward Model [[Paper](https://arxiv.org/abs/2305.18290)] [[GitHub](https://github.com/eric-mitchell/direct-preference-optimization)]
- *(arXiv 2024.01)* Self-Rewarding Language Models [[Paper](https://arxiv.org/abs/2401.10020)] [[GitHub](https://github.com/gagan3012/self_rewarding_models)]
- *(arXiv 2024.02)* Noise Contrastive Alignment of Language Models with Explicit Rewards [[Paper](https://arxiv.org/abs/2402.05369)] [[GitHub](https://github.com/thu-ml/Noise-Contrastive-Alignment)]

### Applications 💻

#### Web Agents

- 🔥 *(NeurIPS'2023)* Mind2Web: Towards a Generalist Agent for the Web [[Paper](https://arxiv.org/abs/2306.06070)] [[GitHub](https://github.com/OSU-NLP-Group/Mind2Web)]
- *(NeurIPS'2023 workshops)* LASER: LLM Agent with State-Space Exploration for Web Navigation [[Paper](https://arxiv.org/abs/2309.08172)]
- *(ICLR'2024)* A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis [[Paper](https://arxiv.org/abs/2307.12856)] [[GitHub]()]

#### Recommender Agents

- *(arXiv 2023.08)* RecMind: Large Language Model Powered Agent For Recommendation [[Paper](https://arxiv.org/abs/2308.14296)]

- *(arXiv 2023.10)* On Generative Agents in Recommendation [[paper](https://arxiv.org/abs/2310.10108)] [[GitHub](https://github.com/LehengTHU/Agent4Rec)]
- *(arXiv 2023.10)* AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems [[Paper](https://arxiv.org/abs/2310.09233)]

#### Paper Review Agents

- *(arXiv 2023.10)* Can large language models provide useful feedback on research papers? A large-scale empirical analysis [[Paper](https://arxiv.org/abs/2310.01783)] [[GitHub](https://github.com/Weixin-Liang/LLM-scientific-feedback)]
- *(arXiv 2024.01)* MARG: Multi-Agent Review Generation for Scientific Papers [[Paper](https://arxiv.org/abs/2401.04259)] [[GitHub](https://github.com/allenai/marg-reviewer)]
- *(arXiv 2024.02)* Reviewer2: Optimizing Review Generation Through Prompt Generation [[Paper](https://arxiv.org/abs/2402.10886)] [[GitHub](https://github.com/ZhaolinGao/Reviewer2)]
- *(CHI'2024)* A Design Space for Intelligent and Interactive Writing Assistants [[Paper](https://arxiv.org/abs/2403.14117)] [[GitHub](https://github.com/writing-assistant/writing-assistant.github.io)] [[Website](https://writing-assistant.github.io/)]

#### Trading Agents

- *(ICLR'2024)* SocioDojo: Building Lifelong Analytical Agents with Real-world Text and Time Series [[Paper](https://openreview.net/forum?id=s9z0HzWJJp)] [[GitHub](https://github.com/chengjunyan1/SocioDojo)]
- *(ICLR'2024 workshops)* FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design [[Paper](https://openreview.net/forum?id=sstfVOwbiG&referrer=%5Bthe%20profile%20of%20Jordan%20W.%20Suchow%5D(%2Fprofile%3Fid%3D~Jordan_W._Suchow1))]

#### Others

- 🚀 *(UIST'2023)* Generative Agents: Interactive Simulacra of Human Behavior [[Paper](https://arxiv.org/abs/2304.03442)] [[GitHub](https://github.com/joonspk-research/generative_agents)]

- 🚀 *(NeurIPS'2023)* HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face. [[Paper](https://arxiv.org/abs/2303.17580)] [[GitHub](https://github.com/microsoft/JARVIS)]

- 🔥 *(ICLR'2024)* ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving [[Paper](https://arxiv.org/abs/2309.17452)] [[GitHub](https://github.com/microsoft/ToRA)]

- *(arXiv 2023.04)* Octopus v2: On-device language model for super agent [[Paper](https://arxiv.org/abs/2404.01744v1)]

  

## Open-Source Projects

### LLM Platform

|     Title      |                             Link                             |                         Description                          |
| :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    FastChat    | [lm-sys/FastChat](https://github.com/lm-sys/FastChat)  ![GitHub stars](https://img.shields.io/github/stars/lm-sys/FastChat) | An open platform for training, serving, and evaluating large language models. Release repo for Vicuna and Chatbot Arena. |
|  🦜️🔗 LangChain  | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) ![GitHub stars](https://img.shields.io/github/stars/langchain-ai/langchain) |        🦜🔗 Build context-aware reasoning applications         |
| 🗂️ LlamaIndex 🦙 | [run-llama/llama_index](https://github.com/run-llama/llama_index) ![GitHub stars](https://img.shields.io/github/stars/run-llama/llama_index) |   LlamaIndex is a data framework for your LLM applications   |
| LLaMA-Factory  | [hiyouga/Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) ![GitHub stars](https://img.shields.io/github/stars/hiyouga/Llama-Factory) |           Unify Efficient Fine-Tuning of 100+ LLMs           |
|    Petals🌸     | [bigscience-workshop/petals](https://github.com/bigscience-workshop/petals) ![GitHub stars](https://img.shields.io/github/stars/bigscience-workshop/petals) | 🌸 Run LLMs at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading |
| Open-Assistant | [LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant) ![GitHub stars](https://img.shields.io/github/stars/LAION-AI/Open-Assistant) | OpenAssistant is a chat-based assistant that understands tasks, can interact with third-party systems, and retrieve information dynamically to do so. |

### Multi-Agent Framework

|     Title     |                             Link                             |                         Description                          |
| :-----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    CAMEL🐫     | [camel-ai/camel](https://github.com/camel-ai/camel) ![GitHub stars](https://img.shields.io/github/stars/camel-ai/camel) | 🐫 CAMEL: Communicative Agents for “Mind” Exploration of Large Language Model Society |
|    AutoGen    | [microsoft/autogen](https://github.com/microsoft/autogen) ![GitHub stars](https://img.shields.io/github/stars/microsoft/autogen) |           A programming framework for agentic AI.            |
| 🤖 AgentVerse🪐 | [OpenBMB/AgentVerse](https://github.com/OpenBMB/AgentVerse) ![GitHub stars](https://img.shields.io/github/stars/OpenBMB/AgentVerse) | 🤖 AgentVerse 🪐 is designed to facilitate the deployment of multiple LLM-based agents in various applications, which primarily provides two frameworks: task-solving and simulation |

### Vector Database

| Title  |                             Link                             |                         Description                          |
| :----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Chroma | [chroma-core/chroma](https://github.com/chroma-core/chroma) ![GitHub stars](https://img.shields.io/github/stars/chroma-core/chroma) |         the AI-native open-source embedding database         |
| Faiss  | [facebookresearch/faiss](https://github.com/facebookresearch/faiss)![GitHub stars](https://img.shields.io/github/stars/facebookresearch/faiss) | A library for efficient similarity search and clustering of dense vectors. |

