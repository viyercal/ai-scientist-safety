Title: Improving Multi-Step Tool-Use Reasoning via Adaptive Curriculum and Self-Reflection

Keywords
tool-use, function calling, curriculum learning, self-reflection, large language models

TL;DR
Can we get LLMs to actually use tools in long, multi-step workflows without derping out? We propose combining an adaptive difficulty curriculum with self-reflection on failed tool traces to make tool-using models more robust, efficient, and generalizable.

Abstract
Language models augmented with tools and function-calling interfaces show strong capabilities on tasks like retrieval, code execution, and environment interaction. Yet they often fail in scenarios requiring multi-step tool-use, recovery from earlier mistakes, and correct composition of several tools in sequence. We propose a training framework that combines an adaptive curriculum over tool-use difficulty with self-reflective error correction. The curriculum gradually increases complexity along axes such as number of tools, depth of the call graph, and presence of noisy or distractor tools. For each episode, the model generates a tool-use trace; when the trace fails, a self-reflection module produces a diagnosis of what went wrong (e.g., incorrect tool choice, wrong arguments, missing intermediate step) and a revised plan, which is then used as an additional supervision signal during fine-tuning.

We hypothesize that this joint curriculum-plus-reflection approach will improve multi-step tool-use success rates, reduce brittle failure patterns, and generalize better to unseen tool topologies. We evaluate on synthetic tool-use benchmarks with controllable DAGs, as well as on existing function-calling and tool-use datasets, comparing against baselines trained with static datasets and no reflection. We further analyze learned behaviors through error-type clustering and examine robustness to spurious or distracting tools. Our results aim to clarify how curriculum design and self-critique interact to shape procedural competence in tool-using language models and to highlight remaining failure modes that persist even under this enhanced training regime.