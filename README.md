# LLM Fine-Tuning From Scratch to Production

This repository is my hands-on exploration of how pretrained Large Language Models are adapted for specific tasks, domains, behaviors, and production use cases.

The goal is to understand fine-tuning beyond simply calling a training API.

I want to understand what happens when a pretrained LLM is taken from its original state and adapted using domain data, instruction datasets, parameter-efficient methods, and preference data.

The repository starts with the fundamentals and gradually moves toward practical and production-oriented fine-tuning workflows.

```text
Pretrained LLM
      ↓
Dataset Preparation
      ↓
Continued Pretraining
      ↓
Supervised Fine-Tuning
      ↓
PEFT
      ↓
LoRA
      ↓
QLoRA
      ↓
Preference Optimization
      ↓
DPO
      ↓
Evaluation
      ↓
Safety Evaluation
      ↓
Model Optimization
      ↓
Inference
      ↓
Deployment
      ↓
Monitoring
```

---

# Why Fine-Tuning?

A pretrained LLM already contains general language knowledge.

However, a general model may not perform well enough for a specific domain, task, style, or instruction-following requirement.

For example:

```text
General LLM
    ↓
Medical Documentation
    ↓
Legal Documents
    ↓
Customer Support
    ↓
Code Generation
    ↓
Enterprise Knowledge
```

Fine-tuning can adapt the model to a particular use case.

The important question is not simply:

```text
How do I fine-tune a model?
```

I want to understand:

```text
When should I fine-tune?

What data should I use?

How much data is required?

Which fine-tuning method should I choose?

Should I update all model parameters?

When should I use LoRA?

When should I use QLoRA?

When is continued pretraining better than SFT?

How do I evaluate whether fine-tuning actually improved the model?

What happens if fine-tuning makes the model worse?

How do I deploy the resulting model?
```

---

# Fine-Tuning Landscape

The repository covers multiple ways of adapting an LLM.

```text
                         PRETRAINED LLM
                              |
              +---------------+---------------+
              |                               |
              ↓                               ↓
      Continued Pretraining          Supervised Fine-Tuning
              |                               |
              |                    +----------+----------+
              |                    |                     |
              |                    ↓                     ↓
              |                   Full FT              PEFT
              |                                          |
              |                              +-----------+-----------+
              |                              |                       |
              |                              ↓                       ↓
              |                            LoRA                    QLoRA
              |                                                     
              +----------------------+-------------------------------+
                                     |
                                     ↓
                              Preference Data
                                     |
                                     ↓
                                    DPO
                                     |
                                     ↓
                              Evaluation
                                     |
                                     ↓
                                Production
```

---

# 1. Pretrained Model

Fine-tuning normally starts from an existing pretrained model.

```text
Large-Scale Text
      ↓
Pretraining
      ↓
Base LLM
      ↓
Fine-Tuning
      ↓
Task / Domain Specific LLM
```

I will explore:

* Base models
* Causal language models
* Model configurations
* Model parameters
* Context length
* Tokenizers
* Model loading
* Checkpoints

---

# 2. Continued Pretraining

Sometimes supervised instruction data is not the best starting point.

If the target domain contains specialized language, continued pretraining can expose the model to additional domain-specific text.

```text
Base LLM
   ↓
Domain Text
   ↓
Continued Pretraining
   ↓
Domain-Adapted LLM
```

Examples:

```text
General LLM
     ↓
Financial Documents
     ↓
Financial Domain Model
```

or:

```text
General LLM
     ↓
Medical Literature
     ↓
Medical Domain Model
```

I will explore:

* Domain-adaptive pretraining
* Continued causal language modeling
* Dataset preparation
* Learning-rate selection
* Catastrophic forgetting
* Domain adaptation

---

# 3. Supervised Fine-Tuning

Supervised Fine-Tuning, or SFT, trains a pretrained model using instruction and response examples.

Example:

```text
Instruction:

Explain gradient descent.

Response:

Gradient descent is an optimization algorithm...
```

The dataset becomes:

```text
Instruction
     +
Response
     ↓
Training Example
```

The model learns to produce the expected response.

---

# 4. SFT Data

Data quality is one of the most important parts of fine-tuning.

I will explore datasets containing:

```text
Instruction
Input
Context
Response
Conversation
System Message
User Message
Assistant Message
```

For conversational models:

```text
System
   ↓
User
   ↓
Assistant
   ↓
User
   ↓
Assistant
```

I will explore how these conversations are converted into the format expected by the model.

---

# 5. Chat Templates

Different LLMs use different conversation formats.

Conceptually:

```text
System
User
Assistant
```

may be converted into a model-specific token sequence.

I will explore:

* Chat templates
* Special tokens
* Role tokens
* BOS
* EOS
* Padding
* Attention masks

The goal is to understand what the model actually receives during training.

---

# 6. Loss Masking

Not every token necessarily needs to contribute to the training loss.

For example:

```text
User:
Explain machine learning.

Assistant:
Machine learning is...
```

We may want the model to learn primarily from the assistant response.

Conceptually:

```text
User Tokens       → Ignore
Assistant Tokens  → Calculate Loss
```

I will explore:

* Label masking
* Ignore index
* Instruction masking
* Response-only loss
* Full-sequence loss

---

# 7. Full Fine-Tuning

In full fine-tuning, most or all model parameters are updated.

```text
Base Model
    ↓
Update All / Most Parameters
    ↓
Fine-Tuned Model
```

Advantages:

* Maximum adaptation capacity
* Direct parameter updates

Limitations:

* High GPU memory usage
* Large optimizer state
* Expensive training
* Large model checkpoints

I will experiment with the practical tradeoffs.

---

# 8. Parameter-Efficient Fine-Tuning

PEFT reduces the number of parameters that need to be trained.

Instead of updating the entire model:

```text
Full Model
    ↓
Update Billions of Parameters
```

PEFT can do:

```text
Full Model
    ↓
Freeze Base Parameters
    ↓
Train Small Additional Parameters
```

This can significantly reduce:

* Memory
* Training cost
* Checkpoint size

---

# 9. LoRA

LoRA, or Low-Rank Adaptation, adds trainable low-rank matrices to selected model layers.

Conceptually:

```text
Original Weight
      +
Low-Rank Update
      ↓
Adapted Weight
```

Instead of directly changing:

```text
W
```

LoRA learns:

```text
W + ΔW
```

where the update can be represented using lower-rank matrices.

```text
ΔW = BA
```

This allows the base model to remain frozen while a much smaller number of parameters are trained.

---

# 10. LoRA Architecture

A simplified view:

```text
                  Input
                    |
              +-----+-----+
              |           |
              ↓           ↓
        Frozen Weight   LoRA A
              |           |
              |           ↓
              |         LoRA B
              |           |
              +-----+-----+
                    |
                    ↓
                  Output
```

I will explore LoRA placement in:

* Query projection
* Key projection
* Value projection
* Output projection
* Feed-forward layers

and compare different configurations.

---

# 11. LoRA Hyperparameters

Important LoRA parameters include:

```text
Rank
Alpha
Dropout
Target Modules
Learning Rate
```

I will experiment with how changing these values affects:

* Trainable parameters
* Memory usage
* Training behavior
* Final model quality

---

# 12. QLoRA

QLoRA combines quantization with LoRA.

The basic idea is:

```text
Pretrained Model
      ↓
Quantized Base Model
      ↓
LoRA Adapters
      ↓
Fine-Tuning
```

The base model can be stored in lower precision while LoRA parameters remain trainable.

This makes fine-tuning larger models possible with significantly lower memory requirements.

---

# 13. Quantization

I will explore the relationship between fine-tuning and quantization.

Topics include:

```text
FP32
FP16
BF16
INT8
INT4
```

and:

* Weight quantization
* Activation quantization
* Quantization-aware training
* Post-training quantization
* Quantization errors
* Memory savings

The important question is:

```text
How much quality am I trading for memory savings?
```

---

# 14. PEFT Methods

LoRA and QLoRA are not the only parameter-efficient approaches.

I will explore the broader PEFT ecosystem, including:

* LoRA
* QLoRA
* Adapters
* Prefix tuning
* Prompt tuning
* P-Tuning
* IA3

The goal is to understand the differences rather than treating PEFT as a single technique.

---

# 15. Dataset Quality

Fine-tuning quality depends heavily on the dataset.

I will investigate:

```text
Dataset
   ↓
Cleaning
   ↓
Filtering
   ↓
Deduplication
   ↓
Formatting
   ↓
Quality Evaluation
   ↓
Training
```

Important considerations include:

* Duplicate examples
* Incorrect answers
* Inconsistent formatting
* Low-quality responses
* Data leakage
* Imbalanced tasks
* Excessively long examples
* Synthetic data quality

---

# 16. Dataset Splitting

The dataset should normally be separated into different sets.

```text
Dataset
   ↓
+-------------------+
| Training          |
| Validation        |
| Test              |
+-------------------+
```

I will explore how these splits are created and why data leakage between them can produce misleading evaluation results.

---

# 17. Fine-Tuning Pipeline

The complete SFT pipeline becomes:

```text
Raw Dataset
     ↓
Cleaning
     ↓
Filtering
     ↓
Formatting
     ↓
Chat Template
     ↓
Tokenization
     ↓
Sequence Construction
     ↓
Batching
     ↓
LLM
     ↓
Loss
     ↓
Backpropagation
     ↓
Parameter Update
     ↓
Checkpoint
```

---

# 18. Training Configuration

I will experiment with:

```text
Learning Rate
Batch Size
Gradient Accumulation
Epochs
Warmup
Weight Decay
Sequence Length
Precision
Optimizer
Scheduler
LoRA Rank
LoRA Alpha
```

The purpose is to understand how each configuration affects training.

---

# 19. Catastrophic Forgetting

Fine-tuning can improve a model on a target task while reducing some of its original capabilities.

For example:

```text
General Model
      ↓
Heavy Domain Fine-Tuning
      ↓
Better Domain Performance
      ↓
Potential General Capability Loss
```

I will investigate:

* Learning rate
* Dataset size
* Training duration
* Domain shift
* Continued pretraining
* Regularization

---

# 20. Overfitting

A fine-tuned model can memorize the training dataset.

The training process may look like:

```text
Training Loss
      ↓
      ↓
      ↓
Very Low
```

while validation performance stops improving.

I will explore:

* Training loss
* Validation loss
* Evaluation metrics
* Dataset size
* Epochs
* Learning rate
* Regularization

---

# 21. Evaluation

Fine-tuning should not be judged only by training loss.

I will compare the model before and after fine-tuning.

```text
Base Model
    ↓
Evaluation
    ↓
Fine-Tuning
    ↓
Evaluation
    ↓
Compare
```

I will evaluate:

* Task accuracy
* Generation quality
* Instruction following
* Domain performance
* General capability
* Safety
* Hallucination behavior

---

# 22. Before vs After Evaluation

A major experiment in this repository will be:

```text
                   Base Model
                       |
                Evaluation
                       |
                       v
                 Fine-Tuning
                       |
                       v
                Fine-Tuned Model
                       |
                  Evaluation
                       |
                       v
                  Comparison
```

The objective is to determine whether fine-tuning actually improved the model.

---

# 23. Preference Optimization

SFT teaches the model using example responses.

Preference optimization introduces comparisons between responses.

Example:

```text
Prompt
  ↓
Response A
Response B
  ↓
Human / Preference Label
  ↓
Preferred Response
```

The model can then be optimized toward preferred behavior.

---

# 24. DPO

Direct Preference Optimization provides a way to optimize a model using preference data without requiring a traditional reinforcement-learning pipeline.

Conceptually:

```text
Prompt
   ↓
Chosen Response
Rejected Response
   ↓
Preference Objective
   ↓
Model Update
```

I will explore:

* Preference datasets
* Chosen responses
* Rejected responses
* Reference model
* DPO objective
* Temperature / beta
* Training stability
* Evaluation

---

# 25. SFT vs DPO

I will compare the different purposes.

```text
SFT

Instruction
   ↓
Expected Response
   ↓
Learn Behavior
```

versus:

```text
DPO

Prompt
   ↓
Chosen vs Rejected
   ↓
Learn Preference
```

The goal is to understand when each approach makes sense.

---

# 26. Continued Pretraining vs SFT

These techniques solve different problems.

```text
Continued Pretraining
        ↓
Learn More Domain Language
```

while:

```text
SFT
        ↓
Learn Instructions / Tasks / Behavior
```

I will experiment with both approaches and understand their tradeoffs.

---

# 27. Full Fine-Tuning vs LoRA vs QLoRA

One of the major comparisons in this repository will be:

```text
                 Fine-Tuning
                      |
        +-------------+-------------+
        |             |             |
        ↓             ↓             ↓
     Full FT         LoRA         QLoRA
        |             |             |
     High Memory   Lower Memory   Lowest Memory
        |             |             |
     Large Model   Adapter        Quantized Base
```

I will compare:

* Trainable parameters
* GPU memory
* Training speed
* Checkpoint size
* Model quality
* Deployment complexity

---

# 28. Adapter Management

One advantage of PEFT is that multiple adapters can be trained for the same base model.

```text
                    Base LLM
                       |
          +------------+------------+
          |            |            |
          ↓            ↓            ↓
      Adapter A    Adapter B    Adapter C
       Finance       Medical       Code
```

This allows different specialized behaviors without maintaining a completely separate copy of the base model for every task.

---

# 29. Adapter Merging

I will explore how trained adapters can be:

```text
Base Model
    +
Adapter
    ↓
Merged Model
```

and compare:

```text
Base + Adapter
```

against:

```text
Merged Weights
```

I will investigate the practical implications for deployment.

---

# 30. Multi-Task Fine-Tuning

A model can be fine-tuned using multiple tasks.

```text
                  Base LLM
                     |
          +----------+----------+
          |          |          |
          ↓          ↓          ↓
       Task A     Task B     Task C
          |          |          |
          +----------+----------+
                     |
                     ↓
              Fine-Tuned Model
```

I will explore:

* Task mixing
* Dataset proportions
* Task interference
* Generalization

---

# 31. Synthetic Data

Fine-tuning datasets can also be generated using models.

```text
Seed Data
   ↓
Teacher Model
   ↓
Synthetic Examples
   ↓
Filtering
   ↓
Quality Control
   ↓
Fine-Tuning
```

I will explore both the advantages and risks of synthetic training data.

---

# 32. Data Contamination

Fine-tuning and evaluation can become unreliable if the same information appears in both datasets.

I will investigate:

* Duplicate detection
* Train-test leakage
* Benchmark contamination
* Memorization
* Evaluation contamination

The objective is to make evaluation results meaningful.

---

# 33. Safety During Fine-Tuning

Fine-tuning can change model behavior.

A dataset can unintentionally teach undesirable behavior.

I will explore:

```text
Training Data
     ↓
Fine-Tuning
     ↓
Behavior Change
     ↓
Safety Evaluation
```

Topics include:

* Harmful examples
* Data poisoning
* Prompt injection
* Jailbreak behavior
* Safety regression
* Output filtering

The deeper security work will be covered in my separate:

`ai-safety-and-security`

repository.

---

# 34. Production Fine-Tuning

The final stage is turning a fine-tuned model into a usable production model.

```text
Dataset
   ↓
Fine-Tuning
   ↓
Evaluation
   ↓
Model Selection
   ↓
Optimization
   ↓
Quantization
   ↓
Model Packaging
   ↓
Inference
   ↓
Serving
   ↓
Monitoring
```

---

# 35. Model Registry

Fine-tuned models should be versioned.

```text
Base Model
    ↓
Experiment 01
    ↓
Checkpoint
    ↓
Evaluation
    ↓
Model Version
```

I will track:

* Base model
* Dataset version
* Training configuration
* Adapter version
* Evaluation results
* Model artifacts

---

# 36. Experiment Tracking

Every fine-tuning experiment should be reproducible.

I will track:

```text
Model
Dataset
Hyperparameters
GPU
Training Steps
Training Loss
Validation Loss
Evaluation Metrics
Checkpoint
```

The experiment flow becomes:

```text
Configuration
     ↓
Training
     ↓
Metrics
     ↓
Artifact
     ↓
Evaluation
     ↓
Comparison
```

---

# 37. Inference After Fine-Tuning

After training, the model needs to be tested in inference.

```text
User Prompt
     ↓
Tokenizer
     ↓
Fine-Tuned LLM
     ↓
Generation
     ↓
Response
```

I will compare:

```text
Base Model
     vs
Fine-Tuned Model
```

using the same prompts.

---

# 38. Fine-Tuning Tradeoffs

There is no single best fine-tuning method.

The choice depends on:

```text
Model Size
Dataset Size
GPU Memory
Budget
Task
Quality Requirement
Deployment Requirements
```

A simplified view:

```text
Full Fine-Tuning
    ↓
Maximum Parameter Updates
    ↓
Higher Cost

LoRA
    ↓
Fewer Trainable Parameters
    ↓
Lower Cost

QLoRA
    ↓
Quantized Base + LoRA
    ↓
Lower Memory Requirement
```

The repository will focus on understanding these tradeoffs through experiments.

---

# 39. Complete Fine-Tuning Flow

```text
                    PRETRAINED LLM
                           |
                           ↓
                    Domain Analysis
                           |
                           ↓
                    Dataset Creation
                           |
                           ↓
                    Data Cleaning
                           |
                           ↓
                    Data Filtering
                           |
                           ↓
                    Data Validation
                           |
                           ↓
                  Continued Pretraining
                           |
                           ↓
                   Supervised Fine-Tuning
                           |
                           ↓
                         PEFT
                           |
                 +---------+---------+
                 |                   |
                 ↓                   ↓
                LoRA               QLoRA
                 |                   |
                 +---------+---------+
                           |
                           ↓
                  Preference Dataset
                           |
                           ↓
                          DPO
                           |
                           ↓
                     Evaluation
                           |
                           ↓
                    Safety Testing
                           |
                           ↓
                    Model Selection
                           |
                           ↓
                      Optimization
                           |
                           ↓
                       Inference
                           |
                           ↓
                       Serving
                           |
                           ↓
                      Monitoring
                           |
                           ↓
                      Production
```

---

# Repository Structure

```text
llm-finetuning-from-scratch-to-production/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
│
├── 01_pretrained_models/
│   ├── loading/
│   ├── configuration/
│   ├── checkpoints/
│   └── tokenizer/
│
├── 02_data/
│   ├── collection/
│   ├── cleaning/
│   ├── filtering/
│   ├── deduplication/
│   ├── formatting/
│   ├── chat_templates/
│   └── quality/
│
├── 03_continued_pretraining/
│   ├── domain_adaptation/
│   ├── dataset/
│   ├── training/
│   └── evaluation/
│
├── 04_supervised_finetuning/
│   ├── instruction_data/
│   ├── conversation_data/
│   ├── loss_masking/
│   ├── training/
│   └── evaluation/
│
├── 05_full_finetuning/
│   ├── training/
│   ├── mixed_precision/
│   ├── distributed/
│   └── experiments/
│
├── 06_peft/
│   ├── adapters/
│   ├── prompt_tuning/
│   ├── prefix_tuning/
│   ├── ia3/
│   └── experiments/
│
├── 07_lora/
│   ├── implementation/
│   ├── target_modules/
│   ├── hyperparameters/
│   ├── training/
│   └── experiments/
│
├── 08_qlora/
│   ├── quantization/
│   ├── nf4/
│   ├── double_quantization/
│   ├── training/
│   └── experiments/
│
├── 09_preference_optimization/
│   ├── preference_data/
│   ├── dpo/
│   ├── evaluation/
│   └── experiments/
│
├── 10_synthetic_data/
│   ├── generation/
│   ├── filtering/
│   └── quality/
│
├── 11_evaluation/
│   ├── task_evaluation/
│   ├── generation/
│   ├── regression/
│   ├── safety/
│   └── comparison/
│
├── 12_model_optimization/
│   ├── quantization/
│   ├── merging/
│   ├── inference/
│   └── performance/
│
├── 13_production/
│   ├── packaging/
│   ├── serving/
│   ├── api/
│   ├── monitoring/
│   └── versioning/
│
├── 14_experiments/
│   ├── full_finetuning/
│   ├── lora/
│   ├── qlora/
│   ├── dpo/
│   └── comparisons/
│
├── src/
│   └── finetuning/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── peft/
│       ├── evaluation/
│       ├── optimization/
│       └── serving/
│
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── notebooks/
├── scripts/
├── tests/
├── checkpoints/
└── artifacts/
```

---

# Technology

The main technologies explored in this repository include:

### Programming

* Python
* PyTorch

### LLM

* Hugging Face Transformers
* Hugging Face Datasets
* Tokenizers

### Fine-Tuning

* Supervised Fine-Tuning
* PEFT
* LoRA
* QLoRA
* DPO

### Optimization

* BF16
* FP16
* Quantization
* Gradient Accumulation
* Gradient Checkpointing

### Training

* PyTorch Distributed
* DDP
* FSDP
* Mixed Precision

### Experimentation

* Experiment Tracking
* Model Checkpoints
* Evaluation Pipelines

### Production

* FastAPI
* Docker
* GPU Inference
* Model Serving
* Monitoring

---

# From Scratch to Production

The repository follows this progression:

```text
Understand
    ↓
Prepare Data
    ↓
Choose Base Model
    ↓
Continued Pretraining
    ↓
SFT
    ↓
PEFT
    ↓
LoRA
    ↓
QLoRA
    ↓
DPO
    ↓
Evaluate
    ↓
Optimize
    ↓
Deploy
    ↓
Monitor
```

---

# What I Want to Learn

By completing this repository, I want to understand the complete process of adapting an LLM.

Not just:

```python
trainer.train()
```

but:

```text
Why this dataset?

Why this format?

Why continued pretraining?

Why SFT?

Why full fine-tuning?

Why PEFT?

Why LoRA?

Why QLoRA?

Why this LoRA rank?

Why this learning rate?

Why this sequence length?

Why this precision?

Why DPO?

Did the model actually improve?

Did it forget anything?

How much GPU memory was required?

How much did training cost?

How should the resulting model be deployed?
```

---

# Final Goal

The final goal is to understand the complete path:

```text
                     BASE LLM
                        ↓
                 Domain Adaptation
                        ↓
                       SFT
                        ↓
                      PEFT
                        ↓
                  LoRA / QLoRA
                        ↓
                       DPO
                        ↓
                    Evaluation
                        ↓
                  Safety Testing
                        ↓
                   Optimization
                        ↓
                    Inference
                        ↓
                    Deployment
                        ↓
                    Monitoring
```

This repository is my exploration of how a general pretrained LLM becomes a specialized model that can be evaluated, optimized, deployed, and maintained in a real production environment.
