# Extending the Context Window of SPECTER2 via Direct Weight Transfer and Fine-tuning
The goal of this project is to start moving towards embedding full scientific documents by growing the original 512 token window of SPECTER2 to 4096 tokens via BigBird's attention mechanisms. 4096 tokens is likely not going to be enough to embed the full document. After this project we can explore further extending this window, though we must be mindful of the diminishing returns of this approach - or by performing clever summarizations.
## Transferring Weights from SPECTER2 to BigBird
- **Weight Transfer Pipeline Overview:**
  - **Load both models:**  
    Load SPECTER2 (BERT-based with adapters) and BigBird (with sparse attention and larger positional embeddings).
  - **Copy Compatible Layers:**  
    - **Word & Token Type Embeddings:** Copy directly (or partially, if vocab sizes differ).
    - **Positional Embeddings:** Use position interpolation to upsample SPECTER2’s 512 learned position embeddings to BigBird’s larger context (e.g., 4096).
    - **Transformer Encoder Layers:** Traverse each layer and copy weights for attention (query, key, value), feed‑forward, and layer norms.
  - **Insert Specialized Layers:**  
    Modify BigBird’s architecture with Adapter-compatible alternatives, where possible and modify the modules to include the SPECTER2 specific layers.
- **Fine‑Tuning:**  
  After transferring weights and modifying the architecture, fine‑tune the resulting BigBird model SciRepEval
  
## Next Steps
- **Code Modifications**
  - Use adapter-compatible layers in the BigBird model. It's mainly the attention and the input layers that should be retained from BigBird but everything else should be drop-in compatible with the adapter versions
  - Add in modules and variables for the required adapter layers. Use the adapters library as reference on how to wire the layers in.
- **Data Setup and Pre-Processing**
  - Use the SciRepEval dataset and repository to load training and validation data. Conveniently, they provide us with all the tools needed, most likely as long as our architecture is similar enough to SPECTER2's
  - Depending on how the data looks, we intend to parse the PDFs into Markdown using Nougat. This step may take some time, or maybe the full text is already extracted.
  - If we're attempting to train on the full text, most likely we'll have to do some truncation, which can make the data noisy.
  - During this step, we may already have to employ some summarization techniques
- **Continued Fine-Tuning**
  - After constructing the dataset, fine tune a general task model and save the checkpoint
  - Use the built-in adapters finetuning tools to create proximity and ad-hoc search adapters
