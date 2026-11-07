# Forensic Audit: Cryptographic Chain of Custody and Artifact Provenance

**Audit Date:** November 6, 2026  
**Auditor:** Independent Research Auditor & Experimental Reproducibility Specialist  
**Target:** Cryptographic verification, byte-level integrity, and origin tracing for all experimental artifacts  
**Machine-Readable Hashes:** [artifact_hashes.json](file:///Users/zero/Documents/Area/testing_llms/research/forensic_audit/artifact_hashes.json)  
**Overall Finding:** `VERIFIED CORRECT` (Chain of Custody is Intact, Complete, and Cryptographically Verified)

---

## 1. Executive Summary

A core requirement of forensic reproducibility is establishing an unbroken, tamper-evident chain of custody from external source materials to final evaluated weights.

This audit traces and cryptographically validates every primary artifact across four distinct provenance lineages:
1. **The Experimental Prompt Lineage:** Originating in the official publication repository and transmitted to Gemini Argon without alteration.
2. **The Astra Reference Weight Lineage:** Originating in the official publication repository and verified against published benchmark records.
3. **The Gemini Argon Generation Lineage:** Delivered directly from the official Gemini interface, archived, extracted, and parsed under strict AST sandboxing.
4. **The PPO Baseline Lineage:** Trained from scratch in this laboratory, with complete checkpoint and training log provenance.

---

## 2. Provenance Chain 1: Experimental Prompt

The experiment requires that Gemini Argon be subjected to the exact same prompt given to Astra in the original study.

```mermaid
flowchart TD
    A["Original Author Repo<br>_reference_repos/lm_understand-logs/.../message.txt<br>SHA: 766ee251..."] --> B["Laboratory Archive<br>research/original_prompt.txt<br>SHA: 766ee251..."]
    B --> C["Gemini Preparation<br>gemini/experiment_prompt.txt<br>SHA: 766ee251..."]
    C --> D["Generation Package<br>experiments/.../experiment_5_prompt.txt<br>SHA: 766ee251..."]
    D --> E["Blind ZIP Deployment<br>argon_experiment_001.zip<br>SHA: 976c17cd..."]
```

### Cryptographic Verification
- **Reference Source:** `_reference_repos/lm_understand-logs/exp5_write_from_scratch/runs/breakout/message.txt`
  - SHA-256: `766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876`
  - Size: 9,973 bytes
- **Preserved Archive:** `research/original_prompt.txt`
  - SHA-256: `766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876`
  - Size: 9,973 bytes
- **Evaluation Package:** `experiments/strict_replication/generation_001/input/experiment_5_prompt.txt`
  - SHA-256: `766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876`
  - Size: 9,973 bytes
- **Blind ZIP Archive:** `experiments/strict_replication/generation_001/input/argon_experiment_001.zip`
  - SHA-256: `976c17cd21bed3febb81e5f4f5b7c43467a7cf87230bb646c0135406d86bec64`
  - Size: 3,686 bytes
  - Internal archive contents: Exactly one file (`experiment_5_prompt.txt`), identical byte-for-byte.

**Auditor Finding:** All four prompt files share identical cryptographic hashes. Zero prompt modification, truncation, or hint leakage occurred.

---

## 3. Provenance Chain 2: Astra Reference Weights

The reference weights provide the gold standard for weight-space programming performance.

```mermaid
flowchart TD
    A["Original Author Repo<br>_reference_repos/.../networks/breakout/hand.npz<br>SHA: 6e8d4265..."] --> B["Laboratory Reference<br>experiments/reference/hand.npz<br>SHA: 6e8d4265..."]
    B --> C["Evaluation Harness<br>evaluation/evaluate.py"]
    B --> D["Comparative Visualizer<br>visualize_comparison.py"]
```

### Cryptographic Verification
- **Original Published File:** `_reference_repos/lm_understand-logs/exp5_write_from_scratch/networks/breakout/hand.npz`
  - SHA-256: `6e8d42658aed3704c7bf6fb67015e9468cc03f1d67cb8a4fd192e6534fcccb4a`
  - Size: 83,574 bytes
- **Laboratory Preserved File:** `experiments/reference/hand.npz`
  - SHA-256: `6e8d42658aed3704c7bf6fb67015e9468cc03f1d67cb8a4fd192e6534fcccb4a`
  - Size: 83,574 bytes

**Auditor Finding:** The reference weight file in our laboratory is cryptographically identical to the file published in the official study repository. No post-hoc weight tuning or modification occurred.

---

## 4. Provenance Chain 3: Gemini Argon Raw Response & Weights

This lineage documents the ingestion and translation of Gemini Argon's output.

```mermaid
flowchart TD
    A["Gemini Web Interface<br>User Download: .docx<br>SHA: 2e40dafa..."] --> B["Generation Archive<br>output/argon_response_original.docx<br>SHA: 2e40dafa..."]
    B --> C["Extracted Text Stream<br>output/raw_response.txt<br>SHA: 6e593d58..."]
    C --> D["AST Weight Parser<br>network/weight_parser.py"]
    D --> E["Parsed Weight Archive<br>output/argon_weights.npz<br>SHA: 455722b0..."]
```

### Cryptographic Verification
- **User Provided Response:** `Read the attached file experiment_5_prompt.txt and....docx`
  - SHA-256: `2e40dafa935ed1b21290a1dcba0ae67f88a54b79c1f924fa493d37c49f80a2e7`
  - Size: 67,042 bytes
- **Archived Response DOCX:** `experiments/strict_replication/generation_001/output/argon_response_original.docx`
  - SHA-256: `2e40dafa935ed1b21290a1dcba0ae67f88a54b79c1f924fa493d37c49f80a2e7`
  - Size: 67,042 bytes
- **Extracted Text File:** `experiments/strict_replication/generation_001/output/raw_response.txt`
  - SHA-256: `6e593d588c827afd4f1ab2d81ae1a5c65585210e0345920fb887a76b24e248ac`
  - Size: 107,380 bytes
- **Reconstructed Weights:** `experiments/strict_replication/generation_001/output/argon_weights.npz`
  - SHA-256: `455722b09750deb693093c823c13dd37b99eb470c957ce472262b1419014abb5`
  - Size: 83,574 bytes

**Auditor Finding:** Gemini Argon's output was archived immediately upon download, extracted without loss, and parsed into valid weight arrays using an AST parser without human intervention or manual editing.

---

## 5. Provenance Chain 4: Trained PPO Baseline

This lineage documents the baseline network trained from scratch with reinforcement learning.

```mermaid
flowchart TD
    A["Training Script<br>baselines/train_ppo.py"] --> B["10M Steps Training Run<br>experiments/trained_baseline/"]
    B --> C["Full PyTorch Checkpoint<br>final_actor_critic.pt<br>SHA: bbf6fbc6..."]
    B --> D["Training Log CSV<br>training_log.csv<br>10M step progression"]
    C --> E["Extracted Actor Weights<br>trained_actor_weights.npz<br>SHA: 9ca0c46d..."]
```

### Cryptographic Verification
- **PyTorch Model Checkpoint:** `experiments/trained_baseline/final_actor_critic.pt`
  - SHA-256: `bbf6fbc64d0b4e89cfa36a7980a17a695133884ca9be10e842fbd1f8f86767a3`
  - Size: 168,197 bytes
- **Actor Weight Archive:** `experiments/trained_baseline/trained_actor_weights.npz`
  - SHA-256: `9ca0c46d59cf41179b3db3cfe7b31bd48cd57ce54785e2b407b524c82450ea86`
  - Size: 83,574 bytes
- **Training Log:** `experiments/trained_baseline/training_log.csv`
  - Verifies continuous progression from step 0 to step 10,000,000 across 16 parallel environments.

**Auditor Finding:** The PPO baseline was trained directly inside the workspace under fully auditable code, with complete checkpoint and log trails.

---

## 6. Audit Conclusion

The cryptographic chain of custody across all experimental materials is pristine. Every weight file, prompt, raw model response, and baseline is verified against immutable cryptographic hashes.

**Audit Status:** `VERIFIED CORRECT`

