

<div align="center">

# Investigating the Possibility of Improving Persian Automatic Speech Recognition by Combining Outputs of Existing Models

**A Multi-View Fusion Study of Whisper and MMS for Persian Post-ASR Correction**


*Bachelor's Thesis — Isfahan University of Technology*

*Author: Zahra Tavakoli · Supervisor: Dr. Zeinab Maleki*

</div>

---


This repository contains my Bachelor's thesis project titled **"Investigating the Possibility of Improving Persian Automatic Speech Recognition by Combining Outputs of Existing Models"**, carried out at Isfahan University of Technology under the supervision of Dr. Zeinab Maleki.

## Project Overview

In this project, the outputs of two multilingual ASR models (Whisper and MMS) are treated as two "views" of the same audio input, and a Gated Fusion architecture is proposed and evaluated for combining these two views into a corrected transcription. The main question was whether the multi-view fusion architecture that had been successful for Rajasthani could also improve Persian ASR.

## What Was Done

- **Dataset preparation:** Generated first-pass outputs from Whisper and MMS on the Persian Common Voice dataset, normalized the text using the Shekar library, and **corrected misspelled labels in the dataset** using Regex-based rules and exception lists.
- **Two models implemented:** A **subword-level model** using SentencePiece and a **character-level model**, along with a full comparison of the two approaches.
- **Two alignment strategies:** **Alignment with a null character (Ø)** and **alignment by substituting from the longer string**, with an evaluation of each strategy's effect on output quality.
- **Result analysis:** Studied gate behavior, computed raw and partial correlations between the gate and position/view correctness, and provided a theoretical analysis of why the architecture fails for Persian based on the information-theoretic bound and the nature of homophone errors.

## Main Finding

The project yields an important **negative result**: the gate mechanism collapses into a position-dependent heuristic for Persian (r = +0.77 with position, partial r ≈ 0 with content), and the multi-view fusion cannot outperform the best single view (MMS). The root cause is the **lexical and memory-based nature of Persian ASR errors** (e.g., homophones like ذ/ز/ض/ظ) as opposed to the positional nature of errors in the Devanagari script.

### Lessons Learned

> **"Success of an architecture in one language is not a guarantee of success in another."**

## More Information

For full details on the methodology, numerical results, plots, and statistical analysis, please refer to the **complete report** and **presentation slides** in the `docs/` folder:

- `docs/Bachlor_project_report.pdf` — Bachelor's thesis report
- `docs/Presentations.pdf` — Defense presentation slides

## Contact

- **Zahra Tavakoli** — zahratavakoli763@gmail.com

---

*This project was completed in Summer 1405 (2026) as part of a Bachelor's degree in the Department of Electrical and Computer Engineering at Isfahan University of Technology.*

