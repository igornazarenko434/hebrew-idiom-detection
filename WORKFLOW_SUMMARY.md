# Complete Project Workflow Summary
**Date:** November 8, 2025
**Purpose:** Clear overview of what runs where, what uses Optuna, and mission flow

---

## 🎯 **Quick Reference: What Runs Where?**

### **PyCharm (Local Development)**
✅ All code development  
✅ Phase 1: Environment setup  
✅ Phase 2: Data preparation (all missions)  
✅ Phase 3: Zero-shot evaluation (optional - can use VAST.ai for speed)  
✅ Mission 4.1: Create config files  
✅ Mission 4.2: Develop training code + **test locally first** (100 samples, CPU)  
✅ Mission 4.3: Develop Optuna code + **test locally** (3 trials)  
✅ **Phase 5: LLM evaluation (ALL missions)** - just API calls, no GPU  
✅ Mission 6.3: Hyperparameter sensitivity analysis (just analyzing Optuna results)  
✅ Phase 7-9: Analysis, paper writing, visualization  

### **VAST.ai (GPU Training)**
🚀 Mission 4.5: **HPO** - 150 trials, ~50-75 GPU hours, ~$20-30  
🚀 Mission 4.6: **Final training** - 30 models, ~15 GPU hours, ~$6  
🚀 Mission 6.2, 6.4: Optional ablations (training required)  

---

## 🔄 **What Uses Optuna?**

### **Uses Optuna (Running HPO)**
✅ **Mission 4.5 ONLY** - 10 HPO studies (5 models × 2 tasks), 15 trials each

### **Uses Results from Optuna (NOT running HPO again)**
✅ Mission 4.6: Final training (loads best params from 4.5)  
✅ Mission 6.2: Frozen backbone (uses best params from 4.5)  
✅ Mission 6.3: Sensitivity analysis (analyzes Optuna databases from 4.5)  
✅ Mission 6.4: Data size ablation (uses best params from 4.5)  

**Key Point:** You only run Optuna ONCE in Mission 4.5. All other missions use those results!

---

## 📊 **Complete Mission Flow with Execution Location**

### **PHASE 1: Environment Setup (Week 1)**
| Mission | Where | What |
|---------|-------|------|
| 1.1-1.5 | PyCharm | Install packages, setup Git, test GPU, Docker |

### **PHASE 2: Data Preparation (Week 1-2)**
| Mission | Where | What |
|---------|-------|------|
| 2.1-2.6 | PyCharm | Load data, validate, split, analyze |

### **PHASE 3: Zero-Shot Baseline (Week 2-3)**
| Mission | Where | What |
|---------|-------|------|
| 3.1 | PyCharm | Download 5 models |
| 3.2 | PyCharm | Create idiom_experiment.py skeleton (all modes) |
| 3.3 | **PyCharm OR VAST.ai** | Run zero-shot (10 evaluations) - CPU ok but slower |
| 3.4 | PyCharm | Analyze results |

### **PHASE 4: Full Fine-Tuning (Week 4-6)**
| Mission | Where | What | Time/Cost |
|---------|-------|------|-----------|
| 4.1 | PyCharm | Create 2 config files + config loading | - |
| 4.2 | **PyCharm** | Implement training + **test locally** (100 samples) | - |
| 4.3 | **PyCharm** | Implement Optuna mode + **test locally** (3 trials) | - |
| 4.4 | **VAST.ai** | Rent GPU instance, setup environment | - |
| 4.5 | **VAST.ai** | **Run HPO** (150 trials) | 50-75 hrs, $20-30 |
| 4.6 | **VAST.ai** | Final training (30 models) | 15 hrs, $6 |
| 4.7 | PyCharm | Analyze results | - |

**Total Phase 4 Cost:** ~$26-36

### **PHASE 5: LLM Evaluation (Week 7)**
| Mission | Where | What | Cost |
|---------|-------|------|------|
| 5.1-5.5 | **PyCharm (LOCAL)** | Setup API, design prompts, run evaluation | $50-200 API |

**Key:** NO GPU needed - just makes API calls!

### **PHASE 6: Ablation Studies (Week 8)**
| Mission | Where | What | Uses Optuna? |
|---------|-------|------|--------------|
| 6.1 | PyCharm or VAST.ai | Token importance (inference only) | No |
| 6.2 | **VAST.ai recommended** | Frozen backbone training | No - uses params from 4.5 |
| 6.3 | **PyCharm (LOCAL)** | Sensitivity analysis | No - analyzes 4.5 results |
| 6.4 | **VAST.ai recommended** | Data size ablation | No - uses params from 4.5 |

### **PHASE 7-9: Analysis & Paper (Week 9-12)**
| Phase | Where | What |
|-------|-------|------|
| 7 | PyCharm | Comprehensive analysis, visualizations |
| 8 | PyCharm | Write paper |
| 9 | PyCharm | Submit, release models/data |

---

## 🚀 **Workflow: Development → Training → Analysis**

### **Step 1: Develop Locally (PyCharm)**
```bash
# Phase 1-2: Setup + data prep
# Phase 3: Create code
python src/idiom_experiment.py --mode zero_shot ...  # Optional local test

# Phase 4.1-4.3: Create configs + training code
python src/idiom_experiment.py --mode full_finetune --max_samples 100 --device cpu  # Test
python src/idiom_experiment.py --mode hpo --max_samples 500 --device cpu  # Test Optuna
```

### **Step 2: Train on VAST.ai**
```bash
# Rent VAST.ai instance (Mission 4.4)
# Upload code, download data

# Mission 4.5: Run HPO (150 trials)
bash scripts/run_all_hpo.sh  # Or run 10 commands manually
# Takes 50-75 hours, costs $20-30

# Mission 4.6: Final training (30 models)
bash scripts/run_all_experiments.sh
# Takes 15 hours, costs $6

# Sync results to Google Drive
bash scripts/sync_to_gdrive.sh
```

### **Step 3: Evaluate LLM Locally (PyCharm)**
```bash
# Phase 5: No VAST.ai needed!
python src/llm_evaluation.py ...  # Just makes API calls
```

### **Step 4: Optional Ablations (VAST.ai or PyCharm)**
```bash
# Mission 6.2, 6.4: Can use VAST.ai for speed
python src/idiom_experiment.py --mode frozen_backbone ...  # Uses params from 4.5
```

### **Step 5: Analyze and Write (PyCharm)**
```bash
# Phase 7-9: All local
# Create figures, tables, write paper
```

---

## 💰 **Cost Breakdown**

| Phase | Where | Cost |
|-------|-------|------|
| Phase 1-3 | PyCharm | $0 |
| Phase 4 HPO (4.5) | VAST.ai | $20-30 |
| Phase 4 Training (4.6) | VAST.ai | $6 |
| Phase 5 LLM | PyCharm (API) | $50-200 |
| Phase 6 (optional) | VAST.ai | $5-10 |
| **TOTAL** | | **$81-246** |

Very affordable for a complete research project!

---

## ✅ **Key Principles**

1. **Develop locally, train remotely** - Write code in PyCharm, train on VAST.ai
2. **Test locally first** - Always test with small data before VAST.ai
3. **Optuna runs once** - Mission 4.5 only, all others use results
4. **LLM runs locally** - No GPU needed for API calls
5. **Batch scripts** - Use `run_all_hpo.sh` and `run_all_experiments.sh`
6. **Google Drive sync** - Backup results after training

---

## 🎯 **Mission Dependencies**

```
Phase 1 (Setup)
  ↓
Phase 2 (Data Prep)
  ↓
Phase 3 (Zero-Shot) - Creates idiom_experiment.py skeleton
  ↓
Phase 4.1-4.2 (Training Code) - Test locally
  ↓
Phase 4.3 (Optuna Code) - Test locally
  ↓
Phase 4.4 (VAST.ai Setup)
  ↓
Phase 4.5 (HPO on VAST.ai) ← ONLY time Optuna runs!
  ↓
Phase 4.6 (Final Training on VAST.ai) ← Uses best params from 4.5
  ↓
Phase 5 (LLM - LOCAL) - Can run in parallel with Phase 6
  ↓
Phase 6 (Ablations) ← Uses best params from 4.5
  ↓
Phase 7-9 (Analysis & Paper)
```

---

## 🛠️ **Tools by Phase**

| Tool | Used In |
|------|---------|
| **PyCharm** | All development, Phases 1-3, 5, 7-9 |
| **VAST.ai** | Phase 4.5, 4.6, optional 6.2/6.4 |
| **Optuna** | Phase 4.5 only |
| **LLM APIs** | Phase 5 only |
| **Google Drive** | Backup after Phase 4, 6 |
| **Git** | All phases (version control) |

---

**This workflow is simple, cost-effective, and clear!** ✅
