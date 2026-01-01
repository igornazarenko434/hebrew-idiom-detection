# Documentation Index
# Hebrew Idiom Detection Project

**Last Updated:** December 31, 2025
**Documentation Version:** 4.0

---

## 🎯 START HERE: The One Document You Need

### **IMPLEMENTATION_ROADMAP.md** ⭐⭐⭐

**This is your primary document. Follow it step-by-step to complete all analysis missions.**

- ✅ Covers ALL Phase 5-7 missions from STEP_BY_STEP_MISSIONS.md
- ✅ Includes ready-to-run code snippets
- ✅ Ensures compatibility with partner's results
- ✅ Follows NLP best practices
- ✅ Uses correct error taxonomy
- ✅ Step-by-step action plan (STEP 1 → STEP 2 → STEP 3 → STEP 4)

**If you only read one document, read this one.**

---

## 📚 Core Documentation (Essential)

### 1. **IMPLEMENTATION_ROADMAP.md** (Primary)
**When to use:** Every day - your action plan
**Contains:**
- What to do next (STEP 1-4)
- Ready-to-run code for each task
- Week-by-week timeline
- Exact commands to execute

**Start here:** Task 1.1 - Run `python src/analyze_finetuning_results.py`

---

### 2. **EVALUATION_STANDARDIZATION_GUIDE.md** (Reference)
**When to use:** When you need to understand standards, metrics, or protocols
**Contains:**
- Exact metric definitions (Span F1, Macro F1)
- 12-category error taxonomy
- Statistical testing protocols
- Visualization standards
- Reproducibility requirements

**Example use cases:**
- "What is Exact Span F1?" → Section 3.2
- "How do I categorize errors?" → Section 4
- "What statistical test should I use?" → Section 15

**You reference this FROM the Implementation Roadmap when needed.**

---

### 3. **OPERATIONS_GUIDE.md** (Manual)
**When to use:** When you need to understand how to use tools or troubleshoot
**Contains:**
- How to train models (VAST.ai workflow)
- How to run evaluations
- Analysis tools reference
- Troubleshooting guide
- How to add new models

**Example use cases:**
- "How do I train on VAST.ai?" → Sections 4-5
- "What does analyze_generalization.py do?" → Section 9.2
- "How do I add a new model?" → Section 10
- "Something broke, how do I fix it?" → Section 11

**You reference this FROM the Implementation Roadmap when you need tool details.**

---

## ✅ Full Re-Run Checklist (New)

### **FULL_RERUN_CHECKLIST.md**
**When to use:** When you want to re-run the entire pipeline end-to-end  
**Contains:**
- VAST.ai volume + instance setup
- Batch HPO + full training + evaluation
- Google Drive sync + local downloads
- Analysis scripts and outputs

---

## 📋 Supporting Documents

### 4. **STEP_BY_STEP_MISSIONS.md**
**Purpose:** Original mission requirements (don't edit this)
**Use:** Reference to verify you're meeting all requirements
**Note:** IMPLEMENTATION_ROADMAP.md covers all these missions - you don't need to read this unless checking specific mission details

### 5. **VENV_USAGE.md**
**Purpose:** Virtual environment usage guide
**Use:** How to activate venv, install packages, troubleshoot
**Note:** Quick reference for analysis environment setup

### 6. **README.md**
**Purpose:** Project overview
**Use:** Quick introduction to the project

### 7. **FINAL_PRD_Hebrew_Idiom_Detection.md**
**Purpose:** Product Requirements Document
**Use:** Understand project goals and scope

### 8. **PROJECT_CONTEXT.md**
**Purpose:** Session memory for Claude Code
**Use:** Automatically used by Claude (you don't need to read this)

---

## 🗂️ Archived Documentation

**Location:** `archive/old_documentation/`

**Contents:** Old intermediate documents superseded by v4.0

**What's in there:**
- MISSION_4.7_SUMMARY_AND_NEXT_STEPS.md (superseded)
- DOCUMENT_COVERAGE_ANALYSIS.md (analysis file)
- IMPLEMENTATION_GUIDE.md (old version)
- Other intermediate documents

**Should you use these?** NO - kept for historical reference only

---

## 📖 Reference Guides (Optional)

**Location:** `archive/reference_guides/`

**Contents:** Specialized guides (optional reading)

**Available:**
- VAST_AI_PERSISTENT_VOLUME_GUIDE.md
- VAST_AI_QUICK_START.md
- IAA_Report.md
- PATH_REFERENCE.md

**When to use:** Only if you need specific VAST.ai details or IAA information
**Note:** OPERATIONS_GUIDE.md already covers VAST.ai workflow - these are supplementary

---

## 🎯 How to Use This Documentation System

### Scenario 1: "I want to start working on analysis"
```
→ Activate virtual environment: source activate_env.sh
→ Open IMPLEMENTATION_ROADMAP.md
→ Start with STEP 1 Task 1.1
→ Follow tasks in order
```

### Scenario 2: "I don't understand what Span F1 means"
```
→ Open EVALUATION_STANDARDIZATION_GUIDE.md
→ Go to Section 3.2
→ Read metric definition
```

### Scenario 3: "How do I use analyze_generalization.py?"
```
→ Open OPERATIONS_GUIDE.md
→ Go to Section 9.2
→ See usage examples
```

### Scenario 4: "I want to train a new model"
```
→ Open OPERATIONS_GUIDE.md
→ Go to Sections 4-5 (HPO + Training)
→ Follow step-by-step
OR
→ Open OPERATIONS_GUIDE.md
→ Go to Section 10 (Adding New Models)
```

### Scenario 5: "Something broke / not working"
```
→ Open OPERATIONS_GUIDE.md
→ Go to Section 11 (Troubleshooting)
→ Find your error and solution
```

### Scenario 6: "I want to verify I'm meeting mission requirements"
```
→ Open STEP_BY_STEP_MISSIONS.md
→ Find the mission (e.g., Mission 7.1)
→ Compare with IMPLEMENTATION_ROADMAP.md tasks
→ Confirm coverage (all missions are covered)
```

---

## ✅ Document Relationship Diagram

```
                    YOUR DAILY WORK
                          ↓
              ┌───────────────────────┐
              │ IMPLEMENTATION_ROADMAP│ ← START HERE & FOLLOW
              │      (Action Plan)    │
              └───────────────────────┘
                     ↓           ↓
          References when needed:
                ↓                    ↓
    ┌──────────────────┐    ┌──────────────────┐
    │  EVALUATION_     │    │  OPERATIONS_     │
    │  STANDARDIZATION │    │  GUIDE           │
    │  (Standards)     │    │  (How-To)        │
    └──────────────────┘    └──────────────────┘
```

**Simple rule:**
1. **Follow:** IMPLEMENTATION_ROADMAP.md (90% of your time here)
2. **Reference:** EVALUATION_STANDARDIZATION_GUIDE.md (when you need standards)
3. **Reference:** OPERATIONS_GUIDE.md (when you need tool details)

---

## 🎓 Typical Workflow

### Week 1: Quick Wins (2-3 hours)
```
1. Open IMPLEMENTATION_ROADMAP.md
2. Read STEP 1 overview
3. Execute Task 1.1: python src/analyze_finetuning_results.py
4. Check output
5. Execute Task 1.2: python src/analyze_generalization.py
6. Continue through Task 1.3 → 1.4 → 1.5 → 1.6
7. After STEP 1: You have 40% of Phase 7 complete!
```

**When stuck:**
- Error understanding metric → EVALUATION_STANDARDIZATION_GUIDE.md
- Error running tool → OPERATIONS_GUIDE.md Section 11 (Troubleshooting)

### Week 2-3: Ablations (Optional but recommended)
```
1. Open IMPLEMENTATION_ROADMAP.md STEP 2
2. Follow Task 2.1: Frozen backbone comparison
3. For VAST.ai help → OPERATIONS_GUIDE.md Sections 4-5
4. Continue with Task 2.2, 2.3
```

### Week 3: Finalization
```
1. Open IMPLEMENTATION_ROADMAP.md STEP 4
2. Create all publication figures
3. Generate LaTeX tables
4. Write results section
```

---

## 📊 Quick Reference Card

| I need to... | Open this document | Section |
|--------------|-------------------|---------|
| Know what to do next | IMPLEMENTATION_ROADMAP.md | Current STEP |
| Understand Span F1 | EVALUATION_STANDARDIZATION_GUIDE.md | 3.2 |
| Categorize errors | EVALUATION_STANDARDIZATION_GUIDE.md | 4, 14 |
| Use analysis tool | OPERATIONS_GUIDE.md | 9 |
| Train on VAST.ai | OPERATIONS_GUIDE.md | 4-5 |
| Add new model | OPERATIONS_GUIDE.md | 10 |
| Fix something broken | OPERATIONS_GUIDE.md | 11 |
| Do statistical test | EVALUATION_STANDARDIZATION_GUIDE.md | 15 |
| Create visualization | EVALUATION_STANDARDIZATION_GUIDE.md | 16 |
| Verify mission coverage | STEP_BY_STEP_MISSIONS.md | Find mission # |

---

## 🎯 Success Criteria

**You know you're using the docs correctly when:**

✅ You spend 90% of time in IMPLEMENTATION_ROADMAP.md
✅ You only open other docs when referenced or stuck
✅ You follow tasks in order (1.1 → 1.2 → 1.3...)
✅ You run the exact commands from code snippets
✅ You create the expected outputs at each step

---

## 🚀 Your Next Action

**Right now, do this:**

```bash
# 1. Activate virtual environment
source activate_env.sh

# 2. Open the implementation roadmap
open IMPLEMENTATION_ROADMAP.md  # or: code IMPLEMENTATION_ROADMAP.md

# 3. Read STEP 1 overview

# 4. Execute Task 1.1
python src/analyze_finetuning_results.py

# 5. Check the output
cat experiments/results/analysis/finetuning_summary.md

# 6. If successful, move to Task 1.2
# If error, check OPERATIONS_GUIDE.md Section 11
```

**That's it! Follow the roadmap step by step.** 🎯

---

## 💡 Pro Tips

### Tip 1: Bookmark These 3 Documents
- IMPLEMENTATION_ROADMAP.md (daily use)
- EVALUATION_STANDARDIZATION_GUIDE.md (reference)
- OPERATIONS_GUIDE.md (reference)

### Tip 2: Don't Read Everything
You don't need to read all 3 docs cover-to-cover. Just:
- Read IMPLEMENTATION_ROADMAP.md fully
- Reference others when needed

### Tip 3: Follow the Order
Tasks are designed to build on each other. Don't skip around.

### Tip 4: Check Outputs
After each task, verify the output files exist and look correct.

### Tip 5: Ask Questions
If something is unclear:
```bash
echo "Question: [your question]" >> QUESTIONS.md
```
Then search the relevant guide for answers.

---

## 📞 Common Questions

**Q: "Which document should I follow for implementation?"**
A: IMPLEMENTATION_ROADMAP.md (only this one)

**Q: "Do I need to read EVALUATION_STANDARDIZATION_GUIDE.md first?"**
A: No. Start with IMPLEMENTATION_ROADMAP.md. It will tell you when to reference the standards guide.

**Q: "What about STEP_BY_STEP_MISSIONS.md?"**
A: That's the original requirements. IMPLEMENTATION_ROADMAP.md covers all those missions with actual implementation code. You don't need to read STEP_BY_STEP_MISSIONS.md unless verifying specific mission details.

**Q: "I'm confused by too many documents!"**
A: Open only IMPLEMENTATION_ROADMAP.md and follow it. That's all you need to start.

**Q: "How do I know I'm not missing anything?"**
A: IMPLEMENTATION_ROADMAP.md covers 100% of Phase 5-7 missions. If you complete all STEPs 1-4, you're done.

**Q: "What if my partner asks how I computed something?"**
A: Point them to EVALUATION_STANDARDIZATION_GUIDE.md (the shared standard).

---

## 🎉 Final Word

**You have everything you need:**
1. ✅ One clear action plan (IMPLEMENTATION_ROADMAP.md)
2. ✅ Complete standards reference (EVALUATION_STANDARDIZATION_GUIDE.md)
3. ✅ Full operations manual (OPERATIONS_GUIDE.md)

**Your next command:**
```bash
python src/analyze_finetuning_results.py
```

**Then follow IMPLEMENTATION_ROADMAP.md step by step!**

Good luck! 🚀

---

**Last Updated:** December 31, 2025
**Documentation Version:** 4.0
**Status:** Complete and ready to use
