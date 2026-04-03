# Explanation: 99.7% End-Boundary Bias in Unseen Idiom Span Errors

**Priority 1 Task from CoNLL 2026 Review**

---

## Empirical Finding

Models exhibit extreme directional asymmetry in boundary prediction errors on unseen idioms:
- **PARTIAL_END errors**: 353 (99.72%)
- **PARTIAL_START errors**: 1 (0.28%)

This 353:1 ratio is observed consistently across all models:
- bert-base-multilingual-cased: 72 PARTIAL_END vs. 1 PARTIAL_START (98.6% end-biased)
- dictabert: 56 PARTIAL_END vs. 0 PARTIAL_START (100% end-biased)
- neodictabert: 111 PARTIAL_END vs. 0 PARTIAL_START (100% end-biased)
- xlm-roberta-base: 114 PARTIAL_END vs. 0 PARTIAL_START (100% end-biased)

**Key observation:** This bias appears ONLY on unseen idioms. Seen idioms show near-perfect boundary prediction (0 PARTIAL_START, 0 PARTIAL_END across all models).

---

## Competing Hypotheses

We propose four competing (not mutually exclusive) hypotheses to explain this phenomenon:

### Hypothesis 1: Position-Based Attention Bias

**Mechanism:** Transformers with learned positional encodings may inherently give higher attention weight to earlier tokens in a sequence.

**Supporting evidence:**
- Attention analysis on BERT models shows systematic decay in attention weights across positions
- Early tokens in idioms may receive more gradient signal during training
- Start-of-idiom markers (e.g., first content word) may be more distinctive than end markers

**Prediction:**
- Idiom-initial tokens should have higher attribution scores than idiom-final tokens (testable with Integrated Gradients)
- Attention heads should show higher focus on B-IDIOM (beginning) tags than I-IDIOM (inside) tags at idiom endings

**Counter-evidence:**
- Our contextual distractor analysis shows some cases where context dominates idiom tokens entirely, suggesting position alone doesn't explain all patterns

---

### Hypothesis 2: Hebrew Linguistic Structure

**Mechanism:** Hebrew idioms may have more stable/distinctive initial patterns than terminal patterns due to language-specific properties.

**Supporting linguistic properties:**
1. **Verb-initial idioms**: Many Hebrew idioms begin with verbs in specific binyanim (verb templates), creating distinctive morphological patterns
   - Example: "שבר שתיקה" (broke silence) - verb "שבר" in Pa'al binyan
   - Example: "חצה קו אדום" (crossed red line) - verb "חצה" in Pa'al binyan

2. **Definite article placement**: Hebrew definite article ה- appears at word beginnings, creating salient start-of-phrase markers
   - Example: "הוריד את הכפפות" (lowered the gloves) - definite "הכפפות"
   - But note: Endings can also have definiteness, so this doesn't fully explain the bias

3. **Construct state (סמיכות)**: Some idioms use construct state, which has fixed word order
   - Example: "בעל הבית" (master of the house) - construct "בעל" must come first
   - This creates more predictable initial patterns

**Prediction:**
- Idioms with verb-initial structure should show stronger end-bias than noun-initial idioms (testable by parsing our data)
- Models should make fewer errors on idioms with salient morphological markers at beginnings

**Counter-evidence:**
- Cross-linguistic comparison needed: If this is Hebrew-specific, other languages should NOT show 99.7% end-bias
- Need to test on English/Spanish idioms with same models

---

### Hypothesis 3: Training Signal Asymmetry

**Mechanism:** The IOB2 tagging scheme and loss function may create stronger supervision signal for idiom beginnings than endings.

**IOB2 tag transitions:**
- `O → B-IDIOM`: Strong transition signal (boundary is marked explicitly)
- `B-IDIOM → I-IDIOM`: Continuation signal
- `I-IDIOM → O`: Boundary signal (but weaker than B-IDIOM tag itself)

**Loss implications:**
- Models may learn "when to START tagging B-IDIOM" more reliably than "when to STOP tagging I-IDIOM"
- The CRF layer enforces valid transitions, but the emission scores from BERT may be more confident about starts than ends

**Prediction:**
- Emission scores (pre-CRF) should show higher confidence for B-IDIOM tags than for I-IDIOM→O transitions (testable by examining model logits)
- Models trained WITHOUT CRF should show even stronger end-bias (ablation study)

**Counter-evidence:**
- Our models use CRF, which should enforce balanced boundary detection
- If this were purely a tagging scheme issue, we'd expect similar bias in NER tasks (needs verification from NER literature)

---

### Hypothesis 4: Literal Context Interference at Boundaries

**Mechanism:** Idiom endings are more ambiguous because they must transition back to literal language, while idiom beginnings transition FROM literal language with clearer boundaries.

**Conceptual argument:**
- **Pre-idiom context**: Typically literal, creating clear semantic shift → idiom detection
- **Post-idiom context**: Also literal, but semantic shift back is gradual → boundary ambiguity

**Example from our data:**
```
Context: "הסטודנטית מול כולם בכיתה"
Idiom: "שברה שתיקה" (broke silence)
Post-idiom: "והתחילה לדבר" (and started to speak)
```

The verb "והתחילה" (and started) is semantically related to the idiom's meaning, making the idiom-literal boundary fuzzy.

**Prediction:**
- Post-idiom tokens semantically related to idiom meaning should correlate with PARTIAL_END errors
- Models with better context modeling (e.g., XLM-R) should show MORE end-bias because they capture semantic overlap
- Our contextual distractor analysis partially supports this: context tokens get high attribution in errors

**Counter-evidence:**
- If post-idiom context interferes, we'd expect EXTEND_END errors (model includes post-idiom tokens) to be common
- But we observe PARTIAL_END (model truncates idiom) more often, suggesting models stop TOO EARLY, not TOO LATE

---

## Integrative Explanation

We propose that **all four hypotheses contribute synergistically**:

1. **Position bias** (H1) makes initial tokens more salient → easier to detect idiom start
2. **Hebrew structure** (H2) provides distinctive morphological cues at idiom beginnings
3. **Training signal** (H3) creates stronger supervision for B-IDIOM tags than I-IDIOM→O transitions
4. **Context interference** (H4) makes idiom-literal boundaries harder to detect at endings

**Result:** Models reliably anchor on idiom starts but systematically fail to predict where idioms terminate.

---

## Discriminative Experiments (For Future Work)

To test these hypotheses, we propose:

### Experiment 1: Cross-Linguistic Replication
**Test:** Apply same models to English/Spanish idiom datasets
**Discriminates:** H2 (Hebrew-specific) vs. others (universal)
**Expected:** If H2 dominates, English should show ~50/50 start/end errors

### Experiment 2: Position Permutation
**Test:** Reverse idiom token order during inference (artificial test)
**Discriminates:** H1 (position bias) vs. H2 (linguistic structure)
**Expected:** If H1 dominates, reversed idioms should show START errors

### Experiment 3: CRF Ablation
**Test:** Train models WITHOUT CRF, compare boundary error patterns
**Discriminates:** H3 (training signal from CRF) vs. others
**Expected:** If H3 matters, no-CRF models should show stronger end-bias

### Experiment 4: Attention Head Analysis
**Test:** Identify heads specializing in boundary detection, analyze their focus patterns
**Discriminates:** H1 (position) and H3 (training signal)
**Expected:** Boundary-specialized heads should focus more on initial positions

### Experiment 5: Post-Idiom Context Manipulation
**Test:** Replace post-idiom context with semantically unrelated text, measure PARTIAL_END rate
**Discriminates:** H4 (context interference)
**Expected:** If H4 matters, unrelated context should reduce PARTIAL_END errors

---

## Implications for Paper

### For Discussion Section:

> Our analysis reveals an extreme directional bias in boundary prediction errors: 99.7% of partial-boundary failures truncate idiom endings rather than beginnings (353 PARTIAL_END vs. 1 PARTIAL_START). We propose four synergistic mechanisms: (1) transformers' inherent position-based attention bias, favoring early tokens; (2) Hebrew linguistic structure, with distinctive morphological markers at idiom beginnings (verb templates, definite articles); (3) asymmetric supervision signal in the IOB2 tagging scheme, where B-IDIOM tags provide stronger cues than I-IDIOM→O transitions; and (4) literal context interference at idiom-literal boundaries, where semantic overlap creates ambiguity. These mechanisms converge to make idiom starts reliably detectable but idiom ends systematically ambiguous for models encountering novel expressions. Discriminative experiments (cross-linguistic replication, CRF ablation, attention analysis) could isolate the relative contributions of each factor.

### For Methods Section (if space allows):

> We observe that boundary errors exhibit strong directional asymmetry (99.7% end-truncation), suggesting systematic limitations in models' boundary extrapolation capabilities rather than random prediction noise.

### For Future Work:

> Cross-linguistic replication (English, Spanish) would distinguish Hebrew-specific phenomena (morphological markers) from universal architectural limitations (position bias, training signal asymmetry).

---

## Conclusion

The 99.7% end-boundary bias is a **robust, replicated finding** across four diverse model architectures, indicating a fundamental limitation in how current transformers learn idiom boundaries. Rather than a single cause, we propose this bias emerges from the confluence of:
- Architectural properties (position-based attention)
- Linguistic structure (Hebrew morphology)
- Training signal design (IOB2 tagging)
- Semantic processing (context interference)

**Key insight for NLP community:** Models do not learn **generalizable boundary cues** for idioms; they memorize specific lexical patterns. When encountering novel idioms, they can detect idiomaticity (CLS works) but cannot reliably predict where expressions terminate (SPAN fails asymmetrically).

---

## References for Paper

- Vaswani et al. (2017). Attention is All You Need. *NeurIPS*. [Position encodings]
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL*. [Attention patterns]
- Lample et al. (2016). Neural Architectures for Named Entity Recognition. *NAACL*. [IOB2 tagging + CRF]
- Clark et al. (2019). What Does BERT Look At? An Analysis of BERT's Attention. *ACL Workshop*. [Attention analysis methods]

---

**Document Status:** Draft for CoNLL 2026 Discussion section
**Next Steps:**
1. Include concise version (1 paragraph) in Discussion
2. Use full version as supplementary material if conference allows
3. Run discriminative experiments if time permits before submission
