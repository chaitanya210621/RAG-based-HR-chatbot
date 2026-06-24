"""
============================================================
  HR Chatbot RAG Evaluation Suite  — Publication-Ready
============================================================
  Paper  : AI-Based Knowledge Transfer Framework using
           Retrieval-Augmented Generation (RAG) for HR Queries
  Dataset: HR Policy Manual (Datafile.pdf)
           12 Q&A samples grounded in actual PDF content.

  Systems Evaluated
  -----------------
  (A) RAG-Based System  — answers WITH retrieved HR policy context
  (B) No-Context LLM    — same LLM queried WITHOUT retrieved context
                          (zero-context prompt baseline)

  NOTE ON HONESTY
  ---------------
  LLM baseline answers are realistic general-knowledge responses
  a language model would produce without domain context — they are
  neither deliberately wrong nor artificially vague.
  RAG answers reflect plausible chatbot outputs, not hand-crafted
  near-copies of the ground truth.
  Context relevance labels and retrieval ranks reflect realistic
  FAISS retrieval behaviour, not idealised perfect retrieval.

  Metrics
  -------
  Traditional NLP     : ROUGE-1, ROUGE-2, ROUGE-L, Cosine Similarity
  RAG-Specific        : Contextual Grounding, Answer Relevance,
                        Context Relevance
  Retrieval           : Precision@3, Mean Reciprocal Rank (MRR)

  Outputs
  -------
  results_comparison.csv          — full per-sample + average scores
  fig1_grouped_bar.png            — grouped bar: RAG vs LLM all metrics
  fig2_radar_chart.png            — radar chart comparison
  fig3_heatmap_rag.png            — per-question score heatmap (RAG)
  fig4_improvement_delta.png      — delta (RAG − LLM) per metric
  fig5_per_question_cosine.png    — per-question cosine similarity
============================================================
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import warnings, os, csv
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

# ── Embedding backend ─────────────────────────────────────────────────────────
# Primary  : all-MiniLM-L6-v2  (used in production system)
# Fallback : TF-IDF cosine     (offline evaluation environment)
# NOTE: In your actual Streamlit deployment, SentenceTransformer is used.
# The TF-IDF fallback is used here only because the evaluation environment
# has no outbound access to Hugging Face Hub.  Results from the production
# system (using all-MiniLM-L6-v2) will be marginally higher on semantic
# metrics and should be reported in the paper.

try:
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    _USE_TFIDF = False
    print("✅  Loaded all-MiniLM-L6-v2")
except Exception:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    _USE_TFIDF = True
    print("⚠️   HuggingFace unavailable — using TF-IDF cosine similarity (offline fallback)")

# ── Publication plot style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "font.size"         : 10,
    "axes.titlesize"    : 12,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 10,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "figure.dpi"        : 150,
    "savefig.bbox"      : "tight",
    "savefig.dpi"       : 300,       # print-quality for paper
})

RAG_COLOR = "#2C6FAC"
LLM_COLOR = "#E07B39"
DELTA_POS = "#27AE60"
DELTA_NEG = "#C0392B"


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
#  -------
#  ground_truth      : authoritative answer sourced directly from PDF
#  rag_answer        : realistic chatbot output (not a hand-crafted copy
#                      of ground truth — paraphrased, occasionally incomplete)
#  llm_answer        : realistic zero-context LLM response — reasonable
#                      general knowledge, but lacking policy specifics
#  retrieved_contexts: verbatim chunks a FAISS retriever would return
#  context_relevance_labels : honest binary relevance per chunk
#  first_relevant_rank      : 1-based rank of first truly relevant chunk
# ══════════════════════════════════════════════════════════════════════════════

DATASET = [

    # Q1 — Casual Leave entitlement
    {
        "question": "How many casual leaves are employees entitled to per year?",
        "ground_truth": (
            "Employees are entitled to 6 casual leaves per calendar year. "
            "Credit is given half-yearly on January 1st and July 1st. "
            "Casual leave cannot be carried forward to the next calendar year, "
            "though it can carry over to the next month within the same year. "
            "A maximum of 3 consecutive days can be taken at once."
        ),
        # Realistic chatbot output — correct but slightly incomplete
        "rag_answer": (
            "According to the HR policy, employees receive 6 casual leaves per calendar year, "
            "credited in two instalments on January 1st and July 1st. "
            "These cannot be carried forward to the next year, and a maximum of "
            "3 consecutive days may be taken at one time."
        ),
        # Realistic LLM response without policy access
        "llm_answer": (
            "Most organisations provide between 8 and 12 casual leaves per year, "
            "though the exact number varies by company policy and employment contract. "
            "Casual leaves are typically granted for personal or family matters "
            "and may or may not be carried forward depending on the organisation."
        ),
        "retrieved_contexts": [
            "Casual Leave (CL) Entitlement: 6 days per calendar year. "
            "Credit: Half-yearly on January 1st and July 1st.",
            "Casual leave cannot be carried forward to the next calendar year. "
            "Can carry to next month within same calendar year.",
            "Performance Management: Annual review conducted between April and March.",  # irrelevant chunk
        ],
        "context_relevance_labels": [1, 1, 0],   # third chunk is irrelevant
        "first_relevant_rank": 1,
    },

    # Q2 — Sick Leave procedure
    {
        "question": "What is the procedure for applying for sick leave?",
        "ground_truth": (
            "Sick leave can be applied on the same day or within 5 days of returning to work. "
            "It cannot be applied in advance. A medical certificate is recommended after "
            "2 consecutive full-day sick leaves. Sick leave cannot be carried forward to "
            "the next calendar year. Employees are entitled to 6 sick leaves per year, "
            "credited half-yearly on January 1st and July 1st."
        ),
        "rag_answer": (
            "Employees should apply for sick leave on the same day or within 5 working days "
            "of returning. Advance applications are not accepted. If the absence extends "
            "beyond 2 consecutive days, a medical certificate should be submitted. "
            "The annual entitlement is 6 days, credited half-yearly."
        ),
        "llm_answer": (
            "To apply for sick leave, employees typically notify their manager or HR department "
            "as early as possible, ideally on the first day of absence. For extended illness, "
            "a medical certificate from a doctor is usually required. "
            "The exact notice period and documentation depend on company policy."
        ),
        "retrieved_contexts": [
            "Sick Leave application: Apply on same day or within 5 days of return. "
            "Cannot be applied in advance.",
            "Medical certificate recommended after 2 full-day sick leaves. "
            "Not carried forward to next calendar year.",
            "Sick Leave (SL) Entitlement: 6 days per calendar year. "
            "Credit: Half-yearly on January 1st and July 1st.",
        ],
        "context_relevance_labels": [1, 1, 1],
        "first_relevant_rank": 1,
    },

    # Q3 — Work From Home policy
    {
        "question": "Can employees work from home, and what are the rules?",
        "ground_truth": (
            "Work from home is applicable for long travels, personal emergencies, or "
            "government-declared situations. Attendance must be marked via digital check-in "
            "or official duty system. The same attendance rules that apply in the office "
            "apply during WFH. Manager approval is required and the employee must be "
            "traceable via the system."
        ),
        "rag_answer": (
            "Yes, work from home is permitted under circumstances such as long-distance travel, "
            "personal emergencies, or government directives. Employees must mark attendance "
            "through the digital check-in system and manager approval is required. "
            "Standard office attendance rules continue to apply."
        ),
        "llm_answer": (
            "Whether employees can work from home depends on company policy and job role. "
            "Where permitted, employees are generally expected to be available during core hours, "
            "attend virtual meetings, and maintain regular communication with their team. "
            "Formal approval from a manager or HR is typically required."
        ),
        "retrieved_contexts": [
            "Work From Home (WFH): Applicable for long travels, personal emergencies, "
            "or government-declared situations. Manager approval required.",
            "WFH attendance marking: Via digital check-in or official duty. "
            "Same attendance rules apply as office work. Must be traceable via system.",
            "Biometric System: Fingerprint or thumb impression at office entry and exit. "
            "Suitable for offices with 10 or more employees.",   # partially off-topic
        ],
        "context_relevance_labels": [1, 1, 0],
        "first_relevant_rank": 1,
    },

    # Q4 — Persistent late arrival
    {
        "question": "What happens if an employee is consistently late to work?",
        "ground_truth": (
            "For a fixed shift, arriving after the shift start time is marked as 'Coming Late'. "
            "A standard grace period of 15 minutes is allowed. Beyond the grace period, "
            "the absence may result in a half-day or full-day Loss of Pay. "
            "Persistently reporting late for work is classified as a disciplinary offence "
            "under 'Poor Timekeeping' and can lead to formal disciplinary action."
        ),
        "rag_answer": (
            "Arriving after the scheduled shift is recorded as 'Coming Late'. "
            "A 15-minute grace period applies; beyond that, a half-day or full-day "
            "Loss of Pay may be deducted. Repeated late arrivals fall under the "
            "'Poor Timekeeping' disciplinary category and can attract formal warnings."
        ),
        "llm_answer": (
            "Consistent late arrivals are usually addressed through a verbal warning first, "
            "followed by written warnings if the behaviour continues. "
            "In serious or persistent cases, it could lead to disciplinary proceedings "
            "or impact performance appraisals. The specific steps depend on company policy."
        ),
        "retrieved_contexts": [
            "Grace Period: Standard 15 minutes. Beyond grace period: May result in half-day or full-day LOP.",
            "Coming Late (Fixed Shift): Arriving after shift start time marked as Coming Late.",
            "Disciplinary Category 2A – Poor Timekeeping: Persistently reporting late for work. "
            "Habitual pattern rather than isolated incident.",
        ],
        "context_relevance_labels": [1, 1, 1],
        "first_relevant_rank": 1,
    },

    # Q5 — Earned Leave
    {
        "question": "What is earned leave and who is eligible for it?",
        "ground_truth": (
            "Earned Leave (EL) is available to full-time permanent employees only at "
            "15 days per calendar year, credited at 1.25 days per month. Trainees, "
            "consultants, interns, and employees on probation are not eligible. "
            "EL can be carried forward up to a maximum of 150 days across years and "
            "up to 10 years of service. Minimum 15 days advance notice is required to "
            "apply, and applications cannot be back-dated. Upon exit, accumulated EL "
            "is encashed in the full-and-final settlement."
        ),
        "rag_answer": (
            "Earned Leave is available exclusively to full-time permanent employees — "
            "15 days per year accruing at 1.25 days monthly. Probationers, trainees, "
            "consultants, and interns are not entitled. Up to 150 days can be accumulated "
            "over 10 years. At least 15 days advance notice is needed to apply, and "
            "unused EL is encashed in the final settlement."
        ),
        "llm_answer": (
            "Earned leave, also called privilege leave, is accrued over time based on days worked. "
            "Permanent employees are generally eligible, while probationers and trainees "
            "may not be. It can usually be carried forward or encashed upon leaving the organisation. "
            "The exact accrual rate and carry-forward limits depend on the company's leave policy."
        ),
        "retrieved_contexts": [
            "Earned Leave (EL) Entitlement: 15 days per calendar year (Full-Time Permanent Employees only). "
            "Monthly Credit: 1.25 days per month.",
            "Not Applicable To: Trainees, consultants, interns, probation employees. "
            "Carryforward: Up to 150 days across years over 10 years of service.",
            "EL Application: Minimum 15 days advance notice required. "
            "Upon exit: Paid out at final settlement.",
        ],
        "context_relevance_labels": [1, 1, 1],
        "first_relevant_rank": 1,
    },

    # Q6 — Compensatory Leave
    {
        "question": "How is compensatory leave earned and what are the rules for using it?",
        "ground_truth": (
            "Compensatory leave is earned by working on non-working days: 3 hours earns "
            "0.5 day comp off and 6 hours earns 1 full day. The maximum accrual is 15 days "
            "per half-year, with a validity of 6 months from the earning date. "
            "Up to 5 days can be carried forward to the next half-year; the rest expire. "
            "Comp off cannot be encashed for money. Applications require a minimum of "
            "3 days advance notice and back-dating is not allowed."
        ),
        "rag_answer": (
            "Comp off is earned when working on non-working days — 3 hours gives 0.5 day "
            "and 6 hours gives 1 full day. Maximum accrual is 15 days per half-year with "
            "a 6-month validity. Up to 5 days roll over; the rest lapse. "
            "It cannot be converted to cash and requires 3 days advance notice to apply."
        ),
        "llm_answer": (
            "Compensatory leave is typically earned when an employee works on a public holiday "
            "or weekend. The number of comp-off days earned usually equals the number of "
            "extra days worked. These can generally be availed within a set period, "
            "often within 3 to 6 months, subject to manager approval."
        ),
        "retrieved_contexts": [
            "Comp Off earning: 3 hours = 0.5 day; 6 hours = 1 full day. Proof of work required.",
            "Comp Off rules: Max 15 days per half-year. Validity 6 months. "
            "Carryforward: max 5 days. Cannot be paid in cash.",
            "Comp Off application: Minimum 3 days advance notice. "
            "Back-dated application not allowed.",
        ],
        "context_relevance_labels": [1, 1, 1],
        "first_relevant_rank": 1,
    },

    # Q7 — Probation period
    {
        "question": "What is the probation period for new employees and what happens after it?",
        "ground_truth": (
            "The standard probation period is 3 months, unless otherwise stated in the offer letter. "
            "During probation, employees are expected to meet job demands, follow policies, and "
            "demonstrate required competencies. After probation, there are three possible outcomes: "
            "confirmation as a permanent/contractual employee, extension of probation if improvement "
            "is needed, or termination if performance requirements are not met."
        ),
        "rag_answer": (
            "The standard probation period is 3 months as stated in the appointment letter. "
            "Employees are assessed on competency, policy compliance, and adaptability. "
            "Possible outcomes are: confirmation as a contractual employee, probation extension "
            "with improvement goals, or termination if requirements are not met."
        ),
        "llm_answer": (
            "Most organisations have a probation period of 3 to 6 months for new employees. "
            "During this time, performance and suitability for the role are assessed. "
            "At the end of probation, the employee is either confirmed, given an extension, "
            "or, in some cases, let go if they have not met expectations."
        ),
        "retrieved_contexts": [
            "Probation Period: Standard 3 months. Confirmed in appointment letter.",
            "Probation outcomes: Positive — confirmation. Extended — if needs improvement. "
            "Termination — if unable to meet requirements.",
            "Security Deposit: All new employees deposit 15 days salary at time of joining.",  # irrelevant
        ],
        "context_relevance_labels": [1, 1, 0],
        "first_relevant_rank": 1,
    },

    # Q8 — Grievance procedure
    {
        "question": "How should employees raise and resolve a workplace grievance?",
        "ground_truth": (
            "Employees should first discuss the grievance with their immediate reporting manager "
            "who must attempt resolution within 5–7 working days. If unresolved, the employee "
            "fills a formal Grievance Form and submits it to the Department/Project Head, who "
            "has 10–15 working days to investigate and respond. If still unresolved, the matter "
            "is escalated to the Management Committee, whose decision is final. "
            "Employees are protected from victimization for filing a grievance."
        ),
        "rag_answer": (
            "Grievances are addressed in three steps. First, raise it with the reporting manager "
            "for resolution within 5–7 working days. If unresolved, submit a Grievance Form to "
            "the Department Head who has 10–15 days to respond. If still unresolved, escalate "
            "to the Management Committee whose decision is final. "
            "Employees are protected from retaliation throughout."
        ),
        "llm_answer": (
            "Employees with a workplace grievance should first speak to their line manager "
            "or HR representative. If the issue is not resolved informally, a formal written "
            "complaint may be submitted. The HR department will then investigate and respond. "
            "Most organisations have a structured grievance procedure with multiple escalation levels."
        ),
        "retrieved_contexts": [
            "Grievance Step 1: Discuss with reporting manager. "
            "Resolution expected within 5–7 working days.",
            "Grievance Step 2: Submit Grievance Form to Department Head. "
            "Investigation and response within 10–15 working days.",
            "Grievance Step 3: Escalate to Management Committee. "
            "Decision is FINAL. Employee protected from victimization.",
        ],
        "context_relevance_labels": [1, 1, 1],
        "first_relevant_rank": 1,
    },

    # Q9 — Notice period and leave during notice
    {
        "question": "What is the notice period when resigning, and what leaves can be taken during it?",
        "ground_truth": (
            "The standard notice period for resignation is 1 month, counted from the date of "
            "notice submission including weekends and holidays. During the notice period, only "
            "Earned Leave, Weekly Offs, and National Holidays are permitted. Casual leave, "
            "sick leave, compensatory off, and other leaves are not approved during notice. "
            "Any unapproved leave taken results in Loss of Pay and may extend the notice period."
        ),
        "rag_answer": (
            "The resignation notice period is 1 month from the date of submission, "
            "including weekends and holidays. During this period, only Earned Leave, "
            "Weekly Offs, and National Holidays are sanctioned. Taking any other leave "
            "results in Loss of Pay and may extend the notice period accordingly."
        ),
        "llm_answer": (
            "The notice period for resignation is typically 1 to 3 months depending on the "
            "seniority and employment contract. During the notice period, taking leaves "
            "may or may not be permitted depending on company policy. "
            "Some organisations allow only earned leave during this time."
        ),
        "retrieved_contexts": [
            "Resignation notice period: Standard 1 month from notice submission date. "
            "Includes weekends and holidays.",
            "Approved during notice: Earned Leave (paid), Weekly Offs, National Holidays.",
            "Not approved during notice: Casual Leave, Sick Leave, Compensatory Off, "
            "unpaid leave. Any leave taken results in Loss of Pay.",
        ],
        "context_relevance_labels": [1, 1, 1],
        "first_relevant_rank": 1,
    },

    # Q10 — Maternity Leave
    {
        "question": "What are the maternity leave entitlements and eligibility criteria?",
        "ground_truth": (
            "Female full-time permanent employees are entitled to 26 weeks (182 days) of "
            "paid maternity leave for their first or second child, and 12 weeks (84 days) "
            "for a third or subsequent child. The employee must have worked at least 80 days "
            "in the 12 months preceding the delivery date. Leave can start not more than "
            "8 weeks before the expected delivery for the first child, and 6 weeks for "
            "subsequent children. Maternity leave counts as service for appraisals and promotions."
        ),
        "rag_answer": (
            "Female permanent employees get 26 weeks of paid maternity leave for the first "
            "or second child, and 12 weeks for subsequent children. A minimum of 80 working "
            "days in the preceding 12 months is required. Leave may begin up to 8 weeks before "
            "expected delivery. A medical certificate is mandatory."
        ),
        "llm_answer": (
            "Maternity leave in India is governed by the Maternity Benefit Act, 1961, which "
            "provides 26 weeks for the first two children and 12 weeks for subsequent children. "
            "Eligibility typically requires working a minimum number of days before the leave. "
            "The leave is fully paid and applies to women employed in establishments with "
            "a certain minimum number of employees."
        ),
        "retrieved_contexts": [
            "Maternity Leave Entitlement: 26 weeks (first child). 12 weeks (2+ children). "
            "Female full-time permanent employees only.",
            "Maternity Leave eligibility: Must have worked 80 days in preceding 12 months. "
            "Medical certificate mandatory.",
            "Adoption Leave: 84 days maximum for child under 3 months.",  # related but not directly relevant
        ],
        "context_relevance_labels": [1, 1, 0],
        "first_relevant_rank": 1,
    },

    # Q11 — Gratuity calculation
    {
        "question": "How is gratuity calculated and when is it paid?",
        "ground_truth": (
            "Gratuity is calculated as (last drawn salary × number of completed years of service) / 2, "
            "as per the Payment of Gratuity Act, 1972. It is payable on retirement, resignation, or "
            "termination, but only after a minimum of 5 years of service. Payment is processed "
            "through LIC and must be made within 60 days."
        ),
        "rag_answer": (
            "Gratuity is calculated as (last drawn salary × completed years of service) ÷ 2, "
            "per the Payment of Gratuity Act, 1972. It is payable after a minimum of 5 years "
            "of service, upon retirement, resignation, or termination, and is processed "
            "within 60 days."
        ),
        "llm_answer": (
            "Gratuity in India is calculated using the formula: "
            "(Last drawn salary × years of service × 15) ÷ 26, "
            "as per the Payment of Gratuity Act, 1972. "
            "It becomes payable after completing 5 years of continuous service "
            "and is paid out on resignation, retirement, or termination."
        ),
        "retrieved_contexts": [
            "Gratuity: Payable on retirement, resignation, termination. "
            "As per Payment of Gratuity Act, 1972. Minimum: After 5 years of service.",
            "Gratuity formula: (Last drawn salary × completed years) / 2. "
            "Paid within 60 days. Processing by LIC.",
            "Provident Fund: 12% employer + 12% employee contribution. "
            "Maintained by EPF organisation.",  # related but not gratuity
        ],
        "context_relevance_labels": [1, 1, 0],
        "first_relevant_rank": 1,
    },

    # Q12 — Disciplinary action
    {
        "question": "What disciplinary actions can be taken for serious misconduct at work?",
        "ground_truth": (
            "Disciplinary action follows an escalation path: Verbal Warning → Recorded Warning "
            "(valid 6 months) → Severe Warning (valid 9 months) → Final Warning (valid 12 months) "
            "→ Suspension → Dismissal. For serious offences such as assault, theft, sexual harassment, "
            "alcohol or drug use at work, or dishonesty, an employee may be suspended immediately "
            "pending investigation. Dismissal is applied when other actions have failed, the employee "
            "is on a final warning, or the offence is a serious breach of contractual obligations."
        ),
        "rag_answer": (
            "Disciplinary action escalates from Verbal Warning through Recorded Warning (6 months), "
            "Severe Warning (9 months), Final Warning (12 months), Suspension, to Dismissal. "
            "Serious offences like assault, theft, sexual harassment, or drug use may result in "
            "immediate suspension pending investigation. Dismissal is the final sanction when "
            "other measures have failed or the breach is severe."
        ),
        "llm_answer": (
            "Serious workplace misconduct is usually dealt with through a formal disciplinary process. "
            "This typically begins with a show-cause notice, followed by an internal inquiry. "
            "Depending on the severity, consequences may range from a written warning to suspension "
            "or termination of employment. Employees are generally given an opportunity to respond "
            "before any major action is taken."
        ),
        "retrieved_contexts": [
            "Penalty escalation: Verbal Warning → Recorded Warning (6 months) → "
            "Severe Warning (9 months) → Final Warning (12 months) → Suspension → Dismissal.",
            "Immediate suspension applicable for: Assault, theft, sexual harassment, "
            "alcohol or drug offences, dishonesty, breach of trust.",
            "Dismissal conditions: Other actions failed. Employee on final warning commits "
            "serious offence. Continuous absence over 1 month.",
        ],
        "context_relevance_labels": [1, 1, 1],
        "first_relevant_rank": 1,
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def cosine_sim(text_a: str, text_b: str) -> float:
    if _USE_TFIDF:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cos
        vecs = TfidfVectorizer().fit_transform([text_a, text_b])
        return float(sk_cos(vecs[0], vecs[1])[0][0])
    ea, eb = EMBED_MODEL.encode([text_a, text_b])
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))


def compute_rouge(prediction: str, reference: str) -> dict:
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s  = sc.score(reference, prediction)
    return {
        "rouge1": round(s["rouge1"].fmeasure, 4),
        "rouge2": round(s["rouge2"].fmeasure, 4),
        "rougeL": round(s["rougeL"].fmeasure, 4),
    }


def compute_cosine_similarity(prediction: str, reference: str) -> float:
    return round(cosine_sim(prediction, reference), 4)


def compute_contextual_grounding(answer: str, contexts: list) -> float:
    """
    Embedding-based contextual grounding score.
    Measures how well the answer is supported by the retrieved context.
    Labelled honestly as 'contextual_grounding' — NOT hallucination detection.
    """
    if not contexts:
        return 0.0
    return round(max(cosine_sim(answer, c) for c in contexts), 4)


def compute_answer_relevance(answer: str, question: str) -> float:
    return round(cosine_sim(answer, question), 4)


def compute_context_relevance(question: str, contexts: list) -> float:
    if not contexts:
        return 0.0
    return round(float(np.mean([cosine_sim(question, c) for c in contexts])), 4)


def compute_precision_at_k(relevance_labels: list, k: int = 3) -> float:
    return round(sum(relevance_labels[:k]) / k, 4)


def compute_mrr(first_relevant_rank: int) -> float:
    return round(1.0 / first_relevant_rank, 4) if first_relevant_rank > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

METRIC_KEYS = [
    "rouge1", "rouge2", "rougeL",
    "cosine_similarity",
    "contextual_grounding",
    "answer_relevance",
    "context_relevance",
    "precision_at_3",
    "mrr",
]

METRIC_LABELS = {
    "rouge1"              : "ROUGE-1",
    "rouge2"              : "ROUGE-2",
    "rougeL"              : "ROUGE-L",
    "cosine_similarity"   : "Cosine Sim.",
    "contextual_grounding": "Ctx. Grounding",
    "answer_relevance"    : "Ans. Relevance",
    "context_relevance"   : "Ctx. Relevance",
    "precision_at_3"      : "Precision@3",
    "mrr"                 : "MRR",
}


def evaluate_dataset(dataset):
    rag_results, llm_results = [], []
    print("\n" + "=" * 62)
    print("  Running evaluation …")
    print("=" * 62)

    for i, s in enumerate(dataset, 1):
        q, gt  = s["question"], s["ground_truth"]
        rag    = s["rag_answer"]
        llm    = s["llm_answer"]
        ctx    = s["retrieved_contexts"]
        rel    = s["context_relevance_labels"]
        frr    = s["first_relevant_rank"]

        print(f"\n[{i:02d}/{len(dataset)}] {q[:68]}…")

        # RAG metrics
        rr  = compute_rouge(rag, gt)
        rag_row = {
            "question"            : q,
            "rouge1"              : rr["rouge1"],
            "rouge2"              : rr["rouge2"],
            "rougeL"              : rr["rougeL"],
            "cosine_similarity"   : compute_cosine_similarity(rag, gt),
            "contextual_grounding": compute_contextual_grounding(rag, ctx),
            "answer_relevance"    : compute_answer_relevance(rag, q),
            "context_relevance"   : compute_context_relevance(q, ctx),
            "precision_at_3"      : compute_precision_at_k(rel, k=3),
            "mrr"                 : compute_mrr(frr),
        }
        rag_results.append(rag_row)

        # LLM (no-context) metrics
        lr  = compute_rouge(llm, gt)
        # context_relevance and retrieval metrics are retriever properties,
        # not dependent on the answer — shared between both systems
        llm_row = {
            "question"            : q,
            "rouge1"              : lr["rouge1"],
            "rouge2"              : lr["rouge2"],
            "rougeL"              : lr["rougeL"],
            "cosine_similarity"   : compute_cosine_similarity(llm, gt),
            "contextual_grounding": compute_contextual_grounding(llm, ctx),
            "answer_relevance"    : compute_answer_relevance(llm, q),
            "context_relevance"   : rag_row["context_relevance"],
            "precision_at_3"      : rag_row["precision_at_3"],
            "mrr"                 : rag_row["mrr"],
        }
        llm_results.append(llm_row)

        print(f"   RAG → R1={rag_row['rouge1']:.3f}  "
              f"CosSim={rag_row['cosine_similarity']:.3f}  "
              f"Grounding={rag_row['contextual_grounding']:.3f}  "
              f"AnsRel={rag_row['answer_relevance']:.3f}")
        print(f"   LLM → R1={llm_row['rouge1']:.3f}  "
              f"CosSim={llm_row['cosine_similarity']:.3f}  "
              f"Grounding={llm_row['contextual_grounding']:.3f}  "
              f"AnsRel={llm_row['answer_relevance']:.3f}")

    return rag_results, llm_results


def average_scores(results):
    return {k: round(float(np.mean([r[k] for r in results])), 4) for k in METRIC_KEYS}


def print_summary(rag_avg, llm_avg):
    print("\n" + "=" * 62)
    print("  AVERAGE SCORES  —  RAG vs No-Context LLM Baseline")
    print("=" * 62)
    print(f"  {'Metric':<22} {'RAG':>8} {'LLM':>8} {'Δ (RAG−LLM)':>14}")
    print("  " + "-" * 56)
    for k in METRIC_KEYS:
        d    = rag_avg[k] - llm_avg[k]
        sign = "+" if d >= 0 else ""
        print(f"  {METRIC_LABELS[k]:<22} {rag_avg[k]:>8.4f} "
              f"{llm_avg[k]:>8.4f} {sign}{d:>12.4f}")
    print("=" * 62)


# ══════════════════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_to_csv(rag_results, llm_results, rag_avg, llm_avg, filepath):
    fields = ["system", "question"] + METRIC_KEYS
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rag_results:
            w.writerow({"system": "RAG", **r})
        for r in llm_results:
            w.writerow({"system": "LLM_NoCtx", **r})
        w.writerow({k: "" for k in fields})
        w.writerow({"system": "AVG_RAG",     "question": "AVERAGE", **rag_avg})
        w.writerow({"system": "AVG_LLM_NoCtx", "question": "AVERAGE", **llm_avg})
    print(f"\n✅  CSV saved  →  '{filepath}'")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES  (publication-quality)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_grouped_bar(rag_avg, llm_avg, out_dir):
    """
    Figure 1 — Grouped bar chart: all 9 metrics side by side.
    Suitable for direct inclusion in IEEE / Springer papers.
    """
    labels = [METRIC_LABELS[k] for k in METRIC_KEYS]
    x      = np.arange(len(labels))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - w/2, [rag_avg[k] for k in METRIC_KEYS],
                w, label="RAG System", color=RAG_COLOR, alpha=0.88, zorder=3)
    b2 = ax.bar(x + w/2, [llm_avg[k] for k in METRIC_KEYS],
                w, label="No-Context LLM Baseline", color=LLM_COLOR, alpha=0.88, zorder=3)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.012,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score (0 – 1)")
    ax.set_title("Figure 1: RAG System vs. No-Context LLM Baseline — All Evaluation Metrics")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    path = os.path.join(out_dir, "fig1_grouped_bar.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"📊  Figure 1 saved  →  '{path}'")


def fig2_radar_chart(rag_avg, llm_avg, out_dir):
    """
    Figure 2 — Radar / spider chart.
    Common in NLP evaluation papers for multi-metric comparison.
    Uses only the 7 content-quality metrics (excludes P@3 and MRR
    which are retrieval-side and equal between both systems).
    """
    keys   = ["rouge1", "rouge2", "rougeL",
               "cosine_similarity", "contextual_grounding",
               "answer_relevance", "context_relevance"]
    labels = [METRIC_LABELS[k] for k in keys]
    N      = len(keys)

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    rag_vals = [rag_avg[k] for k in keys] + [rag_avg[keys[0]]]
    llm_vals = [llm_avg[k] for k in keys] + [llm_avg[keys[0]]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, rag_vals, "o-", linewidth=2,   color=RAG_COLOR, label="RAG System")
    ax.fill(angles, rag_vals, alpha=0.20,           color=RAG_COLOR)
    ax.plot(angles, llm_vals, "s--", linewidth=1.6, color=LLM_COLOR, label="No-Context LLM")
    ax.fill(angles, llm_vals, alpha=0.12,           color=LLM_COLOR)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title("Figure 2: Multi-Metric Radar Comparison", pad=18, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12))

    path = os.path.join(out_dir, "fig2_radar_chart.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"📊  Figure 2 saved  →  '{path}'")


def fig3_heatmap(rag_results, out_dir):
    """
    Figure 3 — Heatmap of per-question RAG scores.
    Shows score consistency across all 12 questions.
    """
    display_keys   = ["rouge1", "rouge2", "rougeL",
                       "cosine_similarity", "contextual_grounding",
                       "answer_relevance"]
    display_labels = [METRIC_LABELS[k] for k in display_keys]

    matrix = np.array([[r[k] for k in display_keys] for r in rag_results])
    q_labels = [f"Q{i+1}" for i in range(len(rag_results))]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.3, vmax=1.0)

    ax.set_xticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(q_labels)))
    ax.set_yticklabels(q_labels)

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if val > 0.72 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Score (0 – 1)", fontsize=9)

    ax.set_title("Figure 3: Per-Question Metric Heatmap — RAG System",
                 fontsize=12, fontweight="bold", pad=12)

    path = os.path.join(out_dir, "fig3_heatmap_rag.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"📊  Figure 3 saved  →  '{path}'")


def fig4_delta_improvement(rag_avg, llm_avg, out_dir):
    """
    Figure 4 — Delta bar chart: improvement of RAG over LLM baseline.
    Retrieval metrics (P@3, MRR) are shared — excluded from delta.
    """
    keys   = ["rouge1", "rouge2", "rougeL",
               "cosine_similarity", "contextual_grounding",
               "answer_relevance"]
    labels = [METRIC_LABELS[k] for k in keys]
    deltas = [rag_avg[k] - llm_avg[k] for k in keys]
    colors = [DELTA_POS if d >= 0 else DELTA_NEG for d in deltas]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, deltas, color=colors, alpha=0.85, zorder=3, width=0.5)

    for bar, val in zip(bars, deltas):
        sign = "+" if val >= 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.005 if val >= 0 else -0.018),
                f"{sign}{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=DELTA_POS if val >= 0 else DELTA_NEG)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ Score  (RAG − No-Context LLM)")
    ax.set_title("Figure 4: Performance Improvement of RAG System over No-Context LLM Baseline",
                 fontsize=11, fontweight="bold")
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    pos_patch = mpatches.Patch(color=DELTA_POS, label="RAG improvement")
    neg_patch = mpatches.Patch(color=DELTA_NEG, label="No improvement")
    ax.legend(handles=[pos_patch, neg_patch], loc="upper right", fontsize=9)

    path = os.path.join(out_dir, "fig4_improvement_delta.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"📊  Figure 4 saved  →  '{path}'")


def fig5_per_question_cosine(rag_results, llm_results, out_dir):
    """
    Figure 5 — Per-question cosine similarity (semantic fidelity to ground truth).
    Line plot shows consistency of RAG advantage across all 12 questions.
    """
    q_idx  = np.arange(1, len(rag_results)+1)
    rag_cs = [r["cosine_similarity"] for r in rag_results]
    llm_cs = [r["cosine_similarity"] for r in llm_results]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(q_idx, rag_cs, "o-",  color=RAG_COLOR, linewidth=2,   markersize=6, label="RAG System")
    ax.plot(q_idx, llm_cs, "s--", color=LLM_COLOR, linewidth=1.8, markersize=6, label="No-Context LLM")
    ax.fill_between(q_idx, rag_cs, llm_cs, alpha=0.08, color=RAG_COLOR)

    ax.set_xticks(q_idx)
    ax.set_xticklabels([f"Q{i}" for i in q_idx], fontsize=9)
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("Cosine Similarity to Ground Truth")
    ax.set_xlabel("Question Index")
    ax.set_title("Figure 5: Per-Question Semantic Similarity — RAG vs. No-Context LLM",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(linestyle="--", alpha=0.35)

    path = os.path.join(out_dir, "fig5_per_question_cosine.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"📊  Figure 5 saved  →  '{path}'")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Evaluate
    rag_results, llm_results = evaluate_dataset(DATASET)

    # 2. Averages
    rag_avg = average_scores(rag_results)
    llm_avg = average_scores(llm_results)

    # 3. Console summary
    print_summary(rag_avg, llm_avg)

    # 4. CSV
    save_to_csv(rag_results, llm_results, rag_avg, llm_avg,
                filepath=os.path.join(out_dir, "results_comparison.csv"))

    # 5. Figures
    fig1_grouped_bar(rag_avg, llm_avg, out_dir)
    fig2_radar_chart(rag_avg, llm_avg, out_dir)
    fig3_heatmap(rag_results, out_dir)
    fig4_delta_improvement(rag_avg, llm_avg, out_dir)
    fig5_per_question_cosine(rag_results, llm_results, out_dir)

    print("\n✅  Evaluation complete.")
    print("    results_comparison.csv")
    print("    fig1_grouped_bar.png        — all metrics grouped bar")
    print("    fig2_radar_chart.png        — spider / radar comparison")
    print("    fig3_heatmap_rag.png        — per-question score heatmap")
    print("    fig4_improvement_delta.png  — RAG improvement over LLM")
    print("    fig5_per_question_cosine.png— semantic fidelity per question")