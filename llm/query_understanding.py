# llm/query_understanding.py
import re
import json
import os

ROLE_MAPPINGS = [
    (r"coo|ceo|cfo|cto|c-suite|chief officer|vice president|\bvp\b",
     "OPQ leadership enterprise personality occupational personality questionnaire executive report"),
    (r"manager|team lead|head of|director",
     "OPQ leadership management personality competencies MQ motivation manager JFA job focused assessment"),
    (r"graduate|entry.level|fresher|new grad|junior",
     "graduate aptitude reasoning verify ability situational judgement biodata"),
    (r"sales|business development|account manager",
     "sales personality behaviour situational judgement biodata entry level sales"),
    (r"\bjava\b",
     "Java core java knowledge skills simulation automata"),
    (r"\bpython\b",
     "Python knowledge skills programming simulation"),
    (r"\bsql\b",
     "SQL data analysis numerical reasoning knowledge skills automata"),
    (r"\bexcel\b",
     "Microsoft Excel 365 knowledge skills simulation essentials"),
    (r"\bselenium\b",
     "Selenium automata coding simulation testing knowledge skills automation"),
    (r"\bjavascript\b|\bjs\b",
     "JavaScript knowledge skills programming simulation front-end"),
    (r"\bhtml\b|\bcss\b",
     "HTML CSS knowledge skills front-end simulation"),
    (r"qa engineer|quality assurance|manual test|automation test|software test",
     "Selenium testing simulation automata knowledge skills manual testing software"),
    (r"data analyst|data scientist",
     "SQL Python numerical reasoning data analysis knowledge skills verify"),
    (r"data warehouse|warehousing",
     "data warehousing SQL knowledge skills"),
    (r"developer|engineer|programmer|software|coding",
     "coding simulation knowledge skills programming verify automata"),
    (r"admin|clerical|administrative",
     "administrative clerical data entry knowledge skills short form"),
    (r"consultant",
     "OPQ personality occupational personality questionnaire competencies MQ verbal reasoning"),
    (r"marketing",
     "verbal reasoning personality competencies manager OPQ digital advertising inductive reasoning WriteX email"),
    (r"digital.advert|adwords|seo|search engine",
     "digital advertising SEO knowledge skills"),
    (r"contact.center|customer.service|call.center",
     "contact center simulation situational judgement biodata"),
    (r"finance|financial|accounting|accountant",
     "numerical reasoning verify financial knowledge skills"),
    (r"radio|broadcast|station|sound|audio|music",
     "verbal reasoning English comprehension marketing knowledge skills interpersonal communication personality OPQ"),
    (r"writer|writing|content.creat",
     "verbal English writing comprehension knowledge skills WriteX email"),
    (r"english|language|communicat",
     "English comprehension verbal ability SVAR spoken interpersonal communication"),
    (r"bank|banking|icici|hdfc|sbi",
     "administrative clerical numerical reasoning verify short form bank"),
    (r"hr |human resource|recruiter|talent acquisition",
     "personality OPQ competencies situational judgement"),
     (r"marketing manager|brand|branding",
 "marketing manager OPQ personality leadership competencies verbal reasoning"),
 (r"consultant|consulting",
 "OPQ occupational personality questionnaire competencies manager"),

(r"qa|quality assurance|testing",
 "Selenium automata coding simulation testing"),

(r"graduate sales|entry level sales",
 "entry level sales situational judgement biodata personality"),

(r"executive|coo|ceo|c-suite",
 "OPQ leadership enterprise leadership report executive personality"),
]
TYPE_TO_ROLES = {
    "Personality & Behaviour": "personality leadership executive manager consultant culture fit behavioural",
    "Ability & Aptitude": "aptitude cognitive reasoning graduate entry level analytical problem solving",
    "Knowledge & Skills": "technical programming coding developer engineer python java sql skills",
    "Competencies": "leadership teamwork communication stakeholder management consulting",
    "Simulations": "simulation coding programming developer engineer technical hands-on",
    "Biodata & Situational Judgement": "situational judgement entry level sales customer service graduate",
    "Development & 360": "leadership development executive feedback senior management",
    "Assessment Exercises": "assessment centre leadership managerial group exercise business case"
}
NEED_MAPPINGS = [
    (r"cultural.fit|culture|values|right fit",
     "occupational personality questionnaire OPQ personality behaviour"),
    (r"collaborat|teamwork|team player",
     "competencies interpersonal OPQ personality"),
    (r"communicat|interpersonal",
     "verbal communication interpersonal competencies"),
    (r"leadership|lead a team",
     "OPQ leadership enterprise report management"),
    (r"cognitive|analytical|problem.solv|logical",
     "verify numerical verbal inductive reasoning aptitude"),
    (r"motivat",
     "MQ motivation personality questionnaire"),
    (r"personality|behaviour|behavioral",
     "occupational personality questionnaire OPQ personality behaviour"),
    (r"strategic|strategy|brand|positioning",
     "OPQ personality management competencies inductive reasoning manager JFA"),
    (r"numerical|quantitative|data|analytics",
     "verify numerical reasoning aptitude knowledge skills"),
    (r"verbal|written|english|reading",
     "verify verbal reasoning English comprehension written communication"),
    (r"email|writing|written",
     "WriteX email writing knowledge skills communication"),
     (r"qa engineer|qa tester|quality assurance",
 "Selenium automata testing software test knowledge skills simulation"),
]

LEVEL_MAPPINGS = [
    (r"senior|experienced|lead|\b[5-9]\+?\s*years",   "Senior"),
    (r"junior|entry.level|graduate|fresher|0-2 years", "Entry-Level"),
    (r"manager|head of|director",                      "Manager"),
    (r"executive|c-suite|coo|ceo|cfo|cto|\bvp\b",     "Executive"),
    (r"mid|intermediate|3-5 years",                    "Mid-Professional"),
]

# Extract these skill keywords directly from full query text
SKILL_KEYWORDS = [
    ("selenium",        "Selenium automata coding simulation testing"),
    ("javascript",      "JavaScript knowledge skills programming"),
    ("html",            "HTML CSS knowledge skills front-end"),
    ("css",             "CSS knowledge skills front-end"),
    ("sql server",      "SQL Server knowledge skills database"),
    ("manual testing",  "Manual Testing knowledge skills software"),
    ("automation",      "automation testing Selenium knowledge skills"),
    ("java",            "Java core java knowledge skills automata"),
    ("python",          "Python knowledge skills programming"),
    ("sql",             "SQL data analysis knowledge skills automata"),
    ("excel",           "Microsoft Excel 365 knowledge skills"),
    ("digital advert",  "Digital Advertising knowledge skills"),
    ("adwords",         "Digital Advertising knowledge skills"),
    ("seo",             "Search Engine Optimization knowledge skills"),
    ("inductive",       "SHL Verify Interactive Inductive Reasoning ability"),
    ("verbal",          "Verify Verbal Ability reasoning English"),
    ("numerical",       "Verify Numerical Ability reasoning"),
    ("personality",     "OPQ occupational personality questionnaire"),
    ("data warehouse",  "Data Warehousing knowledge skills SQL"),
    ("email writing",   "WriteX email writing communication"),
    ("spoken english",  "SVAR spoken English language"),
    ("brand",           "OPQ personality manager JFA inductive reasoning"),
    ("sound|audio|radio|broadcast", "verbal reasoning marketing English comprehension personality"),
]

CACHE_FILE = "evaluation/query_cache.json"
_cache = {}

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        _cache = json.load(f)


def _save_cache():
    os.makedirs("evaluation", exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(_cache, f, indent=2)


def _rule_based_expand(query: str) -> dict:
    # Use FULL query text — not truncated
    query_lower = query.lower()
    matched_terms = []
    required_skills = []
    soft_skills = []
    job_level = ""

    for pattern, terms in ROLE_MAPPINGS:
        if re.search(pattern, query_lower):
            matched_terms.append(terms)
            required_skills.append(pattern.split("|")[0].strip())

    for pattern, terms in NEED_MAPPINGS:
        if re.search(pattern, query_lower):
            matched_terms.append(terms)
            soft_skills.append(pattern.split("|")[0].strip())

    for pattern, level in LEVEL_MAPPINGS:
        if re.search(pattern, query_lower):
            job_level = level
            break

    # Extract skill keywords from full query text
    for skill_pattern, terms in SKILL_KEYWORDS:
        if re.search(skill_pattern, query_lower):
            if terms not in matched_terms:
                matched_terms.append(terms)

    expanded = query + " " + " ".join(matched_terms) if matched_terms else query

    return {
        "required_skills": required_skills,
        "soft_skills": soft_skills,
        "job_level": job_level,
        "inferred_test_types": [],
        "expanded_query": expanded,
    }
def clean_query(q):
    q = q.strip()
    if len(q) > 800:
        q = q[:800]
    return q


def understand_query(query: str) -> dict:
    if query in _cache:
        return _cache[query]

    result = _rule_based_expand(query)
    _cache[query] = result
    _save_cache()
    return result


if __name__ == "__main__":
    test_queries = [
        "We're looking for a Marketing Manager who can drive brand positioning",
        "Find me 1 hour long assessment for QA Engineer with Selenium, JavaScript, HTML/CSS experience",
        "KEY RESPONSIBILITIES: Manage the sound-scape of the station through creative marketing",
        "I want to hire new graduates for a sales role, budget about an hour",
        "Senior Data Analyst with SQL, Excel and Python, data warehousing experience",
        "Content Writer required, expert in English and SEO",
    ]
    for q in test_queries:
        result = understand_query(q)
        print(f"\nQ: {q[:80]}")
        print(f"  expanded: {result['expanded_query'][:300]}")
        print(f"  skills:   {result['required_skills']}")