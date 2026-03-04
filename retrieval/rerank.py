import re
from llm.query_understanding import understand_query


def normalize(text):
    return re.sub(r"[^\w\s]", " ", text.lower())


def extract_duration(query):
    m = re.search(r"(\d+)\s*(minute|min)", query.lower())
    if m:
        return int(m.group(1))
    return None


def compute_score(item, query, structured):

    score = 0
    q = normalize(query)

    name = normalize(item.get("name", ""))
    desc = normalize(item.get("description", ""))
    url = item.get("url", "").lower()

    # --------------------
    # Skill match
    # --------------------
    for skill in structured["required_skills"]:
        skill_clean = skill.replace("\\b", "")
        if skill_clean in name or skill_clean in desc:
            score += 3

    for skill in structured["soft_skills"]:
        if skill in name or skill in desc:
            score += 1.5

    # --------------------
    # Technical query boost
    # --------------------
    if any(k in q for k in ["python", "java", "sql", "javascript", "coding"]):
        if any(k in url for k in ["automata", "python", "java", "sql", "framework"]):
            score += 3

    # --------------------
    # Communication query boost
    # --------------------
    if any(k in q for k in ["english", "communication", "call center", "customer"]):
        if any(k in url for k in ["verbal", "communication", "svar"]):
            score += 3

    # --------------------
    # Sales queries
    # --------------------
    if "sales" in q and "sales" in url:
        score += 2

    # --------------------
    # Analyst queries
    # --------------------
    if any(k in q for k in ["analyst", "data", "finance"]):
        if any(k in url for k in ["numerical", "reasoning", "analysis"]):
            score += 2

    # --------------------
    # Duration match
    # --------------------
    requested = extract_duration(query)
    duration = item.get("duration")

    if requested and duration:
        if duration <= requested:
            score += 2
        else:
            score -= 1

    # --------------------
    # Penalize reports
    # --------------------
    if "report" in url:
        score -= 3

    return score


# --------------------
# Skill priority ordering
# --------------------
def reorder_by_skill_priority(results, query):

    q = query.lower()

    skill_order = []

    if "python" in q:
        skill_order.append("python")

    if "sql" in q:
        skill_order.append("sql")

    if "java" in q:
        skill_order.append("java")

    if "javascript" in q:
        skill_order.append("javascript")

    if not skill_order:
        return results

    grouped = {skill: [] for skill in skill_order}
    others = []

    for r in results:

        name = r.get("name", "").lower()
        url = r.get("url", "").lower()

        placed = False

        for skill in skill_order:
            if skill in name or skill in url:
                grouped[skill].append(r)
                placed = True
                break

        if not placed:
            others.append(r)

    ordered = []

    for skill in skill_order:
        ordered.extend(grouped[skill])

    ordered.extend(others)

    return ordered


# --------------------
# Main reranker
# --------------------
def balance_and_rerank(candidates, query, top_k=10):

    structured = understand_query(query)

    scored = []

    for item in candidates:
        s = compute_score(item, query, structured)
        scored.append((s, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    # -----------------------------
    # Extract skill order from query
    # -----------------------------
    q = query.lower()

    skill_order = []

    for skill in structured["required_skills"]:
        s = skill.replace("\\b", "").lower()
        if s not in skill_order:
            skill_order.append(s)

    skill1 = skill_order[0] if len(skill_order) >= 1 else None
    skill2 = skill_order[1] if len(skill_order) >= 2 else None

    both = []
    skill1_only = []
    skill2_only = []
    others = []

    for _, item in scored:

        name = item.get("name", "").lower()
        url = item.get("url", "").lower()

        has1 = skill1 and (skill1 in name or skill1 in url)
        has2 = skill2 and (skill2 in name or skill2 in url)

        if has1 and has2:
            both.append(item)
        elif has1:
            skill1_only.append(item)
        elif has2:
            skill2_only.append(item)
        else:
            others.append(item)

    ordered = both + skill1_only + skill2_only + others

    # limit to top_k
    return ordered[:top_k]