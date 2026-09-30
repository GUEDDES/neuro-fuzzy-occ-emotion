"""Fuzzy OCC Inference Engine (Mamdani FIS).
Implements the 52 OCC rules, Ruspini partitions, soft/hard agent modes,
centroid intensity calculation, decision threshold, and faithful XAI explanations.
"""

from itertools import product
import numpy as np

from config import (
    AGENTS,
    COMPOUND,
    D_NAMES,
    DEPENDS,
    EXTREME,
    GRID,
    INPUT_TERMS,
    OUTPUT_TERMS,
    P_NAMES,
    TO_EMOWOZ,
    TO_ENVENT,
    VALENCE_TERMS,
)

_integrate = getattr(np, "trapezoid", None) or np.trapz


def trapezoid(x, a1, a2, a3, a4):
    """Vectorized trapezoidal membership function."""
    left = 1.0 if a2 == a1 else (x - a1) / (a2 - a1)
    right = 1.0 if a4 == a3 else (a4 - x) / (a4 - a3)
    return np.maximum(0.0, np.minimum(np.minimum(left, 1.0), right))


def intensity_term(emotion, d_term, p_term):
    """Low if no antecedent on V(e) is extreme, High if all are, Medium otherwise."""
    terms = {"d": d_term, "p": p_term}
    relevant = [terms[v] for v in DEPENDS[emotion]]
    k = sum(t in EXTREME for t in relevant)
    if k == 0:
        return "Low"
    return "High" if k == len(relevant) else "Medium"


def build_rules():
    """Build all 52 rules: (agent, d_term, p_term or None, emotion, intensity_term)."""
    rules = []
    for agent, cells in GRID.items():
        for (dv, pv), emotions in cells.items():
            for d_term, p_term in product(VALENCE_TERMS[dv], VALENCE_TERMS[pv]):
                for e in emotions:
                    rules.append((agent, d_term, p_term, e, intensity_term(e, d_term, p_term)))
    for dv, e in (("-", "Distress"), ("+", "Joy")):
        for d_term in VALENCE_TERMS[dv]:
            rules.append(("Circumstance", d_term, None, e, intensity_term(e, d_term, "0")))
    return rules


class FuzzyOCCEngine:
    def __init__(
        self,
        rules=None,
        n_grid=1001,
        use_calibrated=False,
        use_it2=False,
        delta=0.03,
        weights=None,
    ):
        from config import (
            CALIBRATED_WEIGHTS,
            INPUT_TERMS_CALIBRATED,
            IT2_DELTA,
            W_NEUTRAL,
        )

        self.rules = rules or build_rules()
        self.n_grid = n_grid
        self.use_calibrated = use_calibrated
        self.use_it2 = use_it2
        self.delta = delta if delta is not None else IT2_DELTA
        self.y_grid = np.linspace(0.0, 1.0, n_grid)
        self.output_mfs = {
            term: trapezoid(self.y_grid, *params)
            for term, params in OUTPUT_TERMS.items()
        }

        if use_calibrated:
            self.input_terms = INPUT_TERMS_CALIBRATED
            self.weights = weights or CALIBRATED_WEIGHTS
            self.w_neutral = W_NEUTRAL
        else:
            self.input_terms = INPUT_TERMS
            self.weights = weights or {r[3]: 1.0 for r in self.rules}
            self.w_neutral = 0.0

    def mf_eval(self, x, term):
        params = self.input_terms[term]
        if self.use_it2 and self.delta > 0:
            a1, a2, a3, a4 = params
            d = self.delta
            if a1 <= -0.99 and a2 <= -0.99:
                lmf = trapezoid(x, -1.0, -1.0, a3 - d, a4 - d)
                umf = trapezoid(x, -1.0, -1.0, min(1.0, a3 + d), min(1.0, a4 + d))
            elif a3 >= 0.99 and a4 >= 0.99:
                lmf = trapezoid(x, a1 + d, a2 + d, 1.0, 1.0)
                umf = trapezoid(x, max(-1.0, a1 - d), max(-1.0, a2 - d), 1.0, 1.0)
            else:
                lmf = trapezoid(x, a1 + d, a2 + d * 0.5, a3 - d * 0.5, a4 - d)
                umf = trapezoid(x, a1 - d, a2 - d * 0.5, a3 + d * 0.5, a4 + d)
            return float(0.5 * (lmf + umf))
        return float(trapezoid(x, *params))

    def infer(self, d, p, pi, theta=0.0, crisp=False, hard_agent=False):
        """Perform fuzzy inference for single input (d, p, pi).
        pi: dict or array with keys ('Self', 'Other', 'Circumstance')
        """
        if isinstance(pi, (list, tuple, np.ndarray)):
            pi = dict(zip(AGENTS, pi))

        if hard_agent:
            best_agent = max(pi.items(), key=lambda kv: kv[1])[0]
            pi = {a: (1.0 if a == best_agent else 0.0) for a in AGENTS}

        if crisp:
            # Boolean terms
            best_d = max(self.input_terms.keys(), key=lambda t: float(trapezoid(d, *self.input_terms[t])))
            best_p = max(self.input_terms.keys(), key=lambda t: float(trapezoid(p, *self.input_terms[t])))
            best_agent = max(pi.items(), key=lambda kv: kv[1])[0]

            profile = {}
            for agent, d_term, p_term, e, tau in self.rules:
                match = (agent == best_agent) and (d_term == best_d) and (p_term is None or p_term == best_p)
                if match:
                    mid = {"Low": 0.2, "Medium": 0.5, "High": 0.8}[tau]
                    rule = (agent, d_term, p_term, e, tau)
                    profile[e] = (1.0, mid, tau, (1.0, rule))
            return profile

        agg = {}
        rule_firings = {}  # e -> list of (f_k, rule)
        for rule in self.rules:
            agent, d_term, p_term, e, tau = rule
            w_e = self.weights.get(e, 1.0)
            strengths = [self.mf_eval(d, d_term), float(pi[agent])]
            if p_term is not None:
                strengths.append(self.mf_eval(p, p_term))
            f_k = min(strengths) * w_e
            if f_k > 0:
                clipped = np.minimum(f_k, self.output_mfs[tau])
                agg[e] = np.maximum(agg.get(e, 0.0), clipped)
                if e not in rule_firings:
                    rule_firings[e] = []
                rule_firings[e].append((f_k, rule))

        profile = {}
        for e, mu in agg.items():
            alpha = float(np.max(mu))
            if alpha < theta:
                continue
            denom = float(_integrate(mu, self.y_grid))
            if denom > 0:
                intensity = float(_integrate(self.y_grid * mu, self.y_grid) / denom)
            else:
                intensity = 0.5
            term = max(OUTPUT_TERMS, key=lambda t: float(trapezoid(intensity, *OUTPUT_TERMS[t])))
            best_rule_info = max(rule_firings[e], key=lambda x: x[0]) if e in rule_firings else (0.0, None)
            profile[e] = (alpha, intensity, term, best_rule_info)

        return profile

    def decide(self, d, p, pi, theta=0.35, dataset="envent", crisp=False, hard_agent=False):
        """Produce final label, intensity, and explanation."""
        profile = self.infer(d, p, pi, theta=0.0, crisp=crisp, hard_agent=hard_agent)
        mapping = TO_ENVENT if dataset.lower() == "envent" else TO_EMOWOZ

        # Calculate neutral appraisal score if calibrated
        neu_score = 0.0
        if self.use_calibrated and self.w_neutral > 0:
            mu_d0 = self.mf_eval(d, "0")
            mu_p0 = self.mf_eval(p, "0")
            neu_score = self.w_neutral * min(mu_d0, mu_p0)

        max_alpha = max((v[0] for v in profile.values()), default=0.0)

        # Predict neutral if threshold not reached or neutral appraisal exceeds emotion activation
        if max_alpha < theta or (self.use_calibrated and neu_score > max_alpha):
            null_label = mapping[None]
            explanation = "No appraisal pattern reaches activation threshold theta (or neutral appraisal dominates)."
            return null_label, None, float(max_alpha), 0.0, "None", explanation, profile

        # Top-ranked OCC type (break ties with compound)
        best_e = max(profile.keys(), key=lambda e: (profile[e][0], e in COMPOUND))
        alpha, intensity, term, (f_k, rule) = profile[best_e]
        label = mapping.get(best_e, best_e)

        # Build explanation
        explanation = self.explain(best_e, alpha, intensity, term, rule, d, p, pi, profile, theta)
        return label, best_e, alpha, intensity, term, explanation, profile

    def explain(self, e, alpha, intensity, term, rule, d, p, pi, profile, theta):
        """Generate human-readable, faithful explanation based on the fired rule."""
        if rule is None:
            return f"{e} (activation {alpha:.2f}, {term} intensity)"

        agent, d_term, p_term, _, _ = rule
        d_name = D_NAMES.get(d_term, d_term)
        d_deg = self.mf_eval(d, d_term)

        pi_val = pi[agent] if isinstance(pi, dict) else pi[AGENTS.index(agent)]

        parts = [f"{e} (activation {alpha:.2f}, {term} intensity):"]
        parts.append(f"the event is evaluated as {d_name} ({d_deg:.2f}),")
        parts.append(f"{agent.lower()} is responsible (prob {pi_val:.2f})")

        if p_term is not None:
            p_name = P_NAMES.get(p_term, p_term)
            p_deg = self.mf_eval(p, p_term)
            parts.append(f"and the action is evaluated as {p_name} ({p_deg:.2f}).")
        else:
            parts[-1] += "."

        # Secondary emotions
        secondaries = [
            f"{sec_e} ({profile[sec_e][0]:.2f})"
            for sec_e in profile
            if sec_e != e and profile[sec_e][0] >= theta
        ]
        if secondaries:
            parts.append(f"Secondary emotions: {', '.join(secondaries)}.")

        return " ".join(parts)


if __name__ == "__main__":
    engine = FuzzyOCCEngine()
    print(f"Loaded {len(engine.rules)} rules in FuzzyOCCEngine.")
    # Quick test
    label, occ_e, alpha, intensity, term, exp, prof = engine.decide(
        -0.80, -0.75, {"Self": 0.05, "Other": 0.90, "Circumstance": 0.05}, theta=0.3
    )
    print("Test decision:", label, occ_e, alpha, intensity, term)
    print("Explanation:", exp)
