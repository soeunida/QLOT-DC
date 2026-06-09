"""Equal-budget accept-only selection (public API).

A candidate policy is accepted only if it beats clean SADND at the *same* FP budget
by a margin; refined / clip-gain candidates must also beat their non-tuned
counterpart. Failed candidates fall back (never fabricated).
"""

from qlot_rms.sadnd_cap import equal_budget_accept_only_select

__all__ = ["equal_budget_accept_only_select"]
