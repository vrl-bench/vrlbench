"""A no-model example of the complete next-trial method interface."""

from vrlbench import ActorContext, CaseConfig, MemoryMethod, TrialRecord


class LastTrialContext(MemoryMethod):
    """Pass the latest failed trial through as one unmodified context.

    This demonstrates the API, not a proposed memory algorithm. A scheduler
    can produce its entire response here without splitting it into components.
    """

    def reset_case(self, config: CaseConfig) -> None:
        self.config = config
        self.last_failure = ""
        self.updates = 0

    def context_for_trial(self, trial_index: int) -> ActorContext:
        return ActorContext(system=self.last_failure if trial_index else "")

    def observe_trial(self, record: TrialRecord) -> None:
        if (record.success or record.trial_index + 1 >= self.config.total_trials
                or record.metadata.get("policy_evidence_eligible") is False):
            return
        self.last_failure = record.trajectory
        self.updates += 1

    def snapshot(self) -> dict:
        return {"method": "last_trial_context", "updates": self.updates,
                "last_failure": self.last_failure}


def create_method() -> LastTrialContext:
    return LastTrialContext()
