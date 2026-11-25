"""Diagnostic agents.

An agent is called once per SDBench turn on one shared encounter:

    agent(messages, *, instruction="", final=False, context=None) -> dict

It returns "text" (one validated SDBench action), "model", and "usage" for its final model
call. Other model calls it made for the same action go in "previous_attempts" (retries) or
"agent_calls" (team members), each with its own "usage". MedicalPolicy wraps every agent, so
each design faces the same budget, turn and test limits; only its model calls differ.
"""

import json

from memory import DEFAULT_CARDS, ExperienceMemory
from models import client_for

from .evidence import EvidenceAgent
from .panel_team import PanelTeam
from .single import SingleAgent
from .teams import CentralizedTeam, DecentralizedTeam, IndependentTeam

PLAYBOOK = DEFAULT_CARDS.with_name("playbook.json")
EMBEDDING_CLIENT = "gpt-6-luna"  # retrieval uses the OpenAI key, whatever the agent model


def load_playbook():
    return json.loads(PLAYBOOK.read_text())["rules"]


def evidence_reconciled(client, model, agents=1):
    """The evidence-first agent: case state, closure gate and final reconciliation."""
    return EvidenceAgent(client, model, agents, reconcile=True)


def evidence_with_playbook(client, model, agents=1):
    """Ablation: the evidence-first agent plus the playbook rules, no retrieval."""
    return EvidenceAgent(client, model, agents, reconcile=True, playbook=load_playbook())


def evidence_with_memory(client, model, agents=1):
    """The final design: the evidence-first agent plus the playbook and 3 retrieved lesson cards."""
    return EvidenceAgent(client, model, agents, reconcile=True, playbook=load_playbook(),
                         memory=ExperienceMemory(client_for(EMBEDDING_CLIENT)))


def panel_team_with_memory(client, model, agents=5):
    """The panel team plus Dr. Experience (retrieval) and the playbook."""
    return PanelTeam(client, model, agents, memory=ExperienceMemory(client_for(EMBEDDING_CLIENT)),
                     playbook=load_playbook())


AGENTS = {"single": SingleAgent, "evidence_reconcile": evidence_reconciled,
          "evidence_playbook": evidence_with_playbook, "evidence_memory": evidence_with_memory,
          "independent": IndependentTeam, "decentralized": DecentralizedTeam, "centralized": CentralizedTeam,
          "panel_team": PanelTeam, "panel_team_memory": panel_team_with_memory}
SINGLE_DESIGNS = {"single", "evidence_reconcile", "evidence_playbook", "evidence_memory"}  # one agent each


def make_agent(design, model, agents=1):
    """Build an agent design with a number of agents, for a model from models.json."""
    return AGENTS[design](client_for(model), model, agents)
