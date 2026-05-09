# Agent package.
#
# Keep this file light-weight so importing submodules like ``agent.q_agent``
# does not eagerly import optional dependencies (e.g., PyTorch for DQN).

__all__ = [
	"QLearningAgent",
	"DoubleQLearningAgent",
	"DQNAgent",
	"FixedTimeAgent",
]


def __getattr__(name):
	if name == "QLearningAgent":
		from .q_agent import QLearningAgent
		return QLearningAgent
	if name == "DoubleQLearningAgent":
		from .double_q_agent import DoubleQLearningAgent
		return DoubleQLearningAgent
	if name == "DQNAgent":
		from .dqn_agent import DQNAgent
		return DQNAgent
	if name == "FixedTimeAgent":
		from .fixed_time_agent import FixedTimeAgent
		return FixedTimeAgent
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
