from dataclasses import dataclass
from configs.definitions import CodesaveConfig

# Disables all codesaving functionality (no hassle, no reproducibility)
@dataclass
class NoCodesaveConfig(CodesaveConfig):
    force_manual_commit: bool = False
    autocommit: bool = False
    codesave_to_logs: bool = False

# Autocommits in every run (best code tracking, more cluttered git log)
@dataclass
class AutocommitCodesaveConfig(CodesaveConfig):
    force_manual_commit: bool = False,
    autocommit: bool = True,
    autocommit_push: bool = False,
    codesave_to_logs: bool = False,


# Commits to logs in every run (ok code tracking, cluttered git log in experiment logs)
@dataclass
class LogsCodesaveConfig(CodesaveConfig):
    force_manual_commit: bool = False
    autocommit: bool = False
    codesave_to_logs: bool = True
    codesave_push: bool = False

# Commits and pushes logs in every run (good code tracking, slightly slower due to pushing)
@dataclass
class PushLogsCodesaveConfig(CodesaveConfig):
    force_manual_commit: bool = False
    autocommit: bool = False
    codesave_to_logs: bool = True
    codesave_push: bool = True
