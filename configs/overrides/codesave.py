from dataclasses import dataclass

# Also imports resolvers "resolve_commit_hash" and "resolve_codesave"
from configs.definitions import CodesaveConfig

# Disables all codesaving functionality (no hassle, no reproducibility)
@dataclass
class NoCodesaveConfig(CodesaveConfig):
    force_manual_commit = False
    autocommit = False
    codesave_to_logs = False

# Autocommits in every run (best code tracking, more cluttered git log)
@dataclass
class AutocommitCodesaveConfig(CodesaveConfig):
    force_manual_commit = False
    autocommit = True
    autocommit_push = False

# Commits to logs in every run (ok code tracking, cluttered git log in experiment logs)
@dataclass
class LogsCodesaveConfig(CodesaveConfig):
    force_manual_commit = False
    autocommit = False
    codesave_to_logs = True

# Commits and pushes logs in every run (good code tracking, slightly slower due to pushing)
@dataclass
class PushLogsCodesaveConfig(CodesaveConfig):
    force_manual_commit = False
    autocommit = False
    codesave_to_logs = True
    codesave_push = True